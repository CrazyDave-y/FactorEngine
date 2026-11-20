#pragma once

#include "factor_engine/cuda_runtime.hpp"
#include "factor_engine/plan_validation.hpp"
#include "factor_engine/scalar_math.hpp"

#include <cuda_runtime.h>
#include <cub/device/device_segmented_radix_sort.cuh>
#include "metric_dispatch.cuh"
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <cfloat>
#include <climits>
#include "factor_engine/centered_pair.hpp"
#include "factor_engine/centered_unary.hpp"

#include "pointwise.cuh"
#include "temporal.cuh"
#include "rank_segmented.cuh"

namespace {
using namespace factor_engine::cuda;

// 普通 RankIC 输入超出融合内核容量或共享内存配额时，才使用分段排序。
bool supports_pair_metrics(int time, int max_shared) {
    if (time > factor_engine::metrics::LongCapacity)
        return false;
    const size_t dynamic = time <= factor_engine::metrics::ShortCapacity
                               ? factor_engine::metrics::ShortSharedBytes
                               : factor_engine::metrics::LongSharedBytes;
    return factor_engine::shared_storage_fits(
        dynamic, sizeof(factor_engine::metrics::detail::RankMetadata), max_shared);
}

inline cudaError_t consume_cuda_oom(cudaError_t error) {
    if (error != cudaErrorMemoryAllocation)
        return error;
    // 已知 OOM 会留在线程错误状态中；先读取并清除该状态，才能安全缩批重试。
    // 若同时发现其他设备错误，应返回该错误，不能按可恢复的 OOM 处理。
    const auto pending = cudaGetLastError();
    return pending == cudaSuccess ? error : pending;
}

template <typename T> void release_workspace(T*& pointer) {
    if (pointer)
        cudaFree(pointer);
    pointer = nullptr;
}

__global__ void pack_roots_kernel(const float* const* roots, float* packed, long long cells,
                                  long long total) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total)
        packed[i] = roots[i / cells][i % cells];
}

struct RankWorkspace {
    float* keys_in = nullptr;
    float* keys_out = nullptr;
    int* indices_in = nullptr;
    int* indices_out = nullptr;
    int* offsets = nullptr;
    int* counts = nullptr;
    double* lane_scores = nullptr;
    void* temporary = nullptr;
    size_t temporary_bytes = 0;
    size_t memory_bytes = 0;
    int root_capacity = 0;
    int segments = 0;
    int items = 0;

    void reset() {
        release_workspace(keys_in);
        release_workspace(keys_out);
        release_workspace(indices_in);
        release_workspace(indices_out);
        release_workspace(offsets);
        release_workspace(counts);
        release_workspace(lane_scores);
        release_workspace(temporary);
        *this = {};
    }
};

struct Runtime {
    int lanes = 0, time = 0;
    long long cells = 0;
    long long storage_cells = 0;

    long long allocation_cells() const {
        return storage_cells ? storage_cells : cells;
    }

    cudaError_t allocation_error = cudaSuccess;

    template <typename T> bool allocate_workspace(T** pointer, size_t bytes) {
        allocation_error = consume_cuda_oom(cudaMalloc((void**)pointer, bytes));
        return allocation_error == cudaSuccess;
    }

    int allocation_status(int fallback) const {
        return allocation_error == cudaSuccess
                   ? fallback
                   : factor_engine::runtime_status::CudaErrorBase - (int)allocation_error;
    }

    std::vector<float*> fields;
    float* workspace = nullptr;
    int slots = 0;
    double* prefix_sum = nullptr;
    uint64_t* prefix_meta = nullptr;
    float* packed_roots = nullptr;
    int packed_root_capacity = 0;
    const float** root_ptrs_device = nullptr;
    int root_capacity = 0;
    std::vector<const float*> root_ptrs_host;
    RankWorkspace rank;
    double* queue_front = nullptr;
    double* queue_prefix = nullptr;
    factor_engine::UnaryMoments* std_front = nullptr;
    factor_engine::UnaryMoments* std_prefix = nullptr;
    int sms = 0, max_shared = 0;
    std::vector<factor_engine::PlanNodePOD> prepared_nodes;
    std::vector<int> prepared_roots;
    std::vector<unsigned char> fusion_schedule;

    struct Counters {
        long long kernel_launches = 0;
        long long h2d_bytes = 0;
        long long root_export_bytes = 0;
        long long rank_workspace_bytes = 0;
    } telemetry;

    ~Runtime() {
        release_workspace(std_front);
        release_workspace(std_prefix);
        release_workspace(queue_front);
        release_workspace(queue_prefix);
        release_workspace(workspace);
        release_workspace(prefix_sum);
        release_workspace(prefix_meta);
        release_workspace(packed_roots);
        release_workspace(root_ptrs_device);
        rank.reset();
        // fields 只是输入区的地址，输入区由 CUDABuffers 释放。
    }

    template <typename T>
    bool reserve_workspace(T*& pointer, int& capacity, int requested, size_t bytes) {
        if (requested <= capacity)
            return true;
        release_workspace(pointer);
        capacity = 0;
        if (!allocate_workspace(&pointer, bytes))
            return false;
        capacity = requested;
        return true;
    }

    template <typename T, typename U> bool allocate_pair(T*& first, U*& second, size_t bytes) {
        if (!allocate_workspace(&first, bytes))
            return false;
        if (allocate_workspace(&second, bytes))
            return true;
        release_workspace(first);
        return false;
    }

    bool ensure_slots(int requested) {
        if (!factor_engine::valid_float_storage(allocation_cells(), requested))
            return false;
        return reserve_workspace(workspace, slots, requested,
                                 (size_t)requested * allocation_cells() * sizeof(float));
    }

    bool ensure_pair_queue(int requested) {
        if (requested > time || queue_front)
            return true;
        if ((unsigned long long)allocation_cells() >
            std::numeric_limits<size_t>::max() / sizeof(factor_engine::PairMoments))
            return false;
        return allocate_pair(queue_front, queue_prefix,
                             (size_t)allocation_cells() * sizeof(factor_engine::PairMoments));
    }

    bool ensure_std_queue(int requested) {
        if (requested > time || std_front)
            return true;
        if ((unsigned long long)allocation_cells() >
            std::numeric_limits<size_t>::max() / sizeof(factor_engine::UnaryMoments))
            return false;
        return allocate_pair(std_front, std_prefix,
                             (size_t)allocation_cells() * sizeof(factor_engine::UnaryMoments));
    }

    bool ensure_prefix() {
        if (prefix_sum)
            return true;
        if ((unsigned long long)allocation_cells() >
            std::numeric_limits<size_t>::max() / sizeof(double))
            return false;
        return allocate_pair(prefix_sum, prefix_meta, (size_t)allocation_cells() * sizeof(double));
    }

    bool ensure_root_capacity(int requested) {
        if (requested <= 0)
            return false;
        if (requested <= root_capacity)
            return true;
        if ((size_t)requested > std::numeric_limits<size_t>::max() / sizeof(float*))
            return false;
        root_ptrs_host.resize((size_t)requested);
        return reserve_workspace(root_ptrs_device, root_capacity, requested,
                                 (size_t)requested * sizeof(float*));
    }

    bool ensure_packed_capacity(int requested) {
        if (requested <= packed_root_capacity)
            return true;
        if (!factor_engine::valid_float_storage(allocation_cells(), requested))
            return false;
        return reserve_workspace(packed_roots, packed_root_capacity, requested,
                                 (size_t)requested * allocation_cells() * sizeof(float));
    }

    bool ensure_rank_capacity(int requested) {
        long long item_count = 0, segment_count = 0;
        const bool fused = supports_pair_metrics(time, max_shared);
        const int capacity_lanes = (int)(allocation_cells() / time);
        if (!(fused ? factor_engine::rank_block_layout(requested, capacity_lanes, time, item_count,
                                                       segment_count)
                    : factor_engine::rank_layout(requested, capacity_lanes, time, item_count,
                                                 segment_count)))
            return false;
        if (requested <= rank.root_capacity)
            return true;
        if (!ensure_root_capacity(requested))
            return false;

        rank.reset();
        rank.items = fused ? 0 : (int)item_count;
        rank.segments = (int)segment_count;
        if (fused) {
            rank.temporary_bytes = factor_engine::metrics::pair_metric_scratch_bytes(
                requested, capacity_lanes, time, 1);
            const size_t output_bytes =
                (size_t)segment_count * factor_engine::metrics::MetricCount * sizeof(double);
            if (!allocate_workspace(&rank.lane_scores, output_bytes) ||
                (rank.temporary_bytes &&
                 !allocate_workspace(&rank.temporary, rank.temporary_bytes))) {
                rank.reset();
                return false;
            }
            rank.memory_bytes = output_bytes + rank.temporary_bytes;
            rank.root_capacity = requested;
            return true;
        }

        // 分段排序仍按原时间位置保存因子秩；block 排序不需要这份暂存。
        if (!ensure_packed_capacity(requested))
            return false;
        auto allocate = [&](void** pointer, size_t bytes) {
            if (allocate_workspace(pointer, bytes))
                return true;
            rank.reset();
            return false;
        };
        if (!allocate((void**)&rank.keys_in, (size_t)rank.items * sizeof(float)))
            return false;
        if (!allocate((void**)&rank.keys_out, (size_t)rank.items * sizeof(float)))
            return false;
        if (!allocate((void**)&rank.indices_in, (size_t)rank.items * sizeof(int)))
            return false;
        if (!allocate((void**)&rank.indices_out, (size_t)rank.items * sizeof(int)))
            return false;
        if (!allocate((void**)&rank.offsets, ((size_t)rank.segments + 1) * sizeof(int)))
            return false;
        if (!allocate((void**)&rank.counts, (size_t)rank.segments * sizeof(int)))
            return false;
        if (!allocate((void**)&rank.lane_scores, (size_t)rank.segments * sizeof(double)))
            return false;
        if (cub::DeviceSegmentedRadixSort::SortPairs(
                nullptr, rank.temporary_bytes, rank.keys_in, rank.keys_out, rank.indices_in,
                rank.indices_out, rank.items, rank.segments, rank.offsets,
                rank.offsets + 1) != cudaSuccess) {
            rank.reset();
            return false;
        }
        if (!allocate(&rank.temporary, rank.temporary_bytes))
            return false;

        rank.memory_bytes = (size_t)rank.items * (sizeof(float) * 2 + sizeof(int) * 2) +
                            ((size_t)rank.segments + 1) * sizeof(int) +
                            (size_t)rank.segments * (sizeof(int) + sizeof(double)) +
                            rank.temporary_bytes;

        rank_offsets_kernel<<<factor_engine::ceil_blocks((long long)rank.segments + 1), THREADS>>>(
            rank.offsets, rank.segments, time);
        if (cudaGetLastError() != cudaSuccess) {
            rank.reset();
            return false;
        }
        rank.root_capacity = requested;
        return true;
    }

    void count_kernel() {
        ++telemetry.kernel_launches;
    }

    // 只核算自身拥有的分配；fields 借用输入区，不在这里重复计费。
    size_t workspace_bytes() const {
        const auto capacity = (size_t)allocation_cells();
        return (std_front ? capacity * sizeof(factor_engine::UnaryMoments) * 2 : 0) +
               (queue_front ? capacity * sizeof(factor_engine::PairMoments) * 2 : 0) +
               (size_t)slots * capacity * sizeof(float) +
               (prefix_sum ? capacity * (sizeof(double) + sizeof(uint64_t)) : 0) +
               (size_t)packed_root_capacity * capacity * sizeof(float) +
               (size_t)root_capacity * sizeof(float*) + rank.memory_bytes;
    }
};

// 两个调用方已执行 validate_plan_nodes，节点输入、字段和 slot 均已通过校验。
int execute_nodes(Runtime* rt, const factor_engine::PlanNodePOD* nodes, int n_nodes, int slot_count,
                  std::vector<Value>& values, const int* roots, int n_roots) {
    if (rt->sms <= 0)
        return factor_engine::runtime_status::InvalidArgument;

    rt->allocation_error = cudaSuccess;
    if (!rt->ensure_slots(slot_count))
        return rt->allocation_status(factor_engine::runtime_status::InvalidWorkspace);
    values.assign((size_t)n_nodes, {});

    // 同一计划跨 batch 复用调度；比较内容，避免调用方释放后地址被复用。
    if (rt->prepared_nodes.size() != (size_t)n_nodes ||
        rt->prepared_roots.size() != (size_t)n_roots ||
        std::memcmp(rt->prepared_nodes.data(), nodes, n_nodes * sizeof(*nodes)) != 0 ||
        !std::equal(rt->prepared_roots.begin(), rt->prepared_roots.end(), roots)) {
        // 所有分配完成后再替换，分配失败不能留下半张新 schedule。
        auto next_fusion = build_pointwise_fusion_schedule(nodes, n_nodes, roots, n_roots);
        std::vector<factor_engine::PlanNodePOD> next_nodes(nodes, nodes + n_nodes);
        std::vector<int> next_roots(roots, roots + n_roots);
        rt->fusion_schedule.swap(next_fusion);
        rt->prepared_nodes.swap(next_nodes);
        rt->prepared_roots.swap(next_roots);
    }

    const auto& fusion = rt->fusion_schedule;
    for (int i = 0; i < n_nodes; ++i) {
        const auto& n = nodes[i];
        const auto op = static_cast<PlanOp>(n.op);
        if (fusion[i] >= 2 && i + fusion[i] <= n_nodes) {
            const auto error = launch_pointwise_fusion(nodes, i, fusion[i], values, rt->workspace,
                                                       rt->cells, rt->allocation_cells());
            if (error != cudaSuccess)
                return factor_engine::runtime_status::CudaErrorBase - (int)error;
            rt->count_kernel();
            i += fusion[i] - 1;
            continue;
        }
        if (op == PlanOp::Terminal) {
            values[i] = {false, 0, rt->fields[n.field_index]};
            continue;
        }
        if (op == PlanOp::Const) {
            values[i] = {true, n.param_f, nullptr};
            continue;
        }

        auto in = [&](int j) -> const Value& { return values[j]; };
        if (op >= PlanOp::Neg && op <= PlanOp::SignedLog1p && in(n.in0).scalar) {
            values[i] = {true, factor_engine::scalar_math::scalar_unary(op, in(n.in0).s), nullptr};
            continue;
        }
        if (op >= PlanOp::Add && op <= PlanOp::Max && in(n.in0).scalar && in(n.in1).scalar) {
            values[i] = {true,
                         factor_engine::scalar_math::scalar_binary(op, in(n.in0).s, in(n.in1).s),
                         nullptr};
            continue;
        }

        float* out = rt->workspace + (long long)n.slot * rt->allocation_cells();

        if (op >= PlanOp::Neg && op <= PlanOp::Max) {
            const Value& a = in(n.in0);
            const Value b = (n.in1 >= 0 ? in(n.in1) : Value{});
            const int blocks = (int)((rt->cells + THREADS - 1) / THREADS);
            elementwise_kernel<<<blocks, THREADS>>>(op, a.p, a.s, a.scalar, b.p, b.s, b.scalar, out,
                                                    rt->cells);
            rt->count_kernel();
        } else if (op == PlanOp::Delay || op == PlanOp::Diff) {
            const int blocks = (int)((rt->cells + THREADS - 1) / THREADS);
            delay_diff_kernel<<<blocks, THREADS>>>(in(n.in0).p, out, rt->lanes, rt->time, n.param_i,
                                                   op == PlanOp::Diff);
            rt->count_kernel();
        } else if (op == PlanOp::Ema) {
            ema_tiled_kernel<<<rt->lanes, EMA_THREADS>>>(in(n.in0).p, out, rt->lanes, rt->time,
                                                         n.param_i);
            rt->count_kernel();
        } else if (op == PlanOp::RollMin || op == PlanOp::RollMax) {
            const int w = n.param_i;
            if (w > DIRECT_ROLLING_WINDOW_LIMIT && w <= rt->time) {
                if (!rt->ensure_prefix())
                    return rt->allocation_status(
                        factor_engine::runtime_status::TemporalLaunchFailed);
                const long long epochs = ((long long)rt->time + w - 1) / w;
                const int per_block = std::max(1, THREADS / w);
                const long long groups = (epochs + per_block - 1) / per_block;
                const long long grid = 2 * (long long)rt->lanes * groups;
                if (grid > std::numeric_limits<int>::max())
                    return factor_engine::runtime_status::InvalidGrid;
                // 时间算子在同一 stream 顺序执行；极值复用两个前缀缓冲，无需第三份空间。
                rolling_extrema_summaries_kernel<<<(int)grid, THREADS>>>(
                    in(n.in0).p, reinterpret_cast<float*>(rt->prefix_sum),
                    reinterpret_cast<float*>(rt->prefix_meta), rt->lanes, rt->time, w,
                    op == PlanOp::RollMin);
                rt->count_kernel();
            }
            rolling_extrema_output_kernel<<<factor_engine::ceil_blocks(rt->cells), THREADS>>>(
                in(n.in0).p, reinterpret_cast<float*>(rt->prefix_sum),
                reinterpret_cast<float*>(rt->prefix_meta), out, rt->lanes, rt->time, w,
                op == PlanOp::RollMin);
            rt->count_kernel();
        } else if (op >= PlanOp::RollSum && op <= PlanOp::RollStd) {
            if (op == PlanOp::RollStd) {
                if (!rt->ensure_std_queue(n.param_i))
                    return rt->allocation_status(
                        factor_engine::runtime_status::TemporalLaunchFailed);
                const long long epochs =
                    n.param_i > rt->time ? 1 : ((long long)rt->time + n.param_i - 1) / n.param_i;
                const long long units = 2 * (long long)rt->lanes * epochs,
                                target = (long long)rt->sms * 8;
                int threads = 1;
                while (threads < 128 && threads < (units + target - 1) / target)
                    threads *= 2;
                const long long grid = (units + threads - 1) / threads;
                if (grid > std::numeric_limits<int>::max())
                    return factor_engine::runtime_status::InvalidGrid;
                const int blocks = (int)grid;
                if (n.param_i <= rt->time) {
                    rolling_std_summaries_kernel<<<blocks, threads>>>(
                        in(n.in0).p, rt->std_front, rt->std_prefix, rt->lanes, rt->time, n.param_i);
                    if (cudaGetLastError() != cudaSuccess)
                        return factor_engine::runtime_status::TemporalLaunchFailed;
                    rt->count_kernel();
                }
                const long long output_grid = (rt->cells + THREADS - 1) / THREADS;
                if (output_grid > std::numeric_limits<int>::max())
                    return factor_engine::runtime_status::InvalidGrid;
                rolling_std_native_order_kernel<<<(int)output_grid, THREADS>>>(
                    out, rt->std_front, rt->std_prefix, rt->lanes, rt->time, n.param_i);
                rt->count_kernel();
            } else if (n.param_i > DIRECT_ROLLING_WINDOW_LIMIT) {
                const int blocks = rt->lanes;
                if (!rt->ensure_prefix())
                    return rt->allocation_status(
                        factor_engine::runtime_status::TemporalLaunchFailed);
                rolling_unary_prefix_kernel<<<blocks, THREADS>>>(in(n.in0).p, out, rt->prefix_sum,
                                                                 rt->prefix_meta, rt->lanes,
                                                                 rt->time, n.param_i, op);
                rt->count_kernel();
            } else {
                const int blocks = (int)((rt->cells + THREADS - 1) / THREADS);
                rolling_unary_output_kernel<<<blocks, THREADS>>>(in(n.in0).p, out, rt->lanes,
                                                                 rt->time, n.param_i, op);
                rt->count_kernel();
            }
        } else if (op == PlanOp::Cov || op == PlanOp::Corr) {
            {
                if (!rt->ensure_pair_queue(n.param_i))
                    return rt->allocation_status(
                        factor_engine::runtime_status::PairWorkspaceFailed);
                const long long epochs =
                    n.param_i > rt->time ? 1 : ((long long)rt->time + n.param_i - 1) / n.param_i;
                const long long units = 2 * (long long)rt->lanes * epochs,
                                target = (long long)rt->sms * 8;
                int threads = 1;
                while (threads < 128 && threads < (units + target - 1) / target)
                    threads *= 2;
                const long long grid = (units + threads - 1) / threads;
                if (grid > std::numeric_limits<int>::max())
                    return factor_engine::runtime_status::InvalidGrid;
                const int blocks = (int)grid;
                if (n.param_i <= rt->time) {
                    rolling_pair_summaries_kernel<<<blocks, threads>>>(
                        in(n.in0).p, in(n.in1).p, rt->queue_front, rt->queue_prefix, rt->lanes,
                        rt->time, n.param_i);
                    rt->count_kernel();
                }
                const long long output_grid = (rt->cells + THREADS - 1) / THREADS;
                if (output_grid > std::numeric_limits<int>::max())
                    return factor_engine::runtime_status::InvalidGrid;
                rolling_pair_centered_kernel<<<(int)output_grid, THREADS>>>(
                    out, rt->queue_front, rt->queue_prefix, rt->lanes, rt->time, n.param_i,
                    op == PlanOp::Corr);
                rt->count_kernel();
            }
        } else
            return factor_engine::runtime_status::UnsupportedOperation;

        const auto error = cudaGetLastError();
        if (error != cudaSuccess)
            return factor_engine::runtime_status::CudaErrorBase - (int)error;
        values[i] = {false, 0, out};
    }
    return factor_engine::runtime_status::Success;
}

int validate_roots(const std::vector<Value>& values, const int* roots, int n_roots) {
    for (int r = 0; r < n_roots; ++r) {
        if (roots[r] < 0 || roots[r] >= (int)values.size() || values[roots[r]].scalar ||
            !values[roots[r]].p)
            return factor_engine::runtime_status::InvalidRoot;
    }
    return factor_engine::runtime_status::Success;
}

int prepare_root_pointers(Runtime* rt, const std::vector<Value>& values, const int* roots,
                          int n_roots) {
    if (!rt->ensure_root_capacity(n_roots))
        return rt->allocation_status(factor_engine::runtime_status::InvalidGrid);

    for (int r = 0; r < n_roots; ++r)
        rt->root_ptrs_host[(size_t)r] = values[roots[r]].p;
    if (cudaMemcpy(rt->root_ptrs_device, rt->root_ptrs_host.data(),
                   (size_t)n_roots * sizeof(float*), cudaMemcpyHostToDevice) != cudaSuccess)
        return factor_engine::runtime_status::PointerCopyFailed;

    rt->telemetry.h2d_bytes += (long long)((size_t)n_roots * sizeof(float*));
    return factor_engine::runtime_status::Success;
}

int pack_root_values(Runtime* rt, const std::vector<Value>& values, const int* roots, int n_roots) {
    const int rc = prepare_root_pointers(rt, values, roots, n_roots);
    if (rc)
        return rc;
    if (!rt->ensure_packed_capacity(n_roots))
        return rt->allocation_status(factor_engine::runtime_status::InvalidGrid);

    const long long total = (long long)n_roots * rt->cells;
    const int blocks = (int)((total + THREADS - 1) / THREADS);
    pack_roots_kernel<<<blocks, THREADS>>>(rt->root_ptrs_device, rt->packed_roots, rt->cells,
                                           total);
    if (cudaGetLastError() != cudaSuccess)
        return factor_engine::runtime_status::RootPackFailed;
    rt->count_kernel();
    return factor_engine::runtime_status::Success;
}

// target 和因子已经在设备上，结果保留在 rank.lane_scores，交给外层一次回传。
int score_packed_rank_ic(Runtime* rt, int n_roots, const float* target, const uint8_t* eligible) {
    const auto pointers = rt->root_ptrs_device;
    auto& rank = rt->rank;
    if (supports_pair_metrics(rt->time, rt->max_shared)) {
        const int error = factor_engine::metrics::launch_pair_metrics(
            pointers, target, eligible, rank.lane_scores, n_roots, rt->lanes, rt->time, 1,
            rank.temporary, rank.temporary_bytes);
        if (error != cudaSuccess)
            return factor_engine::runtime_status::CudaErrorBase - error;
        rt->count_kernel();
        rt->telemetry.rank_workspace_bytes = (long long)rank.memory_bytes;
        return factor_engine::runtime_status::Success;
    }

    const int cells = static_cast<int>(rt->cells);
    const int items = n_roots * cells, segments = n_roots * rt->lanes;
    const int item_blocks = factor_engine::ceil_blocks(items),
              segment_blocks = factor_engine::ceil_blocks(segments);

    factor_rank_keys_kernel<<<item_blocks, THREADS>>>(pointers, target, rank.keys_in,
                                                      rank.indices_in, cells, items);
    rt->count_kernel();
    size_t temporary_bytes = rank.temporary_bytes;
    if (cub::DeviceSegmentedRadixSort::SortPairs(rank.temporary, temporary_bytes, rank.keys_in,
                                                 rank.keys_out, rank.indices_in, rank.indices_out,
                                                 items, segments, rank.offsets, rank.offsets + 1, 0,
                                                 32) != cudaSuccess)
        return factor_engine::runtime_status::FactorSortFailed;
    rank_counts_kernel<<<segment_blocks, THREADS>>>(rank.keys_out, rank.counts, segments, rt->time);
    rt->count_kernel();
    average_tie_ranks_kernel<<<item_blocks, THREADS>>>(
        rank.keys_out, rank.indices_out, reinterpret_cast<uint32_t*>(rt->packed_roots), items,
        rt->time);
    rt->count_kernel();

    // target 也按各因子的有效交集取点；当前实现为每个 root/lane 单独排序。
    target_rank_keys_kernel<<<item_blocks, THREADS>>>(reinterpret_cast<uint32_t*>(rt->packed_roots),
                                                      target, rank.keys_in, rank.indices_in, cells,
                                                      items);
    rt->count_kernel();

    temporary_bytes = rank.temporary_bytes;
    if (cub::DeviceSegmentedRadixSort::SortPairs(rank.temporary, temporary_bytes, rank.keys_in,
                                                 rank.keys_out, rank.indices_in, rank.indices_out,
                                                 items, segments, rank.offsets, rank.offsets + 1, 0,
                                                 32) != cudaSuccess)
        return factor_engine::runtime_status::TargetSortFailed;
    average_tie_ranks_kernel<<<item_blocks, THREADS>>>(rank.keys_out, rank.indices_out,
                                                       reinterpret_cast<uint32_t*>(rank.indices_in),
                                                       items, rt->time);
    rt->count_kernel();

    lane_rank_ic_kernel<<<segments, THREADS>>>(reinterpret_cast<uint32_t*>(rt->packed_roots),
                                               reinterpret_cast<uint32_t*>(rank.indices_in),
                                               rank.counts, rank.lane_scores, segments, rt->time);
    rt->count_kernel();
    if (cudaGetLastError() != cudaSuccess)
        return factor_engine::runtime_status::RankLaunchFailed;

    rt->telemetry.rank_workspace_bytes = (long long)(rank.memory_bytes);
    return factor_engine::runtime_status::Success;
}

int execute_score(Runtime* rt, const factor_engine::PlanNodePOD* nodes, int n_nodes, int slot_count,
                  const int* roots, int n_roots, const float* device_target,
                  const uint8_t* eligible) {
    if (!rt || !nodes || n_nodes <= 0 || slot_count < 0 || !roots || n_roots <= 0 || !device_target)
        return factor_engine::runtime_status::InvalidArgument;

    long long items = 0, segments = 0;
    // 在遍历 root 前先检查规模，避免按错误长度读取指针或分配工作区。
    const bool block = supports_pair_metrics(rt->time, rt->max_shared);
    if (!(block ? factor_engine::rank_block_layout(n_roots, rt->lanes, rt->time, items, segments)
                : factor_engine::rank_layout(n_roots, rt->lanes, rt->time, items, segments)))
        return factor_engine::runtime_status::InvalidRankLayout;
    int rc = factor_engine::validate_plan_nodes(nodes, n_nodes, (int)rt->fields.size(), slot_count,
                                                roots, n_roots);
    if (rc)
        return rc;
    if (!factor_engine::valid_float_storage(rt->cells, slot_count))
        return factor_engine::runtime_status::InvalidWorkspace;

    rt->telemetry = {};
    std::vector<Value> values;
    if ((rc = execute_nodes(rt, nodes, n_nodes, slot_count, values, roots, n_roots)))
        return rc;
    if ((rc = validate_roots(values, roots, n_roots)))
        return rc;
    if (!rt->ensure_rank_capacity(n_roots))
        return rt->allocation_status(factor_engine::runtime_status::InvalidRankLayout);
    if ((rc = prepare_root_pointers(rt, values, roots, n_roots)))
        return rc;
    return score_packed_rank_ic(rt, n_roots, device_target, eligible);
}

int workspace_requirement(const Runtime& rt, int slots, int roots, int unary_prefix,
                          int max_pair_window, int max_std_window, long long& bytes) {
    if (slots < 0 || max_pair_window < 0 || max_pair_window > rt.time || max_std_window < 0 ||
        max_std_window > rt.time || (unary_prefix != 0 && unary_prefix != 1))
        return factor_engine::runtime_status::InvalidArgument;

    long long items = 0, segments = 0;
    const bool fused = supports_pair_metrics(rt.time, rt.max_shared);
    const long long capacity = rt.allocation_cells();
    const int capacity_lanes = (int)(capacity / rt.time);
    if (!(fused ? factor_engine::rank_block_layout(roots, capacity_lanes, rt.time, items, segments)
                : factor_engine::rank_layout(roots, capacity_lanes, rt.time, items, segments)))
        return factor_engine::runtime_status::InvalidRankLayout;
    const int slot_capacity = std::max(slots, rt.slots),
              root_capacity = std::max(roots, rt.root_capacity);
    if (!factor_engine::valid_float_storage(capacity, slot_capacity))
        return factor_engine::runtime_status::InvalidWorkspace;

    unsigned long long desired_rank = rt.rank.memory_bytes;
    const int packed_capacity = std::max(rt.packed_root_capacity, fused ? 0 : roots);
    if (roots > rt.rank.root_capacity && fused) {
        desired_rank =
            (unsigned long long)segments * factor_engine::metrics::MetricCount * sizeof(double) +
            factor_engine::metrics::pair_metric_scratch_bytes(roots, capacity_lanes, rt.time, 1);
    } else if (roots > rt.rank.root_capacity) {
        size_t temporary = 0;
        // 用 CUB 的空存储查询拿到空间大小，不分配设备缓冲区。
        // 传带类型的空指针，避免对空指针做偏移计算。
        if (cub::DeviceSegmentedRadixSort::SortPairs(
                nullptr, temporary, (const float*)nullptr, (float*)nullptr, (const int*)nullptr,
                (int*)nullptr, (int)items, (int)segments, (const int*)nullptr,
                (const int*)nullptr) != cudaSuccess)
            return factor_engine::runtime_status::FactorSortFailed;
        desired_rank = (unsigned long long)items * (2 * sizeof(float) + 2 * sizeof(int)) +
                       ((unsigned long long)segments + 1) * sizeof(int) +
                       (unsigned long long)segments * (sizeof(int) + sizeof(double)) + temporary;
    }

    // DAG 和前缀扩容时还留着旧排序空间；替换排序缓冲区前会先释放旧的。
    unsigned long long total = std::max(desired_rank, (unsigned long long)rt.rank.memory_bytes);
    const auto add = [&](unsigned long long value) {
        if (value > (unsigned long long)std::numeric_limits<long long>::max() - total)
            return false;
        total += value;
        return true;
    };
    if (!add((unsigned long long)slot_capacity * capacity * sizeof(float)) ||
        !add((unsigned long long)packed_capacity * capacity * sizeof(float)) ||
        !add((unsigned long long)root_capacity * sizeof(float*)))
        return factor_engine::runtime_status::InvalidWorkspace;
    if ((rt.prefix_sum || unary_prefix) &&
        !add((unsigned long long)capacity * (sizeof(double) + sizeof(uint64_t))))
        return factor_engine::runtime_status::InvalidWorkspace;
    if (max_pair_window || rt.queue_front) {
        if (!add((unsigned long long)capacity * sizeof(factor_engine::PairMoments)) ||
            !add((unsigned long long)capacity * sizeof(factor_engine::PairMoments)))
            return factor_engine::runtime_status::InvalidWorkspace;
    }
    if ((max_std_window || rt.std_front) &&
        !add((unsigned long long)capacity * sizeof(factor_engine::UnaryMoments) * 2))
        return factor_engine::runtime_status::InvalidWorkspace;

    bytes = (long long)total;
    return factor_engine::runtime_status::Success;
}

int export_roots(Runtime& rt, const factor_engine::PlanNodePOD* nodes, int n_nodes, int slot_count,
                 const int* roots, int n_roots, float* host_out) {
    if (!nodes || n_nodes <= 0 || slot_count < 0 || !roots || n_roots <= 0 || !host_out)
        return factor_engine::runtime_status::InvalidArgument;
    if (!factor_engine::valid_float_storage(rt.cells, slot_count) ||
        !factor_engine::valid_float_storage(rt.cells, n_roots))
        return factor_engine::runtime_status::InvalidGrid;
    const long long total = (long long)n_roots * rt.cells;
    if ((total + THREADS - 1) / THREADS > std::numeric_limits<int>::max())
        return factor_engine::runtime_status::InvalidGrid;
    int rc = factor_engine::validate_plan_nodes(nodes, n_nodes, (int)rt.fields.size(), slot_count,
                                                roots, n_roots);
    if (rc)
        return rc;

    rt.telemetry = {};
    std::vector<Value> values;
    rc = execute_nodes(&rt, nodes, n_nodes, slot_count, values, roots, n_roots);
    if (rc)
        return rc;
    if ((rc = validate_roots(values, roots, n_roots)))
        return rc;

    const size_t bytes = (size_t)rt.cells * sizeof(float);
    if ((size_t)rt.cells > std::numeric_limits<size_t>::max() / sizeof(float) / (size_t)n_roots)
        return factor_engine::runtime_status::InvalidGrid;
    const size_t export_bytes = (size_t)n_roots * bytes;
    if (n_roots == 1) {
        if (cudaMemcpy(host_out, values[roots[0]].p, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
            return factor_engine::runtime_status::OutputCopyFailed;
    } else {
        if ((rc = pack_root_values(&rt, values, roots, n_roots)))
            return rc;
        if (cudaMemcpy(host_out, rt.packed_roots, export_bytes, cudaMemcpyDeviceToHost) !=
            cudaSuccess)
            return factor_engine::runtime_status::OutputCopyFailed;
    }

    rt.telemetry.root_export_bytes = (long long)export_bytes;
    return factor_engine::runtime_status::Success;
}

} // 命名空间
