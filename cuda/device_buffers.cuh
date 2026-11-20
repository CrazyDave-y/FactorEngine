#pragma once

// 一次分配输入区；一个对象管理输入和计算工作区。
#include "resident_runtime.cuh"
#include "metric_workspace.cuh"
#include <memory>

namespace {
struct CUDABuffers {
    int lanes = 0, time = 0, sms = 0, active_lanes = 0, max_shared = 0;
    int next_upload = 0;
    bool ready = false;
    long long cells = 0, bytes = 0, budget = 0;
    void* inputs = nullptr;
    std::vector<float*> fields;
    uint8_t* valid = nullptr;
    float* target = nullptr;
    std::unique_ptr<Runtime> work;
    std::unique_ptr<factor_engine::metrics::MetricWorkspace> metrics;

    ~CUDABuffers() {
        work.reset();
        if (inputs)
            cudaFree(inputs);
    }

    int set_batch_width(int width) {
        if (work && work->lanes == width)
            return factor_engine::runtime_status::Success;
        if (cudaDeviceSynchronize() != cudaSuccess)
            return factor_engine::runtime_status::DeviceError;
        if (work) {
            // 只改变有效范围；slot 和算子临时区按输入容量保留，尾批无需重建。
            work->lanes = width;
            work->cells = (long long)width * time;
            return factor_engine::runtime_status::Success;
        }

        auto next = std::make_unique<Runtime>();
        next->lanes = width;
        next->time = time;
        next->cells = (long long)width * time;
        next->storage_cells = cells;
        next->fields.resize(fields.size(), nullptr);
        next->sms = sms;
        next->max_shared = max_shared;
        work = std::move(next);
        return factor_engine::runtime_status::Success;
    }

    long long workspace_bytes() const {
        if (!work)
            return 0;
        return (long long)work->workspace_bytes();
    }
};

__global__ void sanitize_field_kernel(float* field, uint8_t* valid, long long cells,
                                      bool last_field) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells)
        return;

    if (!valid[i] || !isfinite(field[i]))
        field[i] = CUDART_NAN_F;

    // 各字段依次清洗；最后一个字段清洗时复用 mask 空间，省去额外缓冲和清零 kernel。
    if (last_field)
        valid[i] = 1;
}

bool input_layout(int fields, int lanes, int time, long long& cells, long long& bytes) {
    if (fields < 1 || lanes < 1 || time < factor_engine::MinCorrelationPairs)
        return false;

    cells = (long long)lanes * time;
    const long long stride = (long long)fields * sizeof(float) + sizeof(float) + sizeof(uint8_t);
    if (!factor_engine::valid_float_storage(cells, fields) ||
        cells > std::numeric_limits<long long>::max() / stride ||
        (cells + THREADS - 1) / THREADS > std::numeric_limits<int>::max())
        return false;

    bytes = cells * stride;
    return true;
}

} // 命名空间

extern "C" int fe_cuda_backend_allocate_checked(int n_fields, int lanes, int time, long long budget,
                                                void** output) {
    using factor_engine::CudaBackendStatus;
    if (!output)
        return CudaBackendStatus::InvalidArgument;
    *output = nullptr;
    long long cells = 0, bytes = 0;
    if (budget < 1 || !input_layout(n_fields, lanes, time, cells, bytes))
        return CudaBackendStatus::InvalidArgument;
    if (bytes > budget)
        return CudaBackendStatus::AllocationFailed;

    try {
        auto b = std::make_unique<CUDABuffers>();
        b->lanes = lanes;
        b->time = time;
        b->cells = cells;
        b->bytes = bytes;
        b->budget = budget;
        b->fields.resize(n_fields, nullptr);

        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&b->sms, cudaDevAttrMultiProcessorCount, device) !=
                cudaSuccess ||
            b->sms < 1 ||
            cudaDeviceGetAttribute(&b->max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                   device) != cudaSuccess)
            return CudaBackendStatus::DeviceError;

        // 只有明确的分配 OOM 可以触发缩批；设备和参数错误保持独立。
        const auto allocation = consume_cuda_oom(cudaMalloc(&b->inputs, (size_t)bytes));
        if (allocation != cudaSuccess)
            return allocation == cudaErrorMemoryAllocation ? CudaBackendStatus::AllocationFailed
                                                           : CudaBackendStatus::DeviceError;

        auto* data = static_cast<float*>(b->inputs);
        for (int i = 0; i < n_fields; ++i)
            b->fields[i] = data + (long long)i * cells;
        b->target = data + (long long)n_fields * cells;
        b->valid = reinterpret_cast<uint8_t*>(b->target + cells);
        if (cudaDeviceSynchronize() != cudaSuccess)
            return CudaBackendStatus::DeviceError;

        *output = b.release();
        return CudaBackendStatus::Success;
    } catch (...) {
        return CudaBackendStatus::InternalError;
    }
}

extern "C" long long fe_cuda_runtime_input_bytes(void* handle) {
    const auto* b = static_cast<CUDABuffers*>(handle);
    return b ? b->bytes : 0;
}

extern "C" int fe_cuda_runtime_requirement(void* handle, const factor_engine::PlanNodePOD* nodes,
                                           int n_nodes, int slots, int roots, long long* bytes) {
    auto* b = static_cast<CUDABuffers*>(handle);
    if (!b || !b->work || !nodes || n_nodes < 1 || slots < 0 || roots < 1 || !bytes)
        return factor_engine::runtime_status::InvalidArgument;

    auto& rt = *b->work;
    using factor_engine::PlanOp;
    int unary = 0, max_pair = 0, max_std = 0;
    for (int i = 0; i < n_nodes; ++i) {
        const auto& n = nodes[i];
        const auto op = static_cast<PlanOp>(n.op);
        if (((op == PlanOp::RollSum || op == PlanOp::RollMean) &&
             n.param_i > DIRECT_ROLLING_WINDOW_LIMIT) ||
            ((op == PlanOp::RollMin || op == PlanOp::RollMax) &&
             n.param_i > DIRECT_ROLLING_WINDOW_LIMIT && n.param_i <= rt.time))
            unary = 1;
        if ((op == PlanOp::Cov || op == PlanOp::Corr) && n.param_i <= rt.time)
            max_pair = std::max(max_pair, n.param_i);
        if (op == PlanOp::RollStd && n.param_i <= rt.time)
            max_std = std::max(max_std, n.param_i);
    }

    return workspace_requirement(rt, slots, roots, unary, max_pair, max_std, *bytes);
}

extern "C" int fe_cuda_runtime_export_requirement(void* handle,
                                                  const factor_engine::PlanNodePOD* nodes,
                                                  int n_nodes, int slots, int roots,
                                                  long long* bytes) {
    const int rc = fe_cuda_runtime_requirement(handle, nodes, n_nodes, slots, roots, bytes);
    if (rc)
        return rc;

    const auto& rt = *static_cast<CUDABuffers*>(handle)->work;
    const bool fused = supports_pair_metrics(rt.time, rt.max_shared);
    const int extra = fused && roots > 1 ? std::max(0, roots - rt.packed_root_capacity) : 0;
    const auto added =
        (unsigned long long)extra * (unsigned long long)rt.allocation_cells() * sizeof(float);
    if (added > (unsigned long long)std::numeric_limits<long long>::max() - *bytes)
        return factor_engine::runtime_status::InvalidWorkspace;

    *bytes += (long long)added;
    return factor_engine::runtime_status::Success;
}

extern "C" int fe_cuda_runtime_shape_requirement(int fields, int lanes, int time, int slots,
                                                 int roots, int unary, int pair_window,
                                                 int std_window, long long* bytes) {
    long long cells = 0, input_bytes = 0;
    if (!bytes || !input_layout(fields, lanes, time, cells, input_bytes))
        return factor_engine::runtime_status::InvalidArgument;

    Runtime rt;
    rt.lanes = lanes;
    rt.time = time;
    rt.cells = cells;
    rt.fields.resize(fields, nullptr);

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&rt.sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&rt.max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) !=
            cudaSuccess)
        return factor_engine::runtime_status::DeviceError;

    int rc = workspace_requirement(rt, slots, roots, unary, pair_window, std_window, *bytes);
    if (rc)
        return rc;

    // 输入区和计算工作区分别由各自所有者核算。
    if (*bytes > std::numeric_limits<long long>::max() - input_bytes)
        return factor_engine::runtime_status::InvalidWorkspace;
    *bytes += input_bytes;
    return factor_engine::runtime_status::Success;
}

extern "C" long long fe_cuda_runtime_free_bytes() {
    size_t available = 0, total = 0;
    if (cudaMemGetInfo(&available, &total) != cudaSuccess ||
        available > (size_t)std::numeric_limits<long long>::max())
        return factor_engine::runtime_status::InvalidArgument;
    return (long long)available;
}
