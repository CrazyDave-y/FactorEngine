#pragma once
#include "device_common.cuh"
#include "factor_engine/centered_pair.hpp"
#include "factor_engine/centered_unary.hpp"
#include <cub/block/block_scan.cuh>
#include <cstdint>
#include <cfloat>

namespace factor_engine::cuda {
__global__ void delay_diff_kernel(const float* x, float* out, int lanes, int time, int lag,
                                  int do_diff) {
    const long long n = (long long)lanes * time,
                    i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const int t = (int)(i % time);
    if (t < lag) {
        out[i] = qnan();
        return;
    }

    const float old = x[i - lag];
    if (!isfinite(old)) {
        out[i] = qnan();
        return;
    }

    if (!do_diff) {
        out[i] = old;
        return;
    }

    const float cur = x[i];
    out[i] = (isfinite(cur) && isfinite(cur - old)) ? cur - old : qnan();
}

__global__ void ema_tiled_kernel(const float* x, float* out, int lanes, int time, int span) {
    const int lane = blockIdx.x, tid = threadIdx.x;
    if (lane >= lanes)
        return;

    __shared__ float tile[EMA_TILE_SIZE];
    const float alpha = 2.0f / (span + 1.0f);
    const long long base = (long long)lane * time;
    bool init = false;
    float state = 0.0f;

    for (long long chunk = 0; chunk < time; chunk += EMA_TILE_SIZE) {
        const int count = (int)min((long long)EMA_TILE_SIZE, (long long)time - chunk);
        for (int k = tid; k < count; k += blockDim.x)
            tile[k] = x[base + chunk + k];
        __syncthreads();

        if (tid == 0) {
            for (int k = 0; k < count; ++k) {
                const float input = tile[k];
                tile[k] = qnan();
                if (!isfinite(input))
                    continue;
                if (!init) {
                    state = input;
                    init = true;
                } else {
                    const float next =
                        __fmaf_rn(alpha, input, __fmul_rn(__fsub_rn(1.0f, alpha), state));
                    if (isfinite(next))
                        state = next;
                }
                tile[k] = state;
            }
        }
        __syncthreads();

        for (int k = tid; k < count; k += blockDim.x)
            out[base + chunk + k] = tile[k];
        __syncthreads();
    }
}

struct RollingExtremaCombine {
    bool do_min;
    bool reverse;

    __device__ __forceinline__ float operator()(float a, float b) const {
        // 反向扫描仍按真实时间合并；相等时取最新值，保留 CPU 的零符号。
        if (reverse) {
            const float old = a;
            a = b;
            b = old;
        }
        if (!isfinite(a) || !isfinite(b))
            return qnan();
        return do_min ? (a < b ? a : b) : (a > b ? a : b);
    }
};

struct RollingExtremaScanState {
    float value;
    int head;
};

struct RollingExtremaSegmentedCombine {
    RollingExtremaCombine extrema;

    __device__ __forceinline__ RollingExtremaScanState operator()(RollingExtremaScanState a,
                                                                  RollingExtremaScanState b) const {
        return {b.head ? b.value : extrema(a.value, b.value), a.head | b.head};
    }
};

// 按长度 w 分段，分别做前缀和后缀扫描；有限值与有效性共用一个 float。
// 短分段合并到同一 block；grid 为 2*lanes*ceil(ceil(time/w)/max(1,256/w))。
// 固定 256 线程，scratch 各含 lanes*time 个 float。
__global__ void rolling_extrema_summaries_kernel(const float* x, float* prefix, float* suffix,
                                                 int lanes, int time, int w, bool do_min) {
    if (w <= 1 || w > time)
        return;

    const long long epochs = ((long long)time + w - 1) / w;
    const int epochs_per_block = w < ROLLING_EXTREMA_THREADS ? ROLLING_EXTREMA_THREADS / w : 1;
    const long long groups = (epochs + epochs_per_block - 1) / epochs_per_block;
    const long long tasks = (long long)lanes * groups;
    const long long unit = blockIdx.x;
    if (unit >= 2 * tasks)
        return;

    const bool reverse = unit >= tasks;
    const long long task = reverse ? unit - tasks : unit;
    const long long lane = task / groups, group = task % groups;
    const long long begin = group * epochs_per_block * w;
    const long long end = min((long long)time, begin + (long long)epochs_per_block * w);
    const long long base = lane * time;
    const RollingExtremaSegmentedCombine combine{{do_min, reverse}};
    using Scan = cub::BlockScan<RollingExtremaScanState, ROLLING_EXTREMA_THREADS>;
    __shared__ typename Scan::TempStorage scan_storage;
    __shared__ RollingExtremaScanState carry;

    for (long long offset = 0; offset < end - begin; offset += ROLLING_EXTREMA_THREADS) {
        const int count = (int)min((long long)ROLLING_EXTREMA_THREADS, end - begin - offset);
        const bool active = threadIdx.x < count;
        const long long t = reverse ? end - 1 - offset - threadIdx.x : begin + offset + threadIdx.x;
        // 无效线程位于扫描尾部，不会影响任何有效线程的前缀。
        float value = active ? x[base + t] : 0.0f;
        value = isfinite(value) ? value : qnan();
        const int head = active && (reverse ? (t == end - 1 || (t + 1) % w == 0) : t % w == 0);
        RollingExtremaScanState scanned;
        Scan(scan_storage).InclusiveScan(RollingExtremaScanState{value, head}, scanned, combine);
        if (offset)
            scanned = combine(carry, scanned);

        // 所有线程读完旧 carry 后，再更新 carry 并复用 CUB 临时区。
        __syncthreads();
        if (active) {
            (reverse ? suffix : prefix)[base + t] = scanned.value;
            if (threadIdx.x == count - 1)
                carry = scanned;
        }
        __syncthreads();
    }
}

// 一个窗口最多跨两个分段，每个输出只需读取两个扫描结果。
// 短窗口直接比较，省去扫描临时区和额外的 kernel 启动。
__global__ void rolling_extrema_output_kernel(const float* x, const float* prefix,
                                              const float* suffix, float* out, int lanes, int time,
                                              int w, bool do_min) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long cells = (long long)lanes * time;
    if (i >= cells)
        return;

    const int t = (int)(i % time);
    if (t + 1 < w) {
        out[i] = qnan();
        return;
    }

    if (w == 1) {
        const float value = x[i];
        out[i] = isfinite(value) ? value : qnan();
        return;
    }

    if (w <= DIRECT_ROLLING_WINDOW_LIMIT) {
        float value = do_min ? CUDART_INF_F : -CUDART_INF_F;
        for (int k = w - 1; k >= 0; --k) {
            const float next = x[i - k];
            if (!isfinite(next)) {
                out[i] = qnan();
                return;
            }
            // 按时间从旧到新，相等时留下新值，也保留它的零符号。
            value = do_min ? (value < next ? value : next) : (value > next ? value : next);
        }
        out[i] = value;
        return;
    }

    const int start = t - w + 1;
    if (start % w == 0) {
        out[i] = prefix[i];
        return;
    }

    const RollingExtremaCombine combine{do_min, false};
    out[i] = combine(suffix[i - w + 1], prefix[i]);
}

__global__ void rolling_unary_output_kernel(const float* x, float* out, int lanes, int time, int w,
                                            PlanOp op) {
    const long long n = (long long)lanes * time,
                    i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const int t = (int)(i % time);
    if (t + 1 < w) {
        out[i] = qnan();
        return;
    }

    bool constant = true;
    double sum = 0;
    for (int k = 0; k < w; ++k) {
        const float v = x[i - k];
        if (!isfinite(v)) {
            out[i] = qnan();
            return;
        }
        constant &= (v == x[i]);
        sum += v;
    }

    double value = 0;
    if (constant) {
        out[i] = clean(op == PlanOp::RollSum ? (float)((double)x[i] * w) : x[i]);
        return;
    }

    value = op == PlanOp::RollSum ? sum : sum / w;
    out[i] = isfinite(value) ? (float)value : qnan();
}

// 分段重建 CPU 队列的合并顺序，合并的括号位置也要一致。
// 方差继续使用中心矩，避免两个大数相减丢精度。
__global__ void rolling_std_summaries_kernel(const float* x, factor_engine::UnaryMoments* front,
                                             factor_engine::UnaryMoments* prefix, int lanes,
                                             int time, int w) {
    const long long unit = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long epochs = ((long long)time + w - 1) / w, tasks = (long long)lanes * epochs;
    if (unit >= 2 * tasks)
        return;

    const int direction = (int)(unit / tasks), lane = (int)(unit % lanes),
              epoch = (int)((unit % tasks) / lanes);
    const long long base = (long long)lane * time;
    const int begin = epoch * w, end = (int)min((long long)time, (long long)begin + w);

    if (!direction) {
        if (!epoch)
            return;
        auto* suffix = front + ((long long)(epoch - 1) * lanes + lane) * w;
        factor_engine::UnaryMoments aggregate{};
        for (int j = 0; j < w; ++j) {
            const auto leaf = factor_engine::unary_leaf(x[base + begin - 1 - j]);
            aggregate = j ? factor_engine::merge_unary(leaf, aggregate) : leaf;
            suffix[j] = aggregate;
        }
        return;
    }

    factor_engine::UnaryMoments back{};
    for (int t = begin; t < end; ++t) {
        const auto leaf = factor_engine::unary_leaf(x[base + t]);
        back = t == begin ? leaf : factor_engine::merge_unary(back, leaf);
        prefix[base + t] = back;
    }
}

__global__ void rolling_std_native_order_kernel(float* out,
                                                const factor_engine::UnaryMoments* front,
                                                const factor_engine::UnaryMoments* prefix,
                                                int lanes, int time, int w) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)lanes * time)
        return;
    const int t = (int)(i % time), lane = (int)(i / time);
    out[i] = qnan();
    if (t + 1 < w)
        return;

    const int epoch = t / w, remaining = epoch ? w - 1 - (t % w) : 0;
    const auto previous = remaining
                              ? front[((long long)(epoch - 1) * lanes + lane) * w + remaining - 1]
                              : factor_engine::UnaryMoments{};
    const auto moments = factor_engine::merge_unary(previous, prefix[i]);
    if (moments.count != w)
        return;

    out[i] = clean((float)sqrt(fmax(0.0, moments.m2 / w)));
}

struct UnaryPrefixValue {
    double sum;
    int invalid, changes;
};

struct UnaryPrefixAdd {
    __device__ __forceinline__ UnaryPrefixValue operator()(const UnaryPrefixValue& a,
                                                           const UnaryPrefixValue& b) const {
        return {a.sum + b.sum, a.invalid + b.invalid, a.changes + b.changes};
    }
};

__global__ void rolling_unary_prefix_kernel(const float* x, float* out, double* prefix_sum,
                                            uint64_t* prefix_meta, int lanes, int time, int w,
                                            PlanOp op) {
    const int lane = blockIdx.x, tid = threadIdx.x;
    if (lane >= lanes)
        return;

    const long long base = (long long)lane * time;
    using Scan = cub::BlockScan<UnaryPrefixValue, THREADS, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename Scan::TempStorage scan_storage;
    __shared__ UnaryPrefixValue carry;
    if (tid == 0)
        carry = {};
    __syncthreads();

    for (long long chunk = 0; chunk < time; chunk += THREADS) {
        const long long t = chunk + tid;
        const bool active = t < time;
        const float v = active ? x[base + t] : 0.0f;
        const bool good = active && isfinite(v);
        UnaryPrefixValue value = {good ? (double)v : 0.0, active && !good,
                                  active && t > 0 && v != x[base + t - 1]};
        UnaryPrefixValue prefix, aggregate;
        Scan(scan_storage).InclusiveScan(value, prefix, UnaryPrefixAdd{}, aggregate);
        prefix = UnaryPrefixAdd{}(carry, prefix);
        if (active) {
            const long long i = base + t;
            prefix_sum[i] = prefix.sum;
            prefix_meta[i] = ((uint64_t)(uint32_t)prefix.invalid << 32) | (uint32_t)prefix.changes;
        }
        __syncthreads();
        if (active) {
            const long long i = base + t;
            out[i] = qnan();
            if (t + 1 >= w) {
                const int previous = t - w, first = t - w + 1;
                const int invalid = prefix.invalid -
                                    (previous >= 0 ? (int)(prefix_meta[base + previous] >> 32) : 0);
                if (invalid == 0) {
                    if (prefix.changes - (int)(uint32_t)prefix_meta[base + first] == 0)
                        out[i] = clean(op == PlanOp::RollSum ? (float)((double)v * w) : v);
                    else {
                        const double sum =
                            prefix.sum - (previous >= 0 ? prefix_sum[base + previous] : 0.0);
                        const double result = op == PlanOp::RollSum ? sum : sum / w;
                        out[i] = isfinite(result) ? (float)result : qnan();
                    }
                }
            }
        }
        __syncthreads();

        if (tid == 0)
            carry = UnaryPrefixAdd{}(carry, aggregate);
        __syncthreads();
    }
}

// 临时中心矩按 [字段, 时间, lane] 存放。同一 warp 的相邻 lane 读写相邻的 double。
// count 也用 double 保存：它不超过 int 范围，转换不会损失整数精度。
enum PairColumn { PairMeanA, PairMeanB, PairM2A, PairM2B, PairCross, PairCount, PairColumnCount };

static_assert(PairColumnCount * sizeof(double) == sizeof(factor_engine::PairMoments));

__device__ __forceinline__ void store_pair_columns(double* data, long long cells, long long i,
                                                   const factor_engine::PairMoments& value) {
    data[PairMeanA * cells + i] = value.mean_a;
    data[PairMeanB * cells + i] = value.mean_b;
    data[PairM2A * cells + i] = value.m2_a;
    data[PairM2B * cells + i] = value.m2_b;
    data[PairCross * cells + i] = value.cross;
    data[PairCount * cells + i] = value.count;
}

__device__ __forceinline__ factor_engine::PairMoments
load_pair_columns(const double* data, long long cells, long long i) {
    return {data[PairMeanA * cells + i], data[PairMeanB * cells + i],
            data[PairM2A * cells + i],   data[PairM2B * cells + i],
            data[PairCross * cells + i], (int)data[PairCount * cells + i]};
}

// 每个 W 边界的转移只需前 W 个样本，按反序构建即可让各段独立计算。
// 保留原来的合并顺序，后栈当前的累计值放在寄存器中。
// 每个样本只加入一次后缀和一次后栈累计，整体为 O(T)。
__global__ void rolling_pair_summaries_kernel(const float* a, const float* b, double* front,
                                              double* prefix, int lanes, int time, int w) {
    const long long unit = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long epochs = ((long long)time + w - 1) / w, tasks = (long long)lanes * epochs;
    if (unit >= 2 * tasks)
        return;

    const int direction = (int)(unit / tasks), lane = (int)(unit % lanes),
              epoch = (int)((unit % tasks) / lanes);
    const long long base = (long long)lane * time;
    const long long cells = (long long)lanes * time;
    const int begin = epoch * w, end = (int)min((long long)time, (long long)begin + w);

    if (!direction) {
        if (!epoch)
            return;
        factor_engine::PairMoments aggregate{};
        for (int j = 0; j < w; ++j) {
            const int t = begin - 1 - j;
            const auto leaf = factor_engine::pair_leaf(a[base + t], b[base + t]);
            aggregate = j ? factor_engine::merge_pair(leaf, aggregate) : leaf;
            const long long output = ((long long)(epoch - 1) * w + j) * lanes + lane;
            store_pair_columns(front, cells, output, aggregate);
        }
        return;
    }

    factor_engine::PairMoments back{};
    for (int t = begin; t < end; ++t) {
        const float av = a[base + t], bv = b[base + t];
        const auto leaf = factor_engine::pair_leaf(av, bv);
        back = t == begin ? leaf : factor_engine::merge_pair(back, leaf);
        store_pair_columns(prefix, cells, (long long)t * lanes + lane, back);
    }
}

// 各窗口最后的合并和归一化互不依赖，拆出来并行计算。
__global__ void rolling_pair_centered_kernel(float* out, const double* front, const double* prefix,
                                             int lanes, int time, int w, int corr) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)lanes * time)
        return;
    const int t = (int)(i / lanes), lane = (int)(i % lanes);
    const long long cells = (long long)lanes * time, output = (long long)lane * time + t;
    out[output] = qnan();
    if (t + 1 < w)
        return;

    const int epoch = t / w, remaining = epoch ? w - 1 - (t % w) : 0;
    const long long front_index = ((long long)(epoch - 1) * w + remaining - 1) * lanes + lane;
    const auto previous =
        remaining ? load_pair_columns(front, cells, front_index) : factor_engine::PairMoments{};
    const auto moments = factor_engine::merge_pair(previous, load_pair_columns(prefix, cells, i));
    if (moments.count != w)
        return;

    double result = moments.cross / w;
    if (corr) {
        const double den = sqrt((moments.m2_a / w) * (moments.m2_b / w));
        if (!(den > factor_engine::CorrelationDenominatorEpsilon))
            return;
        result /= den;
    }

    out[output] = isfinite(result) ? (float)result : qnan();
}

} // 命名空间 factor_engine::cuda
