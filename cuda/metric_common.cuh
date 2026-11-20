#pragma once

#include "factor_engine/metric.hpp"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <climits>
#include <type_traits>

namespace factor_engine::metrics::detail {
constexpr unsigned Full = 0xffffffffu;
constexpr unsigned Valid = SignBit;

struct RankMetadata {
    int count;
    unsigned partial[MaxWarps];
    long long correction_x;
    long long correction_partial[MaxWarps];
};

static_assert(sizeof(RankMetadata) == 400);

struct WarpRange {
    int base, items;
};

// 每个 warp 按原顺序处理若干组连续的 32 项，保证各轮排序稳定。
template <int Threads> __device__ __forceinline__ WarpRange warp_range(int size) {
    constexpr int Warps = Threads / WarpThreads;
    const int warp = int(threadIdx.x) / WarpThreads;
    const int rounds = (size + WarpThreads - 1) / WarpThreads;
    const int q = rounds / Warps, r = rounds % Warps;
    return {WarpThreads * (warp * q + min(warp, r)) + int(threadIdx.x) % WarpThreads,
            q + (warp < r)};
}

__device__ __forceinline__ bool finite_bits(unsigned value) {
    return (value & ExponentMask) != ExponentMask;
}

__device__ __forceinline__ unsigned ordered(unsigned value) {
    if ((value & MagnitudeMask) == 0)
        value = 0; // 正负零共用排序键。
    return value ^ ((value & Valid) ? Full : Valid);
}

__device__ __forceinline__ unsigned lower_lanes() {
    const unsigned warp_lane = threadIdx.x % WarpThreads;
    return (1u << warp_lane) - 1u;
}

// 一个 word 打包两个 16 位值；调用处完全展开循环，让编译器确定要写哪个半字。
__device__ __forceinline__ unsigned replace_half(unsigned word, unsigned value, int half) {
    return __byte_perm(word, value, half ? ReplaceHighHalfBytes : ReplaceLowHalfBytes);
}

// 索引在低半字，秩和有效位在高半字。
__device__ __forceinline__ unsigned pack_state(unsigned payload, unsigned upper) {
    return __byte_perm(payload, upper, JoinIndexAndStateBytes);
}

template <class T> __device__ __forceinline__ T warp_sum(T value) {
#pragma unroll
    for (int delta = WarpThreads / 2; delta; delta >>= 1)
        value += __shfl_down_sync(Full, value, delta);
    return value;
}

// 计算每个桶在各 warp 之前的计数，以及所有更小桶的总数，用于稳定排序。
template <int Threads>
__device__ __forceinline__ void radix_prefix(unsigned* table, unsigned* histogram,
                                             RankMetadata& meta) {
    constexpr int Warps = Threads / WarpThreads;
    const int tid = int(threadIdx.x), warp = tid / WarpThreads, warp_lane = tid % WarpThreads;

    __syncthreads();

    for (int digit = tid; digit < RadixBuckets; digit += Threads) {
        unsigned running = 0;

#pragma unroll
        for (int owner = 0; owner < Warps; ++owner) {
            const int address = owner * RadixBuckets + (digit ^ owner);
            const unsigned value = table[address];
            table[address] = running;
            running += value;
        }

        histogram[digit] = running;
    }

    __syncthreads();

    if (warp < BucketsPerWarp) {
        const int digit = warp * WarpThreads + warp_lane;
        const unsigned value = histogram[digit];
        unsigned inclusive = value;

#pragma unroll
        for (int delta = 1; delta < WarpThreads; delta *= 2) {
            const unsigned other = __shfl_up_sync(Full, inclusive, delta);
            if (warp_lane >= delta)
                inclusive += other;
        }

        histogram[digit] = inclusive - value;
        if (warp_lane == WarpThreads - 1)
            meta.partial[warp] = inclusive;
    }

    __syncthreads();

    if (warp < BucketsPerWarp) {
        // partial 保留桶组总数；各桶 warp 独立扫描，省去发布组偏移的 CTA 同步。
        const unsigned group = warp_lane < BucketsPerWarp ? meta.partial[warp_lane] : 0;
        unsigned group_inclusive = group;

#pragma unroll
        for (int delta = 1; delta < BucketsPerWarp; delta *= 2) {
            const unsigned other = __shfl_up_sync(Full, group_inclusive, delta);
            if (warp_lane >= delta)
                group_inclusive += other;
        }

        const unsigned preceding = __shfl_sync(Full, group_inclusive - group, warp);
        const unsigned count = __shfl_sync(Full, group_inclusive, BucketsPerWarp - 1);
        // 组内前缀已写入 histogram，直接读取即可，不必保留前一步的临时值。
        histogram[warp * WarpThreads + warp_lane] += preceding;
        if (tid == 0)
            meta.count = int(count);
    }

    __syncthreads(); // 等待所有线程读完 partial，再计算目标地址或进入下一轮。
}

struct RankInfo {
    unsigned rank_twice;
    bool group_start;
    long long correction;
};

// 双倍平均秩为 first + last - 1；每组并列值只在首项累计一次 m³-m。
template <bool Bounded = false>
__device__ __forceinline__ RankInfo rank_info(const unsigned* keys, int p, int count) {
    const unsigned value = keys[p];
    const bool head = p == 0 || keys[p - 1] != value;
    const bool next_equal = p + 1 < count && keys[p + 1] == value;
    using Index = std::conditional_t<Bounded, unsigned, int>;
    int first = p, last = p + 1;
    if (!head) {
        Index lo = 0, hi = Index(p);
        while (lo < hi) {
            const Index mid = lo + (hi - lo) / 2;
            if (keys[mid] < value)
                lo = mid + 1;
            else
                hi = mid;
        }

        first = lo;
    }

    if (next_equal) {
        Index lo = Index(p + 1), hi = Index(count);
        while (lo < hi) {
            const Index mid = lo + (hi - lo) / 2;
            if (value < keys[mid])
                hi = mid;
            else
                lo = mid + 1;
        }

        last = int(lo);
    }

    long long correction = 0;
    if (head && next_equal) {
        if constexpr (Bounded) {
            // 长序列容量不超过 23552，m² 在无符号 32 位范围内精确表示。
            const unsigned m = unsigned(last - p);
            correction = static_cast<long long>(m) * (m * m - 1u);
        } else {

            const long long m = last - p;
            correction = m * m * m - m;
        }
    }

    return {unsigned(first + last - 1), head, correction};
}

// 均值表示为 anchor + residual；mx/my 仅保存 residual，不能先舍入成单 float。
struct PearsonMoments {
    float n;
    float mx, my;
    float xx, yy, xy;
    float anchor_x, anchor_y;
};

struct PearsonMean {
    float anchor, residual;
};

// 通用 TwoSum 不要求 |a| >= |b|；均值跨零时同样保留加法的舍入残差。
// 显式指定 RN 舍入，避免编译器重结合后丢失加法残差。
__device__ __forceinline__ PearsonMean pearson_two_sum(float a, float b) {
    const float sum = __fadd_rn(a, b);
    const float offset = __fsub_rn(sum, a);
    const float error = __fadd_rn(__fsub_rn(a, __fsub_rn(sum, offset)), __fsub_rn(b, offset));
    return {sum, error};
}

__device__ __forceinline__ PearsonMean pearson_mean_delta(float aa, float ar, float ba, float br) {
    const PearsonMean anchors = pearson_two_sum(ba, -aa);
    return pearson_two_sum(anchors.anchor, __fadd_rn(anchors.residual, __fsub_rn(br, ar)));
}

// 先合并 anchor 与加权增量，再汇总三个低部并规范化。
// 单次 fmaf(delta, weight, residual) 仍可能在均值跨零前吞掉 residual。
__device__ __forceinline__ PearsonMean update_pearson_mean(float anchor, float residual,
                                                           PearsonMean delta, float weight) {
    const float increment = __fmul_rn(delta.anchor, weight);
    const float increment_error =
        __fmaf_rn(delta.residual, weight, __fmaf_rn(delta.anchor, weight, -increment));
    const PearsonMean sum = pearson_two_sum(anchor, increment);
    return pearson_two_sum(sum.anchor,
                           __fadd_rn(__fadd_rn(sum.residual, residual), increment_error));
}

__device__ __forceinline__ PearsonMoments merge_pearson(PearsonMoments a, PearsonMoments b) {
    if (a.n == 0)
        return b;
    if (b.n == 0)
        return a;
    const float n = a.n + b.n, weight = b.n / n;
    const PearsonMean delta_x = pearson_mean_delta(a.anchor_x, a.mx, b.anchor_x, b.mx);
    const PearsonMean delta_y = pearson_mean_delta(a.anchor_y, a.my, b.anchor_y, b.my);
    const float dx = delta_x.anchor, dy = delta_y.anchor;
    const float cross = a.n * weight;
    const PearsonMean mx = update_pearson_mean(a.anchor_x, a.mx, delta_x, weight);
    const PearsonMean my = update_pearson_mean(a.anchor_y, a.my, delta_y, weight);
    return {n,
            mx.residual,
            my.residual,
            fmaf(dx * dx, cross, a.xx + b.xx),
            fmaf(dy * dy, cross, a.yy + b.yy),
            fmaf(dx * dy, cross, a.xy + b.xy),
            mx.anchor,
            my.anchor};
}

__device__ __forceinline__ PearsonMoments shuffle_pearson(PearsonMoments a, int delta) {
    return {__shfl_down_sync(Full, a.n, delta),        __shfl_down_sync(Full, a.mx, delta),
            __shfl_down_sync(Full, a.my, delta),       __shfl_down_sync(Full, a.xx, delta),
            __shfl_down_sync(Full, a.yy, delta),       __shfl_down_sync(Full, a.xy, delta),
            __shfl_down_sync(Full, a.anchor_x, delta), __shfl_down_sync(Full, a.anchor_y, delta)};
}

template <int Threads>
__device__ __forceinline__ void finish_rank(int count, long long xy, long long correction_y,
                                            unsigned* arena, const RankMetadata& meta,
                                            double* result) {
    constexpr int Warps = Threads / WarpThreads;
    const int warp_lane = int(threadIdx.x) % WarpThreads, warp = int(threadIdx.x) / WarpThreads;
    correction_y = warp_sum(correction_y);
    xy = warp_sum(xy);

    __syncthreads(); // 所有线程读完 SortedY 后，才能复用这块空间保存归约结果。

    auto* sums = reinterpret_cast<long long*>(arena);
    if (warp_lane == 0) {
        sums[warp] = correction_y;
        sums[Warps + warp] = xy;
    }

    __syncthreads();

    if (warp == 0) {
        const long long correction = warp_sum(warp_lane < Warps ? sums[warp_lane] : 0LL);
        xy = warp_sum(warp_lane < Warps ? sums[Warps + warp_lane] : 0LL);
        const long long n = count, base = n * (n * n - 1);
        const long long xx = (base - meta.correction_x) / 3;
        const long long yy = (base - correction) / 3;
        if (warp_lane == 0) {
            const double den = sqrt(double(xx) * double(yy));
            result[RankIC] = count >= factor_engine::MinCorrelationPairs && den > 0
                                 ? double(xy) / den
                                 : CUDART_NAN;
        }
    }
}
} // 命名空间 factor_engine::metrics::detail
