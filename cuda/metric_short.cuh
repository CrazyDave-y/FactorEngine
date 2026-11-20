#pragma once

#include "metric_common.cuh"

// 完整 key 驻留寄存器；先保留原始 X，完成 Pearson 后原位转为排序键。
// Shared 中的 key、payload 交换区也用于直方图和前缀，覆盖前必须等待读者结束。
namespace factor_engine::metrics::detail {
// 一个线程负责一个桶；八个 warp 各自扫描分组和，直接合成桶前缀。
__device__ __forceinline__ void short_radix_prefix(unsigned* table, unsigned* histogram,
                                                   RankMetadata& meta) {
    constexpr int Warps = ShortThreads / WarpThreads;
    static_assert(ShortThreads == RadixBuckets && Warps == BucketsPerWarp);
    const int tid = int(threadIdx.x), warp = tid / WarpThreads, warp_lane = tid % WarpThreads;

    __syncthreads();

    unsigned total = 0;

#pragma unroll
    for (int owner = 0; owner < Warps; ++owner) {
        const int address = owner * RadixBuckets + (tid ^ owner);
        const unsigned value = table[address];
        table[address] = total;
        total += value;
    }

    unsigned inclusive = total;

#pragma unroll
    for (int delta = 1; delta < WarpThreads; delta *= 2) {
        const unsigned other = __shfl_up_sync(Full, inclusive, delta);
        if (warp_lane >= delta)
            inclusive += other;
    }

    if (warp_lane == WarpThreads - 1)
        meta.partial[warp] = inclusive;

    __syncthreads();

    const unsigned group = warp_lane < Warps ? meta.partial[warp_lane] : 0;
    unsigned group_inclusive = group;

#pragma unroll
    for (int delta = 1; delta < Warps; delta *= 2) {
        const unsigned other = __shfl_up_sync(Full, group_inclusive, delta);
        if (warp_lane >= delta)
            group_inclusive += other;
    }

    const unsigned preceding = __shfl_sync(Full, group_inclusive - group, warp);
    const unsigned count = __shfl_sync(Full, group_inclusive, Warps - 1);
    histogram[tid] = preceding + inclusive - total;
    if (tid == 0)
        meta.count = int(count);

    __syncthreads(); // 等待桶前缀和 warp 前缀写完，再计算目标地址。
}

template <int Shift, bool Last>
__device__ __forceinline__ int short_radix_pass(unsigned (&keys)[ShortItems],
                                                unsigned (&payload)[ShortItems], int input_size,
                                                unsigned* arena, RankMetadata& meta) {
    constexpr int Threads = ShortThreads, Items = ShortItems;
    constexpr int Warps = Threads / WarpThreads, Capacity = Threads * Items,
                  Words = (Items + 1) / 2;
    const int tid = int(threadIdx.x), warp = tid / WarpThreads, warp_lane = tid % WarpThreads;
    unsigned *table = arena, *histogram = arena + Warps * RadixBuckets;
    auto* exchange_payload = arena + Capacity;
    unsigned position[Words]{};
    const WarpRange source = warp_range<Threads>(input_size);

#pragma unroll
    for (int g = 0; g < BucketsPerWarp; ++g)
        table[warp * RadixBuckets + ((g * WarpThreads + warp_lane) ^ warp)] = 0;
    __syncwarp(Full);

#pragma unroll
    for (int k = 0; k < Items; ++k)
        if (k < source.items) {
            const bool active = payload[k] & Valid;
            const unsigned digit = (keys[k] >> Shift) & DigitMask;
            const unsigned peers = __match_any_sync(Full, active ? digit : RadixBuckets);
            const int leader = __ffs(peers) - 1, address = warp * RadixBuckets + (digit ^ warp);
            unsigned previous = 0;
            if (active && warp_lane == leader)
                previous = table[address];
            previous = __shfl_sync(Full, previous, leader);
            if (active && warp_lane == leader)
                table[address] = previous + __popc(peers);
            __syncwarp(Full);
            const unsigned local = previous + __popc(peers & lower_lanes());
            position[k / 2] = replace_half(position[k / 2], local, k & 1);
        }

    short_radix_prefix(table, histogram, meta);

#pragma unroll
    for (int k = 0; k < Items; ++k)
        if (payload[k] & Valid) {
            const unsigned digit = (keys[k] >> Shift) & DigitMask;
            const unsigned local = (position[k / 2] >> (PayloadBits * (k & 1))) & PayloadMask;
            const unsigned destination =
                histogram[digit] + table[warp * RadixBuckets + (digit ^ warp)] + local;
            position[k / 2] = replace_half(position[k / 2], destination, k & 1);
        }

    const int count = meta.count;

    __syncthreads(); // 所有线程读完前缀后，才能覆盖交换区。

#pragma unroll
    for (int k = 0; k < Items; ++k)
        if (payload[k] & Valid) {
            const unsigned destination = (position[k / 2] >> (PayloadBits * (k & 1))) & PayloadMask;
            arena[destination] = keys[k];
            exchange_payload[destination] = payload[k];
        }

    __syncthreads(); // 等待 key 和 payload 写完，再读取排序结果。

    // 最后一轮直接用 shared memory 中的结果计算秩，不再搬回寄存器。
    if constexpr (!Last) {
        const WarpRange output = warp_range<Threads>(count);

#pragma unroll
        for (int k = 0; k < Items; ++k) {
            const int p = output.base + WarpThreads * k;
            const bool active = k < output.items && p < count;
            keys[k] = active ? arena[p] : 0;
            payload[k] = active ? exchange_payload[p] : 0;
        }

        __syncthreads(); // 所有线程读完排序结果后，下一轮才能覆盖直方图区。
    }

    return count;
}

__device__ __forceinline__ int short_radix_sort(unsigned (&keys)[ShortItems],
                                                unsigned (&payload)[ShortItems], int count,
                                                unsigned* arena, RankMetadata& meta) {
    count = short_radix_pass<0, false>(keys, payload, count, arena, meta);
    count = short_radix_pass<RadixBits, false>(keys, payload, count, arena, meta);
    count = short_radix_pass<2 * RadixBits, false>(keys, payload, count, arena, meta);
    return short_radix_pass<3 * RadixBits, true>(keys, payload, count, arena, meta);
}

// 在 shared memory 的 payload 区原位保存 X 双倍秩；不展开循环，减少同时占用的寄存器。
struct XRankSums {
    long long correction;
};

__device__ __forceinline__ XRankSums short_rank_x(unsigned* keys, unsigned* payload, int count) {
    constexpr int Threads = ShortThreads, Items = ShortItems;
    XRankSums sum{};
    const WarpRange owned = warp_range<Threads>(count);

#pragma unroll 1
    for (int k = 0; k < Items; ++k) {
        const int p = owned.base + WarpThreads * k;
        if (k < owned.items && p < count) {
            const unsigned original = payload[p] & PayloadMask;
            const RankInfo info = rank_info(keys, p, count);
            sum.correction += info.correction;
            payload[p] = pack_state(original, Valid | (info.rank_twice << PayloadBits));
        }
    }

    return sum;
}

struct YRankSums {
    long long xy, correction;
};

__device__ __forceinline__ YRankSums short_rank_y(const unsigned* keys, const unsigned* payload,
                                                  int count) {
    constexpr int Threads = ShortThreads, Items = ShortItems;
    YRankSums sum{};
    const WarpRange owned = warp_range<Threads>(count);

#pragma unroll 1
    for (int k = 0; k < Items; ++k) {
        const int p = owned.base + WarpThreads * k;
        if (k < owned.items && p < count) {
            const unsigned rx = (payload[p] >> PayloadBits) & PositionMask;
            const RankInfo info = rank_info(keys, p, count);
            const int a = int(rx) - (count - 1), b = int(info.rank_twice) - (count - 1);
            sum.xy += a * b;
            sum.correction += info.correction;
        }
    }

    return sum;
}

// 先在寄存器中保留原始 X；完成 Pearson 缩放和 Chan 归约后，复用这些寄存器保存排序键。
__device__ __forceinline__ unsigned short_pearson(const float* x, const float* y,
                                                  const uint8_t* eligible, int time,
                                                  unsigned (&keys)[ShortItems], unsigned* arena,
                                                  double* result) {
    constexpr int Threads = ShortThreads, Items = ShortItems, Warps = Threads / WarpThreads;
    const int warp_lane = int(threadIdx.x) % WarpThreads, warp = int(threadIdx.x) / WarpThreads;
    unsigned joint_bits = 0;
    unsigned pearson_max_x = 0, pearson_max_y = 0;
    int pearson_first = time;
    {
        const WarpRange input = warp_range<Threads>(time);

#pragma unroll
        for (int k = 0; k < Items; ++k) {
            const int t = input.base + WarpThreads * k;
            unsigned key = 0;
            bool active = false;
            if (k < input.items && t < time) {
                // 三路读取互不依赖，资格判断不再阻塞 X/Y 的加载。
                const unsigned a = __float_as_uint(x[t]);
                const unsigned b = __float_as_uint(y[t]);
                active = eligible[t] && finite_bits(a) && finite_bits(b);
                if (active) {
                    key = a;
                    joint_bits |= 1u << k;
                    pearson_max_x = max(pearson_max_x, a & MagnitudeMask);
                    pearson_max_y = max(pearson_max_y, b & MagnitudeMask);
                    pearson_first = min(pearson_first, t);
                }
            }

            keys[k] = key;
        }
    }

    {
        unsigned* tmp = arena;
        for (int d = WarpThreads / 2; d; d >>= 1) {
            pearson_max_x = max(pearson_max_x, __shfl_down_sync(Full, pearson_max_x, d));
            pearson_max_y = max(pearson_max_y, __shfl_down_sync(Full, pearson_max_y, d));
            pearson_first = min(pearson_first, __shfl_down_sync(Full, pearson_first, d));
        }

        if (warp_lane == 0) {
            tmp[warp] = pearson_max_x;
            tmp[Warps + warp] = pearson_max_y;
            tmp[2 * Warps + warp] = unsigned(pearson_first);
        }

        __syncthreads();

        if (warp == 0) {
            unsigned ax = warp_lane < Warps ? tmp[warp_lane] : 0,
                     ay = warp_lane < Warps ? tmp[Warps + warp_lane] : 0;
            int first = warp_lane < Warps ? int(tmp[2 * Warps + warp_lane]) : time;
            for (int d = WarpThreads / 2; d; d >>= 1) {
                ax = max(ax, __shfl_down_sync(Full, ax, d));
                ay = max(ay, __shfl_down_sync(Full, ay, d));
                first = min(first, __shfl_down_sync(Full, first, d));
            }

            __syncwarp(Full);
            if (warp_lane == 0) {
                tmp[0] = ax;
                tmp[1] = ay;
                tmp[2] = unsigned(first);
            }
        }

        __syncthreads();

        const int first = int(tmp[2]);
        const float ax = __uint_as_float(tmp[0]), ay = __uint_as_float(tmp[1]);
        const int ex = ax > 0 ? ilogbf(ax) : 0, ey = ay > 0 ? ilogbf(ay) : 0;
        const float ox = first < time ? scalbnf(x[first], -ex) : 0;
        const float oy = first < time ? scalbnf(y[first], -ey) : 0;
        const WarpRange input = warp_range<Threads>(time);
        PearsonMoments m{};

#pragma unroll
        for (int k = 0; k < Items; ++k)
            if (joint_bits & (1u << k)) {
                const int t = input.base + WarpThreads * k;
                // 保留减去全 CTA 原点时的舍入残差，再构造单点统计。
                const PearsonMean vx = pearson_two_sum(scalbnf(__uint_as_float(keys[k]), -ex), -ox);
                const PearsonMean vy = pearson_two_sum(scalbnf(y[t], -ey), -oy);
                m = merge_pearson(m, {1, vx.residual, vy.residual, 0, 0, 0, vx.anchor, vy.anchor});
            }

        for (int d = WarpThreads / 2; d; d >>= 1) {
            const PearsonMoments b = shuffle_pearson(m, d);
            if (warp_lane + d < WarpThreads)
                m = merge_pearson(m, b);
        }

        __syncthreads(); // 所有线程读完缩放参数后，才能复用临时区。

        auto* partial = reinterpret_cast<PearsonMoments*>(tmp);
        if (warp_lane == 0)
            partial[warp] = m;

        __syncthreads();

        if (warp == 0) {
            m = warp_lane < Warps ? partial[warp_lane] : PearsonMoments{};
            for (int d = WarpThreads / 2; d; d >>= 1) {
                const PearsonMoments b = shuffle_pearson(m, d);
                if (warp_lane + d < WarpThreads)
                    m = merge_pearson(m, b);
            }

            if (warp_lane == 0) {
                const float rho = m.n >= factor_engine::MinCorrelationPairs && m.xx > 0 && m.yy > 0
                                      ? (m.xy / sqrtf(m.xx)) / sqrtf(m.yy)
                                      : CUDART_NAN_F;
                *result = isfinite(rho) ? double(fminf(1, fmaxf(-1, rho))) : CUDART_NAN;
            }
        }

        __syncthreads(); // 所有线程读完 Pearson 临时数据后，再复用工作区排序。
    }

    return joint_bits;
}

// 仅供容量不超过 3840 的短序列调用；调用端负责检查容量。
__global__ __launch_bounds__(ShortThreads, 1) void short_metrics_kernel(const float* const* roots,
                                                                        const float* targets,
                                                                        const uint8_t* eligible,
                                                                        double* output, int lanes,
                                                                        int time, int horizons) {
    constexpr int Threads = ShortThreads, Items = ShortItems, Warps = Threads / WarpThreads;
    extern __shared__ __align__(16) unsigned arena[];
    __shared__ RankMetadata meta;
    const int tid = int(threadIdx.x), warp_lane = tid % WarpThreads, warp = tid / WarpThreads;
    const int pair = int(blockIdx.x), horizon_index = pair % horizons;
    const int lane_index = (pair / horizons) % lanes, root_index = pair / (lanes * horizons);
    const float* x = roots[root_index] + size_t(lane_index) * time;
    const size_t target_base = (size_t(horizon_index) * lanes + lane_index) * time;
    const float* y = targets + target_base;
    double* result = output + size_t(pair) * MetricCount;
    unsigned keys[Items], state[Items];
    const unsigned joint_bits =
        short_pearson(x, y, eligible + target_base, time, keys, arena, result);
    {
        const WarpRange input = warp_range<Threads>(time);

#pragma unroll
        for (int k = 0; k < Items; ++k) {
            const int t = input.base + WarpThreads * k;
            const unsigned key = keys[k];
            const bool active = bool(joint_bits & (1u << k));
            keys[k] = active ? ordered(key) : 0;
            state[k] = pack_state(unsigned(t < time ? t : 0), active ? Valid : 0);
        }
    }

    int count = short_radix_sort(keys, state, time, arena, meta);
    // 末轮 scatter 已同步 SortedX；短路条件保护空集合，整个 CTA 统一退出。
    if (count < factor_engine::MinCorrelationPairs || arena[0] == arena[count - 1]) {
        if (tid == 0)
            result[RankIC] = CUDART_NAN;
        return;
    }

    const WarpRange active = warp_range<Threads>(count);
    {
        const auto sum = short_rank_x(arena, arena + Threads * Items, count);
        long long correction_x = warp_sum(sum.correction);
        if (warp_lane == 0)
            meta.correction_partial[warp] = correction_x;
    }

    __syncthreads(); // 等待 X 秩和各 warp 的统计写完。

    // 先汇总 X 统计，再准备 Y 排序，减少两组临时值同时占用的寄存器。
    if (tid == 0) {
        long long correction = 0;

#pragma unroll
        for (int owner = 0; owner < Warps; ++owner)
            correction += meta.correction_partial[owner];
        meta.correction_x = correction;
    }

#pragma unroll
    for (int k = 0; k < Items; ++k) {
        const int p = active.base + WarpThreads * k;
        const bool valid = k < active.items && p < count;
        const unsigned word = valid ? arena[Threads * Items + p] : 0;
        state[k] = word;
        keys[k] = valid ? ordered(__float_as_uint(y[word & PayloadMask])) : 0;
    }

    __syncthreads();

    short_radix_sort(keys, state, count, arena, meta);
    const auto sum = short_rank_y(arena, arena + Threads * Items, count);
    finish_rank<Threads>(count, sum.xy, sum.correction, arena, meta, result);
}
} // 命名空间 factor_engine::metrics::detail
