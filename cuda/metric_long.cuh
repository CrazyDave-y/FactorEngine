#pragma once

#include "metric_common.cuh"

// Pearson 与 X 排序准备共用输入读取，排序键拆为 D01、D23 两个字节对。
// arena 依次用于归约、直方图、索引交换和有序键；各阶段由 CTA 同步划清边界。
// Y 排序只交换原始位置和 X 双倍秩，不保留整组 Y key。
namespace factor_engine::metrics::detail {
// 两个寄存器保存本轮所需的八个 digit，不预存下一轮 digit。
struct DigitCache {
    unsigned words[DigitCacheWords];

    __device__ __forceinline__ unsigned get(int k) const {
        return (words[k / DigitsPerWord] >> (RadixBits * (k % DigitsPerWord))) & DigitMask;
    }

    __device__ __forceinline__ void put(int k, unsigned digit) {
        const unsigned previous = k % DigitsPerWord ? words[k / DigitsPerWord] : 0;
        const unsigned selector_shift = ByteSelectorBits * (k % DigitsPerWord);
        const unsigned selector = (KeepWordBytes & ~(ByteSelectorMask << selector_shift)) |
                                  (SecondWordFirstByte << selector_shift);
        words[k / DigitsPerWord] = __byte_perm(previous, digit, selector);
    }
};

// 用 12 个 word 打包 23 个 16 位值，分别保存前缀、目标地址或 RankX2。
struct PackedRanks {
    unsigned words[PackedRankWords];

    __device__ __forceinline__ void clear() {
#pragma unroll
        for (int j = 0; j < PackedRankWords; ++j)
            words[j] = 0;
    }

    __device__ __forceinline__ unsigned get(int k) const {
        return (words[k / 2] >> (PayloadBits * (k % 2))) & PayloadMask;
    }

    __device__ __forceinline__ void put(int k, unsigned value) {
        words[k / 2] = replace_half(words[k / 2], value, k % 2);
    }
};

// 直接处理指数和次正规数，保持已验证的缩放与舍入行为。
__device__ __forceinline__ float digitpair_shift(float value, int delta) {
    const unsigned bits = __float_as_uint(value), sign = bits & SignBit;
    const unsigned mag = bits & MagnitudeMask;
    if (!mag || mag >= ExponentMask)
        return value;
    unsigned mant = mag & MantissaMask;
    int exponent = int(mag >> MantissaBits) - ExponentBias;
    if (!(mag >> MantissaBits)) {
        const int leading = __clz(mant) - 8;
        exponent = (1 - int(ExponentBias)) - leading;
        mant = (mant << leading) & MantissaMask;
    }

    exponent += delta;
    if (exponent > int(ExponentBias))
        return __uint_as_float(sign | ExponentMask);
    if (exponent >= 1 - int(ExponentBias))
        return __uint_as_float(sign | (unsigned(exponent + ExponentBias) << MantissaBits) | mant);
    const int shift = (1 - int(ExponentBias)) - exponent;
    if (shift > MantissaBits + 1)
        return __uint_as_float(sign);
    const unsigned sig = HiddenBit | mant, low = sig & ((1u << shift) - 1u),
                   half = 1u << (shift - 1);
    unsigned rounded = sig >> shift;
    rounded += low > half || (low == half && (rounded & 1u));
    return __uint_as_float(sign | rounded);
}

__device__ __forceinline__ int digitpair_exponent(unsigned mag) {
    if (mag >> MantissaBits)
        return int(mag >> MantissaBits) - ExponentBias;
    return mag ? (KeyBits - 1 - __clz(mag)) - (ExponentBias + MantissaBits - 1)
               : -(ExponentBias + MantissaBits - 1);
}

__device__ __forceinline__ PearsonMoments digitpair_merge(PearsonMoments a, int& ax, int& ay,
                                                          PearsonMoments b, int bx, int by) {
    if (!a.n) {
        ax = bx;
        ay = by;
        return b;
    }

    if (!b.n)
        return a;
    const int ex = max(ax, bx), ey = max(ay, by);
    const int dax = ax - ex, day = ay - ey, dbx = bx - ex, dby = by - ey;
    a.mx = digitpair_shift(a.mx, dax);
    a.my = digitpair_shift(a.my, day);
    a.anchor_x = digitpair_shift(a.anchor_x, dax);
    a.anchor_y = digitpair_shift(a.anchor_y, day);
    a.xx = digitpair_shift(a.xx, 2 * dax);
    a.yy = digitpair_shift(a.yy, 2 * day);
    a.xy = digitpair_shift(a.xy, dax + day);
    b.mx = digitpair_shift(b.mx, dbx);
    b.my = digitpair_shift(b.my, dby);
    b.anchor_x = digitpair_shift(b.anchor_x, dbx);
    b.anchor_y = digitpair_shift(b.anchor_y, dby);
    b.xx = digitpair_shift(b.xx, 2 * dbx);
    b.yy = digitpair_shift(b.yy, 2 * dby);
    b.xy = digitpair_shift(b.xy, dbx + dby);
    ax = ex;
    ay = ey;
    return merge_pearson(a, b);
}

__device__ __forceinline__ unsigned chunk_pearson_digitpair(const float* x, const float* y,
                                                            const uint8_t* mask, int time,
                                                            unsigned* arena, uint16_t* d23,
                                                            double* result) {
    constexpr int Chunk = PearsonChunkItems;
    constexpr int Capacity = LongCapacity;
    const int tid = int(threadIdx.x), warp_lane = tid % WarpThreads, warp = tid / WarpThreads;
    const WarpRange input = warp_range<LongThreads>(time);
    auto* pair = reinterpret_cast<uint16_t*>(arena) + Capacity;
    unsigned valid_bits = 0;
    PearsonMoments total{};
    int total_ex = 0, total_ey = 0;

#pragma unroll 1
    for (int first = 0; first < LongItems; first += Chunk) {
        float vx[Chunk], vy[Chunk];
        unsigned good = 0, max_x = 0, max_y = 0;
        int n = 0;
        float sx = 0, sy = 0, anchor_x = 0, anchor_y = 0;

#pragma unroll
        for (int j = 0; j < Chunk; ++j) {
            const int k = first + j, t = input.base + WarpThreads * k;
            const bool in = k < LongItems && k < input.items && t < time;
            const float a = in ? x[t] : 0, b = in ? y[t] : 0;
            const bool active =
                in && mask[t] && finite_bits(__float_as_uint(a)) && finite_bits(__float_as_uint(b));
            vx[j] = a;
            vy[j] = b;
            if (active) {
                good |= 1u << j;
                valid_bits |= 1u << k;
                if (!n) {
                    anchor_x = a;
                    anchor_y = b;
                }

                ++n;
                {
                    max_x = max(max_x, __float_as_uint(a) & MagnitudeMask);
                    max_y = max(max_y, __float_as_uint(b) & MagnitudeMask);
                }
            }

            const unsigned key = active ? ordered(__float_as_uint(a)) : 0;
            if (in) {
                pair[t] = uint16_t(key);
                d23[t] = uint16_t(key >> 16);
            }
        }

        // 按本组最大幅值缩放，避免后续平方和溢出。
        int ex = 0, ey = 0;
        {
            ex = digitpair_exponent(max_x);
            ey = digitpair_exponent(max_y);
            anchor_x = digitpair_shift(anchor_x, -ex);
            anchor_y = digitpair_shift(anchor_y, -ey);

#pragma unroll
            for (int j = 0; j < Chunk; ++j)
                if (good & (1u << j)) {
                    vx[j] = digitpair_shift(vx[j], -ex);
                    vy[j] = digitpair_shift(vy[j], -ey);
                    sx += vx[j] - anchor_x;
                    sy += vy[j] - anchor_y;
                }
        }

        // 计算组内中心矩，再用 Chan 公式与前面的分组合并。
        PearsonMoments part{};
        if (n) {
            const float inv = 1.f / float(n);
            const float dx_mean = sx * inv, dy_mean = sy * inv;
            // TwoSum 规范化初始均值，同时保留近常量输入的小偏移。
            const PearsonMean mx = pearson_two_sum(anchor_x, dx_mean);
            const PearsonMean my = pearson_two_sum(anchor_y, dy_mean);
            part = {float(n), mx.residual, my.residual, 0, 0, 0, mx.anchor, my.anchor};

#pragma unroll
            for (int j = 0; j < Chunk; ++j)
                if (good & (1u << j)) {
                    const float dx = (vx[j] - anchor_x) - dx_mean;
                    const float dy = (vy[j] - anchor_y) - dy_mean;
                    part.xx = fmaf(dx, dx, part.xx);
                    part.yy = fmaf(dy, dy, part.yy);
                    part.xy = fmaf(dx, dy, part.xy);
                }
        }

        total = digitpair_merge(total, total_ex, total_ey, part, ex, ey);
    }

    // 先在 warp 内合并，再通过 shared memory 汇总整个 CTA。
    for (int d = WarpThreads / 2; d; d >>= 1) {
        const PearsonMoments other = shuffle_pearson(total, d);
        const int ex = __shfl_down_sync(Full, total_ex, d);
        const int ey = __shfl_down_sync(Full, total_ey, d);
        if (warp_lane + d < WarpThreads) {
            total = digitpair_merge(total, total_ex, total_ey, other, ex, ey);
        }
    }

    auto* partial = reinterpret_cast<PearsonMoments*>(arena);
    auto* scales = reinterpret_cast<int*>(partial + MaxWarps);
    static_assert(MaxWarps * (sizeof(PearsonMoments) + 2 * sizeof(int)) <=
                  LongCapacity * sizeof(uint16_t));
    if (warp_lane == 0) {
        partial[warp] = total;
        {
            scales[2 * warp] = total_ex;
            scales[2 * warp + 1] = total_ey;
        }
    }

    __syncthreads(); // 等待 D01、D23 和各 warp 的中心矩写完。

    if (warp == 0) {
        total = partial[warp_lane];
        {
            total_ex = scales[2 * warp_lane];
            total_ey = scales[2 * warp_lane + 1];
        }

        for (int d = WarpThreads / 2; d; d >>= 1) {
            const PearsonMoments other = shuffle_pearson(total, d);
            const int ex = __shfl_down_sync(Full, total_ex, d);
            const int ey = __shfl_down_sync(Full, total_ey, d);
            if (warp_lane + d < WarpThreads) {
                total = digitpair_merge(total, total_ex, total_ey, other, ex, ey);
            }
        }

        if (warp_lane == 0) {
            float rho = CUDART_NAN_F;
            if (total.n >= factor_engine::MinCorrelationPairs && total.xx > 0 && total.yy > 0 &&
                isfinite(total.xx) && isfinite(total.yy) && isfinite(total.xy)) {
                const float den = sqrtf(total.xx) * sqrtf(total.yy);
                if (den > 0 && isfinite(den))
                    rho = total.xy / den;
            }

            *result = isfinite(rho) ? double(fminf(1, fmaxf(-1, rho))) : CUDART_NAN;
        }
    }

    __syncthreads(); // 所有线程读完 Pearson 临时区后，才能复用这块空间存放直方图。

    return valid_bits;
}

__device__ __forceinline__ int digitpair_sort(unsigned (&state)[LongItems], int time,
                                              const float* x, unsigned* arena, const uint16_t* d23,
                                              RankMetadata& meta) {
    constexpr int Capacity = LongCapacity;
    const int tid = int(threadIdx.x), warp_lane = tid % WarpThreads, warp = tid / WarpThreads;
    auto* index = reinterpret_cast<uint16_t*>(arena);
    const auto* pair = reinterpret_cast<const uint16_t*>(arena) + Capacity;
    unsigned *table = arena, *histogram = arena + LongHistogramWords;
    int count = time;

#pragma unroll 1
    for (int pass = 0; pass < RadixPasses; ++pass) {
        // 低两字节已排完，将本 CTA 的高两字节从 D23 工作区搬入原 D01 区域。
        if (pass == 2) {
            auto* write_pair = reinterpret_cast<uint16_t*>(arena) + Capacity;
            for (int t = tid; t < time; t += LongThreads)
                write_pair[t] = d23[t];

            __syncthreads();
        }

        const int shift = RadixBits * (pass & 1);
        const WarpRange source = warp_range<LongThreads>(count);
        DigitCache current;

#pragma unroll
        for (int k = 0; k < CachedDigits; ++k) {
            const unsigned original = state[k] & PayloadMask;
            current.put(k, (state[k] & Valid) ? (pair[original] >> shift) & DigitMask : 0);
        }

#pragma unroll
        for (int g = 0; g < BucketsPerWarp; ++g)
            table[warp * RadixBuckets + ((g * WarpThreads + warp_lane) ^ warp)] = 0;
        __syncwarp(Full);

#pragma unroll
        for (int k = 0; k < LongItems; ++k)
            if (k < source.items) {
                const bool active = state[k] & Valid;
                const unsigned original = state[k] & PayloadMask;
                const unsigned digit = k < CachedDigits ? current.get(k)
                                       : active         ? (pair[original] >> shift) & DigitMask
                                                        : 0;
                const unsigned peers = __match_any_sync(Full, active ? digit : RadixBuckets);
                const int leader = __ffs(peers) - 1, offset = warp * RadixBuckets + (digit ^ warp);
                unsigned previous = 0;
                if (active && warp_lane == leader)
                    previous = table[offset];
                previous = __shfl_sync(Full, previous, leader);
                if (active && warp_lane == leader)
                    table[offset] = previous + __popc(peers);
                __syncwarp(Full);
                state[k] = pack_state(state[k], ((previous + __popc(peers & lower_lanes())) << 16) |
                                                    (active ? Valid : 0));
            }

        radix_prefix<LongThreads>(table, histogram, meta);

#pragma unroll
        for (int k = 0; k < LongItems; ++k)
            if (state[k] & Valid) {
                const unsigned original = state[k] & PayloadMask;
                const unsigned digit =
                    k < CachedDigits ? current.get(k) : (pair[original] >> shift) & DigitMask;
                const unsigned destination = histogram[digit] +
                                             table[warp * RadixBuckets + (digit ^ warp)] +
                                             ((state[k] >> 16) & PositionMask);
                state[k] = pack_state(original, (destination << 16) | Valid);
            }

        count = meta.count;

        __syncthreads(); // 所有线程读完前缀后，才能覆盖这块空间写入索引。

#pragma unroll
        for (int k = 0; k < LongItems; ++k)
            if (state[k] & Valid)
                index[(state[k] >> 16) & PositionMask] = uint16_t(state[k]);

        __syncthreads();

        const WarpRange output = warp_range<LongThreads>(count);

#pragma unroll
        for (int k = 0; k < LongItems; ++k) {
            const int p = output.base + WarpThreads * k;
            const bool active = k < output.items && p < count;
            const unsigned original = active ? index[p] : 0;
            state[k] = pack_state(original, active ? Valid : 0);
        }

        __syncthreads(); // 等待所有线程读完索引和字节对，再开始下一轮。
    }

    const WarpRange output = warp_range<LongThreads>(count);

#pragma unroll
    for (int k = 0; k < LongItems; ++k) {
        const int p = output.base + WarpThreads * k;
        if (k < output.items && p < count)
            arena[p] = ordered(__float_as_uint(x[state[k] & PayloadMask]));
    }

    __syncthreads(); // 等待 SortedX 写完，后续直接用它计算 X 秩。

    return count;
}

// 只缓存本轮 digit；算完目标地址后就不再使用缓存，避免与 scatter 阶段重叠。
__device__ __forceinline__ void y_payload_positions(unsigned (&payload)[LongItems],
                                                    WarpRange source, volatile uint8_t* plane,
                                                    unsigned* table, unsigned* histogram,
                                                    RankMetadata& meta, PackedRanks& prefix) {
    const int warp_lane = int(threadIdx.x) % WarpThreads, warp = int(threadIdx.x) / WarpThreads;
    DigitCache current;

#pragma unroll
    for (int k = 0; k < CachedDigits; ++k) {
        const unsigned original = payload[k] & PayloadMask;
        current.put(k, original == PayloadMask ? 0 : plane[original]);
    }

#pragma unroll
    for (int g = 0; g < BucketsPerWarp; ++g)
        table[warp * RadixBuckets + ((g * WarpThreads + warp_lane) ^ warp)] = 0;
    __syncwarp(Full);

#pragma unroll
    for (int k = 0; k < LongItems; ++k)
        if (k < source.items) {
            const unsigned original = payload[k] & PayloadMask;
            const bool active = original != PayloadMask;
            const unsigned digit = k < CachedDigits ? current.get(k) : active ? plane[original] : 0;
            const unsigned peers = __match_any_sync(Full, active ? digit : RadixBuckets);
            const int leader = __ffs(peers) - 1, offset = warp * RadixBuckets + (digit ^ warp);
            unsigned previous = 0;
            if (active && warp_lane == leader)
                previous = table[offset];
            previous = __shfl_sync(Full, previous, leader);
            if (active && warp_lane == leader)
                table[offset] = previous + __popc(peers);
            __syncwarp(Full);
            prefix.put(k, previous + __popc(peers & lower_lanes()));
        }

    radix_prefix<LongThreads>(table, histogram, meta);

#pragma unroll
    for (int k = 0; k < LongItems; ++k) {
        const unsigned original = payload[k] & PayloadMask;
        if (original != PayloadMask) {
            const unsigned digit = k < CachedDigits ? current.get(k) : plane[original];
            prefix.put(k, histogram[digit] + table[warp * RadixBuckets + (digit ^ warp)] +
                              prefix.get(k));
        }
    }
}

__device__ __forceinline__ void y_payload_sort(unsigned (&payload)[LongItems], int count, int time,
                                               const float* y, unsigned* arena,
                                               RankMetadata& meta) {
    const int tid = int(threadIdx.x);
    const WarpRange source = warp_range<LongThreads>(count);
    auto* plane = reinterpret_cast<volatile uint8_t*>(arena + LongPrefixWords);

#pragma unroll 4
    for (int pass = 0; pass < RadixPasses; ++pass) {
        const int shift = RadixBits * pass;
        for (int t = tid; t < time; t += LongThreads)
            plane[t] = uint8_t(ordered(__float_as_uint(y[t])) >> shift);

        __syncthreads();

        {
            PackedRanks prefix;
            prefix.clear();
            y_payload_positions(payload, source, plane, arena, arena + LongHistogramWords, meta,
                                prefix);

            __syncthreads(); // 所有线程读完 digit 和前缀表后，才能用 scatter 覆盖工作区。

#pragma unroll
            for (int k = 0; k < LongItems; ++k) {
                const unsigned word = payload[k];
                if ((word & PayloadMask) != PayloadMask)
                    arena[prefix.get(k)] = word;
            }

            __syncthreads(); // 等待所有 payload 写入新位置，再读取排序结果。
        } // 不再使用这 12 个目标地址 word，之后才回读 payload。

        if (pass == 3)
            break; // 最后一轮直接使用 shared memory 中的结果，不再回读整组 payload。

#pragma unroll
        for (int k = 0; k < LongItems; ++k) {
            const int p = source.base + WarpThreads * k;
            const bool active = k < source.items && p < count;
            payload[k] = active ? arena[p] : PayloadMask;
        }

        __syncthreads(); // 所有线程读完排序结果后，才能覆盖交换区。
    }
}

// 每个 CTA 处理一个因子、一个证券 lane、一个期限。
__global__ __launch_bounds__(LongThreads, 1) void long_metrics_kernel(
    const float* const* roots, const float* targets, const uint8_t* eligible, double* output,
    int lanes, int time, int horizons, uint16_t* digit_pairs) {
    extern __shared__ __align__(16) unsigned arena[];
    __shared__ RankMetadata meta;
    const int tid = int(threadIdx.x), warp_lane = tid % WarpThreads, warp = tid / WarpThreads;
    const int pair = int(blockIdx.x), horizon_index = pair % horizons;
    const int lane_index = (pair / horizons) % lanes, root_index = pair / (lanes * horizons);
    const float* x = roots[root_index] + size_t(lane_index) * time;
    const float* y = targets + (size_t(horizon_index) * lanes + lane_index) * time;
    const uint8_t* mask = eligible + (size_t(horizon_index) * lanes + lane_index) * time;
    auto* d23 = digit_pairs + size_t(pair) * LongCapacity;

    // 计算 Pearson，同时将 X 排序键拆成低、高两个字节对。
    const unsigned joint_bits = chunk_pearson_digitpair(
        x, y, mask, time, arena, d23, output + size_t(pair) * MetricCount + Pearson);
    unsigned state[LongItems];
    {
        const WarpRange input = warp_range<LongThreads>(time);

#pragma unroll
        for (int k = 0; k < LongItems; ++k) {
            const int t = input.base + WarpThreads * k;
            state[k] = pack_state(unsigned(t < time ? t : 0), (joint_bits & (1u << k)) ? Valid : 0);
        }
    }

    // X 排序只交换原始位置，结束后在共享区重建完整有序键。
    const int count = digitpair_sort(state, time, x, arena, d23, meta);
    // SortedX 已同步；有效键首尾相等即 X 无方差，短路条件保护空集合。
    if (count < factor_engine::MinCorrelationPairs || arena[0] == arena[count - 1]) {
        if (tid == 0)
            output[size_t(pair) * MetricCount + RankIC] = CUDART_NAN;
        return;
    }

    const WarpRange active = warp_range<LongThreads>(count);
    {
        long long correction_x = 0;

#pragma unroll
        for (int k = 0; k < LongItems; ++k) {
            const int p = active.base + WarpThreads * k;
            const bool valid = k < active.items && p < count;
            unsigned packed = PayloadMask;
            if (valid) {
                const unsigned original = state[k] & PayloadMask;
                const RankInfo info = rank_info<true>(arena, p, count);
                correction_x += info.correction;
                packed = pack_state(original, info.rank_twice << 16);
            }

            state[k] = packed;
        }

        correction_x = warp_sum(correction_x);
        if (warp_lane == 0)
            meta.correction_partial[warp] = correction_x;
    }

    __syncthreads(); // 等待所有线程读完 SortedX，并写完各 warp 的统计。

    if (tid == 0) {
        long long correction = 0;

#pragma unroll
        for (int owner = 0; owner < WarpThreads; ++owner)
            correction += meta.correction_partial[owner];
        meta.correction_x = correction;
    }

    __syncthreads();

    // Y 排序携带原始位置和 X 双倍秩，最终只保留 RankIC 所需的统计。
    PackedRanks rank_x_pairs;
    y_payload_sort(state, count, time, y, arena, meta);
    rank_x_pairs.clear();

#pragma unroll
    for (int k = 0; k < LongItems; ++k) {
        const int p = active.base + WarpThreads * k;
        if (k < active.items && p < count) {
            // scatter 已完成；每个位置只由一个线程读取 payload，再原位写入 SortedY。
            const unsigned value = arena[p];
            rank_x_pairs.put(k, value >> 16);
            arena[p] = ordered(__float_as_uint(y[value & PayloadMask]));
        }
    }

    __syncthreads();

    long long xy = 0, correction_y = 0;

#pragma unroll
    for (int k = 0; k < LongItems; ++k) {
        const int p = active.base + WarpThreads * k;
        if (k < active.items && p < count) {
            const RankInfo info = rank_info<true>(arena, p, count);
            correction_y += info.correction;
            const unsigned rx = rank_x_pairs.get(k);
            const int a = int(rx) - (count - 1), b = int(info.rank_twice) - (count - 1);
            xy += a * b;
        }
    }

    finish_rank<LongThreads>(count, xy, correction_y, arena, meta,
                             output + size_t(pair) * MetricCount);
}

} // 命名空间 factor_engine::metrics::detail
