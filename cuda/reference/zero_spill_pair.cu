// 两项 IC 的 Zero-Spill 研究对照，不接入生产调度。
#include "../metric_dispatch.cuh"

namespace factor_engine::reference::zero_spill {
using namespace factor_engine::metrics;
using namespace factor_engine::metrics::detail;

enum Storage : int {
    OffloadedPayloads = LongCapacity * sizeof(uint16_t) / (LongThreads * sizeof(unsigned))
};

static_assert(OffloadedPayloads * LongThreads * sizeof(unsigned) <=
              LongCapacity * sizeof(uint16_t));

// D23 在 X 排序后已退休；按工作区容量卸载 payload，其余保持静态寄存器访问。
struct Payload {
    unsigned (&words)[LongItems];
    volatile unsigned* scratch;

    __device__ __forceinline__ unsigned get(int k) const {
        return k < OffloadedPayloads ? scratch[k * LongThreads + int(threadIdx.x)] : words[k];
    }

    __device__ __forceinline__ void put(int k, unsigned value) {
        if (k < OffloadedPayloads)
            scratch[k * LongThreads + int(threadIdx.x)] = value;
        else
            words[k] = value;
    }

    __device__ __forceinline__ bool active(int k, WarpRange source, int count) const {
        return k < source.items && source.base + WarpThreads * k < count;
    }
};

// 分组只缩短临时值存活时间，仍按原顺序调用一次 rank_info。
template <int Begin, int Length>
__device__ __forceinline__ void x_rank_group(unsigned (&state)[LongItems], const unsigned* arena,
                                             WarpRange active, int count, long long& correction,
                                             volatile unsigned* scratch) {
#pragma unroll
    for (int item = 0; item < Length; ++item) {
        const int k = Begin + item, p = active.base + WarpThreads * k;
        unsigned word = PayloadMask;
        if (k < active.items && p < count) {
            const unsigned original = state[k] & PayloadMask;
            const RankInfo info = rank_info<true>(arena, p, count);
            correction += info.correction;
            word = pack_state(original, info.rank_twice << PayloadBits);
        }
        if (k < OffloadedPayloads) {
            scratch[k * LongThreads + int(threadIdx.x)] = word;
            state[k] = PayloadMask;
        } else {
            state[k] = word;
        }
    }
}

__device__ __forceinline__ void positions(Payload& payload, WarpRange source, int count,
                                          volatile uint8_t* plane, unsigned* table,
                                          unsigned* histogram, RankMetadata& meta,
                                          PackedRanks& prefix) {
    const int warp_lane = int(threadIdx.x) % WarpThreads, warp = int(threadIdx.x) / WarpThreads;
    DigitCache current;
#pragma unroll
    for (int k = 0; k < CachedDigits; ++k) {
        const unsigned original = payload.get(k) & PayloadMask;
        current.put(k, original == PayloadMask ? 0 : plane[original]);
    }
#pragma unroll
    for (int g = 0; g < BucketsPerWarp; ++g)
        table[warp * RadixBuckets + ((g * WarpThreads + warp_lane) ^ warp)] = 0;
    __syncwarp(Full);
#pragma unroll
    for (int k = 0; k < LongItems; ++k) {
        if (k < source.items) {
            const bool active = payload.active(k, source, count);
            const unsigned digit = k < CachedDigits ? current.get(k)
                                   : active         ? plane[payload.get(k) & PayloadMask]
                                                    : 0;
            const unsigned peers = __match_any_sync(Full, active ? digit : RadixBuckets);
            const int leader = __ffs(peers) - 1;
            const int offset = warp * RadixBuckets + (digit ^ warp);
            unsigned previous = 0;
            if (active && warp_lane == leader)
                previous = table[offset];
            previous = __shfl_sync(Full, previous, leader);
            if (active && warp_lane == leader)
                table[offset] = previous + __popc(peers);
            __syncwarp(Full);
            prefix.put(k, previous + __popc(peers & lower_lanes()));
        }
    }
    radix_prefix<LongThreads>(table, histogram, meta);
#pragma unroll
    for (int k = 0; k < LongItems; ++k) {
        if (payload.active(k, source, count)) {
            const unsigned digit =
                k < CachedDigits ? current.get(k) : plane[payload.get(k) & PayloadMask];
            prefix.put(k, histogram[digit] + table[warp * RadixBuckets + (digit ^ warp)] +
                              prefix.get(k));
        }
    }
}

__device__ __forceinline__ void sort_y(unsigned (&words)[LongItems], int count, int time,
                                       const float* y, unsigned* arena, RankMetadata& meta,
                                       volatile unsigned* scratch) {
    const int tid = int(threadIdx.x);
    const WarpRange source = warp_range<LongThreads>(count);
    auto* plane = reinterpret_cast<volatile uint8_t*>(arena + LongPrefixWords);
    Payload payload{words, scratch};
#pragma unroll 4
    for (int pass = 0; pass < RadixPasses; ++pass) {
        const int shift = RadixBits * pass;
        for (int t = tid; t < time; t += LongThreads)
            plane[t] = uint8_t(ordered(__float_as_uint(y[t])) >> shift);
        __syncthreads();
        {
            PackedRanks prefix;
            prefix.clear();
            positions(payload, source, count, plane, arena, arena + LongHistogramWords, meta,
                      prefix);
            __syncthreads(); // digit 和前缀读者全部退休后，再 scatter。
#pragma unroll
            for (int k = 0; k < LongItems; ++k) {
                const unsigned word = payload.get(k);
                if ((word & PayloadMask) != PayloadMask)
                    arena[prefix.get(k)] = word;
            }
            __syncthreads(); // 等待完整 scatter；不能跨过它保留 shared 暂存。
        }
        if (pass == RadixPasses - 1)
            break; // 最后一轮直接消费 shared 中的完整 payload。
#pragma unroll
        for (int k = 0; k < LongItems; ++k) {
            const int p = source.base + WarpThreads * k;
            payload.put(k, payload.active(k, source, count) ? arena[p] : PayloadMask);
        }
        __syncthreads(); // gather 读者退休后才能覆盖前缀区。
    }
}

__global__ __launch_bounds__(LongThreads,
                             1) void kernel(const float* const* roots, const float* targets,
                                            const uint8_t* eligible, double* output, int lanes,
                                            int time, int horizons, uint16_t* digit_pairs) {
    extern __shared__ __align__(16) unsigned arena[];
    __shared__ RankMetadata meta;
    const int tid = int(threadIdx.x), warp_lane = tid % WarpThreads, warp = tid / WarpThreads;
    const int pair = int(blockIdx.x), horizon = pair % horizons;
    const int lane = (pair / horizons) % lanes, root = pair / (lanes * horizons);
    const float* x = roots[root] + size_t(lane) * time;
    const float* y = targets + (size_t(horizon) * lanes + lane) * time;
    const uint8_t* mask = eligible + (size_t(horizon) * lanes + lane) * time;
    auto* d23 = digit_pairs + size_t(pair) * LongCapacity;
    const unsigned joint = chunk_pearson_digitpair(x, y, mask, time, arena, d23,
                                                   output + size_t(pair) * MetricCount + Pearson);
    unsigned state[LongItems];
    {
        const WarpRange input = warp_range<LongThreads>(time);
#pragma unroll
        for (int k = 0; k < LongItems; ++k) {
            const int t = input.base + WarpThreads * k;
            state[k] = pack_state(unsigned(t < time ? t : 0), (joint & (1u << k)) ? Valid : 0);
        }
    }
    const int count = digitpair_sort(state, time, x, arena, d23, meta);
    if (count < factor_engine::MinCorrelationPairs || arena[0] == arena[count - 1]) {
        if (tid == 0)
            output[size_t(pair) * MetricCount + RankIC] = CUDART_NAN;
        return;
    }
    const WarpRange active = warp_range<LongThreads>(count);
    auto* scratch = reinterpret_cast<volatile unsigned*>(d23);
    {
        long long correction = 0;
        x_rank_group<0, 8>(state, arena, active, count, correction, scratch);
        x_rank_group<8, 8>(state, arena, active, count, correction, scratch);
        x_rank_group<16, 7>(state, arena, active, count, correction, scratch);
        correction = warp_sum(correction);
        if (warp_lane == 0)
            meta.correction_partial[warp] = correction;
    }
    __syncthreads();
    if (tid == 0) {
        long long correction = 0;
#pragma unroll
        for (int owner = 0; owner < WarpThreads; ++owner)
            correction += meta.correction_partial[owner];
        meta.correction_x = correction;
    }
    __syncthreads();
    sort_y(state, count, time, y, arena, meta, scratch);
    // 最后一次 scatter 已退休全部 payload 读者；D23 再复用为 SortedY 位置的 RankX2。
    auto* rank_x = reinterpret_cast<volatile uint16_t*>(d23);
#pragma unroll
    for (int k = 0; k < LongItems; ++k) {
        const int p = active.base + WarpThreads * k;
        if (k < active.items && p < count) {
            const unsigned word = arena[p];
            rank_x[p] = uint16_t(word >> PayloadBits);
            arena[p] = ordered(__float_as_uint(y[word & PayloadMask]));
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
            const int a = int(rank_x[p]) - (count - 1);
            const int b = int(info.rank_twice) - (count - 1);
            xy += a * b;
        }
    }
    finish_rank<LongThreads>(count, xy, correction_y, arena, meta,
                             output + size_t(pair) * MetricCount);
}
} // 命名空间 factor_engine::reference::zero_spill

extern "C" size_t zero_spill_scratch_bytes(int roots, int lanes, int time, int horizons) {
    return factor_engine::metrics::pair_metric_scratch_bytes(roots, lanes, time, horizons);
}

extern "C" int launch_zero_spill_pair(const float* const* roots, const float* targets,
                                      const uint8_t* eligible, double* output, int root_count,
                                      int lanes, int time, int horizons, void* scratch,
                                      size_t scratch_bytes) {
    namespace m = factor_engine::metrics;
    const int error =
        m::detail::validate(roots, targets, eligible, output, root_count, lanes, time, horizons);
    if (error != cudaSuccess)
        return error;
    if (time <= m::ShortCapacity || time > m::LongCapacity)
        return int(cudaErrorNotSupported);
    const size_t required = m::pair_metric_scratch_bytes(root_count, lanes, time, horizons);
    if (!required || !scratch || scratch_bytes < required)
        return int(cudaErrorInvalidValue);
    const int segments = root_count * lanes * horizons;
    const auto configured = m::detail::configure(factor_engine::reference::zero_spill::kernel,
                                                 m::LongSharedBytes, segments);
    if (configured != cudaSuccess)
        return int(configured);
    factor_engine::reference::zero_spill::kernel<<<segments, m::LongThreads, m::LongSharedBytes>>>(
        roots, targets, eligible, output, lanes, time, horizons, static_cast<uint16_t*>(scratch));
    return int(cudaGetLastError());
}
