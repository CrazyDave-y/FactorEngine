#pragma once

#include "contracts.hpp"

#include <limits>

namespace factor_engine::metrics {

enum KernelShape : int {
    WarpThreads = 32,
    MaxBlockThreads = 1024,
    MaxWarps = MaxBlockThreads / WarpThreads,
    ShortThreads = 256,
    ShortItems = 15,
    ShortCapacity = ShortThreads * ShortItems,
    LongThreads = 1024,
    LongItems = 23,
    LongCapacity = LongThreads * LongItems,
    PackedRankWords = (LongItems + 1) / 2,
    RadixBits = 8,
    RadixBuckets = 1 << RadixBits,
    KeyBits = 32,
    PayloadBits = 16,
    RadixPasses = KeyBits / RadixBits,
    CachedDigits = 8,
    PearsonChunkItems = 8,
    DigitsPerWord = KeyBits / RadixBits,
    DigitCacheWords = (CachedDigits + DigitsPerWord - 1) / DigitsPerWord,
    BucketsPerWarp = RadixBuckets / WarpThreads,
    LongHistogramWords = (LongThreads / WarpThreads) * RadixBuckets,
    LongPrefixWords = LongHistogramWords + RadixBuckets,
    MaxRankTime = 24576
};

enum Fp32Bits : unsigned {
    SignBit = 0x80000000u,
    MagnitudeMask = 0x7fffffffu,
    ExponentMask = 0x7f800000u,
    MantissaMask = 0x7fffffu,
    HiddenBit = 0x800000u,
    PayloadMask = 0xffffu,
    PositionMask = PayloadMask >> 1,
    DigitMask = RadixBuckets - 1
};

enum Fp32Shape : int { MantissaBits = 23, ExponentBias = 127 };

// __byte_perm 的选择器：每个十六进制位指定一个输出字节的来源。
enum PackedByteOrder : unsigned {
    ByteSelectorBits = 4,
    ByteSelectorMask = (1u << ByteSelectorBits) - 1u,
    SecondWordFirstByte = KeyBits / RadixBits,
    KeepWordBytes = 0x3210,
    ReplaceLowHalfBytes = 0x3254,
    ReplaceHighHalfBytes = 0x5410,
    JoinIndexAndStateBytes = 0x7610
};

enum Metric { Pearson = 0, RankIC = 1, MetricCount = 2 };

constexpr bool metric_layout_fits(int roots, int lanes, int time, int horizons) {
    if (roots < 1 || lanes < 1 || time < 1 || horizons < 1 || time > MaxRankTime)
        return false;
    constexpr auto MaxBytes = std::numeric_limits<std::ptrdiff_t>::max();
    return static_cast<std::uint64_t>(roots) * lanes <=
               MaxBytes / sizeof(double) / horizons / MetricCount &&
           static_cast<std::uint64_t>(horizons) * lanes <= MaxBytes / sizeof(float) / time;
}
}

// 输出 [root,lane]，eligible 可空；两边必须同时为有限值。
extern "C" int fe_pearson_ic_lanes_cpu(const float* const* roots, int n_roots, const float* target,
                                       const uint8_t* eligible, int lanes, int time,
                                       double* output);

// 输出 [root,lane,horizon,2]，依次为 PearsonIC、RankIC；两项使用相同的有效样本。
// RankIC 使用双倍平均秩和 INT64 归约。
extern "C" int fe_metric_lanes_cpu(const float* const* roots, int n_roots, const float* targets,
                                   const uint8_t* eligible, int lanes, int time, int horizons,
                                   double* output);

namespace factor_engine::rank_math {

// 双倍中心秩逐项用 int32，累计用 int64，转 double 时仍能精确表示。
constexpr bool rank_integer_sums_are_exact(int time) {
    if (time < 1 || time > factor_engine::metrics::MaxRankTime)
        return false;
    const long long centered_max = time - 1;
    return centered_max * centered_max <= std::numeric_limits<std::int32_t>::max() &&
           time * centered_max * centered_max <
               static_cast<long long>(factor_engine::MaxExactDoubleInteger);
}
} // 命名空间 factor_engine::rank_math
