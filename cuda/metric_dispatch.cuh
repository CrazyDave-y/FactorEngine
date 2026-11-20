#pragma once
#include "metric_short.cuh"
#include "metric_long.cuh"

namespace factor_engine::metrics::detail {
inline int validate(const float* const* roots, const float* targets, const uint8_t* eligible,
                    const double* output, int root_count, int lanes, int time, int horizons) {
    if (time > MaxRankTime)
        return int(cudaErrorNotSupported);
    if (!roots || !targets || !eligible || !output || root_count <= 0 || lanes <= 0 || time <= 0 ||
        horizons <= 0)
        return int(cudaErrorInvalidValue);

    // 分步检查，避免维度乘积在验证过程本身溢出。
    if ((long long)root_count * lanes > INT_MAX / horizons)
        return int(cudaErrorInvalidValue);
    return int(cudaSuccess);
}

template <class Kernel> inline cudaError_t configure(Kernel kernel, size_t dynamic, int segments) {
    int device = 0, available = 0, max_grid = 0;
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess)
        return error;
    error = cudaDeviceGetAttribute(&available, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (error != cudaSuccess)
        return error;
    error = cudaDeviceGetAttribute(&max_grid, cudaDevAttrMaxGridDimX, device);
    if (error != cudaSuccess)
        return error;
    if (segments > max_grid)
        return cudaErrorInvalidConfiguration;

    cudaFuncAttributes attributes{};
    error = cudaFuncGetAttributes(&attributes, kernel);
    if (error != cudaSuccess)
        return error;
    if (!factor_engine::shared_storage_fits(dynamic, attributes.sharedSizeBytes, size_t(available)))
        return cudaErrorNotSupported;
    return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(dynamic));
}

} // 命名空间 factor_engine::metrics::detail

namespace factor_engine::metrics {
constexpr size_t ShortSharedBytes = 2 * ShortCapacity * sizeof(uint32_t);
constexpr size_t LongSharedBytes = (LongCapacity + RadixBuckets) * sizeof(uint32_t);

// 前缀与交换区分时复用；字节平面位于长序列前缀区之后。
static_assert((ShortThreads / WarpThreads + 1) * RadixBuckets <= ShortCapacity);
static_assert(LongPrefixWords * sizeof(uint32_t) + LongCapacity <= LongSharedBytes);
static_assert(2 * ShortCapacity - 2 <= PositionMask);
static_assert(2 * LongCapacity - 2 <= PayloadMask);

// 长序列每个 CTA 独占一组 D23；短序列不需要全局工作区。
inline size_t pair_metric_scratch_bytes(int roots, int lanes, int time, int horizons) {
    if (roots < 1 || lanes < 1 || horizons < 1 || time < 1 || time > LongCapacity ||
        (long long)roots * lanes > INT_MAX / horizons || time <= ShortCapacity)
        return 0;
    const size_t segments = size_t(roots) * lanes * horizons;
    constexpr size_t PerSegment = LongCapacity * sizeof(uint16_t);
    if (segments > SIZE_MAX / PerSegment)
        return 0;
    return segments * PerSegment;
}

// 完整融合 Pearson 与 RankIC；超出容量直接报错，由调用者显式选择其他算法。
inline int launch_pair_metrics(const float* const* roots, const float* targets,
                               const uint8_t* eligible, double* output, int root_count, int lanes,
                               int time, int horizons, void* scratch, size_t scratch_bytes) {
    const int error =
        detail::validate(roots, targets, eligible, output, root_count, lanes, time, horizons);
    if (error != cudaSuccess)
        return error;
    if (time > LongCapacity)
        return int(cudaErrorNotSupported);

    const int segments = root_count * lanes * horizons;
    if (time <= ShortCapacity) {
        const auto configured =
            detail::configure(detail::short_metrics_kernel, ShortSharedBytes, segments);
        if (configured != cudaSuccess)
            return int(configured);
        detail::short_metrics_kernel<<<segments, ShortThreads, ShortSharedBytes>>>(
            roots, targets, eligible, output, lanes, time, horizons);
    } else {
        const size_t required = pair_metric_scratch_bytes(root_count, lanes, time, horizons);
        if (!required || !scratch || scratch_bytes < required)
            return int(cudaErrorInvalidValue);
        const auto configured =
            detail::configure(detail::long_metrics_kernel, LongSharedBytes, segments);
        if (configured != cudaSuccess)
            return int(configured);
        detail::long_metrics_kernel<<<segments, LongThreads, LongSharedBytes>>>(
            roots, targets, eligible, output, lanes, time, horizons,
            static_cast<uint16_t*>(scratch));
    }

    return int(cudaGetLastError());
}

} // 命名空间 factor_engine::metrics
