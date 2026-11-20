#pragma once
#include "device_common.cuh"
#include <cstdint>
#include <climits>

namespace factor_engine::cuda {
// rank_layout 已检查总元素数不超过 INT_MAX；排序和秩使用 32 位索引。
__global__ void rank_offsets_kernel(int* offsets, int segments, int time) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= segments)
        offsets[i] = (int)(i * time);
}

__global__ void factor_rank_keys_kernel(const float* const* roots, const float* target, float* keys,
                                        int* indices, int cells, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;

    const float factor = roots[i / cells][i % cells], y = target[i % cells];
    const bool valid = isfinite(factor) && isfinite(y);
    keys[i] = valid ? factor : CUDART_INF_F;
    indices[i] = (int)i;
}

__global__ void target_rank_keys_kernel(const uint32_t* factor_ranks2, const float* target,
                                        float* keys, int* indices, int cells, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;

    const float y = target[i % cells];
    keys[i] = (factor_ranks2[i] != UINT32_MAX && isfinite(y)) ? y : CUDART_INF_F;
    indices[i] = (int)i;
}

__global__ void rank_counts_kernel(const float* sorted_keys, int* counts, int segments, int time) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= segments)
        return;

    const int segment = (int)index, base = segment * time;
    int low = 0, high = time;
    while (low < high) {
        const int mid = low + (high - low) / 2;
        if (isfinite(sorted_keys[base + mid]))
            low = mid + 1;
        else
            high = mid;
    }

    counts[segment] = low;
}

__global__ void average_tie_ranks_kernel(const float* sorted_keys, const int* sorted_indices,
                                         uint32_t* ranks2, int total, int time) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;

    const int position = (int)(i % time), base = (int)(i - position), original = sorted_indices[i];
    const float value = sorted_keys[i];
    if (!isfinite(value)) {
        ranks2[original] = UINT32_MAX;
        return;
    }

    const bool tied_left = position > 0 && sorted_keys[i - 1] == value,
               tied_right = position + 1 < time && sorted_keys[i + 1] == value;
    if (!tied_left && !tied_right) {
        ranks2[original] = 2u * (uint32_t)position;
        return;
    }

    int low = 0, high = position;
    while (low < high) {
        const int mid = low + (high - low) / 2;
        if (sorted_keys[base + mid] < value)
            low = mid + 1;
        else
            high = mid;
    }

    const int first = low;
    low = position + 1;
    high = time;
    while (low < high) {
        const int mid = low + (high - low) / 2;
        if (value < sorted_keys[base + mid])
            high = mid;
        else
            low = mid + 1;
    }

    const int last = low - 1;
    ranks2[original] = (uint32_t)first + (uint32_t)last;
}

__global__ void lane_rank_ic_kernel(const uint32_t* factor_ranks2, const uint32_t* target_ranks2,
                                    const int* counts, double* lane_scores, int segments,
                                    int time) {
    const int segment = blockIdx.x;
    if (segment >= segments)
        return;

    const int tid = threadIdx.x, count = counts[segment];
    const int base = segment * time;
    const double doubled_center = count - 1;
    double xx = 0.0, yy = 0.0, xy = 0.0;
    for (uint32_t t = tid; t < static_cast<uint32_t>(time); t += blockDim.x) {
        const uint32_t xf = factor_ranks2[base + t], yf = target_ranks2[base + t];
        if (xf != UINT32_MAX && yf != UINT32_MAX) {
            const double x = (double)xf - doubled_center, y = (double)yf - doubled_center;
            xx += x * x;
            yy += y * y;
            xy += x * y;
        }
    }

    __shared__ double shared_xx[THREADS], shared_yy[THREADS], shared_xy[THREADS];
    shared_xx[tid] = xx;
    shared_yy[tid] = yy;
    shared_xy[tid] = xy;
    __syncthreads();
    for (int offset = THREADS / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_xx[tid] += shared_xx[tid + offset];
            shared_yy[tid] += shared_yy[tid + offset];
            shared_xy[tid] += shared_xy[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const double denominator = sqrt(shared_xx[0] * shared_yy[0]);
        lane_scores[segment] =
            (count >= 3 && denominator > 0.0) ? shared_xy[0] / denominator : CUDART_NAN;
    }
}

} // 命名空间 factor_engine::cuda
