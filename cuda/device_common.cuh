#pragma once
#include "factor_engine/plan.hpp"
#include "factor_engine/scalar_math.hpp"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace factor_engine::cuda {
using factor_engine::PlanOp;

enum KernelConfig : int {
    THREADS = factor_engine::ElementwiseThreads,
    EMA_TILE_SIZE = 1024,
    EMA_THREADS = 128,
    DIRECT_ROLLING_WINDOW_LIMIT = 32,
    ROLLING_EXTREMA_THREADS = THREADS
};

struct Value {
    bool scalar = false;
    float s = 0;
    const float* p = nullptr;
};

__device__ __forceinline__ float qnan() {
    return CUDART_NAN_F;
}

__device__ __forceinline__ float clean(float x) {
    return isfinite(x) ? x : qnan();
}

} // 命名空间 factor_engine::cuda
