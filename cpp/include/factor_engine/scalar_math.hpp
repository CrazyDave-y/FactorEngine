#pragma once

#include "factor_engine/plan.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace factor_engine::scalar_math {

inline float qnan() {
    return std::numeric_limits<float>::quiet_NaN();
}

inline float clean(float x) {
    return std::isfinite(x) ? x : qnan();
}

inline float scalar_unary(factor_engine::PlanOp op, float x) {
    if (!std::isfinite(x))
        return qnan();
    float v = qnan();
    switch (op) {
    case factor_engine::PlanOp::Neg:
        v = -x;
        break;
    case factor_engine::PlanOp::Abs:
        v = std::fabs(x);
        break;
    case factor_engine::PlanOp::Sign:
        v = (x > 0) - (x < 0);
        break;
    case factor_engine::PlanOp::SignedSqrt:
        v = std::copysign(std::sqrt(std::fabs(x)), x);
        break;
    case factor_engine::PlanOp::SignedLog1p:
        // 先用 double 计算，再舍入为 float，和 CUDA 的计算方式保持一致。
        // 直接用单精度库函数可能使原本并列的值不再相等。
        v = std::copysign(static_cast<float>(std::log1p(static_cast<double>(std::fabs(x)))), x);
        break;
    default:
        break;
    }

    return clean(v);
}

inline float scalar_binary(factor_engine::PlanOp op, float a, float b) {
    if (!std::isfinite(a) || !std::isfinite(b))
        return qnan();
    float v = qnan();
    switch (op) {
    case factor_engine::PlanOp::Add:
        v = a + b;
        break;
    case factor_engine::PlanOp::Sub:
        v = a - b;
        break;
    case factor_engine::PlanOp::Mul:
        v = a * b;
        break;
    case factor_engine::PlanOp::SafeDiv:
        if (std::fabs(b) > factor_engine::SafeDivisionEpsilon)
            v = a / b;
        break;
    case factor_engine::PlanOp::Min:
        v = std::min(a, b);
        break;
    case factor_engine::PlanOp::Max:
        v = std::max(a, b);
        break;
    default:
        break;
    }

    return clean(v);
}
} // 命名空间 factor_engine::scalar_math
