#pragma once

namespace factor_engine {

enum class RollingUnaryOp : int { Sum = 0, Mean = 1, Std = 2, Min = 3, Max = 4 };
enum class RollingPairOp : int { Cov = 0, Corr = 1 };

void rolling_unary_cpu(const float* input, float* output, int lanes, int time, int window,
                       RollingUnaryOp op);

void rolling_pair_cpu(const float* lhs, const float* rhs, float* output, int lanes, int time,
                      int window, RollingPairOp op);

void delay_cpu(const float* input, float* output, int lanes, int time, int lag);

void diff_cpu(const float* input, float* output, int lanes, int time, int lag);

void ema_cpu(const float* input, float* output, int lanes, int time, int span);

} // 命名空间 factor_engine
