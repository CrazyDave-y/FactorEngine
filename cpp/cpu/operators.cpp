#include "factor_engine/operators.hpp"
#include "factor_engine/contracts.hpp"
#include "factor_engine/centered_pair.hpp"
#include "factor_engine/centered_unary.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <exception>
#include <limits>
#include <vector>

namespace factor_engine {

namespace {

inline bool finite(float x) {
    return std::isfinite(x);
}

inline float qnan() {
    return std::numeric_limits<float>::quiet_NaN();
}

void capture_parallel_failure(std::exception_ptr& failure) {
#pragma omp critical(fe_operator_failure)
    {
        if (!failure)
            failure = std::current_exception();
    }
}
} // 命名空间

void rolling_unary_cpu(const float* input, float* output, int lanes, int time, int window,
                       RollingUnaryOp op) {
    if (op == RollingUnaryOp::Std) {
        if (window > time) {
            std::fill(output, output + (long long)lanes * time, qnan());
            return;
        }

        std::exception_ptr failure;
#pragma omp parallel
        {
            std::vector<UnaryEntry> back;
            std::vector<UnaryMoments> front;
            bool ready = false;
            try {
                back.resize(window);
                front.resize(window);
                ready = true;
            } catch (...) {
                capture_parallel_failure(failure);
            }

#pragma omp for schedule(static)
            for (int lane = 0; lane < lanes; ++lane) {
                // 分配失败的线程仍须经过同一个 worksharing 区域和 barrier。
                if (!ready)
                    continue;
                const long long base = (long long)lane * time;
                UnaryWindowQueue queue(back.data(), front.data(), window);

                for (int t = 0; t < time; ++t) {
                    output[base + t] = qnan();
                    queue.push(input[base + t]);

                    if (t + 1 < window)
                        continue;

                    const auto moments = queue.value();
                    if (moments.count != window)
                        continue;

                    const double value = std::sqrt(std::max(0.0, moments.m2 / window));
                    const float result = (float)value;
                    output[base + t] = finite(result) ? result : qnan();
                }
            }
        }
        if (failure)
            std::rethrow_exception(failure);
        return;
    }

    std::exception_ptr failure;
#pragma omp parallel for schedule(static)
    for (int lane = 0; lane < lanes; ++lane) {
        const float* x = input + static_cast<long long>(lane) * time;
        float* y = output + static_cast<long long>(lane) * time;
        std::fill(y, y + time, qnan());

        if (op == RollingUnaryOp::Min || op == RollingUnaryOp::Max) {
            try {
                std::deque<int> extrema_candidates;
                int invalid_count = 0;

                for (int t = 0; t < time; ++t) {
                    const float v = x[t];
                    if (!finite(v)) {
                        ++invalid_count;
                    } else {
                        if (op == RollingUnaryOp::Min)
                            while (!extrema_candidates.empty() && x[extrema_candidates.back()] >= v)
                                extrema_candidates.pop_back();
                        else
                            while (!extrema_candidates.empty() && x[extrema_candidates.back()] <= v)
                                extrema_candidates.pop_back();
                        extrema_candidates.push_back(t);
                    }

                    if (t >= window) {
                        const int expired_t = t - window;
                        if (!finite(x[expired_t]))
                            --invalid_count;
                        if (!extrema_candidates.empty() && extrema_candidates.front() == expired_t)
                            extrema_candidates.pop_front();
                    }

                    if (t + 1 >= window && invalid_count == 0 && !extrema_candidates.empty())
                        y[t] = x[extrema_candidates.front()];
                }
            } catch (...) {
                capture_parallel_failure(failure);
            }
            continue;
        }

        double sum = 0.0;
        int valid_count = 0;
        // 窗口内相邻值变化为零时，直接保留常数路径的精确值。
        int adjacent_changes = 0;

        for (int t = 0; t < time; ++t) {
            const float v = x[t];
            if (t > 0)
                adjacent_changes += (x[t] != x[t - 1]);
            if (t >= window)
                adjacent_changes -= (x[t - window + 1] != x[t - window]);
            if (finite(v)) {
                sum += v;
                ++valid_count;
            }

            if (t >= window) {
                const float expired_value = x[t - window];
                if (finite(expired_value)) {
                    sum -= expired_value;
                    --valid_count;
                }
            }

            if (t + 1 < window || valid_count != window)
                continue;

            if (adjacent_changes == 0) {
                const float exact = op == RollingUnaryOp::Mean
                                        ? v
                                        : static_cast<float>(static_cast<double>(v) * window);
                y[t] = finite(exact) ? exact : qnan();
                continue;
            }

            const double value = op == RollingUnaryOp::Sum ? sum : sum / window;
            y[t] = std::isfinite(value) ? static_cast<float>(value) : qnan();
        }
    }
    if (failure)
        std::rethrow_exception(failure);
}

void rolling_pair_cpu(const float* lhs, const float* rhs, float* output, int lanes, int time,
                      int window, RollingPairOp op) {
    if (window > time) {
        std::fill(output, output + (long long)lanes * time, qnan());
        return;
    }

    std::exception_ptr failure;
#pragma omp parallel
    {
        std::vector<PairEntry> back;
        std::vector<PairMoments> front;
        bool ready = false;
        try {
            back.resize(window);
            front.resize(window);
            ready = true;
        } catch (...) {
            capture_parallel_failure(failure);
        }

#pragma omp for schedule(static)
        for (int lane = 0; lane < lanes; ++lane) {
            if (!ready)
                continue;
            const float* a = lhs + static_cast<long long>(lane) * time;
            const float* b = rhs + static_cast<long long>(lane) * time;
            float* y = output + static_cast<long long>(lane) * time;
            std::fill(y, y + time, qnan());
            PairWindowQueue queue(back.data(), front.data(), window);

            for (int t = 0; t < time; ++t) {
                const float av = a[t], bv = b[t];
                queue.push(av, bv);

                if (t + 1 < window)
                    continue;

                const auto moments = queue.value();
                if (moments.count != window)
                    continue;

                double value = moments.cross / window;
                if (op == RollingPairOp::Corr) {
                    const double den = std::sqrt((moments.m2_a / window) * (moments.m2_b / window));
                    if (!(den > factor_engine::CorrelationDenominatorEpsilon))
                        continue;
                    value /= den;
                }

                y[t] = std::isfinite(value) ? static_cast<float>(value) : qnan();
            }
        }
    }
    if (failure)
        std::rethrow_exception(failure);
}

void delay_cpu(const float* input, float* output, int lanes, int time, int lag) {
#pragma omp parallel for schedule(static)
    for (int lane = 0; lane < lanes; ++lane) {
        const float* x = input + static_cast<long long>(lane) * time;
        float* y = output + static_cast<long long>(lane) * time;
        std::fill(y, y + time, qnan());

        for (int t = lag; t < time; ++t)
            y[t] = finite(x[t - lag]) ? x[t - lag] : qnan();
    }
}

void diff_cpu(const float* input, float* output, int lanes, int time, int lag) {
#pragma omp parallel for schedule(static)
    for (int lane = 0; lane < lanes; ++lane) {
        const float* x = input + static_cast<long long>(lane) * time;
        float* y = output + static_cast<long long>(lane) * time;
        std::fill(y, y + time, qnan());

        for (int t = lag; t < time; ++t) {
            if (finite(x[t]) && finite(x[t - lag])) {
                const float v = x[t] - x[t - lag];
                y[t] = finite(v) ? v : qnan();
            }
        }
    }
}

void ema_cpu(const float* input, float* output, int lanes, int time, int span) {
    const float alpha = 2.0f / (span + 1.0f);

#pragma omp parallel for schedule(static)
    // FMA 固定融合输入项。换一种融合方式，即使只差一位也可能改变排名。
    for (int lane = 0; lane < lanes; ++lane) {
        const float* x = input + static_cast<long long>(lane) * time;
        float* y = output + static_cast<long long>(lane) * time;
        bool has_state = false;
        float state = 0;

        for (int t = 0; t < time; ++t) {
            if (!finite(x[t])) {
                y[t] = qnan();
                continue;
            }

            if (!has_state) {
                state = x[t];
                has_state = true;
            } else {
                const float next_state = std::fma(alpha, x[t], (1 - alpha) * state);
                if (finite(next_state))
                    state = next_state;
            }

            y[t] = state;
        }
    }
}
} // 命名空间 factor_engine
