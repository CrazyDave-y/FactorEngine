#include "factor_engine/contracts.hpp"
#include "factor_engine/rank_ic.hpp"
#include "factor_engine/metric.hpp"

#include <type_traits>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <new>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t target_rank_cache_capacity = 8;

struct SortScratch {
    std::vector<std::pair<float, int>> factor_order, target_order;
    std::vector<double> factor_ranks, target_ranks;
    std::vector<int> valid_times;
};

struct TargetRankCache {
    std::vector<int> valid_times;
    std::vector<double> ranks;
};

std::atomic<std::uint64_t> retained_rank_bytes{0};

struct RankThreadStorage {
    SortScratch scratch;
    // 容量跨调用保留；有效条目只在当前 lane 的各 root 之间复用。
    std::array<TargetRankCache, target_rank_cache_capacity> cache;
    std::uint64_t accounted_bytes = 0;

    void account() noexcept {
        std::uint64_t bytes =
            (scratch.factor_order.capacity() + scratch.target_order.capacity()) *
                sizeof(std::pair<float, int>) +
            (scratch.factor_ranks.capacity() + scratch.target_ranks.capacity()) * sizeof(double) +
            scratch.valid_times.capacity() * sizeof(int);
        for (const auto& entry : cache)
            bytes += entry.valid_times.capacity() * sizeof(int) +
                     entry.ranks.capacity() * sizeof(double);
        if (bytes > accounted_bytes)
            retained_rank_bytes.fetch_add(bytes - accounted_bytes, std::memory_order_relaxed);
        else if (bytes < accounted_bytes)
            retained_rank_bytes.fetch_sub(accounted_bytes - bytes, std::memory_order_relaxed);
        accounted_bytes = bytes;
    }

    ~RankThreadStorage() {
        // 等 vector 的内存真正释放后，再扣掉保留字节数。
        {
            SortScratch released;
            std::swap(released, scratch);
            std::array<TargetRankCache, target_rank_cache_capacity> released_cache;
            released_cache.swap(cache);
        }
        retained_rank_bytes.fetch_sub(accounted_bytes, std::memory_order_relaxed);
    }
};

struct AccountRankCapacity {
    RankThreadStorage& storage;

    ~AccountRankCapacity() {
        storage.account();
    }
};

void average_ranks(std::vector<std::pair<float, int>>& order, std::vector<double>& ranks) {
    std::sort(order.begin(), order.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    for (std::size_t first = 0; first < order.size();) {
        std::size_t end = first + 1;
        while (end < order.size() && order[end].first == order[first].first)
            ++end;

        // 保留两倍平均秩，后面用同一尺度中心化。
        const double doubled_rank = static_cast<double>(first + end - 1);
        for (std::size_t i = first; i < end; ++i)
            ranks[order[i].second] = doubled_rank;
        first = end;
    }
}

// 将 FP32 输入转为 FP64，先求均值，再对中心化后的值累计协方差，避免大数相减。
double lane_pearson(const float* x, const float* y, const uint8_t* eligible, int time) {
    double sx = 0, sy = 0;
    std::size_t count = 0;
    for (int t = 0; t < time; ++t) {
        if ((!eligible || eligible[t]) && std::isfinite(x[t]) && std::isfinite(y[t])) {
            sx += double(x[t]);
            sy += double(y[t]);
            ++count;
        }
    }

    if (count < factor_engine::MinCorrelationPairs)
        return std::numeric_limits<double>::quiet_NaN();

    const double mx = sx / count, my = sy / count;
    double xx = 0, yy = 0, xy = 0;
    for (int t = 0; t < time; ++t) {
        if ((!eligible || eligible[t]) && std::isfinite(x[t]) && std::isfinite(y[t])) {
            const double dx = double(x[t]) - mx, dy = double(y[t]) - my;
            xx += dx * dx;
            yy += dy * dy;
            xy += dx * dy;
        }
    }

    const double denominator = std::sqrt(xx * yy);
    return denominator > 0 ? xy / denominator : std::numeric_limits<double>::quiet_NaN();
}

template <bool WithPearson = false>
double lane_rank_ic(const float* factor, const float* target, int time, SortScratch& scratch,
                    std::array<TargetRankCache, target_rank_cache_capacity>& cache,
                    int& cache_entries, const uint8_t* eligible = nullptr,
                    double* metrics = nullptr) {
    if (!factor)
        return std::numeric_limits<double>::quiet_NaN();

    scratch.factor_order.clear();
    scratch.target_order.clear();
    scratch.valid_times.clear();

    for (int t = 0; t < time; ++t) {
        if ((!eligible || eligible[t]) && std::isfinite(factor[t]) && std::isfinite(target[t])) {
            const int i = static_cast<int>(scratch.factor_order.size());
            scratch.factor_order.emplace_back(factor[t], i);
            scratch.target_order.emplace_back(target[t], i);
            scratch.valid_times.push_back(t);
        }
    }

    const std::size_t n = scratch.factor_order.size();
    if constexpr (!WithPearson)
        if (n < factor_engine::MinCorrelationPairs)
            return std::numeric_limits<double>::quiet_NaN();

    scratch.factor_ranks.resize(n);
    scratch.target_ranks.resize(n);
    average_ranks(scratch.factor_order, scratch.factor_ranks);

    const std::vector<double>* target_ranks = nullptr;
    for (int i = 0; i < cache_entries; ++i) {
        if (cache[i].valid_times == scratch.valid_times) {
            target_ranks = &cache[i].ranks;
            break;
        }
    }

    if (!target_ranks) {
        average_ranks(scratch.target_order, scratch.target_ranks);
        target_ranks = &scratch.target_ranks;
        if (cache_entries < static_cast<int>(cache.size())) {
            cache[cache_entries].valid_times = scratch.valid_times;
            cache[cache_entries].ranks = scratch.target_ranks;
            target_ranks = &cache[cache_entries++].ranks;
        }
    }

    const double center = n ? static_cast<double>(n - 1) : 0;
    using SumType = std::conditional_t<WithPearson, std::int64_t, double>;
    SumType xx = 0, yy = 0, xy = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const SumType x = static_cast<SumType>(scratch.factor_ranks[i] - center);
        const SumType y = static_cast<SumType>((*target_ranks)[i] - center);
        xx += x * x;
        yy += y * y;
        xy += x * y;
    }

    const double denominator = std::sqrt(double(xx) * double(yy));
    const double result = n >= factor_engine::MinCorrelationPairs && denominator > 0
                              ? double(xy) / denominator
                              : std::numeric_limits<double>::quiet_NaN();
    if constexpr (WithPearson) {
        metrics[factor_engine::metrics::Pearson] = lane_pearson(factor, target, eligible, time);
        metrics[factor_engine::metrics::RankIC] = result;
    }

    return result;
}

int rank_ic_reduce(const float* const* roots, int n_roots, const float* target, int lanes, int time,
                   double* stats, bool accumulate, double* lane_output = nullptr) {
    if (!roots || n_roots <= 0 || !target || lanes <= 0 || time <= 0 || !stats)
        return factor_engine::cpu_status::InvalidArgument;

    if (accumulate)
        for (int r = 0; r < n_roots; ++r) {
            const double sum = stats[factor_engine::SumCountWidth * r + factor_engine::Sum],
                         count = stats[factor_engine::SumCountWidth * r + factor_engine::Count];
            if (!std::isfinite(sum) || !std::isfinite(count) || count < 0 ||
                std::floor(count) != count ||
                std::fabs(sum) > count + factor_engine::CorrelationTolerance * count)
                return factor_engine::cpu_status::InvalidStatistics;
        }

    const long long tasks = static_cast<long long>(n_roots) * lanes;
    std::vector<double> lane_scores(static_cast<std::size_t>(tasks));
    std::atomic<int> failure{factor_engine::cpu_status::Success};

#pragma omp parallel
    {
        thread_local RankThreadStorage storage;
        AccountRankCapacity account{storage};
        auto& scratch = storage.scratch;
        auto& cache = storage.cache;

#pragma omp for schedule(static) nowait
        for (int lane = 0; lane < lanes; ++lane) {
            try {
                int cache_entries = 0;
                const long long offset = static_cast<long long>(lane) * time;
                for (int root = 0; root < n_roots; ++root)
                    lane_scores[static_cast<long long>(root) * lanes + lane] =
                        lane_rank_ic(roots[root] ? roots[root] + offset : nullptr, target + offset,
                                     time, scratch, cache, cache_entries);
            } catch (const std::bad_alloc&) {
                failure.store(factor_engine::cpu_status::AllocationFailed,
                              std::memory_order_relaxed);
            } catch (...) {
                failure.store(factor_engine::cpu_status::InternalError, std::memory_order_relaxed);
            }
        }
    }

    if (const int error = failure.load(std::memory_order_relaxed))
        return error;
    if (lane_output)
        std::copy(lane_scores.begin(), lane_scores.end(), lane_output);

    for (int root = 0; root < n_roots; ++root) {
        double sum =
            accumulate ? stats[factor_engine::SumCountWidth * root + factor_engine::Sum] : 0;
        double count =
            accumulate ? stats[factor_engine::SumCountWidth * root + factor_engine::Count] : 0;
        for (int lane = 0; lane < lanes; ++lane) {
            const double value = lane_scores[static_cast<long long>(root) * lanes + lane];
            if (std::isfinite(value)) {
                sum += value;
                ++count;
            }
        }
        stats[factor_engine::SumCountWidth * root + factor_engine::Sum] = sum;
        stats[factor_engine::SumCountWidth * root + factor_engine::Count] = count;
    }

    return factor_engine::cpu_status::Success;
}
} // 命名空间

extern "C" std::uint64_t fe_cpu_rank_scratch_retained_bytes() {
    return retained_rank_bytes.load(std::memory_order_relaxed);
}

extern "C" int fe_rank_ic_lanes_cpu(const float* const* roots, int n_roots, const float* target,
                                    int lanes, int time, double* lane_output) {
    if (!lane_output || n_roots < 1)
        return factor_engine::cpu_status::InvalidArgument;

    try {
        std::vector<double> totals(static_cast<std::size_t>(n_roots) *
                                   factor_engine::SumCountWidth);
        return rank_ic_reduce(roots, n_roots, target, lanes, time, totals.data(), false,
                              lane_output);
    } catch (const std::bad_alloc&) {
        return factor_engine::cpu_status::AllocationFailed;
    } catch (...) {
        return factor_engine::cpu_status::InternalError;
    }
}

extern "C" int fe_rank_ic_stats_cpu(const float* const* roots, int n_roots, const float* target,
                                    int lanes, int time, double* stats) {
    try {
        return rank_ic_reduce(roots, n_roots, target, lanes, time, stats, false);
    } catch (const std::bad_alloc&) {
        return factor_engine::cpu_status::AllocationFailed;
    } catch (...) {
        return factor_engine::cpu_status::InternalError;
    }
}

extern "C" int fe_rank_ic_accumulate_cpu(const float* const* roots, int n_roots,
                                         const float* target, int lanes, int time, double* stats) {
    try {
        return rank_ic_reduce(roots, n_roots, target, lanes, time, stats, true);
    } catch (const std::bad_alloc&) {
        return factor_engine::cpu_status::AllocationFailed;
    } catch (...) {
        return factor_engine::cpu_status::InternalError;
    }
}

extern "C" int fe_pearson_ic_lanes_cpu(const float* const* roots, int n_roots, const float* target,
                                       const uint8_t* eligible, int lanes, int time,
                                       double* output) {
    if (!roots || n_roots < 1 || !target || !output || lanes < 1 || time < 1)
        return factor_engine::cpu_status::InvalidArgument;

    for (int root = 0; root < n_roots; ++root)
        if (!roots[root])
            return factor_engine::cpu_status::MissingField;

#pragma omp parallel for schedule(static)
    for (int lane = 0; lane < lanes; ++lane) {
        const auto base = static_cast<std::int64_t>(lane) * time;
        for (int root = 0; root < n_roots; ++root)
            output[static_cast<std::int64_t>(root) * lanes + lane] = lane_pearson(
                roots[root] + base, target + base, eligible ? eligible + base : nullptr, time);
    }

    return factor_engine::cpu_status::Success;
}

extern "C" int fe_metric_lanes_cpu(const float* const* roots, int n_roots, const float* targets,
                                   const uint8_t* eligible, int lanes, int time, int horizons,
                                   double* output) {
    if (!roots || !targets || !eligible || !output ||
        !factor_engine::metrics::metric_layout_fits(n_roots, lanes, time, horizons))
        return factor_engine::cpu_status::InvalidArgument;

    for (int root = 0; root < n_roots; ++root)
        if (!roots[root])
            return factor_engine::cpu_status::MissingField;

    std::atomic<int> failure{factor_engine::cpu_status::Success};

#pragma omp parallel
    {
        thread_local RankThreadStorage storage;
        AccountRankCapacity account{storage};

#pragma omp for schedule(static)
        for (int lane = 0; lane < lanes; ++lane) {
            // OpenMP 区域中异常不能跨线程传播；错误时整批输出作废。
            try {
                for (int h = 0; h < horizons; ++h) {
                    int cache_entries = 0;
                    const auto target_base = (static_cast<std::int64_t>(h) * lanes + lane) * time;
                    for (int root = 0; root < n_roots; ++root) {
                        const auto result =
                            ((static_cast<std::int64_t>(root) * lanes + lane) * horizons + h) *
                            factor_engine::metrics::MetricCount;
                        lane_rank_ic<true>(roots[root] + static_cast<std::int64_t>(lane) * time,
                                           targets + target_base, time, storage.scratch,
                                           storage.cache, cache_entries, eligible + target_base,
                                           output + result);
                    }
                }
            } catch (const std::bad_alloc&) {
                failure.store(factor_engine::cpu_status::AllocationFailed,
                              std::memory_order_relaxed);
            } catch (...) {
                failure.store(factor_engine::cpu_status::InternalError, std::memory_order_relaxed);
            }
        }
    }

    return failure.load(std::memory_order_relaxed);
}
