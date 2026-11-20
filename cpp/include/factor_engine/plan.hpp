#pragma once

#include <cstdint>
#include "factor_engine/contracts.hpp"
#include "factor_engine/rank_ic.hpp"
#include "factor_engine/metric.hpp"

namespace factor_engine {

enum class PlanOp : std::int32_t {
    Terminal = 0,
    Const = 1,
    Neg = 2,
    Abs = 3,
    Sign = 4,
    SignedSqrt = 5,
    SignedLog1p = 6,
    Add = 7,
    Sub = 8,
    Mul = 9,
    SafeDiv = 10,
    Min = 11,
    Max = 12,
    Delay = 13,
    Diff = 14,
    Ema = 15,
    RollSum = 16,
    RollMean = 17,
    RollStd = 18,
    RollMin = 19,
    RollMax = 20,
    Cov = 21,
    Corr = 22
};

enum class PlanValueType : std::int32_t { Series = 0, Scalar = 1 };

struct PlanNodePOD {
    std::int32_t op;
    std::int32_t in0;
    std::int32_t in1;
    std::int32_t param_i;
    std::int32_t field_index;
    std::int32_t slot;
    std::int32_t value_type;
    float param_f;
};

} // 命名空间 factor_engine

extern "C" {
struct FeCpuLayout {
    int float_bytes;
    int double_bytes;
    int int_bytes;
    int sort_pair_bytes;
    int unary_entry_bytes;
    int unary_moments_bytes;
    int pair_entry_bytes;
    int pair_moments_bytes;
    int max_threads;
};

int fe_execute_plan_cpu(const factor_engine::PlanNodePOD* nodes, int n_nodes,
                        const float* const* fields, int n_fields, int lanes, int time,
                        int slot_count, float* workspace, std::int64_t workspace_floats,
                        const int* root_ids, int n_roots, float* root_outputs);

int fe_score_plan_lanes_cpu(const factor_engine::PlanNodePOD* nodes, int n_nodes,
                            const float* const* fields, int n_fields, int lanes, int time,
                            int slot_count, float* workspace, std::int64_t workspace_floats,
                            const int* root_ids, int n_roots, const float* target,
                            double* lane_output);

int fe_score_plan_rank_ic_cpu(const factor_engine::PlanNodePOD* nodes, int n_nodes,
                              const float* const* fields, int n_fields, int lanes, int time,
                              int slot_count, float* workspace, std::int64_t workspace_floats,
                              const int* root_ids, int n_roots, const float* target, double* stats);

int fe_score_plan_rank_ic_accumulate_cpu(const factor_engine::PlanNodePOD* nodes, int n_nodes,
                                         const float* const* fields, int n_fields, int lanes,
                                         int time, int slot_count, float* workspace,
                                         std::int64_t workspace_floats, const int* root_ids,
                                         int n_roots, const float* target, double* stats);

// 单次 DAG 执行后直接以 root 指针计算 Pearson 和 RankIC，不导出完整因子数组。
int fe_score_plan_metrics_cpu(const factor_engine::PlanNodePOD* nodes, int n_nodes,
                              const float* const* fields, int n_fields, int lanes, int time,
                              int slot_count, float* workspace, std::int64_t workspace_floats,
                              const int* root_ids, int n_roots, const float* targets,
                              const uint8_t* eligible, int horizons, double* output);

// 返回当前动态库中的实际 sizeof，供 Python 估算内存。
int fe_cpu_layout(FeCpuLayout* out);
}
