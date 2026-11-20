#include "factor_engine/plan.hpp"
#include "factor_engine/contracts.hpp"
#include "factor_engine/scalar_math.hpp"
#include "factor_engine/operators.hpp"
#include "factor_engine/rank_ic.hpp"
#include "factor_engine/plan_validation.hpp"

#include <algorithm>
#include <cmath>
#include <new>
#include <vector>

namespace {

using factor_engine::scalar_math::scalar_binary;
using factor_engine::scalar_math::scalar_unary;

struct Value {
    bool scalar = false;
    float s = 0.0f;
    const float* p = nullptr;
};

int execute_plan(const factor_engine::PlanNodePOD* nodes, int n_nodes, const float* const* fields,
                 int n_fields, int lanes, int time, int slot_count, float* workspace,
                 std::int64_t workspace_floats, const int* root_ids, int n_roots,
                 std::vector<Value>& values) {
    if (!nodes || n_nodes <= 0 || !fields || n_fields <= 0 || lanes <= 0 || time <= 0 ||
        slot_count < 0 || !root_ids || n_roots <= 0)
        return factor_engine::cpu_status::InvalidArgument;

    const long long cells = static_cast<long long>(lanes) * time;
    const int validation =
        factor_engine::validate_plan_nodes(nodes, n_nodes, n_fields, slot_count, root_ids, n_roots);
    if (validation)
        return validation;
    if (!factor_engine::valid_float_storage(cells, slot_count) ||
        !factor_engine::valid_float_storage(cells, n_roots))
        return factor_engine::cpu_status::InvalidWorkspace;

    for (int i = 0; i < n_fields; ++i)
        if (!fields[i])
            return factor_engine::cpu_status::MissingField;

    const std::int64_t required_workspace = static_cast<std::int64_t>(slot_count) * cells;
    if (required_workspace > 0 && (!workspace || workspace_floats < required_workspace))
        return factor_engine::cpu_status::InvalidWorkspace;

    values.resize(static_cast<std::size_t>(n_nodes));
    using factor_engine::PlanOp;
    using factor_engine::PlanValueType;

    // 上面已检查字段、输入、slot 和根节点的下标，执行时直接取值。
    for (int i = 0; i < n_nodes; ++i) {
        const auto& n = nodes[i];
        const PlanOp op = static_cast<PlanOp>(n.op);

        if (op == PlanOp::Terminal) {
            values[i] = {false, 0.0f, fields[n.field_index]};
            continue;
        }

        if (op == PlanOp::Const) {
            values[i] = {true, n.param_f, nullptr};
            continue;
        }

        const Value& a = values[n.in0];
        if (static_cast<PlanValueType>(n.value_type) == PlanValueType::Scalar) {
            const float result = op <= PlanOp::SignedLog1p
                                     ? scalar_unary(op, a.s)
                                     : scalar_binary(op, a.s, values[n.in1].s);
            values[i] = {true, result, nullptr};
            continue;
        }

        float* out = workspace + static_cast<long long>(n.slot) * cells;
        switch (op) {
        case PlanOp::Neg:
        case PlanOp::Abs:
        case PlanOp::Sign:
        case PlanOp::SignedSqrt:
        case PlanOp::SignedLog1p: {
#pragma omp parallel for schedule(static)
            for (long long k = 0; k < cells; ++k)
                out[k] = scalar_unary(op, a.p[k]);
            break;
        }
        case PlanOp::Add:
        case PlanOp::Sub:
        case PlanOp::Mul:
        case PlanOp::SafeDiv:
        case PlanOp::Min:
        case PlanOp::Max: {
            const Value& b = values[n.in1];
#pragma omp parallel for schedule(static)
            for (long long k = 0; k < cells; ++k)
                out[k] = scalar_binary(op, a.scalar ? a.s : a.p[k], b.scalar ? b.s : b.p[k]);
            break;
        }
        case PlanOp::Delay:
            factor_engine::delay_cpu(a.p, out, lanes, time, n.param_i);
            break;
        case PlanOp::Diff:
            factor_engine::diff_cpu(a.p, out, lanes, time, n.param_i);
            break;
        case PlanOp::Ema:
            factor_engine::ema_cpu(a.p, out, lanes, time, n.param_i);
            break;
        case PlanOp::RollSum:
        case PlanOp::RollMean:
        case PlanOp::RollStd:
        case PlanOp::RollMin:
        case PlanOp::RollMax: {
            auto rop = factor_engine::RollingUnaryOp::Sum;
            switch (op) {
            case PlanOp::RollMean:
                rop = factor_engine::RollingUnaryOp::Mean;
                break;
            case PlanOp::RollStd:
                rop = factor_engine::RollingUnaryOp::Std;
                break;
            case PlanOp::RollMin:
                rop = factor_engine::RollingUnaryOp::Min;
                break;
            case PlanOp::RollMax:
                rop = factor_engine::RollingUnaryOp::Max;
                break;
            default:
                break;
            }
            factor_engine::rolling_unary_cpu(a.p, out, lanes, time, n.param_i, rop);
            break;
        }
        case PlanOp::Cov:
        case PlanOp::Corr:
            factor_engine::rolling_pair_cpu(a.p, values[n.in1].p, out, lanes, time, n.param_i,
                                            op == PlanOp::Cov ? factor_engine::RollingPairOp::Cov
                                                              : factor_engine::RollingPairOp::Corr);
            break;
        default:
            return factor_engine::cpu_status::UnsupportedOperation;
        }

        values[i] = {false, 0.0f, out};
    }

    return factor_engine::cpu_status::Success;
}
} // 命名空间

extern "C" int fe_execute_plan_cpu(const factor_engine::PlanNodePOD* nodes, int n_nodes,
                                   const float* const* fields, int n_fields, int lanes, int time,
                                   int slot_count, float* workspace, std::int64_t workspace_floats,
                                   const int* root_ids, int n_roots, float* root_outputs) {
    if (!root_outputs)
        return factor_engine::cpu_status::InvalidArgument;

    try {
        std::vector<Value> values;
        const int rc = execute_plan(nodes, n_nodes, fields, n_fields, lanes, time, slot_count,
                                    workspace, workspace_floats, root_ids, n_roots, values);
        if (rc)
            return rc;

        const long long cells = static_cast<long long>(lanes) * time;
        for (int r = 0; r < n_roots; ++r) {
            const Value& v = values[root_ids[r]];
            float* dst = root_outputs + static_cast<long long>(r) * cells;
            if (v.scalar)
                std::fill(dst, dst + cells, v.s);
            else
                std::copy(v.p, v.p + cells, dst);
        }
        return factor_engine::cpu_status::Success;
    } catch (const std::bad_alloc&) {
        return factor_engine::cpu_status::AllocationFailed;
    } catch (...) {
        return factor_engine::cpu_status::InternalError;
    }
}

extern "C" int fe_score_plan_lanes_cpu(const factor_engine::PlanNodePOD* nodes, int n_nodes,
                                       const float* const* fields, int n_fields, int lanes,
                                       int time, int slot_count, float* workspace,
                                       std::int64_t workspace_floats, const int* root_ids,
                                       int n_roots, const float* target, double* lane_output) {
    if (!target || !lane_output)
        return factor_engine::cpu_status::InvalidArgument;

    try {
        std::vector<Value> values;
        const int rc = execute_plan(nodes, n_nodes, fields, n_fields, lanes, time, slot_count,
                                    workspace, workspace_floats, root_ids, n_roots, values);
        if (rc)
            return rc;

        std::vector<const float*> roots(static_cast<std::size_t>(n_roots));
        for (int r = 0; r < n_roots; ++r)
            roots[r] = values[root_ids[r]].scalar ? nullptr : values[root_ids[r]].p;
        return fe_rank_ic_lanes_cpu(roots.data(), n_roots, target, lanes, time, lane_output);
    } catch (const std::bad_alloc&) {
        return factor_engine::cpu_status::AllocationFailed;
    } catch (...) {
        return factor_engine::cpu_status::InternalError;
    }
}

extern "C" int fe_score_plan_rank_ic_cpu(const factor_engine::PlanNodePOD* nodes, int n_nodes,
                                         const float* const* fields, int n_fields, int lanes,
                                         int time, int slot_count, float* workspace,
                                         std::int64_t workspace_floats, const int* root_ids,
                                         int n_roots, const float* target, double* stats) {
    if (!target || !stats)
        return factor_engine::cpu_status::InvalidArgument;

    try {
        std::vector<Value> values;
        const int rc = execute_plan(nodes, n_nodes, fields, n_fields, lanes, time, slot_count,
                                    workspace, workspace_floats, root_ids, n_roots, values);
        if (rc)
            return rc;

        std::vector<const float*> roots(static_cast<std::size_t>(n_roots));
        for (int r = 0; r < n_roots; ++r)
            roots[r] = values[root_ids[r]].scalar ? nullptr : values[root_ids[r]].p;
        return fe_rank_ic_stats_cpu(roots.data(), n_roots, target, lanes, time, stats);
    } catch (const std::bad_alloc&) {
        return factor_engine::cpu_status::AllocationFailed;
    } catch (...) {
        return factor_engine::cpu_status::InternalError;
    }
}

extern "C" int fe_score_plan_rank_ic_accumulate_cpu(
    const factor_engine::PlanNodePOD* nodes, int n_nodes, const float* const* fields, int n_fields,
    int lanes, int time, int slot_count, float* workspace, std::int64_t workspace_floats,
    const int* root_ids, int n_roots, const float* target, double* stats) {
    if (!target || !stats || n_roots <= 0)
        return factor_engine::cpu_status::InvalidArgument;
    for (int r = 0; r < n_roots; ++r) {
        const double sum = stats[factor_engine::SumCountWidth * r + factor_engine::Sum],
                     count = stats[factor_engine::SumCountWidth * r + factor_engine::Count];
        if (!std::isfinite(sum) || !std::isfinite(count) || count < 0 ||
            std::floor(count) != count ||
            std::fabs(sum) > count + factor_engine::CorrelationTolerance * count)
            return factor_engine::cpu_status::InvalidStatistics;
    }

    try {
        std::vector<Value> values;
        const int rc = execute_plan(nodes, n_nodes, fields, n_fields, lanes, time, slot_count,
                                    workspace, workspace_floats, root_ids, n_roots, values);
        if (rc)
            return rc;

        std::vector<const float*> roots(static_cast<std::size_t>(n_roots));
        for (int r = 0; r < n_roots; ++r)
            roots[r] = values[root_ids[r]].scalar ? nullptr : values[root_ids[r]].p;
        return fe_rank_ic_accumulate_cpu(roots.data(), n_roots, target, lanes, time, stats);
    } catch (const std::bad_alloc&) {
        return factor_engine::cpu_status::AllocationFailed;
    } catch (...) {
        return factor_engine::cpu_status::InternalError;
    }
}

extern "C" int fe_score_plan_metrics_cpu(const factor_engine::PlanNodePOD* nodes, int n_nodes,
                                         const float* const* fields, int n_fields, int lanes,
                                         int time, int slots, float* workspace,
                                         std::int64_t workspace_floats, const int* roots,
                                         int n_roots, const float* targets, const uint8_t* eligible,
                                         int horizons, double* output) {
    if (!targets || !eligible || !output ||
        !factor_engine::metrics::metric_layout_fits(n_roots, lanes, time, horizons))
        return factor_engine::cpu_status::InvalidArgument;

    try {
        std::vector<Value> values;
        const int rc = execute_plan(nodes, n_nodes, fields, n_fields, lanes, time, slots, workspace,
                                    workspace_floats, roots, n_roots, values);
        if (rc)
            return rc;

        std::vector<const float*> pointers(n_roots);
        // 多期限评分只接受 Series root；常数 Series 由普通算子产生。
        for (int root = 0; root < n_roots; ++root) {
            const auto& value = values[roots[root]];
            if (value.scalar)
                return factor_engine::cpu_status::InvalidPlan;
            pointers[root] = value.p;
        }
        return fe_metric_lanes_cpu(pointers.data(), n_roots, targets, eligible, lanes, time,
                                   horizons, output);
    } catch (const std::bad_alloc&) {
        return factor_engine::cpu_status::AllocationFailed;
    } catch (...) {
        return factor_engine::cpu_status::InternalError;
    }
}
