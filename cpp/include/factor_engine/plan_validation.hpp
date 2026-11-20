#pragma once

#include "factor_engine/plan.hpp"
#include "factor_engine/contracts.hpp"

#include <limits>
#include <cstddef>

namespace factor_engine {

inline bool valid_float_storage(long long cells, int arrays) {
    return cells > 0 && arrays >= 0 &&
           (!arrays ||
            (unsigned long long)cells <= (unsigned long long)std::numeric_limits<long long>::max() /
                                             sizeof(float) / (unsigned)arrays);
}

inline bool rank_layout(int roots, int lanes, int time, long long& items, long long& segments) {
    if (roots <= 0 || lanes <= 0 || time <= 0)
        return false;

    const long long cells = (long long)lanes * time;
    if (cells > std::numeric_limits<int>::max() / roots)
        return false;
    items = cells * roots;
    segments = (long long)roots * lanes;
    return true;
}

inline bool rank_block_layout(int roots, int lanes, int time, long long& items,
                              long long& segments) {
    if (roots <= 0 || lanes <= 0 || time <= 0)
        return false;
    segments = (long long)roots * lanes;
    if (segments > std::numeric_limits<int>::max())
        return false;
    items = segments * time;
    return true;
}

inline int ceil_blocks(long long count) {
    constexpr int threads_per_block = 256;
    return static_cast<int>((count + threads_per_block - 1) / threads_per_block);
}

inline int validate_plan_nodes(const PlanNodePOD* nodes, int count, int fields, int slots,
                               const int* roots, int n_roots) {
    if (!nodes || count <= 0 || fields <= 0 || slots < 0 || !roots || n_roots <= 0)
        return factor_engine::cpu_status::InvalidArgument;

    for (int i = 0; i < count; ++i) {
        const auto& n = nodes[i];
        const auto op = static_cast<PlanOp>(n.op);
        const auto value_type = static_cast<PlanValueType>(n.value_type);
        if (op < PlanOp::Terminal || op > PlanOp::Corr ||
            (value_type != PlanValueType::Series && value_type != PlanValueType::Scalar))
            return factor_engine::cpu_status::InvalidPlan;

        if (op == PlanOp::Terminal) {
            if (n.field_index < 0 || n.field_index >= fields || value_type != PlanValueType::Series)
                return factor_engine::cpu_status::InvalidPlan;
            continue;
        }

        if (op == PlanOp::Const) {
            if (value_type != PlanValueType::Scalar)
                return factor_engine::cpu_status::InvalidPlan;
            continue;
        }

        if (n.in0 < 0 || n.in0 >= i)
            return factor_engine::cpu_status::InvalidPlan;

        const bool rolling_pair = op == PlanOp::Cov || op == PlanOp::Corr;
        const bool pair = (op >= PlanOp::Add && op <= PlanOp::Max) || rolling_pair;
        if (pair) {
            if (n.in1 < 0 || n.in1 >= i)
                return factor_engine::cpu_status::InvalidPlan;
        } else if (n.in1 != -1)
            return factor_engine::cpu_status::InvalidPlan;

        const auto input_type = static_cast<PlanValueType>(nodes[n.in0].value_type);
        if (op >= PlanOp::Delay) {
            const int minimum_window = rolling_pair ? 2 : 1;
            if (n.param_i < minimum_window || input_type != PlanValueType::Series ||
                (pair &&
                 static_cast<PlanValueType>(nodes[n.in1].value_type) != PlanValueType::Series) ||
                value_type != PlanValueType::Series)
                return factor_engine::cpu_status::InvalidPlan;
        } else {
            const bool scalar = input_type == PlanValueType::Scalar &&
                                (!pair || static_cast<PlanValueType>(nodes[n.in1].value_type) ==
                                              PlanValueType::Scalar);
            const auto expected = scalar ? PlanValueType::Scalar : PlanValueType::Series;
            if (value_type != expected)
                return factor_engine::cpu_status::InvalidPlan;
        }

        if (value_type == PlanValueType::Series && (n.slot < 0 || n.slot >= slots))
            return factor_engine::cpu_status::InvalidPlan;
    }

    for (int i = 0; i < n_roots; ++i)
        if (roots[i] < 0 || roots[i] >= count)
            return factor_engine::cpu_status::InvalidPlan;

    return factor_engine::cpu_status::Success;
}
} // 命名空间 factor_engine
