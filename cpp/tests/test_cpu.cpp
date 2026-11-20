#include "factor_engine/operators.hpp"
#include "factor_engine/plan_validation.hpp"
#include "factor_engine/contracts.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <new>
#include <vector>

namespace {

// 只让调用线程分配失败，此时还没有进入 OpenMP 并行区。
enum AllocationControl : int { AllocationFailureDisabled = -1 };

thread_local int allocations_before_failure = AllocationFailureDisabled;
} // 命名空间

void* operator new(std::size_t size) {
    if (allocations_before_failure == 0) {
        allocations_before_failure = AllocationFailureDisabled;
        throw std::bad_alloc();
    }
    if (allocations_before_failure > 0)
        --allocations_before_failure;
    if (void* memory = std::malloc(size ? size : 1))
        return memory;
    throw std::bad_alloc();
}

void operator delete(void* memory) noexcept {
    std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept {
    std::free(memory);
}

namespace {

namespace status = factor_engine::cpu_status;
using factor_engine::PlanNodePOD;
using factor_engine::PlanOp;
using factor_engine::PlanValueType;

PlanNodePOD terminal_node() {
    PlanNodePOD node{};
    node.op = static_cast<int>(PlanOp::Terminal);
    node.in0 = node.in1 = node.slot = factor_engine::NoIndex;
    node.field_index = 0;
    node.value_type = static_cast<int>(PlanValueType::Series);
    return node;
}

PlanNodePOD add_node() {
    PlanNodePOD node{};
    node.op = static_cast<int>(PlanOp::Add);
    node.in0 = node.in1 = node.slot = 0;
    node.field_index = factor_engine::NoIndex;
    node.value_type = static_cast<int>(PlanValueType::Series);
    return node;
}

struct PlanFixture {
    enum Shape { FieldCount = 1, LaneCount = 1, Time = 3, NodeCount = 2, SlotCount = 1 };

    PlanNodePOD nodes[NodeCount]{terminal_node(), add_node()};
    const float input[Time]{1, 2, 3};
    const float* fields[FieldCount]{input};
    float workspace[Time]{}, output[Time]{};
    int root = NodeCount - 1;

    int execute(float* storage, int storage_size) {
        return fe_execute_plan_cpu(nodes, NodeCount, fields, FieldCount, LaneCount, Time, SlotCount,
                                   storage, storage_size, &root, 1, output);
    }

    int execute() {
        return execute(workspace, Time);
    }
};

void test_plan_boundaries() {
    PlanFixture plan;
    assert(plan.execute() == status::Success);
    for (int i = 0; i < PlanFixture::Time; ++i)
        assert(plan.output[i] == 2 * plan.input[i]);

    plan.nodes[0].field_index = PlanFixture::FieldCount;
    assert(plan.execute() == status::InvalidPlan);

    plan.nodes[0] = terminal_node();
    plan.nodes[1].in0 = 1;
    assert(plan.execute() == status::InvalidPlan);

    plan.nodes[1] = add_node();
    plan.nodes[1].in1 = factor_engine::NoIndex;
    assert(plan.execute() == status::InvalidPlan);

    plan.nodes[1] = add_node();
    plan.nodes[1].slot = PlanFixture::SlotCount;
    assert(plan.execute() == status::InvalidPlan);

    plan.nodes[1] = add_node();
    plan.nodes[1].value_type = static_cast<int>(PlanValueType::Scalar);
    assert(plan.execute() == status::InvalidPlan);

    plan.nodes[1] = add_node();
    plan.root = PlanFixture::NodeCount;
    assert(plan.execute() == status::InvalidPlan);

    plan.root = PlanFixture::NodeCount - 1;
    plan.fields[0] = nullptr;
    assert(plan.execute() == status::MissingField);

    plan.fields[0] = plan.input;
    assert(plan.execute(nullptr, 0) == status::InvalidWorkspace);
}

void test_allocation_errors_stay_inside_c_interface() {
    PlanFixture plan;

    // vector 分配失败时，C 接口应返回错误码，不能把异常抛给 Python。
    allocations_before_failure = 0;
    assert(plan.execute() == status::AllocationFailed);

    double stats[factor_engine::SumCountWidth]{};
    const PlanNodePOD terminal = terminal_node();
    const int terminal_root = 0;
    for (int successful_allocations : {0, 1}) {
        allocations_before_failure = successful_allocations;
        assert(fe_score_plan_rank_ic_cpu(&terminal, 1, plan.fields, PlanFixture::FieldCount,
                                         PlanFixture::LaneCount, PlanFixture::Time, 0, nullptr, 0,
                                         &terminal_root, 1, plan.input,
                                         stats) == status::AllocationFailed);
    }
}

void test_storage_boundaries() {
    enum StorageShape {
        DynamicBytes = 96 * 1024,
        StaticBytes = 1040,
        RequiredBytes = DynamicBytes + StaticBytes,
        Roots = 256,
        Lanes = 32,
        Time = 23400
    };

    // 静态 shared memory 也占 block 配额，只够动态区时仍无法启动 kernel。
    assert(!factor_engine::shared_storage_fits(DynamicBytes, StaticBytes, DynamicBytes));
    assert(!factor_engine::shared_storage_fits(DynamicBytes, StaticBytes, RequiredBytes - 1));
    assert(factor_engine::shared_storage_fits(DynamicBytes, StaticBytes, RequiredBytes));
    assert(!factor_engine::shared_storage_fits(1, StaticBytes, StaticBytes - 1));

    // RankIC 的 32 位上限，与通用设备数组允许的 64 位偏移是两件事。
    long long rank_items = 0, rank_segments = 0;
    constexpr int max_index = std::numeric_limits<int>::max();
    assert(factor_engine::rank_layout(1, 1, max_index, rank_items, rank_segments));
    assert(rank_items == max_index && rank_segments == 1);
    assert(!factor_engine::rank_layout(2, 1, max_index, rank_items, rank_segments));
    assert(factor_engine::rank_layout(Roots, Lanes, Time, rank_items, rank_segments));
    assert(rank_items == 1LL * Roots * Lanes * Time && rank_segments == Roots * Lanes);
    assert(factor_engine::ceil_blocks(max_index) ==
           max_index / factor_engine::ElementwiseThreads + 1);
    assert(factor_engine::valid_float_storage(1LL + max_index, 1));
}

void test_rolling_operators() {
    enum Shape { lanes = 2, time = 5, window = 3 };

    constexpr float tolerance = 1e-6f;
    std::vector<float> x{1, 2, 3, 4, 5, 10, 20, 30, 40, 50};
    std::vector<float> y(x.size());

    factor_engine::rolling_unary_cpu(x.data(), y.data(), lanes, time, window,
                                     factor_engine::RollingUnaryOp::Mean);
    assert(std::isnan(y[0]) && std::isnan(y[1]));
    assert(std::fabs(y[2] - 2.0f) < tolerance);
    assert(std::fabs(y[4] - 4.0f) < tolerance);
    assert(std::fabs(y[7] - 20.0f) < tolerance);

    factor_engine::diff_cpu(x.data(), y.data(), lanes, time, 1);
    assert(std::isnan(y[0]) && std::isnan(y[5]));
    assert(std::fabs(y[4] - 1.0f) < tolerance);
    assert(std::fabs(y[9] - 10.0f) < tolerance);
}

void test_two_ic_joint_samples() {
    using namespace factor_engine::metrics;
    static_assert(MetricCount == 2);
    static_assert(Pearson == 0 && RankIC == 1);
    assert(fe_native_abi_version() == 4);

    constexpr int time = 6;
    constexpr double guard = 1234567;
    constexpr double tolerance = 1e-12;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float factor[time]{1, 2, nan, 4, 5, 6};
    const float target[time]{2, 4, 3, 8, 100, 12};
    const float* roots[]{factor};
    uint8_t eligible[time]{1, 1, 1, 1, 0, 1};
    std::array<double, MetricCount + 2> output{guard, guard, guard, guard};

    // 两项都只使用 joint finite 且 eligible 的样本，不能各自筛选。
    assert(fe_metric_lanes_cpu(roots, 1, target, eligible, 1, time, 1, output.data() + 1) ==
           status::Success);
    assert(std::fabs(output[1 + Pearson] - 1) < tolerance);
    assert(std::fabs(output[1 + RankIC] - 1) < tolerance);
    assert(output.front() == guard && output.back() == guard);

    // 剩余三个样本足以计算 IC，样本占比不构成额外门槛。
    eligible[5] = 0;
    assert(fe_metric_lanes_cpu(roots, 1, target, eligible, 1, time, 1, output.data() + 1) ==
           status::Success);
    assert(std::fabs(output[1 + Pearson] - 1) < tolerance);
    assert(std::fabs(output[1 + RankIC] - 1) < tolerance);

    eligible[3] = 0;
    assert(fe_metric_lanes_cpu(roots, 1, target, eligible, 1, time, 1, output.data() + 1) ==
           status::Success);
    assert(std::isnan(output[1 + Pearson]) && std::isnan(output[1 + RankIC]));
}

void test_two_ic_ties_and_constants() {
    using namespace factor_engine::metrics;
    constexpr int time = 5;
    constexpr double tolerance = 1e-12;
    float factor[time]{1, 1, 2, 3, 4};
    float target[time]{5, 4, 3, 2, 1};
    const float* roots[]{factor};
    const uint8_t eligible[time]{1, 1, 1, 1, 1};
    double output[MetricCount]{};

    // 并列值使用平均秩；删掉 ties 输出不能改变秩本身。
    assert(fe_metric_lanes_cpu(roots, 1, target, eligible, 1, time, 1, output) == status::Success);
    assert(std::fabs(output[Pearson] + 8 / std::sqrt(68.0)) < tolerance);
    assert(std::fabs(output[RankIC] + std::sqrt(0.95)) < tolerance);

    for (float& value : factor)
        value = 1;
    assert(fe_metric_lanes_cpu(roots, 1, target, eligible, 1, time, 1, output) == status::Success);
    assert(std::isnan(output[Pearson]) && std::isnan(output[RankIC]));

    for (int t = 0; t < time; ++t) {
        factor[t] = static_cast<float>(t);
        target[t] = 1;
    }
    assert(fe_metric_lanes_cpu(roots, 1, target, eligible, 1, time, 1, output) == status::Success);
    assert(std::isnan(output[Pearson]) && std::isnan(output[RankIC]));

    // 相邻 FP32 值仍有方差，不得因常数判定或相消误差返回 NaN。
    factor[0] = 1;
    for (int t = 1; t < time; ++t)
        factor[t] = std::nextafter(factor[t - 1], 2.0f);
    for (int t = 0; t < time; ++t)
        target[t] = static_cast<float>(t);
    assert(fe_metric_lanes_cpu(roots, 1, target, eligible, 1, time, 1, output) == status::Success);
    assert(std::fabs(output[Pearson] - 1) < tolerance);
    assert(std::fabs(output[RankIC] - 1) < tolerance);
}

void test_two_ic_layout() {
    using namespace factor_engine::metrics;

    enum Shape { roots_count = 2, lanes = 2, time = 3, horizons = 2 };

    constexpr double guard = 1234567;
    constexpr double tolerance = 1e-12;
    const float increasing[lanes * time]{1, 2, 3, 4, 5, 6};
    const float decreasing[lanes * time]{3, 2, 1, 6, 5, 4};
    const float* roots[roots_count]{increasing, decreasing};
    const float targets[horizons * lanes * time]{1, 2, 3, 4, 5, 6, 3, 2, 1, 6, 5, 4};
    const uint8_t eligible[horizons * lanes * time]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    constexpr int values = roots_count * lanes * horizons * MetricCount;
    std::array<double, values + 2> output;
    output.fill(guard);

    assert(fe_metric_lanes_cpu(roots, roots_count, targets, eligible, lanes, time, horizons,
                               output.data() + 1) == status::Success);
    for (int root = 0; root < roots_count; ++root)
        for (int lane = 0; lane < lanes; ++lane)
            for (int horizon = 0; horizon < horizons; ++horizon) {
                const int base = ((root * lanes + lane) * horizons + horizon) * MetricCount + 1;
                const double expected = root == horizon ? 1 : -1;
                assert(std::fabs(output[base + Pearson] - expected) < tolerance);
                assert(std::fabs(output[base + RankIC] - expected) < tolerance);
            }
    assert(output.front() == guard && output.back() == guard);
}

} // 命名空间

int main() {
    test_plan_boundaries();
    test_allocation_errors_stay_inside_c_interface();
    test_storage_boundaries();
    test_rolling_operators();
    test_two_ic_joint_samples();
    test_two_ic_ties_and_constants();
    test_two_ic_layout();
    return 0;
}
