#pragma once

#include "factor_engine/plan.hpp"

#include <cstddef>
#include <vector>
#ifdef __CUDACC__
#include "device_common.cuh"
#endif

namespace factor_engine::cuda {
enum FusionLimit : int { POINTWISE_FUSION_LIMIT = 8, MAX_KERNEL_ARGUMENT_BYTES = 4096 };

enum OperandSource : int { PreviousStage = 0, InputArray = -1, HostScalar = -2 };

inline bool fusion_pointwise_series(const factor_engine::PlanNodePOD& node) {
    using factor_engine::PlanOp;
    using factor_engine::PlanValueType;
    const auto op = static_cast<PlanOp>(node.op);
    return op >= PlanOp::Neg && op <= PlanOp::Max &&
           static_cast<PlanValueType>(node.value_type) == PlanValueType::Series;
}

// 只省去连续链的中间结果写回：中间值恰好只被下一节点引用一次。
// root 必须写回，即使后续节点仍会使用它。相同值用于二元算子的两端时，按两次引用计算，
// 宁可保留一次 kernel 启动，也不能把多处引用的值当成可省去写回的中间值。
// 调用方已经执行 validate_plan_nodes；时间 O(nodes + roots)，内存 O(nodes)。
inline std::vector<unsigned char>
build_pointwise_fusion_schedule(const factor_engine::PlanNodePOD* nodes, int count,
                                const int* roots, int n_roots) {
    using factor_engine::PlanOp;
    std::vector<unsigned char> spans((std::size_t)count, 0), protected_root((std::size_t)count, 0),
        consumers((std::size_t)count, 0);
    for (int i = 0; i < n_roots; ++i)
        protected_root[roots[i]] = 1;
    for (int i = 0; i < count; ++i) {
        const auto op = static_cast<PlanOp>(nodes[i].op);
        if (op < PlanOp::Neg)
            continue;
        auto consume = [&](int input) {
            if (consumers[input] < 2)
                ++consumers[input];
        };
        consume(nodes[i].in0);
        if ((op >= PlanOp::Add && op <= PlanOp::Max) || op >= PlanOp::Cov)
            consume(nodes[i].in1);
    }

    for (int first = 0; first < count;) {
        if (!fusion_pointwise_series(nodes[first])) {
            ++first;
            continue;
        }
        int length = 1;
        while (length < POINTWISE_FUSION_LIMIT && first + length < count) {
            const int previous = first + length - 1, next = previous + 1;
            if (protected_root[previous] || consumers[previous] != 1 ||
                !fusion_pointwise_series(nodes[next]) ||
                (nodes[next].in0 != previous && nodes[next].in1 != previous))
                break;
            ++length;
        }
        if (length >= 2)
            spans[first] = (unsigned char)length;
        first += length;
    }

    return spans;
}

#ifdef __CUDACC__

struct FusedPointwiseOperand {
    const float* pointer = nullptr;
    float scalar = 0.0f;
    OperandSource source = HostScalar;
};

struct FusedPointwiseStage {
    FusedPointwiseOperand a, b;
    factor_engine::PlanOp op = factor_engine::PlanOp::Terminal;
};

struct FusedPointwiseGroup {
    FusedPointwiseStage stages[POINTWISE_FUSION_LIMIT];
    float* output = nullptr;
};

static_assert(sizeof(FusedPointwiseGroup) <= MAX_KERNEL_ARGUMENT_BYTES,
              "bounded kernel argument payload");

__device__ __forceinline__ float pointwise_apply(factor_engine::PlanOp op, float x, float y) {
    using factor_engine::PlanOp;
    float value = qnan();

    // op 来自统一的启动参数，同一 warp 执行相同算子。
    // 每阶段保持 float32 舍入，避免跨阶段生成 FMA。
    // 溢出和非法结果须立即转为 NaN，避免后续 min/max 将它们变回有限值。
    switch (op) {
    case PlanOp::Neg:
        if (isfinite(x))
            value = -x;
        break;
    case PlanOp::Abs:
        if (isfinite(x))
            value = fabsf(x);
        break;
    case PlanOp::Sign:
        if (isfinite(x))
            value = (x > 0) - (x < 0);
        break;
    case PlanOp::SignedSqrt:
        if (isfinite(x))
            value = copysignf(sqrtf(fabsf(x)), x);
        break;
    case PlanOp::SignedLog1p:
        if (isfinite(x))
            value = copysignf((float)log1p((double)fabsf(x)), x);
        break;
    case PlanOp::Add:
        if (isfinite(x) && isfinite(y))
            value = clean(__fadd_rn(x, y));
        break;
    case PlanOp::Sub:
        if (isfinite(x) && isfinite(y))
            value = clean(__fsub_rn(x, y));
        break;
    case PlanOp::Mul:
        if (isfinite(x) && isfinite(y))
            value = clean(__fmul_rn(x, y));
        break;
    case PlanOp::SafeDiv:
        if (isfinite(x) && isfinite(y) && fabsf(y) > factor_engine::SafeDivisionEpsilon)
            value = clean(x / y);
        break;
    case PlanOp::Min:
        if (isfinite(x) && isfinite(y))
            value = fminf(x, y);
        break;
    case PlanOp::Max:
        if (isfinite(x) && isfinite(y))
            value = fmaxf(x, y);
        break;
    default:
        break;
    }

    return value;
}

__global__ void elementwise_kernel(PlanOp op, const float* a, float sa, int ascalar, const float* b,
                                   float sb, int bscalar, float* out, long long n) {
    // i 是展平后的数组下标，先转成 64 位再相乘；不是把 threadIdx 本身改成 64 位。
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const float x = ascalar ? sa : a[i], y = bscalar ? sb : (b ? b[i] : 0.0f);
    out[i] = pointwise_apply(op, x, y);
}

__device__ __forceinline__ float fused_pointwise_load(const FusedPointwiseOperand& operand,
                                                      float previous, long long cell) {
    return operand.source == PreviousStage
               ? previous
               : (operand.source == HostScalar ? operand.scalar : operand.pointer[cell]);
}

template <int Count>
__global__ void fused_pointwise_kernel(FusedPointwiseGroup group, long long cells) {
    static_assert(Count >= 2 && Count <= POINTWISE_FUSION_LIMIT, "bounded pointwise fusion");
    const long long cell = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cells)
        return;

    float previous = qnan();
#pragma unroll
    for (int stage = 0; stage < Count; ++stage) {
        const auto& operation = group.stages[stage];
        const float x = fused_pointwise_load(operation.a, previous, cell),
                    y = fused_pointwise_load(operation.b, previous, cell);
        previous = pointwise_apply(operation.op, x, y);
    }

    // 只有最终结果写回。不能把内部节点重新读为 workspace 指针：slot 可能复用。
    // 组内没有其他写入；同一 cell 的所有外部输入都已在此处写回前读完。
    group.output[cell] = previous;
}

// 前置条件：count 来自当前已验证 plan 的 schedule[first]，values 中所有外部输入
// 已经就绪，workspace 已按原 slot_count 分配。返回 CUDA 错误，调用方必须检查；
// 本函数只启动一次 kernel，调用方应将 kernel_launches 增加一次。
inline cudaError_t launch_pointwise_fusion(const factor_engine::PlanNodePOD* nodes, int first,
                                           int count, std::vector<Value>& values, float* workspace,
                                           long long cells, long long slot_cells = 0) {
    if (!slot_cells)
        slot_cells = cells;
    if (count < 2 || count > POINTWISE_FUSION_LIMIT || cells <= 0)
        return cudaErrorInvalidValue;

    FusedPointwiseGroup group{};
    for (int stage = 0; stage < count; ++stage) {
        const int current = first + stage;
        const auto& node = nodes[current];
        auto operand = [&](int input) -> FusedPointwiseOperand {
            if (stage && input == current - 1)
                return {nullptr, 0.0f, PreviousStage};
            if (input < 0)
                return {};
            const Value& value = values[input];
            return {value.p, value.s, value.scalar ? HostScalar : InputArray};
        };
        group.stages[stage] = {operand(node.in0), operand(node.in1),
                               static_cast<factor_engine::PlanOp>(node.op)};
        // 这些指针仅用于保留 plan 元数据；组外节点不会读取内部 slot。
        values[current] = {false, 0.0f, workspace + (long long)node.slot * slot_cells};
    }

    group.output = workspace + (long long)nodes[first + count - 1].slot * slot_cells;
    const int blocks = (int)((cells + THREADS - 1) / THREADS);
    switch (count) {
    case 2:
        fused_pointwise_kernel<2><<<blocks, THREADS>>>(group, cells);
        break;
    case 3:
        fused_pointwise_kernel<3><<<blocks, THREADS>>>(group, cells);
        break;
    case 4:
        fused_pointwise_kernel<4><<<blocks, THREADS>>>(group, cells);
        break;
    case 5:
        fused_pointwise_kernel<5><<<blocks, THREADS>>>(group, cells);
        break;
    case 6:
        fused_pointwise_kernel<6><<<blocks, THREADS>>>(group, cells);
        break;
    case 7:
        fused_pointwise_kernel<7><<<blocks, THREADS>>>(group, cells);
        break;
    case 8:
        fused_pointwise_kernel<8><<<blocks, THREADS>>>(group, cells);
        break;
    }

    return cudaGetLastError();
}

#endif

} // 命名空间 factor_engine::cuda
