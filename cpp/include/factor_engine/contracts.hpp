#pragma once

#include <cstddef>
#include <cstdint>

namespace factor_engine {

// C 接口的返回值按所属接口解释，禁止与算子或指标编号混用。
namespace cpu_status {
enum Code : int {
    Success = 0,
    InvalidArgument = -1,
    MissingField = -2,
    UnsupportedOperation = -4,
    InvalidWorkspace = -5,
    InvalidPlan = -6,
    InvalidStatistics = -8,
    AllocationFailed = -12,
    InternalError = -13
};
}

namespace runtime_status {
enum Code : int {
    Success = 0,
    InvalidArgument = -1,
    InvalidWorkspace = -2,
    DeviceError = -3,
    InternalError = -4,
    UnsupportedOperation = -5,
    InvalidRoot = -6,
    OutputCopyFailed = -7,
    TemporalLaunchFailed = -8,
    InvalidGrid = -9,
    PointerCopyFailed = -10,
    RootPackFailed = -11,
    InvalidRankLayout = -12,
    FactorSortFailed = -14,
    TargetSortFailed = -15,
    RankLaunchFailed = -16,
    PairWorkspaceFailed = -18,
    CudaErrorBase = -100
};
}

enum SumCountSlot : int { Sum = 0, Count = 1, SumCountWidth = 2 };

constexpr int MinCorrelationPairs = 3;

enum NativeLimit : int {
    NativeAbiVersion = 4,
    NoIndex = -1,
    ElementwiseThreads = 256,
    CUDA_MAX_BATCH_LANES = 32
};

constexpr double CorrelationTolerance = 1e-10;
constexpr float SafeDivisionEpsilon = 1e-12f;
constexpr double CorrelationDenominatorEpsilon = 1e-12;
constexpr std::uint64_t MaxExactDoubleInteger = std::uint64_t{1} << 53;

// 工作区按 FP64 对齐，内存估算和指针划分使用同一套计算。
constexpr bool shared_storage_fits(std::size_t dynamic, std::size_t fixed, std::size_t available) {
    return fixed <= available && dynamic <= available - fixed;
}

constexpr std::size_t align_double(std::size_t bytes) {
    return (bytes + alignof(double) - 1) & ~(alignof(double) - 1);
}
} // 命名空间 factor_engine

extern "C" int fe_native_abi_version();
