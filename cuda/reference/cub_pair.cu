// CUB 排序与独立 FP64 Pearson 对照，仅用于 T=23400 的分析，不接入生产调度。
#include <cstddef>
#include <cstdint>

constexpr bool rank_shared_storage_fits(std::size_t dynamic, std::size_t fixed,
                                        std::size_t available) {
    return fixed <= available && dynamic <= available - fixed;
}

#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <math_constants.h>

// 每个 block 负责一条 root/lane/horizon；CUB 排序后计算平均秩与 RankIC。
// 当前只实例化 512 线程 × 48 项，每轮 radix 使用 6 bit。
constexpr int RANK_MAX_TIME = 24576;
static_assert(2 * (RANK_MAX_TIME - 1) < 65536);
static_assert((long long)RANK_MAX_TIME * (RANK_MAX_TIME - 1) * (RANK_MAX_TIME - 1) < (1LL << 53));
template <int Items> constexpr int RANK_THREADS = Items >= 32 ? 512 : 256;

template <int Items>
using RankSort =
    cub::BlockRadixSort<float, RANK_THREADS<Items>, Items, uint16_t, (Items >= 32 ? 6 : 4), false>;

__device__ __forceinline__ unsigned rank_tie_twice(const float* ordered, int position, int count) {
    const float value = ordered[position];
    if ((position == 0 || ordered[position - 1] != value) &&
        (position + 1 == count || ordered[position + 1] != value))
        return 2u * position;
    int low = 0, high = position;
    while (low < high) {
        const int mid = low + (high - low) / 2;
        if (ordered[mid] < value)
            low = mid + 1;
        else
            high = mid;
    }
    const int first = low;
    low = position + 1;
    high = count;
    while (low < high) {
        const int mid = low + (high - low) / 2;
        if (value < ordered[mid])
            high = mid;
        else
            low = mid + 1;
    }
    return (unsigned)first + (unsigned)(low - 1);
}

// 维度检查使用 INT_MAX。
#include <climits>

namespace factor_engine::reference {
enum Metric { Pearson = 0, RankIC = 1, MetricCount = 2 };
}

namespace factor_engine::reference {
namespace detail {
template <int Items> union Storage {
    typename RankSort<Items>::TempStorage sort;
    float ordered[RANK_THREADS<Items> * Items];
    long long sums[3][RANK_THREADS<Items>];
};

// 两遍 FP64 中心化；float 极值平方和仍在 double 的有限范围内。
template <int Threads>
__device__ __forceinline__ void pearson(const float* x, const float* y, const uint8_t* eligible,
                                        int time, double* output, double (&scratch)[3][Threads]) {
    const int tid = threadIdx.x;
    double sx = 0, sy = 0, n = 0;
    for (int t = tid; t < time; t += Threads) {
        const float a = x[t], b = y[t];
        if (eligible[t] && isfinite(a) && isfinite(b)) {
            sx += double(a);
            sy += double(b);
            n += 1;
        }
    }
    scratch[0][tid] = sx;
    scratch[1][tid] = sy;
    scratch[2][tid] = n;
    __syncthreads();
    for (int offset = Threads / 2; offset; offset >>= 1) {
        if (tid < offset) {
#pragma unroll
            for (int k = 0; k < 3; ++k)
                scratch[k][tid] += scratch[k][tid + offset];
        }
        __syncthreads();
    }
    n = scratch[2][0];
    const double mx = n > 0 ? scratch[0][0] / n : 0;
    const double my = n > 0 ? scratch[1][0] / n : 0;
    __syncthreads();
    double xx = 0, yy = 0, xy = 0;
    for (int t = tid; t < time; t += Threads) {
        const float a = x[t], b = y[t];
        if (eligible[t] && isfinite(a) && isfinite(b)) {
            const double dx = double(a) - mx, dy = double(b) - my;
            xx += dx * dx;
            yy += dy * dy;
            xy += dx * dy;
        }
    }
    scratch[0][tid] = xx;
    scratch[1][tid] = yy;
    scratch[2][tid] = xy;
    __syncthreads();
    for (int offset = Threads / 2; offset; offset >>= 1) {
        if (tid < offset) {
#pragma unroll
            for (int k = 0; k < 3; ++k)
                scratch[k][tid] += scratch[k][tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const double den = sqrt(scratch[0][0] * scratch[1][0]);
        *output = n >= 3 && den > 0 ? scratch[2][0] / den : CUDART_NAN;
    }
}

template <int Items>
__global__ __launch_bounds__(RANK_THREADS<Items>) void cub_rank_kernel(const float* const* roots,
                                                                       const float* targets,
                                                                       const uint8_t* eligible,
                                                                       double* output, int lanes,
                                                                       int time, int horizons) {
    constexpr int Threads = RANK_THREADS<Items>;
    using Sort = RankSort<Items>;
    extern __shared__ __align__(16) unsigned char multi_storage[];
    auto& shared = *reinterpret_cast<Storage<Items>*>(multi_storage);
    __shared__ int valid_count;
    const int segment = blockIdx.x, h = segment % horizons;
    const int lane = (segment / horizons) % lanes, root = segment / (lanes * horizons);
    const float* x = roots[root] + (long long)lane * time;
    const long long target_base = ((long long)h * lanes + lane) * time;
    const float* y = targets + target_base;
    const uint8_t* mask = eligible + target_base;
    double* result = output + (long long)segment * MetricCount;
    const int tid = threadIdx.x;
    if (tid == 0) {
#pragma unroll
        for (int metric = 0; metric < MetricCount; ++metric)
            result[metric] = 0;
        result[Pearson] = CUDART_NAN;
        result[RankIC] = CUDART_NAN;
    }
    float keys[Items];
    uint16_t payload[Items];
#pragma unroll
    for (int k = 0; k < Items; ++k) {
        const int t = tid + k * Threads;
        float key = CUDART_INF_F;
        if (t < time && mask[t] && isfinite(x[t]) && isfinite(y[t]))
            key = x[t];
        keys[k] = key;
        payload[k] = t < time ? t : 0;
    }
    Sort(shared.sort).SortBlockedToStriped(keys, payload);
    __syncthreads();
#pragma unroll
    for (int k = 0; k < Items; ++k)
        shared.ordered[tid + k * Threads] = keys[k];
    __syncthreads();
    if (tid == 0) {
        int low = 0, high = time;
        while (low < high) {
            const int mid = low + (high - low) / 2;
            if (isfinite(shared.ordered[mid]))
                low = mid + 1;
            else
                high = mid;
        }
        valid_count = low;
    }
    __syncthreads();
    const int count = valid_count;
#pragma unroll
    for (int k = 0; k < Items; ++k) {
        const int position = tid + k * Threads, original = payload[k];
        if (position < count) {
            keys[k] = y[original];
            payload[k] = uint16_t(rank_tie_twice(shared.ordered, position, count));
        } else {
            keys[k] = CUDART_INF_F;
            payload[k] = 0;
        }
    }
    // 常数序列和有效点不足的输入也走相同的两次排序。
    __syncthreads();
    Sort(shared.sort).SortBlockedToStriped(keys, payload);
    __syncthreads();
#pragma unroll
    for (int k = 0; k < Items; ++k)
        shared.ordered[tid + k * Threads] = keys[k];
    __syncthreads();
    const int center = count - 1;
    long long xx = 0, yy = 0, xy = 0;
#pragma unroll
    for (int k = 0; k < Items; ++k) {
        const int position = tid + k * Threads;
        if (position < count) {
            const int a = int(payload[k]) - center;
            const int b = int(rank_tie_twice(shared.ordered, position, count)) - center;
            xx += a * a;
            yy += b * b;
            xy += a * b;
        }
    }
    // 所有 ordered 读取结束后才能复用 union；单项 int32，总和 int64 精确。
    __syncthreads();
    shared.sums[0][tid] = xx;
    shared.sums[1][tid] = yy;
    shared.sums[2][tid] = xy;
    __syncthreads();
    for (int offset = Threads / 2; offset; offset >>= 1) {
        if (tid < offset) {
#pragma unroll
            for (int k = 0; k < 3; ++k)
                shared.sums[k][tid] += shared.sums[k][tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const double den = sqrt(double(shared.sums[0][0]) * double(shared.sums[1][0]));
        result[RankIC] = count >= 3 && den > 0 ? double(shared.sums[2][0]) / den : CUDART_NAN;
    }
}

constexpr int PearsonThreads = 256;

template <int Threads>
__global__ __launch_bounds__(Threads) void pearson_kernel(const float* const* roots,
                                                          const float* targets,
                                                          const uint8_t* eligible, double* output,
                                                          int lanes, int time, int horizons) {
    extern __shared__ __align__(16) unsigned char pearson_storage[];
    auto& scratch = *reinterpret_cast<double(*)[3][Threads]>(pearson_storage);
    const int segment = blockIdx.x, h = segment % horizons;
    const int lane = (segment / horizons) % lanes, root = segment / (lanes * horizons);
    const long long base = ((long long)h * lanes + lane) * time;
    pearson<Threads>(roots[root] + (long long)lane * time, targets + base, eligible + base, time,
                     output + (long long)segment * MetricCount + Pearson, scratch);
}

inline int validate(const float* const* roots, const float* targets, const uint8_t* eligible,
                    const double* output, int root_count, int lanes, int time, int horizons) {
    if (time > RANK_MAX_TIME)
        return int(cudaErrorNotSupported);
    if (!roots || !targets || !eligible || !output || root_count <= 0 || lanes <= 0 || time <= 0 ||
        horizons <= 0)
        return int(cudaErrorInvalidValue);
    // 分步检查，避免维度乘积在验证过程本身溢出。
    if ((long long)root_count * lanes > INT_MAX / horizons)
        return int(cudaErrorInvalidValue);
    return int(cudaSuccess);
}

template <class Kernel> inline cudaError_t configure(Kernel kernel, size_t dynamic, int segments) {
    int device = 0, available = 0, max_grid = 0;
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess)
        return error;
    error = cudaDeviceGetAttribute(&available, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (error != cudaSuccess)
        return error;
    error = cudaDeviceGetAttribute(&max_grid, cudaDevAttrMaxGridDimX, device);
    if (error != cudaSuccess)
        return error;
    if (segments > max_grid)
        return cudaErrorInvalidConfiguration;
    cudaFuncAttributes attributes{};
    error = cudaFuncGetAttributes(&attributes, kernel);
    if (error != cudaSuccess)
        return error;
    if (!rank_shared_storage_fits(dynamic, attributes.sharedSizeBytes, size_t(available)))
        return cudaErrorNotSupported;
    return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(dynamic));
}

} // 命名空间 detail
} // 命名空间 factor_engine::reference

// 分别启动 RankIC 与 Pearson；两项输出共用相同的联合有效集合。

namespace {
thread_local bool prepared = false;
thread_local int maximum_grid = 0;
constexpr size_t RankShared = sizeof(factor_engine::reference::detail::Storage<48>);
constexpr size_t PearsonShared =
    3 * factor_engine::reference::detail::PearsonThreads * sizeof(double);
static_assert(RankShared == 98304);
static_assert(PearsonShared == 6144);
}

// 在计时线程上准备资源，设备变更后需重新调用。CUDA Events 不包含这一步。
extern "C" int prepare_ic_pair() {
    prepared = false;
    int device = 0;
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess)
        return int(error);
    error = cudaDeviceGetAttribute(&maximum_grid, cudaDevAttrMaxGridDimX, device);
    if (error != cudaSuccess)
        return int(error);
    error = factor_engine::reference::detail::configure(
        factor_engine::reference::detail::cub_rank_kernel<48>, RankShared, 1);
    if (error != cudaSuccess)
        return int(error);
    error = factor_engine::reference::detail::configure(
        factor_engine::reference::detail::pearson_kernel<256>, PearsonShared, 1);
    if (error != cudaSuccess)
        return int(error);
    prepared = true;
    return int(cudaSuccess);
}

extern "C" size_t ic_pair_scratch_bytes(int, int, int, int) {
    return 0;
}

// 与生产两项 IC 入口保持相同参数布局；本实现不使用全局工作区。
// 计时应覆盖默认 stream 上的两个 kernel，不能只测其中一个。
extern "C" int launch_ic_pair(const float* const* roots, const float* y, const uint8_t* mask,
                              double* out, int root_count, int lanes, int time, int horizons, void*,
                              size_t) {
    if (!prepared)
        return int(cudaErrorInitializationError);
    if (time != 23400)
        return int(cudaErrorNotSupported);
    const int status = factor_engine::reference::detail::validate(roots, y, mask, out, root_count,
                                                                  lanes, time, horizons);
    if (status != cudaSuccess)
        return status;
    const int segments = root_count * lanes * horizons;
    if (segments > maximum_grid)
        return int(cudaErrorInvalidConfiguration);
    factor_engine::reference::detail::cub_rank_kernel<48>
        <<<segments, 512, RankShared>>>(roots, y, mask, out, lanes, time, horizons);
    auto error = cudaGetLastError();
    if (error != cudaSuccess)
        return int(error);
    factor_engine::reference::detail::pearson_kernel<256>
        <<<segments, 256, PearsonShared>>>(roots, y, mask, out, lanes, time, horizons);
    return int(cudaGetLastError());
}

extern "C" int cub_ic_pair_resources(int stage, int* registers, size_t* local_bytes,
                                     size_t* static_shared, size_t* dynamic_shared, int* threads) {
    if ((stage != 0 && stage != 1) || !registers || !local_bytes || !static_shared ||
        !dynamic_shared || !threads)
        return int(cudaErrorInvalidValue);
    cudaFuncAttributes attributes{};
    const auto error =
        stage == 0 ? cudaFuncGetAttributes(&attributes,
                                           factor_engine::reference::detail::cub_rank_kernel<48>)
                   : cudaFuncGetAttributes(&attributes,
                                           factor_engine::reference::detail::pearson_kernel<256>);
    if (error != cudaSuccess)
        return int(error);
    *registers = attributes.numRegs;
    *local_bytes = attributes.localSizeBytes;
    *static_shared = attributes.sharedSizeBytes;
    *dynamic_shared = stage == 0 ? RankShared : PearsonShared;
    *threads = stage == 0 ? 512 : 256;
    return int(cudaSuccess);
}

extern "C" int ic_pair_resources(int* registers, size_t* local_bytes, size_t* static_shared,
                                 size_t* dynamic_shared, int* threads) {
    return cub_ic_pair_resources(0, registers, local_bytes, static_shared, dynamic_shared, threads);
}
