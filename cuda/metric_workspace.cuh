#pragma once
#include "metric_dispatch.cuh"

namespace factor_engine::metrics {
// 多期限工作区仍归原 CUDABuffers 管理，不另设设备内存所有者。
struct MetricWorkspace {
    int roots = 0, lanes = 0, time = 0, horizons = 0;
    int next_lane = 0;
    bool uploaded = false;
    void* storage = nullptr;
    void* scratch = nullptr;
    size_t storage_bytes = 0, scratch_bytes = 0, requested_scratch = 0;
    size_t prepare_peak_bytes = 0;
    float* targets = nullptr;
    uint8_t* eligible = nullptr;
    double* output = nullptr;
    double* totals = nullptr;

    static size_t bytes(int r, int l, int t, int h) {
        const size_t target = (size_t)h * l * t;
        const size_t aligned =
            factor_engine::align_double(target * (sizeof(float) + sizeof(uint8_t)));
        return aligned + (size_t)r * l * h * MetricCount * sizeof(double) +
               (size_t)r * h * MetricCount * factor_engine::SumCountWidth * sizeof(double);
    }

    size_t peak_bytes(size_t storage_required, size_t scratch_required) const {
        // 保守计入新旧空间同时存活；容量足够时不重复计算有效布局。
        return storage_bytes + scratch_bytes +
               (storage_required > storage_bytes ? storage_required : 0) +
               (scratch_required > scratch_bytes ? scratch_required : 0);
    }

    bool allocate(int r, int l, int t, int h) {
        const size_t required = bytes(r, l, t, h);
        const bool keep_inputs =
            uploaded && required <= storage_bytes && l == lanes && t == time && h == horizons;
        if (required > storage_bytes) {
            void* next = nullptr;
            if (cudaMalloc(&next, required) != cudaSuccess)
                return false;
            if (storage && cudaFree(storage) != cudaSuccess) {
                cudaFree(next);
                return false;
            }
            storage = next;
            storage_bytes = required;
        }

        roots = r;
        lanes = l;
        time = t;
        horizons = h;
        const size_t target = (size_t)h * l * t;
        auto* base = static_cast<unsigned char*>(storage);
        targets = reinterpret_cast<float*>(base);
        eligible = base + target * sizeof(float);
        output = reinterpret_cast<double*>(
            base + (factor_engine::align_double(target * (sizeof(float) + sizeof(uint8_t)))));
        totals = output + (size_t)r * l * h * MetricCount;

        uploaded = keep_inputs;
        next_lane = 0;
        return true;
    }

    bool reserve(size_t required) {
        requested_scratch = required;
        if (required <= scratch_bytes)
            return true;

        void* next = nullptr;
        if (cudaMalloc(&next, required) != cudaSuccess)
            return false;
        if (scratch && cudaFree(scratch) != cudaSuccess) {
            cudaFree(next);
            return false;
        }

        scratch = next;
        scratch_bytes = required;
        return true;
    }

    ~MetricWorkspace() {
        if (scratch)
            cudaFree(scratch);
        if (storage)
            cudaFree(storage);
    }
};

// 每个线程负责一个 root、horizon、metric 组合，按原 lane 顺序累加。
__global__ void ordered_metric_commit(const double* values, double* totals, int roots, int lanes,
                                      int horizons) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= roots * horizons * MetricCount)
        return;

    const int metric = i % MetricCount;
    const int horizon = (i / MetricCount) % horizons;
    const int root = i / (MetricCount * horizons);
    double sum = totals[factor_engine::SumCountWidth * i + factor_engine::Sum],
           count = totals[factor_engine::SumCountWidth * i + factor_engine::Count];

    for (int lane = 0; lane < lanes; ++lane) {
        const double value = values[((size_t)root * lanes + lane) * horizons * MetricCount +
                                    horizon * MetricCount + metric];
        if (isfinite(value)) {
            sum += value;
            count += 1;
        }
    }

    totals[factor_engine::SumCountWidth * i + factor_engine::Sum] = sum;
    totals[factor_engine::SumCountWidth * i + factor_engine::Count] = count;
}

} // 命名空间 factor_engine::metrics
