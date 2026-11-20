// CUDA 后端只有一个资源句柄：输入与计算临时区由它统一释放。
#include "device_buffers.cuh"
#include <climits>

namespace {
using factor_engine::CudaBackendStatus;

bool valid_batch(CUDABuffers* b, int first) {
    return b && b->ready && b->work && first >= 0 && first <= b->active_lanes - b->work->lanes;
}

void select_inputs(CUDABuffers* b, int first) {
    for (int i = 0; i < (int)b->fields.size(); ++i)
        b->work->fields[i] = b->fields[i] + (long long)first * b->time;
}

// 上传失败后不能继续使用旧批次；调用方必须从 first=0 重新上传。
int invalidate_upload(CUDABuffers& b, CudaBackendStatus status) {
    b.next_upload = -1;
    b.ready = false;
    b.active_lanes = 0;
    return status;
}

bool copy_input(void* device, const void* host, size_t bytes, FeCudaUploadStats& stats) {
    if (cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    stats.upload_h2d_bytes += (long long)bytes;
    return true;
}
} // 命名空间

extern "C" int fe_cuda_backend_release(void* handle) {
    auto* b = static_cast<CUDABuffers*>(handle);
    if (!b)
        return CudaBackendStatus::InvalidArgument;
    if (cudaDeviceSynchronize() != cudaSuccess)
        return CudaBackendStatus::DeviceError;
    delete b;
    return CudaBackendStatus::Success;
}

extern "C" int fe_cuda_backend_set_batch_width(void* handle, int width) {
    auto* b = static_cast<CUDABuffers*>(handle);
    // 形状准备不读取输入。ready/active_lanes 仍由实际评分入口检查。
    if (!b || width < 1 || width > b->lanes)
        return CudaBackendStatus::InvalidArgument;

    try {
        return b->set_batch_width(width);
    } catch (...) {
        return CudaBackendStatus::InternalError;
    }
}

// 按顺序上传各批输入和 target。first==0 表示重新上传，旧结果随即失效。
// 最后一批上传成功后才允许评分；传输字节数来自实际 cudaMemcpy。
extern "C" int fe_cuda_backend_upload_window(void* handle, int first, const float* const* fields,
                                             const uint8_t* valid, const float* target,
                                             int n_fields, int lanes, int time, int final_window,
                                             FeCudaUploadStats* stats) {
    auto* b = static_cast<CUDABuffers*>(handle);
    if (!b)
        return CudaBackendStatus::InvalidArgument;

    if (first == 0) {
        b->next_upload = 0;
        b->ready = false;
        b->active_lanes = 0;
    }
    if (!stats || !fields || !valid || !target)
        return invalidate_upload(*b, CudaBackendStatus::InvalidArgument);
    *stats = {};

    // 形状、窗口范围和上传顺序分别检查，避免把不同原因塞进一条条件。
    if (time != b->time || n_fields != (int)b->fields.size())
        return invalidate_upload(*b, CudaBackendStatus::InvalidArgument);
    if (first < 0 || lanes < 1 || lanes > b->lanes || first > b->lanes - lanes)
        return invalidate_upload(*b, CudaBackendStatus::InvalidArgument);
    if (first != b->next_upload || (final_window != 0 && final_window != 1))
        return invalidate_upload(*b, CudaBackendStatus::InvalidArgument);
    for (int i = 0; i < n_fields; ++i)
        if (!fields[i])
            return invalidate_upload(*b, CudaBackendStatus::InvalidArgument);

    b->ready = false;
    b->active_lanes = 0;
    const long long cells = (long long)lanes * time, offset = (long long)first * time;
    if (!copy_input(b->valid + offset, valid, (size_t)cells, *stats) ||
        !copy_input(b->target + offset, target, (size_t)cells * sizeof(float), *stats))
        return invalidate_upload(*b, CudaBackendStatus::OperationFailed);
    for (int i = 0; i < n_fields; ++i) {
        if (!copy_input(b->fields[i] + offset, fields[i], (size_t)cells * sizeof(float), *stats))
            return invalidate_upload(*b, CudaBackendStatus::OperationFailed);
        sanitize_field_kernel<<<factor_engine::ceil_blocks(cells), THREADS>>>(
            b->fields[i] + offset, b->valid + offset, cells, i + 1 == n_fields);
        if (cudaGetLastError() != cudaSuccess)
            return invalidate_upload(*b, CudaBackendStatus::DeviceError);
        ++stats->sanitize_launches;
    }

    // 输入资格已经写入字段的 NaN。delay 等算子的输出资格由结果本身决定，
    // 不能再次套用原时点的输入 mask；清洗的最后一步已将其改成全选资格。
    if (cudaDeviceSynchronize() != cudaSuccess)
        return invalidate_upload(*b, CudaBackendStatus::DeviceError);

    b->next_upload = first + lanes;
    if (final_window) {
        b->active_lanes = b->next_upload;
        b->ready = true;
    }
    return CudaBackendStatus::Success;
}

extern "C" int fe_cuda_backend_score_lanes(void* handle, int first,
                                           const factor_engine::PlanNodePOD* nodes, int count,
                                           int slots, const int* roots, int root_count,
                                           double* lane_output, long long lane_output_bytes,
                                           FeCudaScoreStats* stats) {
    auto* b = static_cast<CUDABuffers*>(handle);
    if (!valid_batch(b, first) || !nodes || count < 1 || !roots || root_count < 1 || !lane_output ||
        !stats ||
        root_count >
            std::numeric_limits<long long>::max() / b->work->lanes / (long long)sizeof(double))
        return CudaBackendStatus::InvalidArgument;

    auto& rt = *b->work;
    const long long expected = (long long)root_count * rt.lanes * sizeof(double);
    if (lane_output_bytes != expected)
        return CudaBackendStatus::InvalidArgument;

    try {
        select_inputs(b, first);
        const int rc = execute_score(&rt, nodes, count, slots, roots, root_count,
                                     b->target + (long long)first * b->time,
                                     b->valid + (long long)first * b->time);
        if (rc)
            return rc;

        const bool fused = supports_pair_metrics(rt.time, rt.max_shared);
        const double* scores = rt.rank.lane_scores + (fused ? factor_engine::metrics::RankIC : 0);
        const size_t pitch = (fused ? factor_engine::metrics::MetricCount : 1) * sizeof(double);
        if (cudaMemcpy2D(lane_output, sizeof(double), scores, pitch, sizeof(double),
                         (size_t)root_count * rt.lanes, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess)
            return CudaBackendStatus::DeviceError;

        const auto& t = rt.telemetry;
        stats->h2d_bytes = t.h2d_bytes;
        stats->kernel_launches = t.kernel_launches;
        stats->root_export_bytes = t.root_export_bytes;
        stats->rank_workspace_bytes = t.rank_workspace_bytes;
        stats->owned_device_bytes = b->bytes + b->workspace_bytes();
        stats->input_target_h2d_bytes = t.h2d_bytes - (long long)root_count * sizeof(float*);
        stats->lane_ic_d2h_bytes = expected;
        return stats->input_target_h2d_bytes == 0 ? CudaBackendStatus::Success
                                                  : CudaBackendStatus::UnexpectedInputCopy;
    } catch (...) {
        return CudaBackendStatus::InternalError;
    }
}

extern "C" int fe_cuda_backend_evaluate(void* handle, int first,
                                        const factor_engine::PlanNodePOD* nodes, int count,
                                        int slots, const int* roots, int root_count, float* output,
                                        long long bytes) {
    auto* b = static_cast<CUDABuffers*>(handle);
    if (!valid_batch(b, first) || root_count < 1 || !output ||
        !factor_engine::valid_float_storage(b->work->cells, root_count))
        return CudaBackendStatus::InvalidArgument;
    const long long required = (long long)root_count * b->work->cells * sizeof(float);
    if (bytes != required)
        return CudaBackendStatus::InvalidExportSize;

    try {
        select_inputs(b, first);
        return export_roots(*b->work, nodes, count, slots, roots, root_count, output);
    } catch (...) {
        return CudaBackendStatus::InternalError;
    }
}

// 多期限指标的工作区与分配峰值。
struct FeMetricResources {
    long long workspace_bytes, requested_scratch_bytes, retained_scratch_bytes, owned_device_bytes;
};

struct FeMetricCapacity {
    long long active_storage_bytes, retained_storage_bytes, prepare_peak_metric_bytes;
    long long prepare_peak_owned_bytes;
};

extern "C" int fe_cuda_metric_prepare(void* handle, int roots, int horizons,
                                      long long available_bytes, FeMetricResources* resources) {
    auto* b = static_cast<CUDABuffers*>(handle);
    if (!b || !b->work || !resources || roots < 1 || horizons < 1 || available_bytes < 0)
        return CudaBackendStatus::InvalidArgument;

    auto& rt = *b->work;
    if (rt.time > factor_engine::metrics::MaxRankTime)
        return CudaBackendStatus::OperationFailed;
    if ((long long)roots * rt.lanes >
        std::numeric_limits<int>::max() / factor_engine::metrics::MetricCount / horizons)
        return CudaBackendStatus::OperationFailed;
    if (rt.time > factor_engine::metrics::LongCapacity)
        return CudaBackendStatus::OperationFailed;
    const size_t required =
        factor_engine::metrics::pair_metric_scratch_bytes(roots, rt.lanes, rt.time, horizons);
    if (!required && rt.time > factor_engine::metrics::ShortCapacity)
        return CudaBackendStatus::OperationFailed;
    const size_t storage =
        factor_engine::metrics::MetricWorkspace::bytes(roots, rt.lanes, rt.time, horizons);
    // 扩容先申请再释放，预算必须容纳新旧空间同时存活的瞬间。
    const size_t peak = b->metrics ? b->metrics->peak_bytes(storage, required) : storage + required;
    if (peak > (size_t)available_bytes)
        return CudaBackendStatus::AllocationFailed;

    try {
        if (!b->metrics) {
            auto next = std::make_unique<factor_engine::metrics::MetricWorkspace>();
            if (!next->allocate(roots, rt.lanes, rt.time, horizons) || !next->reserve(required))
                return CudaBackendStatus::AllocationFailed;
            b->metrics = std::move(next);
        } else if (!b->metrics->allocate(roots, rt.lanes, rt.time, horizons) ||
                   !b->metrics->reserve(required))
            return CudaBackendStatus::AllocationFailed;

        auto& m = *b->metrics;
        m.prepare_peak_bytes = peak;
        resources->workspace_bytes = (long long)(m.storage_bytes + m.scratch_bytes);
        resources->requested_scratch_bytes = (long long)required;
        resources->retained_scratch_bytes = (long long)m.scratch_bytes;
        resources->owned_device_bytes =
            b->bytes + b->workspace_bytes() + resources->workspace_bytes;
        return CudaBackendStatus::Success;
    } catch (...) {
        return CudaBackendStatus::InternalError;
    }
}

// 查询当前布局、保留容量与最近准备过程的分配峰值。
extern "C" int fe_cuda_metric_capacity(void* handle, FeMetricCapacity* capacity) {
    const auto* b = static_cast<CUDABuffers*>(handle);
    if (!b || !b->metrics || !capacity)
        return CudaBackendStatus::InvalidArgument;

    const auto& m = *b->metrics;
    *capacity = {(long long)m.bytes(m.roots, m.lanes, m.time, m.horizons),
                 (long long)m.storage_bytes, (long long)m.prepare_peak_bytes,
                 b->bytes + b->workspace_bytes() + (long long)m.prepare_peak_bytes};
    return CudaBackendStatus::Success;
}

extern "C" int fe_cuda_metric_resources(void* handle, FeMetricResources* resources) {
    const auto* b = static_cast<CUDABuffers*>(handle);
    if (!b || !b->metrics || !resources)
        return CudaBackendStatus::InvalidArgument;

    const auto& m = *b->metrics;
    *resources = {(long long)(m.storage_bytes + m.scratch_bytes), (long long)m.requested_scratch,
                  (long long)m.scratch_bytes,
                  b->bytes + b->workspace_bytes() + (long long)(m.storage_bytes + m.scratch_bytes)};
    return CudaBackendStatus::Success;
}

extern "C" int fe_cuda_metric_inputs_ready(void* handle, int* ready) {
    auto* b = static_cast<CUDABuffers*>(handle);
    if (!b || !b->metrics || !ready)
        return CudaBackendStatus::InvalidArgument;

    *ready = b->metrics->uploaded ? 1 : 0;
    return CudaBackendStatus::Success;
}

extern "C" int fe_cuda_metric_upload(void* handle, const float* targets, const uint8_t* eligible) {
    auto* b = static_cast<CUDABuffers*>(handle);
    if (!b || !b->metrics || !targets || !eligible)
        return CudaBackendStatus::InvalidArgument;

    auto& m = *b->metrics;
    m.uploaded = false;
    const size_t items = (size_t)m.horizons * m.lanes * m.time;
    if (cudaMemcpy(m.targets, targets, items * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(m.eligible, eligible, items, cudaMemcpyHostToDevice) != cudaSuccess)
        return CudaBackendStatus::DeviceError;
    m.uploaded = true;
    return CudaBackendStatus::Success;
}

extern "C" int fe_cuda_metric_score(void* handle, void* accumulator, int first_lane,
                                    const factor_engine::PlanNodePOD* nodes, int count, int slots,
                                    const int* roots, int root_count, double* details) {
    auto* b = static_cast<CUDABuffers*>(handle);
    auto* destination = static_cast<CUDABuffers*>(accumulator);
    if (!valid_batch(b, 0) || !destination || !destination->metrics || !b->metrics ||
        !b->metrics->uploaded || !nodes || !roots)
        return CudaBackendStatus::InvalidArgument;

    auto& rt = *b->work;

    auto& m = *b->metrics;
    auto& total = *destination->metrics;
    if (root_count != m.roots || m.lanes != rt.lanes || m.horizons != total.horizons ||
        m.roots != total.roots || first_lane < 0)
        return CudaBackendStatus::InvalidArgument;
    if (first_lane && first_lane != total.next_lane)
        return CudaBackendStatus::InvalidArgument;

    int rc = factor_engine::validate_plan_nodes(nodes, count, (int)rt.fields.size(), slots, roots,
                                                root_count);
    if (rc)
        return rc;

    try {
        if (first_lane == 0) {
            const size_t bytes = (size_t)m.roots * m.horizons *
                                 factor_engine::metrics::MetricCount * 2 * sizeof(double);
            if (cudaMemset(total.totals, 0, bytes) != cudaSuccess)
                return CudaBackendStatus::DeviceError;
            total.next_lane = 0;
        }

        select_inputs(b, 0);
        std::vector<Value> values;
        {
            if ((rc = execute_nodes(&rt, nodes, count, slots, values, roots, root_count)))
                return rc;
            if ((rc = validate_roots(values, roots, root_count)))
                return rc;
            if ((rc = prepare_root_pointers(&rt, values, roots, root_count)))
                return rc;
        }

        {
            rc = factor_engine::metrics::launch_pair_metrics(
                rt.root_ptrs_device, m.targets, m.eligible, m.output, m.roots, m.lanes, m.time,
                m.horizons, m.scratch, m.scratch_bytes);
            if (rc)
                return CudaBackendStatus::DeviceError;
        }

        {
            const int items = m.roots * m.horizons * factor_engine::metrics::MetricCount;
            factor_engine::metrics::
                ordered_metric_commit<<<(items + THREADS - 1) / THREADS, THREADS>>>(
                    m.output, total.totals, m.roots, m.lanes, m.horizons);
            if (cudaGetLastError() != cudaSuccess)
                return CudaBackendStatus::DeviceError;
        }

        if (details && cudaMemcpy(details, m.output,
                                  (size_t)m.roots * m.lanes * m.horizons *
                                      factor_engine::metrics::MetricCount * sizeof(double),
                                  cudaMemcpyDeviceToHost) != cudaSuccess)
            return CudaBackendStatus::DeviceError;

        total.next_lane += m.lanes;
        return CudaBackendStatus::Success;
    } catch (...) {
        return CudaBackendStatus::InternalError;
    }
}

extern "C" int fe_cuda_metric_read(void* handle, int expected_lanes, double* output) {
    auto* b = static_cast<CUDABuffers*>(handle);
    if (!b || !b->metrics || !output || expected_lanes < 1 ||
        b->metrics->next_lane != expected_lanes)
        return CudaBackendStatus::InvalidArgument;

    auto& m = *b->metrics;
    const size_t bytes =
        (size_t)m.roots * m.horizons * factor_engine::metrics::MetricCount * 2 * sizeof(double);
    return cudaMemcpy(output, m.totals, bytes, cudaMemcpyDeviceToHost) == cudaSuccess
               ? CudaBackendStatus::Success
               : CudaBackendStatus::DeviceError;
}

extern "C" int fe_native_abi_version() {
    return factor_engine::NativeAbiVersion;
}

extern "C" int fe_cuda_runtime_device_info(FeCudaDeviceInfo* info) {
    if (!info)
        return factor_engine::runtime_status::InvalidArgument;

    size_t available = 0, total = 0;
    int device = -1, sms = 0, shared = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaMemGetInfo(&available, &total) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) !=
            cudaSuccess ||
        total > (size_t)LLONG_MAX)
        return factor_engine::runtime_status::DeviceError;

    *info = {(long long)available, (long long)total, sms, shared, device};
    return 0;
}
