#include "factor_engine/cuda_runtime.hpp"
#include "factor_engine/metric.hpp"

#include <cstddef>
#include <iostream>
#include <initializer_list>
#include <utility>

namespace {

void layout(const char* name, std::size_t bytes, std::size_t alignment,
            std::initializer_list<std::pair<const char*, std::size_t>> offsets) {
    std::cout << '"' << name << "\":{\"bytes\":" << bytes << ",\"alignment\":" << alignment
              << ",\"offsets\":{";
    bool first = true;
    for (const auto& field : offsets) {
        if (!first)
            std::cout << ',';
        first = false;
        std::cout << '"' << field.first << "\":" << field.second;
    }
    std::cout << "}}";
}
} // 命名空间

int main() {
    using factor_engine::PlanNodePOD;
    static_assert(factor_engine::metrics::MetricCount == 2);
    static_assert(factor_engine::NativeAbiVersion == 4);
    static_assert(factor_engine::metrics::metric_layout_fits(256, 32, 23400, 9));
    static_assert(!factor_engine::metrics::metric_layout_fits(-1, 32, 23400, 9));
    static_assert(
        !factor_engine::metrics::metric_layout_fits(2147483647, 2147483647, 23400, 2147483647));
    static_assert(sizeof(PlanNodePOD) == 32);
    static_assert(sizeof(FeCudaScoreStats) == 56);
    static_assert(sizeof(FeCudaUploadStats) == 16);

    std::cout << "{\"layouts\":{";
    layout("NativeNode", sizeof(PlanNodePOD), alignof(PlanNodePOD),
           {{"op", offsetof(PlanNodePOD, op)},
            {"in0", offsetof(PlanNodePOD, in0)},
            {"in1", offsetof(PlanNodePOD, in1)},
            {"param_i", offsetof(PlanNodePOD, param_i)},
            {"field_index", offsetof(PlanNodePOD, field_index)},
            {"slot", offsetof(PlanNodePOD, slot)},
            {"value_type", offsetof(PlanNodePOD, value_type)},
            {"param_f", offsetof(PlanNodePOD, param_f)}});
    std::cout << ',';
    layout("DeviceInfo", sizeof(FeCudaDeviceInfo), alignof(FeCudaDeviceInfo),
           {{"free_bytes", offsetof(FeCudaDeviceInfo, free_bytes)},
            {"total_bytes", offsetof(FeCudaDeviceInfo, total_bytes)},
            {"sm_count", offsetof(FeCudaDeviceInfo, sm_count)},
            {"max_shared_bytes", offsetof(FeCudaDeviceInfo, max_shared_bytes)},
            {"device", offsetof(FeCudaDeviceInfo, device)}});
    std::cout << ',';
    layout("ScoreStats", sizeof(FeCudaScoreStats), alignof(FeCudaScoreStats),
           {{"h2d_bytes", offsetof(FeCudaScoreStats, h2d_bytes)},
            {"kernel_launches", offsetof(FeCudaScoreStats, kernel_launches)},
            {"root_export_bytes", offsetof(FeCudaScoreStats, root_export_bytes)},
            {"rank_workspace_bytes", offsetof(FeCudaScoreStats, rank_workspace_bytes)},
            {"owned_device_bytes", offsetof(FeCudaScoreStats, owned_device_bytes)},
            {"input_target_h2d_bytes", offsetof(FeCudaScoreStats, input_target_h2d_bytes)},
            {"lane_ic_d2h_bytes", offsetof(FeCudaScoreStats, lane_ic_d2h_bytes)}});
    std::cout << ',';
    layout("UploadStats", sizeof(FeCudaUploadStats), alignof(FeCudaUploadStats),
           {{"upload_h2d_bytes", offsetof(FeCudaUploadStats, upload_h2d_bytes)},
            {"sanitize_launches", offsetof(FeCudaUploadStats, sanitize_launches)}});
    std::cout << "},\"metric\":[" << factor_engine::metrics::Pearson << ','
              << factor_engine::metrics::RankIC << "],\"cuda_status\":[" << factor_engine::Success
              << ',' << factor_engine::InvalidArgument << ',' << factor_engine::OperationFailed
              << ',' << factor_engine::DeviceError << ',' << factor_engine::InternalError << ','
              << factor_engine::UnexpectedInputCopy << ',' << factor_engine::AllocationFailed << ','
              << factor_engine::InvalidExportSize << "],\"sum_count\":[" << factor_engine::Sum
              << ',' << factor_engine::Count
              << "],\"max_rank_time\":" << factor_engine::metrics::MaxRankTime << "}\n";
}
