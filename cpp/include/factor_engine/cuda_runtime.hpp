#pragma once

#include "factor_engine/plan.hpp"

namespace factor_engine {

// C 接口返回状态；算子种类另用 PlanOp，不能混用。
enum CudaBackendStatus : int {
    Success = 0,
    InvalidArgument = -1,
    OperationFailed = -2,
    DeviceError = -3,
    InternalError = -4,
    UnexpectedInputCopy = -5,
    AllocationFailed = -9,
    InvalidExportSize = -12
};
} // 命名空间 factor_engine

extern "C" {
struct FeCudaUploadStats {
    long long upload_h2d_bytes;
    long long sanitize_launches;
};

struct FeCudaScoreStats {
    long long h2d_bytes;
    long long kernel_launches;
    long long root_export_bytes;
    long long rank_workspace_bytes;
    long long owned_device_bytes;
    long long input_target_h2d_bytes;
    long long lane_ic_d2h_bytes;
};

struct FeCudaDeviceInfo {
    long long free_bytes, total_bytes;
    int sm_count, max_shared_bytes, device;
};

// 分配失败通过返回码说明，输出句柄保持为空。
int fe_cuda_backend_allocate_checked(int fields, int lanes, int time, long long budget,
                                     void** output);

int fe_cuda_backend_upload_window(void* handle, int first, const float* const* fields,
                                  const uint8_t* valid, const float* target, int n_fields,
                                  int lanes, int time, int final_window, FeCudaUploadStats* stats);

int fe_cuda_backend_score_lanes(void* handle, int first, const factor_engine::PlanNodePOD* nodes,
                                int count, int slots, const int* roots, int root_count,
                                double* lane_output, long long lane_output_bytes,
                                FeCudaScoreStats* stats);
}
