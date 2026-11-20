#pragma once

#include <cstdint>

// OpenMP 线程当前保留的 vector 容量，以字节计。
// 分配器等额外开销由调用方另留余量。
extern "C" std::uint64_t fe_cpu_rank_scratch_retained_bytes();

// 数值诊断：导出 [root,lane]，不改变正常评分的累计顺序。
extern "C" int fe_rank_ic_lanes_cpu(const float* const* roots, int n_roots, const float* target,
                                    int lanes, int time, double* lane_output);

// 每个 root 占两个数：[有效 lane 的 IC 总和, 有效 lane 数]。
// root 指针为空时按常数因子处理，相关系数没有定义。
extern "C" int fe_rank_ic_stats_cpu(const float* const* roots, int n_roots, const float* target,
                                    int lanes, int time, double* stats);

// 在传入的总和与计数上继续累加，调用方按原顺序提供后续完整 lane。
// 逐条加入，不能先把一批合并成小计再加入。
extern "C" int fe_rank_ic_accumulate_cpu(const float* const* roots, int n_roots,
                                         const float* target, int lanes, int time, double* stats);
