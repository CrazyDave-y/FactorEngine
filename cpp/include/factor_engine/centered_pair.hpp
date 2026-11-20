#pragma once

#include <cmath>
#include <cfloat>

#ifdef __CUDACC__
#define FE_PAIR_HD __host__ __device__
#else
#define FE_PAIR_HD
#endif

namespace factor_engine {

// 用中心矩累计协方差，避免两个大的原始二阶矩相减。
struct PairMoments {
    double mean_a = 0, mean_b = 0, m2_a = 0, m2_b = 0, cross = 0;
    int count = 0;
};

struct PairEntry {
    PairMoments aggregate;
    float a, b;
};

static_assert(sizeof(PairMoments) == 48, "CUDA epoch workspace byte contract changed");
static_assert(sizeof(PairEntry) + sizeof(PairMoments) <= 128,
              "streaming queue estimate must cover aggregate layout");

FE_PAIR_HD inline PairMoments pair_leaf(float a, float b) {
    if (!(a <= FLT_MAX && a >= -FLT_MAX && b <= FLT_MAX && b >= -FLT_MAX))
        return {};
    return {(double)a, (double)b, 0, 0, 0, 1};
}

FE_PAIR_HD inline PairMoments merge_pair(const PairMoments& lhs, const PairMoments& rhs) {
    if (!lhs.count)
        return rhs;

    if (!rhs.count)
        return lhs;

    const int count = lhs.count + rhs.count;
    const double delta_a = rhs.mean_a - lhs.mean_a, delta_b = rhs.mean_b - lhs.mean_b;
    const double fraction = (double)rhs.count / count,
                 weight = ((double)lhs.count * rhs.count) / count;
    return {::fma(delta_a, fraction, lhs.mean_a),
            ::fma(delta_b, fraction, lhs.mean_b),
            ::fma(delta_a * delta_a, weight, lhs.m2_a + rhs.m2_a),
            ::fma(delta_b * delta_b, weight, lhs.m2_b + rhs.m2_b),
            ::fma(delta_a * delta_b, weight, lhs.cross + rhs.cross),
            count};
}

// 双栈维护滚动窗口，每个样本最多入栈、转移、出栈各一次。
// 总时间 O(T)，空间 O(W)，不做容易丢精度的减量更新。
// 队列借用调用方提供的线程或 lane 缓冲区。
struct PairWindowQueue {
    PairEntry* back;
    PairMoments* front;
    int front_size = 0, back_size = 0, window;

    FE_PAIR_HD PairWindowQueue(PairEntry* back_storage, PairMoments* front_storage, int window_size)
        : back(back_storage), front(front_storage), window(window_size) {}

    FE_PAIR_HD void push(float a, float b) {
        if (front_size + back_size == window) {
            if (!front_size) {
                for (int i = back_size - 1; i >= 0; --i) {
                    const auto leaf = pair_leaf(back[i].a, back[i].b);
                    front[front_size] = front_size ? merge_pair(leaf, front[front_size - 1]) : leaf;
                    ++front_size;
                }
                back_size = 0;
            }
            --front_size;
        }

        const auto leaf = pair_leaf(a, b);
        const auto aggregate = back_size ? merge_pair(back[back_size - 1].aggregate, leaf) : leaf;
        back[back_size++] = {aggregate, a, b};
    }

    FE_PAIR_HD PairMoments value() const {
        const PairMoments a = front_size ? front[front_size - 1] : PairMoments{};
        const PairMoments b = back_size ? back[back_size - 1].aggregate : PairMoments{};
        return merge_pair(a, b);
    }
};
} // 命名空间 factor_engine

#undef FE_PAIR_HD
