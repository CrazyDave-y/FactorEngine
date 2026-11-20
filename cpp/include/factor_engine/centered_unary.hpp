#pragma once

#include <cmath>
#include <cfloat>

#ifdef __CUDACC__
#define FE_UNARY_HD __host__ __device__
#else
#define FE_UNARY_HD
#endif

namespace factor_engine {

// 用均值和中心矩 M2 合并，避免 E[x²]-E[x]² 的消减误差。
struct UnaryMoments {
    double mean = 0, m2 = 0;
    int count = 0;
};

struct UnaryEntry {
    UnaryMoments aggregate;
    float value;
};

static_assert(sizeof(UnaryMoments) == 24, "centered std workspace byte contract");
static_assert(sizeof(UnaryEntry) == 32, "CPU centered std queue byte contract");

FE_UNARY_HD inline UnaryMoments unary_leaf(float value) {
    if (!(value <= FLT_MAX && value >= -FLT_MAX))
        return {};
    return {(double)value, 0, 1};
}

FE_UNARY_HD inline UnaryMoments merge_unary(const UnaryMoments& a, const UnaryMoments& b) {
    if (!a.count)
        return b;

    if (!b.count)
        return a;

    const int count = a.count + b.count;
    const double difference = b.mean - a.mean, fraction = (double)b.count / count;
    const double weight = ((double)a.count * b.count) / count;
    return {::fma(difference, fraction, a.mean),
            ::fma(difference * difference, weight, a.m2 + b.m2), count};
}

// 缓冲区由线程持有，队列只借用。每个样本最多入栈、转移、出栈各一次。
// 总时间 O(T)，空间 O(W)，不靠减去旧样本来更新方差。
struct UnaryWindowQueue {
    UnaryEntry* back;
    UnaryMoments* front;
    int front_size = 0, back_size = 0, window;

    FE_UNARY_HD UnaryWindowQueue(UnaryEntry* back_storage, UnaryMoments* front_storage,
                                 int window_size)
        : back(back_storage), front(front_storage), window(window_size) {}

    FE_UNARY_HD void push(float value) {
        if (front_size + back_size == window) {
            if (!front_size) {
                for (int i = back_size - 1; i >= 0; --i) {
                    const auto leaf = unary_leaf(back[i].value);
                    front[front_size] =
                        front_size ? merge_unary(leaf, front[front_size - 1]) : leaf;
                    ++front_size;
                }
                back_size = 0;
            }
            --front_size;
        }

        const auto leaf = unary_leaf(value);
        const auto aggregate = back_size ? merge_unary(back[back_size - 1].aggregate, leaf) : leaf;
        back[back_size++] = {aggregate, value};
    }

    FE_UNARY_HD UnaryMoments value() const {
        return merge_unary(front_size ? front[front_size - 1] : UnaryMoments{},
                           back_size ? back[back_size - 1].aggregate : UnaryMoments{});
    }
};
} // 命名空间 factor_engine

#undef FE_UNARY_HD
