// 检查生产 RankIC 的整数上限与 CTA 树归约；不证明 GPU 同步或代码生成。
#include "cpp/include/factor_engine/metric.hpp"

#include <cassert>
#include <iostream>
#include <random>
#include <vector>

struct RankSums {
    long long xx = 0, yy = 0, xy = 0;

    RankSums& operator+=(const RankSums& rhs) {
        xx += rhs.xx;
        yy += rhs.yy;
        xy += rhs.xy;
        return *this;
    }
};

static RankSums emulate_block(std::vector<RankSums> partials) {
    assert(partials.size() == 256 || partials.size() == 512);
    for (std::size_t offset = partials.size() / 2; offset; offset /= 2)
        for (std::size_t tid = 0; tid < offset; ++tid)
            partials[tid] += partials[tid + offset];
    return partials[0];
}

int main() {
    static_assert(factor_engine::rank_math::rank_integer_sums_are_exact(24576));
    static_assert(!factor_engine::rank_math::rank_integer_sums_are_exact(0));
    static_assert(!factor_engine::rank_math::rank_integer_sums_are_exact(24577));

    std::mt19937 rng(47);
    int cases = 0;
    for (int threads : {256, 512}) {
        for (int n : {3, 4, 31, 32, 33, 255, 256, 257, 4096, 16384, 23400, 24576}) {
            for (int trial = 0; trial < 24; ++trial) {
                std::vector<RankSums> partials(threads);
                RankSums expected{};
                for (int i = 0; i < n; ++i) {
                    // 覆盖最大平方项，以及 xy 的正负两种符号。
                    int x = trial == 0 ? n - 1 : int(rng() % (2 * n - 1)) - (n - 1);
                    int y = trial == 0 ? -(n - 1) : int(rng() % (2 * n - 1)) - (n - 1);
                    RankSums term{1LL * x * x, 1LL * y * y, 1LL * x * y};
                    expected += term;
                    partials[i % threads] += term;
                }

                const auto actual = emulate_block(partials);
                assert(actual.xx == expected.xx && actual.yy == expected.yy &&
                       actual.xy == expected.xy);
                assert((long long)(double)actual.xx == actual.xx);
                assert((long long)(double)actual.xy == actual.xy);
                ++cases;
            }
        }
    }

    std::cout << "host RankIC integer/CTA checks: " << cases << " passed\n";
}
