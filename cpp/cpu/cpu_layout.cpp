#include "factor_engine/plan.hpp"
#include "factor_engine/contracts.hpp"
#include "factor_engine/centered_pair.hpp"
#include "factor_engine/centered_unary.hpp"

#include <utility>
#include <omp.h>

extern "C" int fe_cpu_layout(FeCpuLayout* out) {
    if (!out)
        return factor_engine::cpu_status::InvalidArgument;

    out->float_bytes = sizeof(float);
    out->double_bytes = sizeof(double);
    out->int_bytes = sizeof(int);
    out->sort_pair_bytes = sizeof(std::pair<float, int>);
    out->unary_entry_bytes = sizeof(factor_engine::UnaryEntry);
    out->unary_moments_bytes = sizeof(factor_engine::UnaryMoments);
    out->pair_entry_bytes = sizeof(factor_engine::PairEntry);
    out->pair_moments_bytes = sizeof(factor_engine::PairMoments);
    out->max_threads = omp_get_max_threads();

    return factor_engine::cpu_status::Success;
}

extern "C" int fe_native_abi_version() {
    return factor_engine::NativeAbiVersion;
}
