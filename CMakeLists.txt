cmake_minimum_required(VERSION 3.24)
project(FactorEngine LANGUAGES CXX)

option(FACTORENGINE_ENABLE_CUDA "Build the current Generic CUDA backend" OFF)
option(FACTORENGINE_BUILD_TESTS "Build native correctness tests" ON)
set(FACTORENGINE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA target architectures")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(OpenMP REQUIRED)

add_library(factor_engine_plan SHARED
    cpp/cpu/operators.cpp
    cpp/cpu/plan_runtime.cpp
    cpp/cpu/rank_ic.cpp
    cpp/cpu/cpu_layout.cpp)
target_include_directories(factor_engine_plan PUBLIC cpp/include)
target_link_libraries(factor_engine_plan PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(factor_engine_plan PRIVATE
    $<$<NOT:$<CONFIG:Debug>>:-O3>
    -march=native)

if(FACTORENGINE_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${FACTORENGINE_CUDA_ARCHITECTURES}")
    enable_language(CUDA)
    add_library(factor_engine_cuda SHARED cuda/backend.cu
        cuda/device_common.cuh cuda/device_buffers.cuh cuda/resident_runtime.cuh
        cuda/pointwise.cuh cuda/temporal.cuh cuda/rank_segmented.cuh
        cuda/metric_workspace.cuh cuda/metric_common.cuh cuda/metric_short.cuh
        cuda/metric_long.cuh cuda/metric_dispatch.cuh)
    target_include_directories(factor_engine_cuda PRIVATE cpp/include)
    target_link_libraries(factor_engine_cuda PRIVATE CUDA::cudart)
    set_target_properties(factor_engine_cuda PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${FACTORENGINE_CUDA_ARCHITECTURES}")
    target_compile_options(factor_engine_cuda PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-lineinfo;-Xptxas=-v>)

endif()

if(FACTORENGINE_BUILD_TESTS)
    enable_testing()
    add_executable(factor_engine_cpu_test cpp/tests/test_cpu.cpp)
    target_compile_options(factor_engine_cpu_test PRIVATE -UNDEBUG)
    target_link_libraries(factor_engine_cpu_test PRIVATE factor_engine_plan)
    add_test(NAME factor_engine_cpu_test COMMAND factor_engine_cpu_test)

    # 主机检查无需 CUDA，设备验证直接执行生产接口。
    add_executable(factor_engine_contracts tests/cpp/test_contracts.cpp)
    target_include_directories(factor_engine_contracts PRIVATE cpp/include)
    add_test(NAME factor_engine_contracts COMMAND factor_engine_contracts)
    add_executable(factor_engine_rank_math tests/cpp/test_rank_math.cpp)
    target_include_directories(factor_engine_rank_math PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(factor_engine_rank_math PRIVATE -UNDEBUG)
    add_test(NAME factor_engine_rank_math COMMAND factor_engine_rank_math)
endif()
