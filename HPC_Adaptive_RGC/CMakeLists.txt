cmake_minimum_required(VERSION 3.16)
project(HPC_Adaptive_RGC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)

add_library(argcsr
    src/csr/csr_matrix.cpp
    src/csr/csr_spmv.cpp
    src/argcsr/argcsr_grouping.cpp
    src/argcsr/argcsr_chunking.cpp
    src/argcsr/argcsr_mapping.cpp
    src/argcsr/argcsr_conversion.cpp
    src/argcsr/argcsr_spmv.cpp
    src/validation/argcsr_validation.cpp
)
target_include_directories(argcsr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(adaptive_rgc src/main.cpp)
target_link_libraries(adaptive_rgc PRIVATE argcsr)
if(OpenMP_CXX_FOUND)
    target_link_libraries(adaptive_rgc PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(adaptive_rgc PRIVATE ADAPTIVE_RGC_HAS_OPENMP=1)
endif()

add_executable(generate_matrices benchmarks/generate_matrices.cpp)

add_executable(sequential_benchmark benchmarks/sequential_benchmark.cpp)
target_link_libraries(sequential_benchmark PRIVATE argcsr)

add_executable(conversion_profile benchmarks/conversion_profile.cpp)
target_link_libraries(conversion_profile PRIVATE argcsr)

enable_testing()
option(ENABLE_HYBRID_CPU_GPU "Build hybrid OpenMP conversion + CUDA SpMV module" OFF)
option(ENABLE_CUDA "Build the optional CUDA Adaptive RGC data-transfer module" OFF)
if(ENABLE_CUDA)
    if(CMAKE_VERSION VERSION_LESS 3.18)
        message(FATAL_ERROR "ENABLE_CUDA requires CMake 3.18 or newer")
    endif()
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "ENABLE_CUDA=ON but no CUDA compiler was found")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit 13 REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_library(argcsr_cuda
        src/cuda/argcsr_device.cu
        src/cuda/argcsr_cuda_pipeline.cu
        src/cuda/argcsr_cuda_grouping.cu
        src/cuda/argcsr_cuda_mapping.cu
        src/cuda/argcsr_cuda_storage.cu
        src/cuda/argcsr_cuda_spmv.cu
    )
    target_include_directories(argcsr_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(argcsr_cuda PUBLIC argcsr CUDA::cudart)
    set_target_properties(argcsr_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    # The configured workstation GPU is an RTX 5070 Laptop (compute capability 12.0).
    set_property(TARGET argcsr_cuda PROPERTY CUDA_ARCHITECTURES 120)

    add_executable(test_argcsr_device tests/test_argcsr_device.cu)
    target_link_libraries(test_argcsr_device PRIVATE argcsr_cuda)
    set_property(TARGET test_argcsr_device PROPERTY CUDA_ARCHITECTURES 120)
    add_test(NAME argcsr_device_representation COMMAND test_argcsr_device)

    add_executable(test_argcsr_cuda_pipeline tests/test_argcsr_cuda_pipeline.cu)
    target_link_libraries(test_argcsr_cuda_pipeline PRIVATE argcsr_cuda)
    set_property(TARGET test_argcsr_cuda_pipeline PROPERTY CUDA_ARCHITECTURES 120)
    add_test(NAME argcsr_cuda_pipeline COMMAND test_argcsr_cuda_pipeline)

    add_executable(cuda_pipeline_benchmark benchmarks/cuda_pipeline_benchmark.cpp)
    target_link_libraries(cuda_pipeline_benchmark PRIVATE argcsr_cuda)
    set_target_properties(cuda_pipeline_benchmark PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    add_library(argcsr_cuda_streams src/cuda/argcsr_cuda_streams.cu)
    target_include_directories(argcsr_cuda_streams PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(argcsr_cuda_streams PUBLIC argcsr_cuda CUDA::cudart)
    set_property(TARGET argcsr_cuda_streams PROPERTY CUDA_ARCHITECTURES 120)

    add_executable(test_argcsr_cuda_streams tests/test_argcsr_cuda_streams.cpp)
    target_link_libraries(test_argcsr_cuda_streams PRIVATE argcsr_cuda_streams)
    add_test(NAME argcsr_cuda_streams COMMAND test_argcsr_cuda_streams)
    set_tests_properties(argcsr_cuda_streams PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_executable(cuda_pipeline_streams_benchmark benchmarks/cuda_pipeline_streams_benchmark.cpp)
    target_link_libraries(cuda_pipeline_streams_benchmark PRIVATE argcsr_cuda_streams)
    set_target_properties(cuda_pipeline_streams_benchmark PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

add_executable(test_csr tests/test_csr.cpp)
target_link_libraries(test_csr PRIVATE argcsr)
add_test(NAME csr_foundation COMMAND test_csr)
add_executable(test_argcsr tests/test_argcsr.cpp)
target_link_libraries(test_argcsr PRIVATE argcsr)
add_test(NAME argcsr_conversion COMMAND test_argcsr)
add_executable(test_argcsr_spmv tests/test_argcsr_spmv.cpp)
target_link_libraries(test_argcsr_spmv PRIVATE argcsr)
add_test(NAME argcsr_spmv COMMAND test_argcsr_spmv)
add_executable(test_argcsr_random tests/test_argcsr_random.cpp)
target_link_libraries(test_argcsr_random PRIVATE argcsr)
add_test(NAME argcsr_random COMMAND test_argcsr_random)
if(ENABLE_HYBRID_CPU_GPU)
    add_subdirectory(hybrid_cpu_gpu)
endif()
