cmake_minimum_required(VERSION 3.16)

if(NOT ENABLE_HYBRID_CPU_GPU)
    return()
endif()

if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "ENABLE_HYBRID_CPU_GPU requires OpenMP")
endif()

set(HYBRID_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(hybrid_cpu_gpu_core
    src/conversion_openmp.cpp
    src/metrics.cpp
)
target_include_directories(hybrid_cpu_gpu_core
    PUBLIC ${HYBRID_INCLUDE}
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(hybrid_cpu_gpu_core PUBLIC argcsr OpenMP::OpenMP_CXX)

add_executable(test_hybrid_conversion_openmp tests/test_conversion_openmp.cpp)
target_link_libraries(test_hybrid_conversion_openmp PRIVATE hybrid_cpu_gpu_core)
add_test(NAME hybrid_conversion_openmp COMMAND test_hybrid_conversion_openmp)

if(ENABLE_CUDA)
    add_library(hybrid_cpu_gpu_cuda STATIC src/pipeline.cu)
    target_include_directories(hybrid_cpu_gpu_cuda
        PUBLIC ${HYBRID_INCLUDE}
        PRIVATE ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(hybrid_cpu_gpu_cuda PUBLIC hybrid_cpu_gpu_core CUDA::cudart)
    set_target_properties(hybrid_cpu_gpu_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    set_property(TARGET hybrid_cpu_gpu_cuda PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

    add_executable(test_hybrid_pipeline tests/test_hybrid_pipeline.cpp)
    target_link_libraries(test_hybrid_pipeline PRIVATE hybrid_cpu_gpu_cuda argcsr argcsr_cuda)
    add_test(NAME hybrid_pipeline COMMAND test_hybrid_pipeline)

    add_executable(hybrid_four_way_benchmark benchmarks/four_way_benchmark.cpp)
    target_link_libraries(hybrid_four_way_benchmark PRIVATE hybrid_cpu_gpu_cuda argcsr argcsr_cuda)
    set_target_properties(hybrid_four_way_benchmark PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()
