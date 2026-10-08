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

enable_testing()
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
