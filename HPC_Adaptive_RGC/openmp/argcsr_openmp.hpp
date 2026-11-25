#pragma once

#include "argcsr/argcsr_types.hpp"
#include "csr/csr_matrix.hpp"

#include <vector>

namespace argcsr_omp {

struct OpenMPConversionTimings {
    double validationMs = 0.0;
    double groupingMs = 0.0;
    double mappingChunkingMs = 0.0;
    double prefixOffsetsMs = 0.0;
    double allocationInitMs = 0.0;
    double storageScatterMs = 0.0;
    double postValidationMs = 0.0;
    double totalMs = 0.0;
};

struct OpenMPArgCSRResult {
    argcsr::ArgCSRMatrix matrix{};
    std::vector<int> rowPrefix;
    std::vector<int> rowGroup;
    OpenMPConversionTimings timings{};
};

// Converts CSRMatrix to ArgCSRMatrix using OpenMP parallelization across
// independent groups for thread-to-row mapping, chunk sizing, offset prefixing,
// slab initialization, and storage scatter. Grouping remains sequential due to
// its strict row-to-row workload dependency.
OpenMPArgCSRResult convertToArgCSROpenMPDetailed(const argcsr::CSRMatrix& csr,
                                                 int desiredChunkSize,
                                                 int blockSize,
                                                 int numThreads = 0,
                                                 bool runPostValidation = false);

argcsr::ArgCSRMatrix convertToArgCSROpenMP(const argcsr::CSRMatrix& csr,
                                           int desiredChunkSize,
                                           int blockSize,
                                           int numThreads = 0,
                                           OpenMPConversionTimings* timings = nullptr,
                                           bool runPostValidation = false);

// OpenMP parallel ArgCSR SpMV: parallelizes across independent row groups
// using a single thread-private partial-sum buffer per OpenMP thread.
std::vector<double> argcsrSpmvOpenMP(const argcsr::ArgCSRMatrix& A,
                                     const std::vector<double>& x,
                                     int numThreads = 0);

}  // namespace argcsr_omp
