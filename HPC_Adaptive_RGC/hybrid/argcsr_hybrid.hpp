#pragma once

#include "argcsr/argcsr_types.hpp"
#include "argcsr_openmp.hpp"
#include "csr/csr_matrix.hpp"

#include <vector>

namespace argcsr_hybrid {

struct HybridConfig {
    int ompThreads = 0;               // 0 -> OpenMP default (omp_get_max_threads)
    int cudaBlockSize = 256;          // CUDA threads per block for SpMV kernel
    bool runPostValidation = false;   // Optional post-conversion CPU structural check
};

struct HybridTimings {
    argcsr_omp::OpenMPConversionTimings ompConversion{};
    double ompTotalMs = 0.0;          // Total CPU OpenMP conversion wall time
    double deviceAllocMs = 0.0;       // cudaMalloc time for device buffers
    double h2dArgCsrMs = 0.0;         // H2D transfer of ArgCSR arrays + rowGroup + rowPrefix
    double h2dInputVectorMs = 0.0;    // H2D transfer of dense vector x
    double h2dTotalMs = 0.0;          // h2dArgCsrMs + h2dInputVectorMs
    double cudaSpmvMs = 0.0;          // CUDA kernel time measured via cudaEvent_t
    double d2hOutputMs = 0.0;         // D2H transfer of output vector y
    double deviceFreeMs = 0.0;        // cudaFree cleanup time
    double totalEndToEndMs = 0.0;     // End-to-end wall time (CSR -> OpenMP conv -> H2D -> SpMV -> D2H)
};

struct HybridPipelineResult {
    std::vector<double> output;
    argcsr::ArgCSRMatrix representation{};
    HybridTimings timings{};
};

// Executes: CSR -> OpenMP ArgCSR Conversion -> H2D Transfer -> CUDA ArgCSR SpMV -> D2H Output.
HybridPipelineResult runArgCsrHybridPipeline(const argcsr::CSRMatrix& csr,
                                             const std::vector<double>& x,
                                             int desiredChunkSize,
                                             int blockSize,
                                             const HybridConfig& config = {},
                                             bool captureRepresentation = false);

// Executes CUDA ArgCSR SpMV multiple times on an already-uploaded representation
// (used for benchmarking amortized SpMV latency separately from conversion/transfer).
double benchmarkHybridSpmvOnly(const argcsr_omp::OpenMPArgCSRResult& converted,
                               const std::vector<double>& x,
                               int cudaBlockSize,
                               std::vector<double>& output);

}  // namespace argcsr_hybrid
