#pragma once

#include "hybrid_cpu_gpu/types.hpp"

#include "csr/csr_matrix.hpp"

namespace hybrid_cpu_gpu {

// CSR -> ArgCSR using the same algorithm as the sequential baseline, with OpenMP
// parallelism over independent row groups (after sequential group formation).
OpenMPConversionResult convertCsrToArgCsrOpenMP(const argcsr::CSRMatrix& csr,
                                                int desiredChunkSize,
                                                int blockSize,
                                                int ompThreads = 0);

}  // namespace hybrid_cpu_gpu
