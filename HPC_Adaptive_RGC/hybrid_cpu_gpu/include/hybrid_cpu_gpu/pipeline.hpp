#pragma once

#include "hybrid_cpu_gpu/types.hpp"

#include "csr/csr_matrix.hpp"

#include <vector>

namespace hybrid_cpu_gpu {

bool cudaDeviceAvailable();

// End-to-end: OpenMP conversion -> H2D -> CUDA ArgCSR SpMV -> D2H.
PipelineResult runHybridPipeline(const argcsr::CSRMatrix& csr,
                                 const std::vector<double>& x,
                                 int desiredChunkSize,
                                 int blockSize,
                                 const PipelineConfig& config = {},
                                 bool keepRepresentation = false);

// Amortized SpMV timing on an already-converted matrix (after warm-up upload).
double benchmarkHybridKernelOnly(const OpenMPConversionResult& converted,
                                 const std::vector<double>& x,
                                 int cudaBlockSize,
                                 std::vector<double>& output,
                                 double& kernelMs,
                                 TransferMetrics* transfer = nullptr);

}  // namespace hybrid_cpu_gpu
