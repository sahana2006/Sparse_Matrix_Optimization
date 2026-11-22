#pragma once

#include "cuda/argcsr_cuda_pipeline.hpp"

namespace argcsr {

// Experimental Phase 6D path. Uses one explicit transfer stream and reuses the
// unchanged Phase 6C CUDA kernels and CPU grouping decision.
CudaPipelineResult runArgCsrCudaStreamsPipeline(const CSRMatrix& csr,
                                                const std::vector<double>& x,
                                                int desiredChunkSize,
                                                int blockSize);

} // namespace argcsr
