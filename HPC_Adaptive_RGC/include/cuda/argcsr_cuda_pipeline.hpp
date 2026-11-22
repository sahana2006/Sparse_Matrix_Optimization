#pragma once

#include "argcsr/argcsr_types.hpp"

#include <vector>

namespace argcsr {

struct CudaPipelineTimings {
    double hostValidationMs = 0.0;
    double h2dCsrMs = 0.0;
    double gpuRowStatisticsMs = 0.0;
    double d2hRowStatisticsMs = 0.0;
    double cpuGroupingMs = 0.0;
    double h2dGroupsMs = 0.0;
    double gpuMappingMs = 0.0;
    double gpuChunkSizingMs = 0.0;
    double d2hChunkMetadataMs = 0.0;
    double d2hStorageStatusMs = 0.0;
    double cpuGroupOffsetsMs = 0.0;
    double h2dFinalMetadataMs = 0.0;
    double gpuStorageConversionMs = 0.0;
    double h2dInputVectorMs = 0.0;
    double gpuSpmvMs = 0.0;
    double d2hOutputMs = 0.0;
    double d2hRepresentationMs = 0.0;
    double totalPipelineMs = 0.0;

    double cpuPreprocessingMs() const noexcept;
    double gpuKernelMs() const noexcept;
    double transferMs() const noexcept;
};

struct CudaPipelineResult {
    std::vector<double> output;
    // Populated only when captureRepresentation is true.
    ArgCSRMatrix representation{};
    CudaPipelineTimings timings;
};

// Runs CSR -> GPU row statistics -> CPU group-boundary decisions -> GPU mapping,
// chunk sizing and storage conversion -> GPU Adaptive RGC SpMV -> host output.
// captureRepresentation copies the GPU-created Adaptive RGC arrays back for
// validation; keep it false in normal timing runs.
CudaPipelineResult runArgCsrCudaPipeline(const CSRMatrix& csr,
                                         const std::vector<double>& x,
                                         int desiredChunkSize,
                                         int blockSize,
                                         bool captureRepresentation = false);

} // namespace argcsr
