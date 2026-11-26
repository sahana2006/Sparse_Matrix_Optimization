#pragma once

#include "hybrid_cpu_gpu/types.hpp"

#include "csr/csr_matrix.hpp"

#include <memory>
#include <vector>

namespace hybrid_cpu_gpu {

struct PipelineDeviceCache {
    struct Storage;

    PipelineDeviceCache();
    ~PipelineDeviceCache();
    PipelineDeviceCache(PipelineDeviceCache&&) noexcept;
    PipelineDeviceCache& operator=(PipelineDeviceCache&&) noexcept;
    PipelineDeviceCache(const PipelineDeviceCache&) = delete;
    PipelineDeviceCache& operator=(const PipelineDeviceCache&) = delete;

    void clear();

    std::unique_ptr<Storage> storage_;
};

bool cudaDeviceAvailable();

PipelineResult runHybridPipeline(const argcsr::CSRMatrix& csr,
                                 const std::vector<double>& x,
                                 int desiredChunkSize,
                                 int blockSize,
                                 const PipelineConfig& config = {},
                                 bool keepRepresentation = false,
                                 PipelineDeviceCache* deviceCache = nullptr);

double benchmarkHybridKernelOnly(const OpenMPConversionResult& converted,
                                 const std::vector<double>& x,
                                 int cudaBlockSize,
                                 std::vector<double>& output,
                                 double& kernelMs,
                                 TransferMetrics* transfer = nullptr);

}  // namespace hybrid_cpu_gpu
