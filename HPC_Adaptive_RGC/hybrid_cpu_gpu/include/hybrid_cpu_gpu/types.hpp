#pragma once

#include "argcsr/argcsr_types.hpp"

#include <vector>

namespace hybrid_cpu_gpu {

struct ConversionTimingsMs {
    double groupingMs = 0.0;
    double mappingMs = 0.0;
    double chunkingMs = 0.0;
    double storageMs = 0.0;
    double metadataMs = 0.0;
    double totalMs = 0.0;
};

struct OpenMPConversionResult {
    argcsr::ArgCSRMatrix matrix{};
    std::vector<int> rowGroup;
    std::vector<int> rowPrefix;
    ConversionTimingsMs timings{};
};

struct PipelineConfig {
    int ompThreads = 0;
    int cudaBlockSize = 256;
    bool validateAgainstSequential = false;
};

struct TransferMetrics {
    std::size_t h2dBytes = 0;
    std::size_t d2hBytes = 0;
    double h2dMs = 0.0;
    double d2hMs = 0.0;
    double h2dBandwidthGbps = 0.0;
    double d2hBandwidthGbps = 0.0;
};

struct PipelineTimingsMs {
    ConversionTimingsMs ompConversion{};
    double deviceAllocMs = 0.0;
    double kernelMs = 0.0;
    TransferMetrics transfer{};
    double totalMs = 0.0;
};

struct PipelineResult {
    std::vector<double> output;
    PipelineTimingsMs timings{};
    argcsr::ArgCSRMatrix representation{};
};

double gigaBytesPerSecond(std::size_t bytes, double milliseconds);

}  // namespace hybrid_cpu_gpu
