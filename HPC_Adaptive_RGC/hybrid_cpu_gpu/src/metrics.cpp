#include "hybrid_cpu_gpu/metrics.hpp"

namespace hybrid_cpu_gpu {

SpeedupRow computeSpeedups(double sequentialMs, double hybridMs, double cudaMs) {
    SpeedupRow row;
    row.sequentialTotalMs = sequentialMs;
    row.hybridTotalMs = hybridMs;
    row.cudaPipelineTotalMs = cudaMs;
    if (hybridMs > 0.0) row.hybridVsSequential = sequentialMs / hybridMs;
    if (cudaMs > 0.0) row.hybridVsCuda = cudaMs / hybridMs;
    if (cudaMs > 0.0) row.cudaVsSequential = sequentialMs / cudaMs;
    return row;
}

}  // namespace hybrid_cpu_gpu
