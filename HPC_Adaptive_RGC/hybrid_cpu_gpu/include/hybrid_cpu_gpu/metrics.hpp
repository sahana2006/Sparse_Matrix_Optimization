#pragma once

#include "hybrid_cpu_gpu/types.hpp"

namespace hybrid_cpu_gpu {

struct SpeedupRow {
    double sequentialTotalMs = 0.0;
    double hybridTotalMs = 0.0;
    double cudaPipelineTotalMs = 0.0;
    double hybridVsSequential = 0.0;
    double hybridVsCuda = 0.0;
    double cudaVsSequential = 0.0;
};

SpeedupRow computeSpeedups(double sequentialMs, double hybridMs, double cudaMs);

}  // namespace hybrid_cpu_gpu
