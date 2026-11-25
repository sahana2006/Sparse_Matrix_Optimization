#pragma once

#include "argcsr/argcsr_types.hpp"
#include <vector>

namespace argcsr {

// Hybrid CPU (OpenMP) + GPU (CUDA) SpMV.
// *A* is an ArgCSR matrix, *x* the input dense vector.
// *gpu_fraction* (0.0‑1.0) determines the fraction of groups that will be processed on the GPU.
// If CUDA is disabled or unavailable, the function falls back to the pure OpenMP version.
// Returns y = A * x.
std::vector<double> argcsr_hybrid_spmv(const ArgCSRMatrix& A,
                                       const std::vector<double>& x,
                                       double gpu_fraction = 0.5);

} // namespace argcsr
