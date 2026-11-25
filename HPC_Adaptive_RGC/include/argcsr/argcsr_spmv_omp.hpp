#pragma once

#include "argcsr/argcsr_types.hpp"
#include <vector>

namespace argcsr {

// OpenMP parallel implementation of ArgCSR SpMV.
// Returns the result vector y = A * x.
// The algorithm follows the sequential reference (argcsr_spmv) but
// parallelises over groups using OpenMP. Each group works on its own
// temporary partial‑sum buffer, so no race conditions occur.
std::vector<double> argcsr_spmv_omp(const ArgCSRMatrix& A,
                                    const std::vector<double>& x);

} // namespace argcsr
