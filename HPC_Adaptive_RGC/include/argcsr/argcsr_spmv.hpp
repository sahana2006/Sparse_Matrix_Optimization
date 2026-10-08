#pragma once

#include "argcsr/argcsr_types.hpp"

#include <vector>

namespace argcsr {

std::vector<double> argcsr_spmv(const ArgCSRMatrix& A, const std::vector<double>& x);

}  // namespace argcsr
