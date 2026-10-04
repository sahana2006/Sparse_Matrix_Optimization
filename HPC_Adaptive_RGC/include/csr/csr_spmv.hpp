#pragma once

#include "csr/csr_matrix.hpp"

#include <vector>

namespace argcsr {

// Function to perform sparse matrix-vector multiplication (SpMV) using the CSR format  
void spmvCSRSerial(const CSRMatrix& A, const std::vector<double>& x, std::vector<double>& y);

}  // namespace argcsr
