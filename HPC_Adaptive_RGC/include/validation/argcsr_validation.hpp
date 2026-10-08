#pragma once

#include "argcsr/argcsr_types.hpp"

#include <vector>

namespace argcsr {

// Throws std::logic_error with a structural explanation when the representation is invalid.
void validateArgCSRStructure(const ArgCSRMatrix& adaptive);

// Checks NNZ counts and that each CSR (value, column) entry appears exactly once in its row.
void validateArgCSRNnzPreservation(const CSRMatrix& csr, const ArgCSRMatrix& adaptive);

// Runs structural, NNZ, and serial SpMV equivalence checks.
void validateArgCSR(const CSRMatrix& csr, const ArgCSRMatrix& adaptive,
                    const std::vector<double>& x,
                    double absoluteTolerance = 1e-9,
                    double relativeTolerance = 1e-9);

}  // namespace argcsr
