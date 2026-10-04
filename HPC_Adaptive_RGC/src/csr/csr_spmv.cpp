#include "csr/csr_spmv.hpp"

#include <stdexcept>

namespace argcsr {

void spmvCSRSerial(const CSRMatrix& A, const std::vector<double>& x, std::vector<double>& y) {
    if (x.size() != static_cast<std::size_t>(A.cols)) {
        throw std::invalid_argument("CSR SpMV input vector length must equal the matrix column count");
    }
    y.assign(static_cast<std::size_t>(A.rows), 0.0);
    for (int row = 0; row < A.rows; ++row) {
        double sum = 0.0;
        for (int i = A.row_pointers[static_cast<std::size_t>(row)];
             i < A.row_pointers[static_cast<std::size_t>(row) + 1]; ++i) {
            sum += A.values[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(A.column_indices[static_cast<std::size_t>(i)])];
        }
        y[static_cast<std::size_t>(row)] = sum;
    }
}

}  // namespace argcsr
