#include "csr/csr_matrix.hpp"
#include "csr/csr_spmv.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool near(double a, double b) { return std::fabs(a - b) < 1e-12; }

argcsr::CSRMatrix makeMatrix() {
    return argcsr::CSRMatrix(4, 4,
        {10.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0},
        {0, 3, 1, 0, 2, 2, 3},
        {0, 2, 3, 5, 7});
}
}

int main() {
    try {
        // Test CSRMatrix construction and SpMV operation
        const auto A = makeMatrix();
        require(A.rows == 4 && A.cols == 4 && A.nnz == 7, "metadata mismatch");
        require(A.row_pointers == std::vector<int>({0, 2, 3, 5, 7}), "row pointers mismatch");
        require(A.column_indices == std::vector<int>({0, 3, 1, 0, 2, 2, 3}), "column indices mismatch");
        require(A.values == std::vector<double>({10, 2, 3, 4, 5, 7, 8}), "values mismatch");

        std::vector<double> y;
        argcsr::spmvCSRSerial(A, {1, 2, 3, 4}, y);
        const std::vector<double> expected{18, 6, 19, 53};
        require(y.size() == expected.size(), "SpMV result size mismatch"); // Check that the size of the result vector matches the expected size
        for (std::size_t i = 0; i < expected.size(); ++i) {
            require(near(y[i], expected[i]), "SpMV result mismatch at row " + std::to_string(i));
        }
    } catch (const std::exception& error) {
        std::cerr << "CSR test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "CSR tests passed\n";
    return 0;
}
