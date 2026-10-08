#include "argcsr/argcsr_spmv.hpp"
#include "csr/csr_spmv.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const std::string& message) { // Throws an exception if the condition is not met
    if (!ok) throw std::runtime_error(message);
}

void compare(const argcsr::CSRMatrix& csr, int desiredChunkSize, int blockSize,
             const std::vector<double>& x) {
    const auto adaptive = argcsr::convertToArgCSR(csr, desiredChunkSize, blockSize); // Convert CSR to Adaptive RGC format
    std::vector<double> expected;
    argcsr::spmvCSRSerial(csr, x, expected); // expected = CSR * x
    const auto actual = argcsr::argcsr_spmv(adaptive, x); // actual = Adaptive RGC * x
    
    require(actual.size() == expected.size(), "output row count mismatch");
    
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double tolerance = 1e-9 * (1.0 + std::abs(expected[i]));
        require(std::abs(actual[i] - expected[i]) <= tolerance,
                "CSR and Adaptive RGC results differ at row " + std::to_string(i));
    }
}
}

int main() {
    try {
        const std::vector<double> x{1, 2, 3, 4};
        // Test 1: A simple 4x4 matrix with a few non-zero elements
        const argcsr::CSRMatrix phase2(4, 4, {10,2,3,4,5,7,8}, {0,3,1,0,2,2,3}, {0,2,3,5,7});
        compare(phase2, 2, 4, x);

        // Test 2: A uniform 4x4 matrix with non-zero elements in every row and column
        const argcsr::CSRMatrix uniform(4, 4, {1,2,3,4,5,6,7,8}, {0,1,1,2,2,3,0,3}, {0,2,4,6,8});
        compare(uniform, 2, 4, x);

        // Test 3: A 4x8 matrix with an irregular row (row 3 has many non-zero elements)
        std::vector<int> rr, cc;
        std::vector<double> vv;
        for (int r = 0; r < 3; ++r) { rr.push_back(r); cc.push_back(r); vv.push_back(r + 1); }
        for (int c = 0; c < 8; ++c) { rr.push_back(3); cc.push_back(c); vv.push_back(c + 10); }
        const auto irregular = argcsr::CSRMatrix::fromCoordinates(4, 8, rr, cc, vv);
        const auto irregularArg = argcsr::convertToArgCSR(irregular, 20, 8);
        require(irregularArg.threadsMapping[3] > 1, "irregular row must receive multiple chunks");
        compare(irregular, 20, 8, {1,2,3,4,5,6,7,8});

        // Test 4: A 5x5 matrix with empty rows (rows 0, 2, and 3 are empty)
        const argcsr::CSRMatrix emptyRows(5, 5, {2,3}, {0,4}, {0,0,1,1,1,2});
        compare(emptyRows, 4, 4, {1,2,3,4,5});
        const auto emptyResult = argcsr::argcsr_spmv(argcsr::convertToArgCSR(emptyRows, 4, 4), {1,2,3,4,5});
        require(emptyResult[0] == 0.0 && emptyResult[2] == 0.0 && emptyResult[3] == 0.0,
                "empty rows must produce zero");

        // Test 5: Randomized tests with different seeds to ensure robustness
        for (unsigned seed = 7; seed < 12; ++seed) {
            std::mt19937 rng(seed);
            std::uniform_int_distribution<int> valueDist(-5, 9);
            std::vector<int> cols;
            std::vector<double> vals;
            std::vector<int> rowPointers{0};
            constexpr int rows = 6, columns = 7;
            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < columns; ++col) {
                    if (rng() % 3 == 0) { cols.push_back(col); vals.push_back(valueDist(rng)); }
                }
                rowPointers.push_back(static_cast<int>(cols.size()));
            }
            const auto randomCsr = argcsr::CSRMatrix(rows, columns, vals, cols, rowPointers);
            compare(randomCsr, 5, 4, {1, -2, 3, 0.5, 4, -1, 2});
        }
    } catch (const std::exception& e) {
        std::cerr << "ArgCSR SpMV tests failed: " << e.what() << '\n';
        return 1;
    }
    std::cout << "ArgCSR SpMV tests passed\n";
    return 0;
}
