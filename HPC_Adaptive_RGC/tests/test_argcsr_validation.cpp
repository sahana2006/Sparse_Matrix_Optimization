#include "validation/argcsr_validation.hpp"
#include "argcsr/argcsr_spmv.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
void check(const argcsr::CSRMatrix& csr, int desired, int block, const std::vector<double>& x) {
    const auto adaptive = argcsr::convertToArgCSR(csr, desired, block);
    argcsr::validateArgCSR(csr, adaptive, x);
}
}

int main() {
    try {
        const argcsr::CSRMatrix phase2(4, 4, {10,2,3,4,5,7,8}, {0,3,1,0,2,2,3}, {0,2,3,5,7});
        check(phase2, 2, 4, {1,2,3,4});
        const argcsr::CSRMatrix uniform(4, 4, {1,2,3,4,5,6,7,8}, {0,1,1,2,2,3,0,3}, {0,2,4,6,8});
        check(uniform, 2, 4, {1,2,3,4});

        std::vector<int> rr, cc;
        std::vector<double> vv;
        for (int row = 0; row < 3; ++row) { rr.push_back(row); cc.push_back(row); vv.push_back(row + 1); }
        for (int col = 0; col < 8; ++col) { rr.push_back(3); cc.push_back(col); vv.push_back(col + 10); }
        const auto irregular = argcsr::CSRMatrix::fromCoordinates(4, 8, rr, cc, vv);
        const auto irregularAdaptive = argcsr::convertToArgCSR(irregular, 20, 8);
        require(irregularAdaptive.threadsMapping[3] > 1, "irregular row did not receive multiple chunks");
        check(irregular, 20, 8, {1,2,3,4,5,6,7,8});

        const argcsr::CSRMatrix emptyRows(5, 5, {2,3}, {0,4}, {0,0,1,1,1,2});
        const auto emptyAdaptive = argcsr::convertToArgCSR(emptyRows, 4, 4);
        argcsr::validateArgCSR(emptyRows, emptyAdaptive, {1,2,3,4,5});
        const auto emptyY = argcsr::argcsr_spmv(emptyAdaptive, {1,2,3,4,5});
        require(emptyY[0] == 0.0 && emptyY[2] == 0.0 && emptyY[3] == 0.0, "empty rows are not zero");

        // Zero-row matrices exercise the valid empty representation.
        const argcsr::CSRMatrix noRows(0, 3, {}, {}, {0});
        check(noRows, 4, 4, {1,2,3});
    } catch (const std::exception& e) {
        std::cerr << "ArgCSR validation tests failed: " << e.what() << '\n';
        return 1;
    }
    std::cout << "ArgCSR validation tests passed\n";
    return 0;
}
