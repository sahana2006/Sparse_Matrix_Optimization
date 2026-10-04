#include "argcsr/argcsr_types.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
void verify(const argcsr::CSRMatrix& csr, int desired, int block) {
    const auto out = argcsr::convertToArgCSR(csr, desired, block);
    require(out.threadsMapping.size() == static_cast<std::size_t>(csr.rows), "mapping row count");
    int nnz = 0;
    for (const auto& g : out.groups) {
        int sum = 0;
        for (int r = g.firstRow; r < g.firstRow + g.size; ++r) sum += out.threadsMapping[r];
        require(sum <= block, "group mapping exceeds block");
        for (int p = g.offset; p < g.offset + g.chunkSize * block; ++p) if (out.columns[p] != -1) ++nnz;
    }
    require(nnz == csr.nnz, "NNZ preservation");
}
}

int main() {
    try {
        const argcsr::CSRMatrix uniform(4, 4, {1,2,3,4,5,6,7,8}, {0,1,1,2,2,3,0,3}, {0,2,4,6,8});
        verify(uniform, 2, 4);

        std::vector<int> rr, cc; std::vector<double> vv;
        for (int r = 0; r < 3; ++r) { rr.push_back(r); cc.push_back(r); vv.push_back(r + 1); }
        for (int c = 0; c < 8; ++c) { rr.push_back(3); cc.push_back(c); vv.push_back(c + 10); }
        const auto irregular = argcsr::CSRMatrix::fromCoordinates(4, 8, rr, cc, vv);
        const auto adaptive = argcsr::convertToArgCSR(irregular, 20, 8);
        verify(irregular, 20, 8);
        require(adaptive.threadsMapping[3] > 1, "extra assignments should go to long row");

        const argcsr::CSRMatrix emptyRows(5, 5, {2,3}, {0,4}, {0,0,1,1,1,2});
        verify(emptyRows, 4, 4);

        const argcsr::CSRMatrix phase2(4, 4, {10,2,3,4,5,7,8}, {0,3,1,0,2,2,3}, {0,2,3,5,7});
        verify(phase2, 2, 4);
        require(argcsr::exclusivePrefixSum({1,1,4,2}) == std::vector<int>({0,1,2,6}), "exclusive prefix");
        argcsr::printArgCSR(adaptive);
    } catch (const std::exception& e) {
        std::cerr << "ArgCSR tests failed: " << e.what() << '\n';
        return 1;
    }
    std::cout << "ArgCSR tests passed\n";
    return 0;
}
