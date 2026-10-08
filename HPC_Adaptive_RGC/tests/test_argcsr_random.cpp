#include "validation/argcsr_validation.hpp"

#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
argcsr::CSRMatrix makeRandom(int rows, int cols, unsigned seed, unsigned threshold) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> values(-9, 12);
    std::vector<double> vv;
    std::vector<int> cc, rp{0};
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (rng() % 100 < threshold) { cc.push_back(c); vv.push_back(values(rng)); }
        }
        rp.push_back(static_cast<int>(cc.size()));
    }
    return argcsr::CSRMatrix(rows, cols, std::move(vv), std::move(cc), std::move(rp));
}
}

int main() {
    try {
        const int sizes[][2] = {{1,1}, {3,7}, {8,5}, {12,12}, {17,9}};
        const unsigned sparsities[] = {5, 25, 55, 80};
        unsigned caseNo = 0;
        for (const auto& size : sizes) {
            for (unsigned sparsity : sparsities) {
                const auto csr = makeRandom(size[0], size[1], 0x51A7u + caseNo++, sparsity);
                const auto adaptive = argcsr::convertToArgCSR(csr, 13, 8);
                std::vector<double> x(static_cast<std::size_t>(size[1]));
                for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<double>(i + 1) / 3.0;
                argcsr::validateArgCSR(csr, adaptive, x);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "ArgCSR random tests failed: " << e.what() << '\n';
        return 1;
    }
    std::cout << "ArgCSR deterministic random tests passed (20 matrices)\n";
    return 0;
}
