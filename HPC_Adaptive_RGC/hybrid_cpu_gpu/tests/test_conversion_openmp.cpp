#include "hybrid_cpu_gpu/conversion_openmp.hpp"

#include "argcsr/argcsr_spmv.hpp"
#include "argcsr/argcsr_types.hpp"
#include "csr/csr_spmv.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool ok, const std::string& msg) {
    if (!ok) throw std::runtime_error(msg);
}

void assertSameMatrix(const argcsr::ArgCSRMatrix& a, const argcsr::ArgCSRMatrix& b,
                      const std::string& tag) {
    require(a.rows == b.rows && a.cols == b.cols && a.nnz == b.nnz &&
                a.blockSize == b.blockSize && a.desiredChunkSize == b.desiredChunkSize,
            tag + ": metadata");
    require(a.groups.size() == b.groups.size(), tag + ": group count");
    for (std::size_t i = 0; i < a.groups.size(); ++i) {
        const auto& ga = a.groups[i];
        const auto& gb = b.groups[i];
        require(ga.firstRow == gb.firstRow && ga.size == gb.size && ga.offset == gb.offset &&
                    ga.chunkSize == gb.chunkSize,
                tag + ": group " + std::to_string(i));
    }
    require(a.threadsMapping == b.threadsMapping, tag + ": mapping");
    require(a.values == b.values, tag + ": values");
    require(a.columns == b.columns, tag + ": columns");
}

void assertVectorsClose(const std::vector<double>& ref, const std::vector<double>& got,
                        const std::string& tag) {
    require(ref.size() == got.size(), tag + ": size");
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double tol = 1e-10 + 1e-9 * std::abs(ref[i]);
        require(std::isfinite(got[i]) && std::abs(ref[i] - got[i]) <= tol,
                tag + ": row " + std::to_string(i));
    }
}

void caseStudy(const std::string& name, const argcsr::CSRMatrix& csr, int desired, int block,
               const std::vector<double>& x) {
    const auto sequential = argcsr::convertToArgCSR(csr, desired, block);
    const auto omp = hybrid_cpu_gpu::convertCsrToArgCsrOpenMP(csr, desired, block, 4);
    assertSameMatrix(sequential, omp.matrix, name);

    const auto ySeq = argcsr::argcsr_spmv(sequential, x);
    const auto yOmp = argcsr::argcsr_spmv(omp.matrix, x);
    assertVectorsClose(ySeq, yOmp, name + " SpMV via sequential kernel");

    std::vector<double> yCsr;
    argcsr::spmvCSRSerial(csr, x, yCsr);
    assertVectorsClose(yCsr, yOmp, name + " vs CSR");
}

}  // namespace

int main() {
    try {
        const argcsr::CSRMatrix tiny(
            4, 4, {10, 2, 3, 4, 5, 7, 8},
            {0, 3, 1, 0, 2, 2, 3}, {0, 2, 3, 5, 7});
        caseStudy("tiny", tiny, 2, 4, {1, 2, 3, 4});

        const auto irregular = argcsr::CSRMatrix::fromCoordinates(
            8, 12,
            {0, 0, 2, 2, 2, 2, 2, 2, 2, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7},
            {0, 8, 0, 1, 2, 3, 4, 5, 6, 1, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 6},
            {1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13});
        caseStudy("irregular", irregular, 3, 4, {1, -2, 3, 4, -5, 6, 7, -8, 9, 10, -11, 12});

        const auto irregularSeq = argcsr::convertToArgCSR(irregular, 3, 4);
        for (int t : {1, 2, 4, 8}) {
            const auto omp = hybrid_cpu_gpu::convertCsrToArgCsrOpenMP(irregular, 3, 4, t);
            assertSameMatrix(irregularSeq, omp.matrix,
                             "irregular threads=" + std::to_string(t));
        }

        std::cout << "Hybrid OpenMP conversion tests passed (no .csrbin required).\n";
    } catch (const std::exception& e) {
        std::cerr << "test_hybrid_conversion_openmp failed: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
