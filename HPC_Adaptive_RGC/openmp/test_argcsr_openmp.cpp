#include "argcsr_openmp.hpp"
#include "argcsr/argcsr_spmv.hpp"
#include "csr/csr_spmv.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& msg) {
    if (!condition) throw std::runtime_error(msg);
}

void assertSameRepresentation(const argcsr::ArgCSRMatrix& seq,
                              const argcsr::ArgCSRMatrix& omp,
                              const std::string& label) {
    require(seq.rows == omp.rows && seq.cols == omp.cols && seq.nnz == omp.nnz &&
            seq.blockSize == omp.blockSize && seq.desiredChunkSize == omp.desiredChunkSize,
            label + ": scalar metadata mismatch");
    require(seq.groups.size() == omp.groups.size(), label + ": group count mismatch");
    for (std::size_t i = 0; i < seq.groups.size(); ++i) {
        const auto& a = seq.groups[i];
        const auto& b = omp.groups[i];
        require(a.firstRow == b.firstRow && a.size == b.size &&
                a.offset == b.offset && a.chunkSize == b.chunkSize,
                label + ": group mismatch at index " + std::to_string(i));
    }
    require(seq.threadsMapping == omp.threadsMapping, label + ": threadsMapping mismatch");
    require(seq.columns == omp.columns, label + ": columns mismatch");
    require(seq.values == omp.values, label + ": values mismatch");
}

void assertSpmvClose(const std::vector<double>& ref,
                     const std::vector<double>& actual,
                     const std::string& label) {
    require(ref.size() == actual.size(), label + ": SpMV output size mismatch");
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double tol = 1e-10 + 1e-9 * std::abs(ref[i]);
        require(std::isfinite(actual[i]) && std::abs(ref[i] - actual[i]) <= tol,
                label + ": SpMV value mismatch at row " + std::to_string(i));
    }
}

argcsr::CSRMatrix loadCsrBinary(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::int32_t rows = 0, cols = 0, nnz = 0;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    in.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));
    std::vector<int> rp(static_cast<std::size_t>(rows) + 1), ci(static_cast<std::size_t>(nnz));
    std::vector<double> vals(static_cast<std::size_t>(nnz));
    in.read(reinterpret_cast<char*>(rp.data()), static_cast<std::streamsize>(rp.size() * sizeof(int)));
    in.read(reinterpret_cast<char*>(ci.data()), static_cast<std::streamsize>(ci.size() * sizeof(int)));
    in.read(reinterpret_cast<char*>(vals.data()), static_cast<std::streamsize>(vals.size() * sizeof(double)));
    if (!in) throw std::runtime_error("truncated matrix " + path);
    return argcsr::CSRMatrix(rows, cols, std::move(vals), std::move(ci), std::move(rp));
}

void verifyCase(const std::string& name, const argcsr::CSRMatrix& csr,
                int desiredChunkSize, int blockSize, const std::vector<double>& x) {
    const auto seq = argcsr::convertToArgCSR(csr, desiredChunkSize, blockSize);
    std::vector<double> csrY;
    argcsr::spmvCSRSerial(csr, x, csrY);
    const auto seqY = argcsr::argcsr_spmv(seq, x);

    for (int t : {1, 2, 4, 8, 16, 24}) {
        const auto ompRes = argcsr_omp::convertToArgCSROpenMPDetailed(
            csr, desiredChunkSize, blockSize, t, true);
        const std::string tag = name + " (threads=" + std::to_string(t) + ")";
        assertSameRepresentation(seq, ompRes.matrix, tag);

        const auto ompY = argcsr_omp::argcsrSpmvOpenMP(ompRes.matrix, x, t);
        assertSpmvClose(csrY, ompY, tag + " vs CSR");
        assertSpmvClose(seqY, ompY, tag + " vs Seq ArgCSR");
    }
}

}  // namespace

int main() {
    try {
        // 1. Zero rows
        verifyCase("zero_rows", argcsr::CSRMatrix(0, 3, {}, {}, {0}), 4, 4, {1.0, 2.0, 3.0});

        // 2. All empty rows
        verifyCase("all_empty", argcsr::CSRMatrix(5, 3, {}, {}, {0, 0, 0, 0, 0, 0}), 4, 4, {2.0, 3.0, 4.0});

        // 3. Mixed empty rows and irregular long row
        const auto irregular = argcsr::CSRMatrix::fromCoordinates(
            8, 12,
            {0,0, 2, 2,2,2,2,2,2, 4,4, 5,5,5,5,5,5,5,5,5,5,5,5, 7},
            {0,8, 0,1,2,3,4,5,6, 1,10, 0,1,2,3,4,5,6,7,8,9,10,11, 6},
            {1,2, 1,2,3,4,5,6,7, 8,9, 1,2,3,4,5,6,7,8,9,10,11,12, 13});
        verifyCase("irregular_empty", irregular, 3, 4, {1,-2,3,4,-5,6,7,-8,9,10,-11,12});

        // 4. Larger blockSize=8 and multiple groups
        std::vector<int> rr, cc;
        std::vector<double> vv;
        for (int r = 0; r < 64; ++r) {
            const int len = (r % 7 == 0) ? 16 : (r % 3 + 1);
            for (int k = 0; k < len; ++k) {
                rr.push_back(r);
                cc.push_back((r * 3 + k) % 32);
                vv.push_back(static_cast<double>(r + k + 1) * 0.25);
            }
        }
        const auto mixed64 = argcsr::CSRMatrix::fromCoordinates(64, 32, rr, cc, vv);
        std::vector<double> x64(32, 1.5);
        verifyCase("mixed64_b8", mixed64, 4, 8, x64);

        // 5. Real benchmark matrices (U1, I1, U3, I3) across all thread counts
        for (const std::string name : {"U1", "I1", "U3", "I3"}) {
            const auto csr = loadCsrBinary("benchmarks/matrices/" + name + ".csrbin");
            std::vector<double> x(static_cast<std::size_t>(csr.cols));
            for (int c = 0; c < csr.cols; ++c) {
                x[static_cast<std::size_t>(c)] = static_cast<double>(c % 100 + 1) / 100.0;
            }
            verifyCase(name, csr, 2, 4, x);
            std::cout << name << ": OpenMP representation & SpMV parity PASS (threads 1..24)\n";
        }

        std::cout << "All OpenMP ArgCSR tests passed!\n";
    } catch (const std::exception& e) {
        std::cerr << "OpenMP ArgCSR test failed: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
