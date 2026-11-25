#include "argcsr_hybrid.hpp"
#include "argcsr/argcsr_spmv.hpp"
#include "argcsr_openmp.hpp"
#include "csr/csr_spmv.hpp"
#include "cuda/argcsr_cuda_pipeline.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& msg) {
    if (!condition) throw std::runtime_error(msg);
}

void assertSameRepresentation(const argcsr::ArgCSRMatrix& ref,
                              const argcsr::ArgCSRMatrix& cand,
                              const std::string& label) {
    require(ref.rows == cand.rows && ref.cols == cand.cols && ref.nnz == cand.nnz &&
            ref.blockSize == cand.blockSize && ref.desiredChunkSize == cand.desiredChunkSize,
            label + ": scalar metadata mismatch");
    require(ref.groups.size() == cand.groups.size(), label + ": group count mismatch");
    for (std::size_t i = 0; i < ref.groups.size(); ++i) {
        const auto& a = ref.groups[i];
        const auto& b = cand.groups[i];
        require(a.firstRow == b.firstRow && a.size == b.size &&
                a.offset == b.offset && a.chunkSize == b.chunkSize,
                label + ": group mismatch at " + std::to_string(i));
    }
    require(ref.threadsMapping == cand.threadsMapping, label + ": threadsMapping mismatch");
    require(ref.columns == cand.columns, label + ": columns mismatch");
    require(ref.values == cand.values, label + ": values mismatch");
}

void assertOutputClose(const std::vector<double>& ref,
                       const std::vector<double>& actual,
                       const std::string& label) {
    require(ref.size() == actual.size(), label + ": output length mismatch");
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double tol = 1e-10 + 1e-9 * std::abs(ref[i]);
        require(std::isfinite(actual[i]) && std::abs(ref[i] - actual[i]) <= tol,
                label + ": output mismatch at row " + std::to_string(i));
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

void verifyHybridCase(const std::string& name, const argcsr::CSRMatrix& csr,
                      int desired, int block, const std::vector<double>& x) {
    const auto seqRep = argcsr::convertToArgCSR(csr, desired, block);
    const auto seqY = argcsr::argcsr_spmv(seqRep, x);
    std::vector<double> csrY;
    argcsr::spmvCSRSerial(csr, x, csrY);
    const auto cudaRes = argcsr::runArgCsrCudaPipeline(csr, x, desired, block, true);

    for (int threads : {1, 4, 8, 24}) {
        for (int cudaBlock : {64, 128, 256, 512}) {
            const argcsr_hybrid::HybridConfig cfg{threads, cudaBlock, true};
            const auto hybridRes = argcsr_hybrid::runArgCsrHybridPipeline(
                csr, x, desired, block, cfg, true);
            const std::string tag = name + " (omp=" + std::to_string(threads) +
                                    ", cuBlk=" + std::to_string(cudaBlock) + ")";
            assertSameRepresentation(seqRep, hybridRes.representation, tag + " vs Seq");
            assertSameRepresentation(cudaRes.representation, hybridRes.representation, tag + " vs PureCUDA");
            assertOutputClose(csrY, hybridRes.output, tag + " vs CSR");
            assertOutputClose(seqY, hybridRes.output, tag + " vs Seq ArgCSR");
            assertOutputClose(cudaRes.output, hybridRes.output, tag + " vs PureCUDA");
        }
    }
}

}  // namespace

int main() {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA device available for hybrid test\n";
        return 1;
    }

    try {
        verifyHybridCase("zero_rows", argcsr::CSRMatrix(0, 3, {}, {}, {0}), 4, 4, {1.0, 2.0, 3.0});
        verifyHybridCase("all_empty", argcsr::CSRMatrix(5, 3, {}, {}, {0, 0, 0, 0, 0, 0}), 4, 4, {2.0, 3.0, 4.0});

        const auto irregular = argcsr::CSRMatrix::fromCoordinates(
            8, 12,
            {0,0, 2, 2,2,2,2,2,2, 4,4, 5,5,5,5,5,5,5,5,5,5,5,5, 7},
            {0,8, 0,1,2,3,4,5,6, 1,10, 0,1,2,3,4,5,6,7,8,9,10,11, 6},
            {1,2, 1,2,3,4,5,6,7, 8,9, 1,2,3,4,5,6,7,8,9,10,11,12, 13});
        verifyHybridCase("irregular_empty", irregular, 3, 4, {1,-2,3,4,-5,6,7,-8,9,10,-11,12});

        for (const std::string name : {"U1", "I1", "U3", "I3"}) {
            const auto csr = loadCsrBinary("benchmarks/matrices/" + name + ".csrbin");
            std::vector<double> x(static_cast<std::size_t>(csr.cols));
            for (int c = 0; c < csr.cols; ++c) {
                x[static_cast<std::size_t>(c)] = static_cast<double>(c % 100 + 1) / 100.0;
            }
            verifyHybridCase(name, csr, 2, 4, x);
            std::cout << name << ": Hybrid OpenMP+CUDA representation & SpMV parity PASS\n";
        }

        std::cout << "All Hybrid OpenMP+CUDA tests passed!\n";
    } catch (const std::exception& e) {
        std::cerr << "Hybrid test failed: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
