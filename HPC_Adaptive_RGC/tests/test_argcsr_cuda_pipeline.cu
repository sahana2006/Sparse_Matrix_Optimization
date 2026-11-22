#include "argcsr/argcsr_spmv.hpp"
#include "cuda/argcsr_cuda_pipeline.hpp"
#include "csr/csr_spmv.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void requireSameRepresentation(const argcsr::ArgCSRMatrix& cpu,
                               const argcsr::ArgCSRMatrix& gpu,
                               const std::string& name) {
    require(cpu.rows == gpu.rows && cpu.cols == gpu.cols && cpu.nnz == gpu.nnz &&
            cpu.blockSize == gpu.blockSize && cpu.desiredChunkSize == gpu.desiredChunkSize,
            name + ": scalar metadata mismatch");
    require(cpu.groups.size() == gpu.groups.size(), name + ": group count mismatch");
    for (std::size_t i = 0; i < cpu.groups.size(); ++i) {
        const auto& a = cpu.groups[i];
        const auto& b = gpu.groups[i];
        require(a.firstRow == b.firstRow && a.size == b.size,
                name + ": grouping stage differs at group " + std::to_string(i));
        require(a.chunkSize == b.chunkSize,
                name + ": chunk sizing differs at group " + std::to_string(i));
        require(a.offset == b.offset,
                name + ": group storage offset differs at group " + std::to_string(i));
    }
    require(cpu.threadsMapping == gpu.threadsMapping, name + ": GPU mapping differs from CPU mapping");
    require(cpu.values == gpu.values, name + ": GPU storage values differ from CPU conversion");
    require(cpu.columns == gpu.columns, name + ": GPU storage columns differ from CPU conversion");
}

void runCase(const std::string& name, const argcsr::CSRMatrix& csr, int desired, int block,
             const std::vector<double>& x) {
    const auto cpu = argcsr::convertToArgCSR(csr, desired, block);
    const auto gpu = argcsr::runArgCsrCudaPipeline(csr, x, desired, block, true);
    requireSameRepresentation(cpu, gpu.representation, name);

    std::vector<double> csrOutput;
    argcsr::spmvCSRSerial(csr, x, csrOutput);
    require(gpu.output.size() == csrOutput.size(), name + ": output row count mismatch");
    for (std::size_t row = 0; row < csrOutput.size(); ++row) {
        const double tolerance = 1e-10 + 1e-9 * std::abs(csrOutput[row]);
        require(std::isfinite(gpu.output[row]) &&
                std::abs(gpu.output[row] - csrOutput[row]) <= tolerance,
                name + ": GPU SpMV output differs at row " + std::to_string(row));
    }
    require(gpu.timings.totalPipelineMs >= gpu.timings.gpuKernelMs(),
            name + ": total pipeline time is smaller than the kernel-time sum");
}
}

int main() {
    int deviceCount = 0;
    const cudaError_t status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess || deviceCount == 0) {
        std::cout << "CUDA Adaptive RGC pipeline test skipped: no usable CUDA device\n";
        (void)cudaGetLastError();
        return 0;
    }

    try {
        const argcsr::CSRMatrix uniform(8, 8,
            {1,2,3,4,5,6,7,8, 2,3,4,5,6,7,8,9},
            {0,2, 1,3, 2,4, 3,5, 4,6, 5,7, 0,6, 1,7},
            {0,2,4,6,8,10,12,14,16});
        runCase("uniform multiple groups", uniform, 4, 4, {1,2,3,4,5,6,7,8});

        const auto irregular = argcsr::CSRMatrix::fromCoordinates(
            8, 12,
            {0,0, 2, 2,2,2,2,2,2, 4,4, 5,5,5,5,5,5,5,5,5,5,5,5, 7},
            {0,8, 0,1,2,3,4,5,6, 1,10, 0,1,2,3,4,5,6,7,8,9,10,11, 6},
            {1,2, 1,2,3,4,5,6,7, 8,9, 1,2,3,4,5,6,7,8,9,10,11,12, 13});
        runCase("irregular long rows and empty rows", irregular, 3, 4,
                {1,-2,3,4,-5,6,7,-8,9,10,-11,12});

        const argcsr::CSRMatrix noNonzeros(5, 3, {}, {}, {0,0,0,0,0,0});
        runCase("all empty rows", noNonzeros, 4, 4, {2,3,4});

        const argcsr::CSRMatrix noRows(0, 3, {}, {}, {0});
        runCase("zero rows", noRows, 4, 4, {2,3,4});
    } catch (const std::exception& error) {
        std::cerr << "CUDA Adaptive RGC pipeline tests failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "CUDA Adaptive RGC pipeline tests passed\n";
    return 0;
}
