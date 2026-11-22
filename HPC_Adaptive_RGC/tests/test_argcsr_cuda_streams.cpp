#include "argcsr/argcsr_spmv.hpp"
#include "cuda/argcsr_cuda_pipeline.hpp"
#include "cuda/argcsr_cuda_streams.h"
#include "csr/csr_spmv.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
argcsr::CSRMatrix load(const std::string& name) {
    const std::string path = "benchmarks/matrices/" + name + ".csrbin";
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::int32_t rows = 0, cols = 0, nnz = 0;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    in.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));
    if (!in || rows < 0 || cols < 0 || nnz < 0) throw std::runtime_error("invalid matrix header: " + path);
    std::vector<int> rp(static_cast<std::size_t>(rows) + 1), ci(static_cast<std::size_t>(nnz));
    std::vector<double> values(static_cast<std::size_t>(nnz));
    in.read(reinterpret_cast<char*>(rp.data()), static_cast<std::streamsize>(rp.size() * sizeof(int)));
    in.read(reinterpret_cast<char*>(ci.data()), static_cast<std::streamsize>(ci.size() * sizeof(int)));
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!in) throw std::runtime_error("truncated matrix: " + path);
    return argcsr::CSRMatrix(rows, cols, std::move(values), std::move(ci), std::move(rp));
}

void compare(const std::vector<double>& expected, const std::vector<double>& actual,
             const std::string& label) {
    if (expected.size() != actual.size()) throw std::runtime_error(label + ": length mismatch");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double tolerance = 1e-10 + 1e-9 * std::abs(expected[i]);
        if (!std::isfinite(actual[i]) || std::abs(expected[i] - actual[i]) > tolerance)
            throw std::runtime_error(label + ": mismatch at row " + std::to_string(i));
    }
}
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cout << "CUDA Streams test skipped: no CUDA device available\n";
        (void)cudaGetLastError();
        return 0;
    }
    try {
        for (const std::string name : {"U1", "U2", "U3", "U4", "I1", "I2", "I3", "I4"}) {
            const auto csr = load(name);
            std::vector<double> x(static_cast<std::size_t>(csr.cols));
            for (int c = 0; c < csr.cols; ++c) x[static_cast<std::size_t>(c)] = (c % 100 + 1) / 100.0;
            std::vector<double> cpu;
            argcsr::spmvCSRSerial(csr, x, cpu);
            const auto baseline = argcsr::runArgCsrCudaPipeline(csr, x, 2, 4, false);
            compare(cpu, baseline.output, name + " CPU/6C");
            for (int repeat = 0; repeat < 2; ++repeat) {
                const auto streams = argcsr::runArgCsrCudaStreamsPipeline(csr, x, 2, 4);
                compare(cpu, streams.output, name + " CPU/Streams");
                compare(baseline.output, streams.output, name + " 6C/Streams");
            }
            std::cout << name << ": CUDA Streams matches CPU and CUDA baseline\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "CUDA Streams test failed: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
