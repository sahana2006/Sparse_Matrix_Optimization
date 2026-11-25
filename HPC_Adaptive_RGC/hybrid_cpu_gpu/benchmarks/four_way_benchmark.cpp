#include "hybrid_cpu_gpu/conversion_openmp.hpp"
#include "hybrid_cpu_gpu/metrics.hpp"
#include "hybrid_cpu_gpu/pipeline.hpp"

#include "argcsr/argcsr_spmv.hpp"
#include "argcsr/argcsr_types.hpp"
#include "cuda/argcsr_cuda_pipeline.hpp"
#include "csr/csr_spmv.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kDesired = 2;
constexpr int kBlock = 4;
constexpr int kConversionRuns = 3;
constexpr int kSpmvRuns = 5;
using Clock = std::chrono::steady_clock;

double ms(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const auto m = v.size() / 2;
    return v.size() % 2 ? v[m] : (v[m - 1] + v[m]) / 2.0;
}

argcsr::CSRMatrix loadBinary(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path.string());
    std::int32_t rows = 0, cols = 0, nnz = 0;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    in.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));
    std::vector<int> rp(static_cast<std::size_t>(rows) + 1);
    std::vector<int> ci(static_cast<std::size_t>(nnz));
    std::vector<double> vals(static_cast<std::size_t>(nnz));
    in.read(reinterpret_cast<char*>(rp.data()), static_cast<std::streamsize>(rp.size() * sizeof(int)));
    in.read(reinterpret_cast<char*>(ci.data()), static_cast<std::streamsize>(ci.size() * sizeof(int)));
    in.read(reinterpret_cast<char*>(vals.data()), static_cast<std::streamsize>(vals.size() * sizeof(double)));
    if (!in) throw std::runtime_error("truncated " + path.string());
    return argcsr::CSRMatrix(rows, cols, std::move(vals), std::move(ci), std::move(rp));
}

std::vector<double> makeX(int cols) {
    std::vector<double> x(static_cast<std::size_t>(cols));
    for (int c = 0; c < cols; ++c) x[static_cast<std::size_t>(c)] = (c % 100 + 1) / 100.0;
    return x;
}

struct Row {
    std::string matrix;
    int ompThreads;
    double sequentialMs;
    double openmpConvertMs;
    double cudaPipelineMs;
    double hybridTotalMs;
    double hybridKernelMs;
    double hybridH2dGbps;
    double hybridD2hGbps;
    double speedupHybridVsSeq;
    double speedupCudaVsSeq;
    double speedupHybridVsCuda;
};

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path matrixDir = "benchmarks/matrices";
    int ompThreads = 8;
    if (argc > 1) matrixDir = argv[1];
    if (argc > 2) ompThreads = std::stoi(argv[2]);

    std::vector<std::string> names;
    for (const auto& entry : std::filesystem::directory_iterator(matrixDir)) {
        if (entry.path().extension() == ".csrbin") {
            names.push_back(entry.path().stem().string());
        }
    }
    std::sort(names.begin(), names.end());
    if (names.empty()) {
        std::cerr << "No .csrbin in " << matrixDir
                  << " — run: cmake --build build --target generate_matrices && ./build/generate_matrices\n";
        return 1;
    }

    if (!hybrid_cpu_gpu::cudaDeviceAvailable()) {
        std::cerr << "CUDA device required for four-way benchmark.\n";
        return 1;
    }

    const std::filesystem::path outCsv =
        "hybrid_cpu_gpu/benchmarks/results/four_way_comparison.csv";
    std::filesystem::create_directories(outCsv.parent_path());
    std::ofstream csv(outCsv);
    csv << "matrix,omp_threads,sequential_total_ms,openmp_convert_ms,cuda_pipeline_ms,"
           "hybrid_total_ms,hybrid_kernel_ms,hybrid_h2d_gbps,hybrid_d2h_gbps,"
           "speedup_hybrid_vs_seq,speedup_cuda_vs_seq,speedup_hybrid_vs_cuda\n";

    try {
        for (const std::string& name : names) {
            const auto csr = loadBinary(matrixDir / (name + ".csrbin"));
            const auto x = makeX(csr.cols);

            std::vector<double> seqConv;
            argcsr::ArgCSRMatrix rep{};
            for (int i = 0; i < kConversionRuns; ++i) {
                const auto t0 = Clock::now();
                rep = argcsr::convertToArgCSR(csr, kDesired, kBlock);
                seqConv.push_back(ms(Clock::now() - t0));
            }
            std::vector<double> seqSpmv;
            for (int i = 0; i < kSpmvRuns; ++i) {
                const auto t0 = Clock::now();
                (void)argcsr::argcsr_spmv(rep, x);
                seqSpmv.push_back(ms(Clock::now() - t0));
            }
            const double sequentialMs = median(seqConv) + median(seqSpmv);

            std::vector<double> ompConv;
            for (int i = 0; i < kConversionRuns; ++i) {
                const auto t0 = Clock::now();
                (void)hybrid_cpu_gpu::convertCsrToArgCsrOpenMP(csr, kDesired, kBlock, ompThreads);
                ompConv.push_back(ms(Clock::now() - t0));
            }
            const double openmpMs = median(ompConv);

            std::vector<double> cudaSamples;
            for (int i = 0; i < kSpmvRuns; ++i) {
                const auto t0 = Clock::now();
                (void)argcsr::runArgCsrCudaPipeline(csr, x, kDesired, kBlock, false);
                cudaSamples.push_back(ms(Clock::now() - t0));
            }
            const double cudaMs = median(cudaSamples);

            hybrid_cpu_gpu::PipelineConfig cfg{ompThreads, 256, false};
            std::vector<double> hybridSamples;
            hybrid_cpu_gpu::PipelineTimingsMs last{};
            for (int i = 0; i < kSpmvRuns; ++i) {
                const auto t0 = Clock::now();
                const auto h = hybrid_cpu_gpu::runHybridPipeline(csr, x, kDesired, kBlock, cfg, false);
                hybridSamples.push_back(ms(Clock::now() - t0));
                last = h.timings;
            }
            const double hybridMs = median(hybridSamples);
            const auto speed = hybrid_cpu_gpu::computeSpeedups(sequentialMs, hybridMs, cudaMs);

            Row r{name,
                  ompThreads,
                  sequentialMs,
                  openmpMs,
                  cudaMs,
                  hybridMs,
                  last.kernelMs,
                  last.transfer.h2dBandwidthGbps,
                  last.transfer.d2hBandwidthGbps,
                  speed.hybridVsSequential,
                  speed.cudaVsSequential,
                  speed.hybridVsCuda};
            csv << r.matrix << ',' << r.ompThreads << ',' << std::setprecision(10) << r.sequentialMs
                << ',' << r.openmpConvertMs << ',' << r.cudaPipelineMs << ',' << r.hybridTotalMs << ','
                << r.hybridKernelMs << ',' << r.hybridH2dGbps << ',' << r.hybridD2hGbps << ','
                << r.speedupHybridVsSeq << ',' << r.speedupCudaVsSeq << ',' << r.speedupHybridVsCuda
                << '\n';
            std::cout << name << " hybrid_kernel_ms=" << r.hybridKernelMs
                      << " speedup_vs_seq=" << r.speedupHybridVsSeq << '\n';
        }
        std::cout << "Wrote " << outCsv << '\n';
    } catch (const std::exception& e) {
        std::cerr << "four_way_benchmark failed: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
