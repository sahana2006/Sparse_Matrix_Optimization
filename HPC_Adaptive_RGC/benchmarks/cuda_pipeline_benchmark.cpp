#include "argcsr/argcsr_spmv.hpp"
#include "argcsr/argcsr_types.hpp"
#include "cuda/argcsr_cuda_pipeline.hpp"
#include "csr/csr_spmv.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

template <class T>
void readExact(std::ifstream& input, std::vector<T>& values, const std::filesystem::path& path) {
    if (values.empty()) return;
    const std::size_t bytes = values.size() * sizeof(T);
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::runtime_error("matrix section too large: " + path.string());
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
    if (!input) throw std::runtime_error("truncated CSR matrix: " + path.string());
}

argcsr::CSRMatrix loadMatrix(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open matrix: " + path.string());
    std::int32_t rows = 0, cols = 0, nnz = 0;
    input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    input.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    input.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));
    if (!input || rows < 0 || cols < 0 || nnz < 0)
        throw std::runtime_error("invalid CSR matrix header: " + path.string());
    std::vector<int> rowPointers(static_cast<std::size_t>(rows) + 1);
    std::vector<int> columns(static_cast<std::size_t>(nnz));
    std::vector<double> values(static_cast<std::size_t>(nnz));
    readExact(input, rowPointers, path);
    readExact(input, columns, path);
    readExact(input, values, path);
    if (input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("trailing data in CSR matrix: " + path.string());
    return argcsr::CSRMatrix(rows, cols, std::move(values), std::move(columns),
                             std::move(rowPointers));
}

double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2.0;
}

template <class Getter>
double medianStage(const std::vector<argcsr::CudaPipelineTimings>& samples, Getter getter) {
    std::vector<double> values;
    values.reserve(samples.size());
    for (const auto& sample : samples) values.push_back(getter(sample));
    return median(std::move(values));
}

std::vector<double> makeVector(int cols) {
    std::vector<double> x(static_cast<std::size_t>(cols));
    for (int col = 0; col < cols; ++col)
        x[static_cast<std::size_t>(col)] = static_cast<double>(col % 100 + 1) / 100.0;
    return x;
}

void compare(const std::vector<double>& reference, const std::vector<double>& actual,
             const std::string& matrixName) {
    if (reference.size() != actual.size()) throw std::runtime_error(matrixName + ": output size mismatch");
    for (std::size_t row = 0; row < reference.size(); ++row) {
        const double tolerance = 1e-10 + 1e-9 * std::abs(reference[row]);
        if (!std::isfinite(actual[row]) || std::abs(reference[row] - actual[row]) > tolerance)
            throw std::runtime_error(matrixName + ": CPU/GPU output mismatch at row " +
                                     std::to_string(row));
    }
}

struct ResultRow {
    std::string matrix;
    int rows;
    int cols;
    int nnz;
    double cpuDecisionGroupingMs;
    double cpuOffsetsMs;
    double cpuPreprocessingMs;
    double gpuRowStatsMs;
    double gpuMappingMs;
    double gpuChunkMs;
    double gpuStorageMs;
    double gpuSpmvMs;
    double gpuKernelTotalMs;
    double h2dMs;
    double d2hMs;
    double transferTotalMs;
    double gpuPipelineTotalMs;
    double cpuReferenceConversionMs;
    double cpuReferenceSpmvMs;
    double cpuReferencePipelineMs;
};

ResultRow runMatrix(const std::string& name, const std::filesystem::path& matrixDir,
                    int runs, int desiredChunkSize, int blockSize) {
    std::filesystem::path path = matrixDir / name;
    if (!path.has_extension()) path += ".csrbin";
    const auto csr = loadMatrix(path);
    const auto x = makeVector(csr.cols);
    std::vector<double> csrReference;
    argcsr::spmvCSRSerial(csr, x, csrReference);

    // Warm CUDA context, allocator and kernels before collecting samples.
    (void)argcsr::runArgCsrCudaPipeline(csr, x, desiredChunkSize, blockSize, false);

    std::vector<argcsr::CudaPipelineTimings> gpuSamples;
    std::vector<double> cpuConversion, cpuSpmv, cpuTotal;
    gpuSamples.reserve(static_cast<std::size_t>(runs));
    cpuConversion.reserve(static_cast<std::size_t>(runs));
    cpuSpmv.reserve(static_cast<std::size_t>(runs));
    cpuTotal.reserve(static_cast<std::size_t>(runs));
    for (int run = 0; run < runs; ++run) {
        const auto gpu = argcsr::runArgCsrCudaPipeline(csr, x, desiredChunkSize, blockSize, false);
        compare(csrReference, gpu.output, name);
        gpuSamples.push_back(gpu.timings);

        const auto cpuStart = Clock::now();
        const auto cpuAdaptive = argcsr::convertToArgCSR(csr, desiredChunkSize, blockSize);
        const auto conversionStop = Clock::now();
        const auto cpuY = argcsr::argcsr_spmv(cpuAdaptive, x);
        const auto cpuStop = Clock::now();
        compare(csrReference, cpuY, name + " CPU Adaptive RGC");
        cpuConversion.push_back(milliseconds(conversionStop - cpuStart));
        cpuSpmv.push_back(milliseconds(cpuStop - conversionStop));
        cpuTotal.push_back(milliseconds(cpuStop - cpuStart));
    }

    const auto cpuConversionMs = median(std::move(cpuConversion));
    const auto cpuSpmvMs = median(std::move(cpuSpmv));
    const auto cpuTotalMs = median(std::move(cpuTotal));
    const double gpuRowStats = medianStage(gpuSamples, [](const auto& t) { return t.gpuRowStatisticsMs; });
    const double gpuMapping = medianStage(gpuSamples, [](const auto& t) { return t.gpuMappingMs; });
    const double gpuChunk = medianStage(gpuSamples, [](const auto& t) { return t.gpuChunkSizingMs; });
    const double gpuStorage = medianStage(gpuSamples, [](const auto& t) { return t.gpuStorageConversionMs; });
    const double gpuSpmv = medianStage(gpuSamples, [](const auto& t) { return t.gpuSpmvMs; });
    const double gpuTotal = medianStage(gpuSamples, [](const auto& t) { return t.totalPipelineMs; });
    const double cpuOffsets = medianStage(gpuSamples, [](const auto& t) { return t.cpuGroupOffsetsMs; });
    const double cpuPreprocessing = medianStage(gpuSamples, [](const auto& t) {
        return t.cpuPreprocessingMs();
    });
    const double h2d = medianStage(gpuSamples, [](const auto& t) {
        return t.h2dCsrMs + t.h2dGroupsMs + t.h2dFinalMetadataMs + t.h2dInputVectorMs;
    });
    const double d2h = medianStage(gpuSamples, [](const auto& t) {
        return t.d2hRowStatisticsMs + t.d2hChunkMetadataMs + t.d2hStorageStatusMs + t.d2hOutputMs;
    });
    const double transfer = medianStage(gpuSamples, [](const auto& t) { return t.transferMs(); });
    const double kernelTotal = medianStage(gpuSamples, [](const auto& t) { return t.gpuKernelMs(); });

    std::cout << name << " (" << csr.rows << " x " << csr.cols << ", nnz=" << csr.nnz << ")\n"
              << "  CPU Adaptive RGC conversion: " << cpuConversionMs << " ms; CPU SpMV: " << cpuSpmvMs
              << " ms; CPU total: " << cpuTotalMs << " ms\n"
              << "  GPU row stats: " << gpuRowStats << " ms; mapping: " << gpuMapping
              << " ms; chunk sizing: " << gpuChunk << " ms; storage: " << gpuStorage
              << " ms; SpMV: " << gpuSpmv << " ms\n"
              << "  GPU kernels: " << kernelTotal << " ms; H2D: " << h2d << " ms; D2H: "
              << d2h << " ms; CPU preprocessing: " << cpuPreprocessing
              << " ms; complete pipeline: " << gpuTotal << " ms\n";
    return {name, csr.rows, csr.cols, csr.nnz,
            medianStage(gpuSamples, [](const auto& t) { return t.cpuGroupingMs; }),
            cpuOffsets, cpuPreprocessing,
            gpuRowStats, gpuMapping, gpuChunk, gpuStorage, gpuSpmv, kernelTotal,
            h2d, d2h, transfer, gpuTotal, cpuConversionMs, cpuSpmvMs, cpuTotalMs};
}

void writeCsv(const std::filesystem::path& path, const std::vector<ResultRow>& rows) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open benchmark CSV: " + path.string());
    output << "matrix,rows,cols,nnz,cpu_grouping_decisions_ms,cpu_group_offsets_ms,cpu_preprocessing_ms,"
              "gpu_row_stats_ms,gpu_mapping_ms,gpu_chunk_size_ms,gpu_storage_ms,gpu_spmv_ms,"
              "gpu_kernel_total_ms,h2d_ms,d2h_ms,transfer_total_ms,gpu_pipeline_total_ms,"
              "cpu_reference_conversion_ms,cpu_reference_spmv_ms,cpu_reference_pipeline_ms\n";
    output << std::setprecision(12);
    for (const auto& r : rows)
        output << r.matrix << ',' << r.rows << ',' << r.cols << ',' << r.nnz << ','
               << r.cpuDecisionGroupingMs << ',' << r.cpuOffsetsMs << ',' << r.cpuPreprocessingMs << ','
               << r.gpuRowStatsMs << ',' << r.gpuMappingMs << ',' << r.gpuChunkMs << ','
               << r.gpuStorageMs << ',' << r.gpuSpmvMs << ',' << r.gpuKernelTotalMs << ','
               << r.h2dMs << ',' << r.d2hMs << ',' << r.transferTotalMs << ','
               << r.gpuPipelineTotalMs << ',' << r.cpuReferenceConversionMs << ','
               << r.cpuReferenceSpmvMs << ',' << r.cpuReferencePipelineMs << '\n';
    if (!output) throw std::runtime_error("failed writing benchmark CSV: " + path.string());
}

void printUsage() {
    std::cout << "Usage: cuda_pipeline_benchmark [matrix-directory] [runs] [desired-chunk-size] "
                 "[block-size] [matrix-name ...]\n"
                 "Defaults: benchmarks/matrices, 3, 2, 4, U1 U2 U3 U4 I1 I2 I3 I4\n";
}
}

int main(int argc, char** argv) {
    try {
        if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
            printUsage();
            return 0;
        }
        const std::filesystem::path matrixDir = argc > 1 ? argv[1] : "benchmarks/matrices";
        const int runs = argc > 2 ? std::stoi(argv[2]) : 3;
        const int desired = argc > 3 ? std::stoi(argv[3]) : 2;
        const int block = argc > 4 ? std::stoi(argv[4]) : 4;
        if (runs <= 0 || desired <= 0 || block <= 0)
            throw std::invalid_argument("runs, desired chunk size, and block size must be positive");

        std::vector<std::string> names;
        if (argc > 5) {
            for (int i = 5; i < argc; ++i) names.emplace_back(argv[i]);
        } else {
            names = {"U1", "U2", "U3", "U4", "I1", "I2", "I3", "I4"};
        }

        std::vector<ResultRow> results;
        results.reserve(names.size());
        for (const auto& name : names)
            results.push_back(runMatrix(name, matrixDir, runs, desired, block));
        const std::filesystem::path resultPath = "benchmarks/results/cuda_pipeline_results.csv";
        std::filesystem::create_directories(resultPath.parent_path());
        writeCsv(resultPath, results);
        std::cout << "Results written to " << resultPath.generic_string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "CUDA pipeline benchmark failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
