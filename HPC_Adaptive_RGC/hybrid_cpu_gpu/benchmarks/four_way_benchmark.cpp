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
constexpr int kWarmupRuns = 2;
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
    double seqConvertMs;
    double seqSpmvMs;
    double openmpConvertMs;
    double cudaPipelineMs;
    double hybridTotalMs;
    double hybridKernelMs;
    double hybridH2dGbps;
    double hybridD2hGbps;
    double speedupHybridVsSeq;
    double speedupCudaVsSeq;
    double speedupHybridVsCuda;
    double speedupOmpConvertVsSeqConvert;
    double speedupHybridKernelVsSeqSpmv;
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

    const std::filesystem::path outDir = "hybrid_cpu_gpu/benchmarks/results";
    std::filesystem::create_directories(outDir);
    const std::filesystem::path outCsv = outDir / "four_way_comparison.csv";
    const std::filesystem::path diagCsv = outDir / "end_to_end_diagnosis.csv";
    const std::filesystem::path scaleCsv = outDir / "omp_thread_scaling.csv";
    std::ofstream csv(outCsv);
    std::ofstream diag(diagCsv);
    csv << "matrix,omp_threads,sequential_total_ms,seq_convert_ms,seq_spmv_ms,openmp_convert_ms,"
           "cuda_pipeline_ms,hybrid_total_ms,hybrid_kernel_ms,hybrid_h2d_gbps,hybrid_d2h_gbps,"
           "speedup_hybrid_vs_seq,speedup_cuda_vs_seq,speedup_hybrid_vs_cuda,"
           "speedup_omp_convert_vs_seq_convert,speedup_hybrid_kernel_vs_seq_spmv\n";
    diag << "matrix,rows,nnz,storage_elements,pad_ratio,"
            "seq_convert_ms,seq_spmv_ms,sequential_total_ms,"
            "omp_convert_ms,grouping_ms,mapping_chunk_wall_ms,mapping_sum_ms,"
            "offset_assign_ms,scatter_ms,metadata_ms,convert_accounted_ms,convert_hidden_ms,"
            "assign_gbs,scatter_useful_gbs,h2d_bytes,h2d_ms,h2d_gbs,d2h_ms,kernel_ms,"
            "device_alloc_ms,hybrid_inner_total_ms,hybrid_outer_median_ms,"
            "hybrid_accounted_ms,hybrid_unexplained_ms,speedup_hybrid_vs_seq\n";

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
            const double seqConvertMs = median(seqConv);
            const double seqSpmvMs = median(seqSpmv);
            const double sequentialMs = seqConvertMs + seqSpmvMs;

            std::vector<double> ompConv;
            hybrid_cpu_gpu::OpenMPConversionResult lastOmp{};
            for (int i = 0; i < kConversionRuns; ++i) {
                const auto t0 = Clock::now();
                lastOmp = hybrid_cpu_gpu::convertCsrToArgCsrOpenMP(csr, kDesired, kBlock, ompThreads);
                ompConv.push_back(ms(Clock::now() - t0));
            }
            const double openmpMs = median(ompConv);

            for (int w = 0; w < kWarmupRuns; ++w) {
                (void)argcsr::runArgCsrCudaPipeline(csr, x, kDesired, kBlock, false);
            }
            std::vector<double> cudaSamples;
            for (int i = 0; i < kSpmvRuns; ++i) {
                const auto t0 = Clock::now();
                (void)argcsr::runArgCsrCudaPipeline(csr, x, kDesired, kBlock, false);
                cudaSamples.push_back(ms(Clock::now() - t0));
            }
            const double cudaMs = median(cudaSamples);

            hybrid_cpu_gpu::PipelineConfig cfg{ompThreads, 256, false};
            hybrid_cpu_gpu::PipelineDeviceCache deviceCache;
            for (int w = 0; w < kWarmupRuns; ++w) {
                (void)hybrid_cpu_gpu::runHybridPipeline(csr, x, kDesired, kBlock, cfg, false, &deviceCache);
            }
            std::vector<double> hybridSamples;
            hybrid_cpu_gpu::PipelineTimingsMs last{};
            for (int i = 0; i < kSpmvRuns; ++i) {
                const auto t0 = Clock::now();
                const auto h = hybrid_cpu_gpu::runHybridPipeline(csr, x, kDesired, kBlock, cfg, false,
                                                                 &deviceCache);
                hybridSamples.push_back(ms(Clock::now() - t0));
                last = h.timings;
            }
            const double hybridMs = median(hybridSamples);
            const auto speed = hybrid_cpu_gpu::computeSpeedups(sequentialMs, hybridMs, cudaMs);

            const double speedOmpConv =
                openmpMs > 0.0 ? seqConvertMs / openmpMs : 0.0;
            const double speedKernel =
                last.kernelMs > 0.0 ? seqSpmvMs / last.kernelMs : 0.0;

            Row r{name,
                  ompThreads,
                  sequentialMs,
                  seqConvertMs,
                  seqSpmvMs,
                  openmpMs,
                  cudaMs,
                  hybridMs,
                  last.kernelMs,
                  last.transfer.h2dBandwidthGbps,
                  last.transfer.d2hBandwidthGbps,
                  speed.hybridVsSequential,
                  speed.cudaVsSequential,
                  speed.hybridVsCuda,
                  speedOmpConv,
                  speedKernel};
            csv << r.matrix << ',' << r.ompThreads << ',' << std::setprecision(10) << r.sequentialMs
                << ',' << r.seqConvertMs << ',' << r.seqSpmvMs << ',' << r.openmpConvertMs << ','
                << r.cudaPipelineMs << ',' << r.hybridTotalMs << ',' << r.hybridKernelMs << ','
                << r.hybridH2dGbps << ',' << r.hybridD2hGbps << ',' << r.speedupHybridVsSeq << ','
                << r.speedupCudaVsSeq << ',' << r.speedupHybridVsCuda << ',' << r.speedupOmpConvertVsSeqConvert
                << ',' << r.speedupHybridKernelVsSeqSpmv << '\n';
            const auto& conv = last.ompConversion.storageElements ? last.ompConversion : lastOmp.timings;
            const double grouping = conv.groupingMs;
            const double mapWall = conv.mappingChunkWallMs;
            const double assignMs = conv.offsetAssignMs;
            const double scatterMs = conv.storageMs;
            const double metaMs = conv.metadataMs;
            const double convertAccounted = grouping + mapWall + assignMs + scatterMs + metaMs;
            const double convertHidden = conv.totalMs - convertAccounted;
            const std::size_t storage = conv.storageElements;
            const std::size_t nnz = conv.nnz ? conv.nnz : static_cast<std::size_t>(csr.nnz);
            const double pad = nnz ? static_cast<double>(storage) / static_cast<double>(nnz) : 0.0;
            const std::size_t assignBytes = storage * (sizeof(double) + sizeof(int));
            const std::size_t scatterBytes = nnz * (sizeof(double) + sizeof(int));
            const double assignGbs = hybrid_cpu_gpu::gigaBytesPerSecond(assignBytes, assignMs);
            const double scatterGbs = hybrid_cpu_gpu::gigaBytesPerSecond(scatterBytes, scatterMs);
            const double hybridAccounted = last.ompConversion.totalMs + last.deviceAllocMs +
                                           last.transfer.h2dMs + last.kernelMs + last.transfer.d2hMs;
            const double hybridUnexplained = last.totalMs - hybridAccounted;

            diag << name << ',' << csr.rows << ',' << nnz << ',' << storage << ','
                 << std::setprecision(10) << pad << ',' << seqConvertMs << ',' << seqSpmvMs << ','
                 << sequentialMs << ',' << openmpMs << ',' << grouping << ',' << mapWall << ','
                 << conv.mappingMs << ',' << assignMs << ',' << scatterMs << ',' << metaMs << ','
                 << convertAccounted << ',' << convertHidden << ',' << assignGbs << ',' << scatterGbs
                 << ',' << last.transfer.h2dBytes << ',' << last.transfer.h2dMs << ','
                 << last.transfer.h2dBandwidthGbps << ',' << last.transfer.d2hMs << ',' << last.kernelMs
                 << ',' << last.deviceAllocMs << ',' << last.totalMs << ',' << hybridMs << ','
                 << hybridAccounted << ',' << hybridUnexplained << ',' << speed.hybridVsSequential
                 << '\n';

            std::cout << name << " hybrid_kernel_ms=" << r.hybridKernelMs
                      << " speedup_vs_seq=" << r.speedupHybridVsSeq
                      << " convert_hidden_ms=" << convertHidden
                      << " hybrid_unexplained_ms=" << hybridUnexplained
                      << " pad_ratio=" << pad
                      << " assign_gbs=" << assignGbs << '\n';
        }
        std::cout << "Wrote " << outCsv << '\n';
        std::cout << "Wrote " << diagCsv << '\n';

        std::ofstream scale(scaleCsv);
        scale << "matrix,omp_threads,convert_ms,speedup_vs_1thread\n";
        std::cout << "\nOpenMP conversion thread sweep (memory-bound test)\n";
        const int sweepThreads[] = {1, 2, 4, 8};
        for (const std::string& focus : {"U3", "U4", "I3", "I4"}) {
            const auto path = matrixDir / (focus + ".csrbin");
            if (!std::filesystem::exists(path)) continue;
            const auto csr = loadBinary(path);
            double t1 = 0.0;
            for (int th : sweepThreads) {
                std::vector<double> samples;
                for (int i = 0; i < kConversionRuns; ++i) {
                    const auto t0 = Clock::now();
                    (void)hybrid_cpu_gpu::convertCsrToArgCsrOpenMP(csr, kDesired, kBlock, th);
                    samples.push_back(ms(Clock::now() - t0));
                }
                const double med = median(samples);
                if (th == 1) t1 = med;
                const double vs1 = (t1 > 0.0) ? t1 / med : 0.0;
                scale << focus << ',' << th << ',' << std::setprecision(10) << med << ',' << vs1 << '\n';
                std::cout << focus << " threads=" << th << " convert_ms=" << med
                          << " vs_1thread=" << vs1 << '\n';
            }
        }
        std::cout << "Wrote " << scaleCsv << '\n';

        const std::filesystem::path blockCsv = outDir / "cuda_block_sweep.csv";
        std::ofstream blockOut(blockCsv);
        blockOut << "matrix,cuda_block_size,kernel_ms_median\n";
        std::cout << "\nCUDA block-size sweep (kernel-only, async stream)\n";
        const int blockCandidates[] = {128, 256, 512, 1024};
        for (const std::string& focus : {"U3", "U4"}) {
            const auto path = matrixDir / (focus + ".csrbin");
            if (!std::filesystem::exists(path)) continue;
            const auto csr = loadBinary(path);
            const auto x = makeX(csr.cols);
            const auto converted =
                hybrid_cpu_gpu::convertCsrToArgCsrOpenMP(csr, kDesired, kBlock, ompThreads);
            for (int cudaBlock : blockCandidates) {
                if (cudaBlock > 1024) continue;
                std::vector<double> samples;
                std::vector<double> scratch;
                for (int w = 0; w < kWarmupRuns; ++w) {
                    double km = 0.0;
                    (void)hybrid_cpu_gpu::benchmarkHybridKernelOnly(converted, x, cudaBlock, scratch, km,
                                                                    nullptr);
                }
                for (int i = 0; i < kSpmvRuns; ++i) {
                    double km = 0.0;
                    (void)hybrid_cpu_gpu::benchmarkHybridKernelOnly(converted, x, cudaBlock, scratch, km,
                                                                    nullptr);
                    samples.push_back(km);
                }
                const double med = median(samples);
                blockOut << focus << ',' << cudaBlock << ',' << std::setprecision(10) << med << '\n';
                std::cout << focus << " cuda_block=" << cudaBlock << " kernel_ms=" << med << '\n';
            }
        }
        std::cout << "Wrote " << blockCsv << " (justify cudaBlockSize — see docs/HYBRID_KERNEL_CONFIG.md)\n";
    } catch (const std::exception& e) {
        std::cerr << "four_way_benchmark failed: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
