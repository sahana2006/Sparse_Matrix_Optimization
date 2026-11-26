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
std::cout << "Wrote " << blockCsv << '\n';
