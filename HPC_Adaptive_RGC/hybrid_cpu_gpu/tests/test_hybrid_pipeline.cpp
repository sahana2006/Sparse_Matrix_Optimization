#include "hybrid_cpu_gpu/pipeline.hpp"

#include "argcsr/argcsr_spmv.hpp"
#include "argcsr/argcsr_types.hpp"
#include "cuda/argcsr_cuda_pipeline.hpp"
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

void assertClose(const std::vector<double>& ref, const std::vector<double>& got, const std::string& tag) {
    require(ref.size() == got.size(), tag + ": size");
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double tol = 1e-10 + 1e-9 * std::abs(ref[i]);
        require(std::isfinite(got[i]) && std::abs(ref[i] - got[i]) <= tol,
                tag + ": row " + std::to_string(i));
    }
}

}  // namespace

int main() {
    if (!hybrid_cpu_gpu::cudaDeviceAvailable()) {
        std::cout << "SKIP: no CUDA device (OK on CPU-only machines)\n";
        return 0;
    }

    try {
        const argcsr::CSRMatrix m(
            4, 4, {10, 2, 3, 4, 5, 7, 8},
            {0, 3, 1, 0, 2, 2, 3}, {0, 2, 3, 5, 7});
        const std::vector<double> x{1, 2, 3, 4};
        const auto ref = argcsr::argcsr_spmv(argcsr::convertToArgCSR(m, 2, 4), x);

        hybrid_cpu_gpu::PipelineConfig cfg{4, 256, false};
        const auto hybrid = hybrid_cpu_gpu::runHybridPipeline(m, x, 2, 4, cfg, false);
        assertClose(ref, hybrid.output, "hybrid vs sequential SpMV");

        const auto cuda = argcsr::runArgCsrCudaPipeline(m, x, 2, 4, false);
        assertClose(cuda.output, hybrid.output, "hybrid vs cuda pipeline");

        std::cout << "Hybrid pipeline GPU test passed.\n";
    } catch (const std::exception& e) {
        std::cerr << "test_hybrid_pipeline failed: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
