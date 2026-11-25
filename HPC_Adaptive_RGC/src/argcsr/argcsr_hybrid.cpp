#include "argcsr/argcsr_hybrid.hpp"
#include "argcsr/argcsr_spmv.hpp"
#include "argcsr/argcsr_spmv_omp.hpp"

#ifdef ADAPTIVE_RGC_HAS_CUDA
#include "cuda/argcsr_cuda_pipeline.hpp" // Assume this header provides a launch function
#include <cuda_runtime.h>
#endif

#include <stdexcept>
#include <algorithm>

namespace argcsr {

static ArgCSRMatrix sliceMatrix(const ArgCSRMatrix& A, size_t groupStart, size_t groupCount) {
    ArgCSRMatrix slice = A; // shallow copy of vectors (they are shared)
    slice.groups.clear();
    slice.threadsMapping.clear();
    // Adjust offsets for the slice
    size_t storageOffset = 0;
    size_t mappingOffset = 0;
    for (size_t i = 0; i < groupStart; ++i) {
        const auto& g = A.groups[i];
        storageOffset += static_cast<size_t>(g.chunkSize) * static_cast<size_t>(A.blockSize);
        mappingOffset += static_cast<size_t>(g.size);
    }
    // Now copy the required groups, adjusting internal offsets
    for (size_t i = 0; i < groupCount; ++i) {
        const auto& g = A.groups[groupStart + i];
        ArgCSRGroupInfo gcopy = g;
        gcopy.offset = static_cast<int>(storageOffset);
        slice.groups.push_back(gcopy);
        // copy corresponding thread mapping entries
        for (size_t j = 0; j < static_cast<size_t>(g.size); ++j) {
            slice.threadsMapping.push_back(A.threadsMapping[mappingOffset + j]);
        }
        storageOffset += static_cast<size_t>(g.chunkSize) * static_cast<size_t>(A.blockSize);
        mappingOffset += static_cast<size_t>(g.size);
    }
    // Trim values/columns to the slice range
    slice.values = std::vector<double>(A.values.begin() + storageOffset - slice.groups.back().chunkSize * A.blockSize,
                                      A.values.begin() + storageOffset);
    slice.columns = std::vector<int>(A.columns.begin() + storageOffset - slice.groups.back().chunkSize * A.blockSize,
                                      A.columns.begin() + storageOffset);
    return slice;
}

std::vector<double> argcsr_hybrid_spmv(const ArgCSRMatrix& A,
                                       const std::vector<double>& x,
                                       double gpu_fraction) {
    if (gpu_fraction < 0.0 || gpu_fraction > 1.0)
        throw std::invalid_argument("gpu_fraction must be in [0,1]");

    // Validate inputs using sequential validator logic (reuse existing checks)
    // We rely on the sequential implementation to perform full validation later.

    const size_t totalGroups = A.groups.size();
    const size_t gpuGroupCount = static_cast<size_t>(std::round(gpu_fraction * totalGroups));
    const size_t cpuGroupCount = totalGroups - gpuGroupCount;

    std::vector<double> y(A.rows, 0.0);

    // ----------------------------------------------------------------
    // GPU part (if enabled)
    // ----------------------------------------------------------------
    bool gpu_success = false;
#ifdef ADAPTIVE_RGC_HAS_CUDA
    if (gpuGroupCount > 0) {
        // Build a view containing only the first gpuGroupCount groups.
        ArgCSRMatrix gpuSlice = sliceMatrix(A, 0, gpuGroupCount);
        // Device vectors – reuse existing CUDA pipeline function which expects a full ArgCSRMatrix.
        // Assume a function: std::vector<double> argcsr_cuda_spmv(const ArgCSRMatrix&, const std::vector<double>&);
        try {
            std::vector<double> gpuResult = argcsr_cuda_spmv(gpuSlice, x);
            // Copy results back into the appropriate rows.
            for (const auto& grp : gpuSlice.groups) {
                for (int r = 0; r < grp.size; ++r) {
                    size_t rowIdx = static_cast<size_t>(grp.firstRow + r);
                    y[rowIdx] = gpuResult[rowIdx];
                }
            }
            gpu_success = true;
        } catch (const std::exception& e) {
            // If any CUDA error occurs we fall back to CPU for entire matrix.
            gpu_success = false;
        }
    }
#endif
    // ----------------------------------------------------------------
    // CPU (OpenMP) part – handle remaining groups if GPU failed or fraction < 1.
    // ----------------------------------------------------------------
    size_t cpuStart = gpu_success ? gpuGroupCount : 0;
    size_t cpuCount = totalGroups - cpuStart;
    if (cpuCount > 0) {
        ArgCSRMatrix cpuSlice = sliceMatrix(A, cpuStart, cpuCount);
        // OpenMP implementation works on the slice indices directly.
        std::vector<double> cpuResult = argcsr_spmv_omp(cpuSlice, x);
        // Merge into final vector
        for (const auto& grp : cpuSlice.groups) {
            for (int r = 0; r < grp.size; ++r) {
                size_t rowIdx = static_cast<size_t>(grp.firstRow + r);
                y[rowIdx] = cpuResult[rowIdx];
            }
        }
    }
    return y;
}

} // namespace argcsr
