#include "argcsr_cuda_pipeline_internal.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace argcsr {
namespace cuda_detail {
namespace {
constexpr int kThreads = 256;

__device__ int findRowForNonzero(const int* rowPointers, int rows, int nonzero) {
    int low = 0;
    int high = rows;
    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (rowPointers[middle + 1] <= nonzero) low = middle + 1;
        else high = middle;
    }
    return low;
}

// One thread owns one CSR nonzero and its unique Adaptive RGC destination.
__global__ void storageScatterKernel(const int* rowPointers, const double* csrValues,
                                     const int* csrColumns, int rows, int nnz,
                                     const ArgCSRGroupInfo* groups, const int* rowGroup,
                                     const int* rowMapping, const int* rowPrefix,
                                     int blockSize, double* values, int* columns,
                                     int* storageError) {

    const std::size_t indexValue = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (indexValue >= static_cast<std::size_t>(nnz)) return;
    const int index = static_cast<int>(indexValue);
    const int row = findRowForNonzero(rowPointers, rows, index);
    const int groupIndex = rowGroup[row];
    const ArgCSRGroupInfo group = groups[groupIndex];
    const int local = index - rowPointers[row]; // The local index of the nonzero within its row
    const int chunk = local / group.chunkSize; // The chunk index of the nonzero within its row
    if (chunk >= rowMapping[row]) {
        atomicExch(storageError, 1);
        return;
    }
    const int withinChunk = local % group.chunkSize;
    // Compute the destination index in the Adaptive RGC storage for this nonzero.
    const std::size_t destination = static_cast<std::size_t>(group.offset) +
        static_cast<std::size_t>(withinChunk) * static_cast<std::size_t>(blockSize) +
        static_cast<std::size_t>(rowPrefix[row] + chunk);
    values[destination] = csrValues[index];
    columns[destination] = csrColumns[index];
}
} // namespace

void launchStorageScatter(const int* rowPointers, const double* csrValues,
                          const int* csrColumns, int rows, int nnz,
                          const ArgCSRGroupInfo* groups, const int* rowGroup,
                          const int* rowMapping, const int* rowPrefix, int blockSize,
                          double* values, int* columns, int* storageError) {
    const unsigned int blocks = static_cast<unsigned int>(
        (static_cast<std::size_t>(nnz) + kThreads - 1) / kThreads);
    storageScatterKernel<<<blocks, kThreads>>>(rowPointers, csrValues, csrColumns, rows, nnz,
        groups, rowGroup, rowMapping, rowPrefix, blockSize, values, columns, storageError);
}

} // namespace cuda_detail
} // namespace argcsr
