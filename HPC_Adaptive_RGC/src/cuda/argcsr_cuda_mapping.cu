#include "argcsr_cuda_pipeline_internal.cuh"

#include <cuda_runtime.h>

namespace argcsr {
namespace cuda_detail {
namespace {
constexpr int kThreads = 256;

__device__ int requiredLanes(int length, int lanes) {
    if (length == 0) return 0;
    return static_cast<int>((static_cast<long long>(length) + lanes - 1) / lanes);
}

// Independent groups run concurrently. Within a group, one thread preserves
// the CPU algorithm's iterative decision order and first-row tie break.
__global__ void mappingKernel(const int* rowLengths, const ArgCSRGroupInfo* groups,
                              int groupCount, int blockSize, int* rowMapping) {
    const int groupIndex = static_cast<int>(blockIdx.x); // Groups are independent, each group run by a single block.
    
    if (groupIndex >= groupCount || threadIdx.x != 0) return;
    const ArgCSRGroupInfo group = groups[groupIndex]; // Gets current group information
    
    for (int i = 0; i < group.size; ++i) rowMapping[group.firstRow + i] = 1; // Initializes row mapping to 1 for each row in the group
    int used = group.size;
    while (used < blockSize) {
        int best = 0; // Thinks that first row 0 is best, compares with row 1 in iteration, if row 1 is better, then best = 1.
        // Finds the row that requires the most lanes to process its non-zero elements
        for (int i = 1; i < group.size; ++i) {
            const int candidate = requiredLanes(rowLengths[group.firstRow + i],
                                                rowMapping[group.firstRow + i]);
            const int current = requiredLanes(rowLengths[group.firstRow + best],
                                              rowMapping[group.firstRow + best]);
            if (candidate > current) best = i;
        }

        const int before = requiredLanes(rowLengths[group.firstRow + best],
                                         rowMapping[group.firstRow + best]);
        if (before <= 1) break;
        ++rowMapping[group.firstRow + best];
        const int after = requiredLanes(rowLengths[group.firstRow + best],
                                        rowMapping[group.firstRow + best]);
        if (after == before) {
            --rowMapping[group.firstRow + best];
            break;
        }
        ++used;
    }
}

// Each group is independent. Threads reduce the per-row required chunk sizes.
__global__ void chunkSizingKernel(const int* rowLengths, const int* rowMapping,
                                  ArgCSRGroupInfo* groups, int groupCount) {
    const int groupIndex = static_cast<int>(blockIdx.x);
    if (groupIndex >= groupCount) return;
    const ArgCSRGroupInfo group = groups[groupIndex]; // Get the group metadata
    __shared__ int maxima[kThreads]; // All threads in same block can access it
    int localMax = 1;
    
    // Each thread computes the maximum required lanes for its assigned rows in the group
    for (int i = static_cast<int>(threadIdx.x); i < group.size; i += blockDim.x) {
        const int row = group.firstRow + i;
        const int needed = requiredLanes(rowLengths[row], rowMapping[row]);
        if (needed > localMax) localMax = needed;
    }
    maxima[threadIdx.x] = localMax;
    __syncthreads(); // Wait
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && maxima[threadIdx.x + stride] > maxima[threadIdx.x])
            maxima[threadIdx.x] = maxima[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) groups[groupIndex].chunkSize = maxima[0]; // Store the maximum required lanes for the group in the chunkSize field of the group metadata
}

// Each block preserves the ordered exclusive prefix within its group.
__global__ void mappingPrefixKernel(const ArgCSRGroupInfo* groups, int groupCount,
                                    const int* rowMapping, int* rowPrefix) {
    const int groupIndex = static_cast<int>(blockIdx.x);
    if (groupIndex >= groupCount || threadIdx.x != 0) return;
    const ArgCSRGroupInfo group = groups[groupIndex];
    int prefix = 0;
    for (int i = 0; i < group.size; ++i) {
        const int row = group.firstRow + i;
        rowPrefix[row] = prefix;
        prefix += rowMapping[row];
    }
}

__global__ void rowGroupKernel(const ArgCSRGroupInfo* groups, int groupCount,
                               int rows, int* rowGroup) {
    const std::size_t rowIndex = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (rowIndex >= static_cast<std::size_t>(rows)) return;
    
    const int row = static_cast<int>(rowIndex);
    int low = 0;
    int high = groupCount;
    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (groups[middle].firstRow <= row) low = middle + 1;
        else high = middle;
    }
    rowGroup[row] = low - 1;
}

} // namespace

void launchLaneMapping(const int* rowLengths, ArgCSRGroupInfo* groups, int groupCount,
                       int blockSize, int* rowMapping) {
    mappingKernel<<<static_cast<unsigned int>(groupCount), 1>>>(
        rowLengths, groups, groupCount, blockSize, rowMapping);
}

void launchChunkSizing(const int* rowLengths, const int* rowMapping,
                       ArgCSRGroupInfo* groups, int groupCount) {
    chunkSizingKernel<<<static_cast<unsigned int>(groupCount), kThreads>>>(
        rowLengths, rowMapping, groups, groupCount);
}

void launchMappingPrefixes(const ArgCSRGroupInfo* groups, int groupCount,
                           const int* rowMapping, int* rowPrefix) {
    mappingPrefixKernel<<<static_cast<unsigned int>(groupCount), 1>>>(
        groups, groupCount, rowMapping, rowPrefix);
}

void launchRowGroups(const ArgCSRGroupInfo* groups, int groupCount, int rows, int* rowGroup) {
    const unsigned int blocks = static_cast<unsigned int>(
        (static_cast<std::size_t>(rows) + kThreads - 1) / kThreads);
    rowGroupKernel<<<blocks, kThreads>>>(groups, groupCount, rows, rowGroup);
}

} // namespace cuda_detail
} // namespace argcsr
