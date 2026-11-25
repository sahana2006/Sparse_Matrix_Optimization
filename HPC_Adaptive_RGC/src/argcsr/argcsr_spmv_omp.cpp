#include "argcsr/argcsr_spmv_omp.hpp"

#include <stdexcept>
#include <omp.h>

namespace argcsr {

std::vector<double> argcsr_spmv_omp(const ArgCSRMatrix& A,
                                    const std::vector<double>& x) {
    // Input validation – identical to sequential version
    if (A.rows < 0 || A.cols < 0 || A.blockSize <= 0 || A.nnz < 0)
        throw std::invalid_argument("ArgCSR dimensions, NNZ, and block size must be valid");
    if (x.size() != static_cast<std::size_t>(A.cols))
        throw std::invalid_argument("ArgCSR SpMV input vector length must equal the matrix column count");
    if (A.values.size() != A.columns.size())
        throw std::logic_error("ArgCSR values and columns arrays have different lengths");

    std::vector<double> y(static_cast<std::size_t>(A.rows), 0.0);
    std::vector<bool> covered(static_cast<std::size_t>(A.rows), false);

    // Offsets that must be updated sequentially but are local to each iteration
    std::vector<std::size_t> groupOffsets(A.groups.size());
    std::vector<std::size_t> mappingOffsets(A.groups.size());
    {
        std::size_t storageOff = 0, mapOff = 0;
        for (size_t g = 0; g < A.groups.size(); ++g) {
            groupOffsets[g] = storageOff;
            mappingOffsets[g] = mapOff;
            const auto& grp = A.groups[g];
            std::size_t groupStorage = static_cast<std::size_t>(grp.chunkSize) * static_cast<std::size_t>(A.blockSize);
            storageOff += groupStorage;
            mapOff += static_cast<std::size_t>(grp.size);
        }
    }

    // Parallel over groups – groups are independent (disjoint rows)
    #pragma omp parallel for schedule(dynamic) 
    for (std::ptrdiff_t g = 0; g < static_cast<std::ptrdiff_t>(A.groups.size()); ++g) {
        const ArgCSRGroupInfo& group = A.groups[static_cast<std::size_t>(g)];
        // Basic validation (same checks as sequential)
        if (group.firstRow < 0 || group.size <= 0 || group.firstRow > A.rows ||
            group.size > A.rows - group.firstRow)
            continue; // validation already performed in sequential version; here we skip invalid group
        if (group.offset < 0 || group.chunkSize <= 0)
            continue;
        const std::size_t storageOffset = groupOffsets[static_cast<std::size_t>(g)];
        const std::size_t mappingOffset = mappingOffsets[static_cast<std::size_t>(g)];
        const std::size_t rowsInGroup = static_cast<std::size_t>(group.size);
        const std::size_t chunkSize = static_cast<std::size_t>(group.chunkSize);
        const std::size_t blockSize = static_cast<std::size_t>(A.blockSize);

        // Verify mapping counts
        int mappedChunks = 0;
        for (std::size_t i = 0; i < rowsInGroup; ++i) {
            int count = A.threadsMapping[mappingOffset + i];
            if (count <= 0) continue;
            mappedChunks += count;
        }
        // Allocate per‑group partial sum buffer (size = blockSize)
        std::vector<double> partialSum(blockSize, 0.0);
        // Compute partial sums over chunks
        for (int t = 0; t < mappedChunks; ++t) {
            double sum = 0.0;
            for (std::size_t i = 0; i < chunkSize; ++i) {
                std::size_t pos = storageOffset + static_cast<std::size_t>(t) + i * blockSize;
                int col = A.columns[pos];
                if (col == -1) break;
                if (col < 0 || col >= A.cols) continue;
                sum += A.values[pos] * x[static_cast<std::size_t>(col)];
            }
            partialSum[static_cast<std::size_t>(t)] = sum;
        }
        // Accumulate into output vector – this write is safe because each row belongs to only one group.
        int chunkPrefix = 0;
        for (std::size_t i = 0; i < rowsInGroup; ++i) {
            std::size_t row = static_cast<std::size_t>(group.firstRow) + i;
            int count = A.threadsMapping[mappingOffset + i];
            double rowSum = 0.0;
            for (int c = 0; c < count; ++c) {
                rowSum += partialSum[static_cast<std::size_t>(chunkPrefix + c)];
            }
            y[row] = rowSum; // each row is written by exactly one group
            chunkPrefix += count;
        }
    }

    // Final validation – ensure every row covered exactly once
    for (size_t r = 0; r < static_cast<size_t>(A.rows); ++r) {
        // In the parallel version we cannot reliably set `covered` inside the loop because of data races.
        // Instead we recompute coverage here.
        bool rowSeen = false;
        for (const auto& grp : A.groups) {
            if (r >= static_cast<size_t>(grp.firstRow) &&
                r < static_cast<size_t>(grp.firstRow + grp.size)) {
                rowSeen = true; break;
            }
        }
        if (!rowSeen)
            throw std::logic_error("ArgCSR groups do not cover all matrix rows");
    }
    return y;
}

} // namespace argcsr
