#include "argcsr_openmp.hpp"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace argcsr_omp {
namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

int resolveThreadCount(int requested) {
    return requested > 0 ? requested : omp_get_max_threads();
}

inline int requiredChunks(int nnzInRow, int assignedThreads) {
    return nnzInRow == 0 ? 0 : (nnzInRow + assignedThreads - 1) / assignedThreads;
}

void validateCsrParallel(const argcsr::CSRMatrix& csr, int numThreads) {
    if (csr.rows < 0 || csr.cols < 0) {
        throw std::invalid_argument("CSR dimensions must be non-negative");
    }
    if (csr.values.size() != csr.column_indices.size()) {
        throw std::invalid_argument("CSR values and column indices must have identical size");
    }
    if (csr.values.size() != static_cast<std::size_t>(csr.nnz)) {
        throw std::invalid_argument("CSR NNZ does not match the values and column arrays");
    }
    if (csr.row_pointers.size() != static_cast<std::size_t>(csr.rows) + 1) {
        throw std::invalid_argument("CSR row pointers size must equal rows + 1");
    }
    if (csr.row_pointers.empty() || csr.row_pointers.front() != 0) {
        throw std::invalid_argument("CSR row pointers must start at zero");
    }
    if (csr.row_pointers.back() != csr.nnz) {
        throw std::invalid_argument("CSR final row pointer must equal NNZ");
    }

    bool rowPtrValid = true;
    bool colIdxValid = true;
    const std::size_t rpSize = csr.row_pointers.size();
    const std::size_t nnzSize = csr.column_indices.size();

    #pragma omp parallel num_threads(numThreads) if(rpSize > 4096 || nnzSize > 4096)
    {
        #pragma omp for schedule(static) reduction(&&:rowPtrValid) nowait
        for (std::size_t i = 1; i < rpSize; ++i) {
            if (csr.row_pointers[i] < csr.row_pointers[i - 1]) {
                rowPtrValid = false;
            }
        }

        #pragma omp for schedule(static) reduction(&&:colIdxValid)
        for (std::size_t i = 0; i < nnzSize; ++i) {
            const int col = csr.column_indices[i];
            if (col < 0 || col >= csr.cols) {
                colIdxValid = false;
            }
        }
    }

    if (!rowPtrValid) {
        throw std::invalid_argument("CSR row pointers must be monotonically non-decreasing");
    }
    if (!colIdxValid) {
        throw std::invalid_argument("CSR column index is outside [0, cols)");
    }
}

std::vector<argcsr::ArgCSRGroupInfo> buildGroupsSequential(const argcsr::CSRMatrix& csr,
                                                           int desiredChunkSize,
                                                           int blockSize) {
    const long long nnzLimit = static_cast<long long>(desiredChunkSize) * blockSize;
    std::vector<argcsr::ArgCSRGroupInfo> groups;
    if (csr.rows > 0) {
        groups.reserve(static_cast<std::size_t>((csr.rows + blockSize - 1) / blockSize));
    }

    int first = 0;
    long long count = 0;
    int size = 0;
    for (int row = 0; row < csr.rows; ++row) {
        const long long rowNnz = csr.row_pointers[static_cast<std::size_t>(row) + 1] -
                                 csr.row_pointers[static_cast<std::size_t>(row)];
        if (size > 0 && (count + rowNnz > nnzLimit || size + 1 > blockSize)) {
            groups.push_back({first, size, 0, 0});
            first = row;
            size = 0;
            count = 0;
        }
        ++size;
        count += rowNnz;
    }
    if (size > 0) {
        groups.push_back({first, size, 0, 0});
    }
    return groups;
}

void validateConvertedParallel(const argcsr::CSRMatrix& csr,
                               const argcsr::ArgCSRMatrix& out,
                               const std::vector<int>& rowPrefix,
                               int numThreads) {
    int expectedFirst = 0;
    for (const auto& group : out.groups) {
        if (group.firstRow != expectedFirst || group.size <= 0 || group.firstRow < 0 ||
            group.firstRow + group.size > csr.rows || group.chunkSize <= 0 || group.offset < 0) {
            throw std::logic_error("invalid group structure or row coverage");
        }
        expectedFirst = group.firstRow + group.size;
    }
    if (expectedFirst != csr.rows) {
        throw std::logic_error("groups do not cover all rows");
    }
    if (csr.rows == 0 && !out.groups.empty()) {
        throw std::logic_error("zero-row matrix must have no groups");
    }

    bool allValid = true;
    const std::size_t groupCount = out.groups.size();
    #pragma omp parallel for schedule(guided, 32) num_threads(numThreads) reduction(&&:allValid) if(groupCount > 64)
    for (std::size_t g = 0; g < groupCount; ++g) {
        if (!allValid) continue;
        const auto& group = out.groups[g];
        int mappedThreads = 0;
        for (int i = 0; i < group.size; ++i) {
            const int row = group.firstRow + i;
            const int threads = out.threadsMapping[static_cast<std::size_t>(row)];
            if (threads < 1) {
                allValid = false;
                break;
            }
            mappedThreads += threads;
            const int rowBasePrefix = rowPrefix[static_cast<std::size_t>(row)];
            int found = 0;
            for (int k = 0; k < group.chunkSize * threads; ++k) {
                const std::size_t pos = static_cast<std::size_t>(group.offset) +
                    static_cast<std::size_t>(k % group.chunkSize) * static_cast<std::size_t>(out.blockSize) +
                    static_cast<std::size_t>(rowBasePrefix + k / group.chunkSize);
                if (out.columns[pos] == -1) continue;
                const int source = csr.row_pointers[static_cast<std::size_t>(row)] + found;
                if (source >= csr.row_pointers[static_cast<std::size_t>(row) + 1] ||
                    out.columns[pos] != csr.column_indices[static_cast<std::size_t>(source)] ||
                    out.values[pos] != csr.values[static_cast<std::size_t>(source)]) {
                    allValid = false;
                    break;
                }
                ++found;
            }
            if (!allValid ||
                found != csr.row_pointers[static_cast<std::size_t>(row) + 1] -
                         csr.row_pointers[static_cast<std::size_t>(row)]) {
                allValid = false;
                break;
            }
        }
        if (mappedThreads > out.blockSize) {
            allValid = false;
        }
    }

    if (!allValid) {
        throw std::logic_error("OpenMP ArgCSR post-conversion validation failed");
    }
}

}  // namespace

OpenMPArgCSRResult convertToArgCSROpenMPDetailed(const argcsr::CSRMatrix& csr,
                                                 int desiredChunkSize,
                                                 int blockSize,
                                                 int numThreads,
                                                 bool runPostValidation) {
    const auto totalStart = Clock::now();
    const int threads = resolveThreadCount(numThreads);
    if (desiredChunkSize <= 0 || blockSize <= 0) {
        throw std::invalid_argument("desiredChunkSize and blockSize must be positive");
    }

    OpenMPArgCSRResult result;

    const auto valStart = Clock::now();
    validateCsrParallel(csr, threads);
    const auto valStop = Clock::now();
    result.timings.validationMs = elapsedMs(valStart, valStop);

    const auto groupStart = Clock::now();
    auto groups = buildGroupsSequential(csr, desiredChunkSize, blockSize);
    const auto groupStop = Clock::now();
    result.timings.groupingMs = elapsedMs(groupStart, groupStop);

    result.matrix = argcsr::ArgCSRMatrix{
        csr.rows, csr.cols, csr.nnz, blockSize, desiredChunkSize,
        std::move(groups), {}, {}, {}
    };
    const std::size_t groupCount = result.matrix.groups.size();
    const std::size_t rowCount = static_cast<std::size_t>(csr.rows);

    result.matrix.threadsMapping.resize(rowCount);
    result.rowPrefix.resize(rowCount);
    result.rowGroup.resize(rowCount);

    // Stage B & C: Parallel mapping + intra-group prefix + chunk size across independent groups.
    const auto mapStart = Clock::now();
    bool mappingOverflow = false;
    bool offsetOverflow = false;
    long long totalSize = 0;

    const int activeThreads = (groupCount == 0) ? 1 : std::min<int>(threads, static_cast<int>(groupCount));
    std::vector<long long> threadStorageTotals(static_cast<std::size_t>(activeThreads), 0);
    std::vector<long long> threadStorageOffsets(static_cast<std::size_t>(activeThreads), 0);
    Clock::time_point mapStop{};
    Clock::time_point prefixStart{};
    Clock::time_point prefixStop{};

    if (groupCount > 0) {
        #pragma omp parallel num_threads(activeThreads)
        {
            const int tid = omp_get_thread_num();
            const int nthreads = omp_get_num_threads();
            const std::size_t gBegin = (groupCount * static_cast<std::size_t>(tid)) / static_cast<std::size_t>(nthreads);
            const std::size_t gEnd = (groupCount * static_cast<std::size_t>(tid + 1)) / static_cast<std::size_t>(nthreads);

            std::vector<int> localMapping(static_cast<std::size_t>(blockSize));
            std::vector<int> localLengths(static_cast<std::size_t>(blockSize));
            long long localStorage = 0;

            for (std::size_t g = gBegin; g < gEnd; ++g) {
                auto& group = result.matrix.groups[g];
                const int gSize = group.size;
                if (gSize > blockSize) {
                    #pragma omp atomic write
                    mappingOverflow = true;
                    continue;
                }

                for (int i = 0; i < gSize; ++i) {
                    const int row = group.firstRow + i;
                    localMapping[static_cast<std::size_t>(i)] = 1;
                    localLengths[static_cast<std::size_t>(i)] =
                        csr.row_pointers[static_cast<std::size_t>(row) + 1] -
                        csr.row_pointers[static_cast<std::size_t>(row)];
                }

                int used = gSize;
                while (used < blockSize) {
                    int best = 0;
                    for (int i = 1; i < gSize; ++i) {
                        if (requiredChunks(localLengths[static_cast<std::size_t>(i)],
                                           localMapping[static_cast<std::size_t>(i)]) >
                            requiredChunks(localLengths[static_cast<std::size_t>(best)],
                                           localMapping[static_cast<std::size_t>(best)])) {
                            best = i;
                        }
                    }
                    const int before = requiredChunks(localLengths[static_cast<std::size_t>(best)],
                                                      localMapping[static_cast<std::size_t>(best)]);
                    if (before <= 1) break;
                    ++localMapping[static_cast<std::size_t>(best)];
                    const int after = requiredChunks(localLengths[static_cast<std::size_t>(best)],
                                                     localMapping[static_cast<std::size_t>(best)]);
                    if (after == before) {
                        --localMapping[static_cast<std::size_t>(best)];
                        break;
                    }
                    ++used;
                }

                int chunk = 0;
                int lanePrefix = 0;
                for (int i = 0; i < gSize; ++i) {
                    const std::size_t rowIdx = static_cast<std::size_t>(group.firstRow + i);
                    const int m = localMapping[static_cast<std::size_t>(i)];
                    result.matrix.threadsMapping[rowIdx] = m;
                    result.rowPrefix[rowIdx] = lanePrefix;
                    result.rowGroup[rowIdx] = static_cast<int>(g);
                    lanePrefix += m;
                    chunk = std::max(chunk, requiredChunks(localLengths[static_cast<std::size_t>(i)], m));
                }

                group.chunkSize = std::max(1, chunk);
                localStorage += static_cast<long long>(group.chunkSize) * blockSize;
            }

            threadStorageTotals[static_cast<std::size_t>(tid)] = localStorage;

            #pragma omp barrier
            #pragma omp single
            {
                mapStop = Clock::now();
                prefixStart = mapStop;
                long long running = 0;
                for (int t = 0; t < nthreads; ++t) {
                    threadStorageOffsets[static_cast<std::size_t>(t)] = running;
                    running += threadStorageTotals[static_cast<std::size_t>(t)];
                }
                totalSize = running;
                if (totalSize > std::numeric_limits<int>::max()) {
                    offsetOverflow = true;
                }
            }

            if (!offsetOverflow) {
                long long groupOffset = threadStorageOffsets[static_cast<std::size_t>(tid)];
                for (std::size_t g = gBegin; g < gEnd; ++g) {
                    auto& group = result.matrix.groups[g];
                    group.offset = static_cast<int>(groupOffset);
                    groupOffset += static_cast<long long>(group.chunkSize) * blockSize;
                }
            }
        }
        prefixStop = Clock::now();
        result.timings.mappingChunkingMs = elapsedMs(mapStart, mapStop);
        result.timings.prefixOffsetsMs = elapsedMs(prefixStart, prefixStop);
    } else {
        mapStop = Clock::now();
        result.timings.mappingChunkingMs = elapsedMs(mapStart, mapStop);
    }

    if (mappingOverflow) {
        throw std::logic_error("group exceeds block size");
    }
    if (offsetOverflow) {
        throw std::overflow_error("ArgCSR storage exceeds int range");
    }

    // Stage D: Slab allocation/initialization + parallel scatter across independent groups.
    const auto allocStart = Clock::now();
    const std::size_t storageCount = static_cast<std::size_t>(totalSize);
    result.matrix.values.resize(storageCount);
    result.matrix.columns.resize(storageCount);
    if (storageCount > 0) {
        #pragma omp parallel for schedule(static) num_threads(threads) if(storageCount > 4096)
        for (std::size_t i = 0; i < storageCount; ++i) {
            result.matrix.values[i] = 0.0;
            result.matrix.columns[i] = -1;
        }
    }
    const auto allocStop = Clock::now();
    result.timings.allocationInitMs = elapsedMs(allocStart, allocStop);

    const auto scatterStart = Clock::now();
    bool scatterOk = true;
    if (groupCount > 0 && csr.nnz > 0) {
        #pragma omp parallel for schedule(guided, 32) num_threads(threads) reduction(&&:scatterOk) if(groupCount > 64)
        for (std::size_t g = 0; g < groupCount; ++g) {
            if (!scatterOk) continue;
            const auto& group = result.matrix.groups[g];
            const std::size_t base = static_cast<std::size_t>(group.offset);

            for (int i = 0; i < group.size; ++i) {
                const int row = group.firstRow + i;
                const int begin = csr.row_pointers[static_cast<std::size_t>(row)];
                const int end = csr.row_pointers[static_cast<std::size_t>(row) + 1];
                const int chunks = result.matrix.threadsMapping[static_cast<std::size_t>(row)];
                const int rowLaneBase = result.rowPrefix[static_cast<std::size_t>(row)];

                for (int k = 0; k < end - begin; ++k) {
                    const int chunk = k / group.chunkSize;
                    if (chunk >= chunks) {
                        scatterOk = false;
                        break;
                    }
                    const int within = k % group.chunkSize;
                    const std::size_t dest = base +
                        static_cast<std::size_t>(within) * static_cast<std::size_t>(blockSize) +
                        static_cast<std::size_t>(rowLaneBase + chunk);
                    result.matrix.values[dest] = csr.values[static_cast<std::size_t>(begin + k)];
                    result.matrix.columns[dest] = csr.column_indices[static_cast<std::size_t>(begin + k)];
                }
                if (!scatterOk) break;
            }
        }
    }
    const auto scatterStop = Clock::now();
    result.timings.storageScatterMs = elapsedMs(scatterStart, scatterStop);

    if (!scatterOk) {
        throw std::logic_error("row data exceeds assigned chunks");
    }

    if (runPostValidation) {
        const auto postStart = Clock::now();
        validateConvertedParallel(csr, result.matrix, result.rowPrefix, threads);
        result.timings.postValidationMs = elapsedMs(postStart, Clock::now());
    }

    result.timings.totalMs = elapsedMs(totalStart, Clock::now());
    return result;
}

argcsr::ArgCSRMatrix convertToArgCSROpenMP(const argcsr::CSRMatrix& csr,
                                           int desiredChunkSize,
                                           int blockSize,
                                           int numThreads,
                                           OpenMPConversionTimings* timings,
                                           bool runPostValidation) {
    auto res = convertToArgCSROpenMPDetailed(csr, desiredChunkSize, blockSize, numThreads, runPostValidation);
    if (timings) *timings = res.timings;
    return std::move(res.matrix);
}

std::vector<double> argcsrSpmvOpenMP(const argcsr::ArgCSRMatrix& A,
                                     const std::vector<double>& x,
                                     int numThreads) {
    if (A.rows < 0 || A.cols < 0 || A.blockSize <= 0 || A.nnz < 0) {
        throw std::invalid_argument("ArgCSR dimensions, NNZ, and block size must be valid");
    }
    if (x.size() != static_cast<std::size_t>(A.cols)) {
        throw std::invalid_argument("ArgCSR SpMV input vector length must equal the matrix column count");
    }
    if (A.values.size() != A.columns.size()) {
        throw std::logic_error("ArgCSR values and columns arrays have different lengths");
    }

    // Fast O(G) structural verification of contiguous group offsets and row coverage.
    int expectedRow = 0;
    std::size_t expectedStorageOffset = 0;
    const std::size_t blockSize = static_cast<std::size_t>(A.blockSize);
    for (const auto& group : A.groups) {
        if (group.firstRow != expectedRow || group.size <= 0 ||
            group.firstRow < 0 || group.size > A.rows - group.firstRow) {
            throw std::logic_error("ArgCSR group has invalid row coverage");
        }
        if (group.offset < 0 || group.chunkSize <= 0 ||
            static_cast<std::size_t>(group.offset) != expectedStorageOffset) {
            throw std::logic_error("ArgCSR group offsets are not contiguous or correctly ordered");
        }
        const std::size_t chunkSize = static_cast<std::size_t>(group.chunkSize);
        if (chunkSize > A.values.size() / blockSize) {
            throw std::logic_error("ArgCSR group storage extent exceeds its arrays");
        }
        const std::size_t groupStorage = chunkSize * blockSize;
        if (expectedStorageOffset > A.values.size() || groupStorage > A.values.size() - expectedStorageOffset) {
            throw std::logic_error("ArgCSR group storage extent exceeds its arrays");
        }
        expectedRow += group.size;
        expectedStorageOffset += groupStorage;
    }
    if (expectedRow != A.rows) {
        throw std::logic_error("ArgCSR groups do not cover all matrix rows");
    }
    if (A.threadsMapping.size() != static_cast<std::size_t>(A.rows)) {
        throw std::logic_error("ArgCSR threadsMapping length does not match rows");
    }
    if (expectedStorageOffset != A.values.size()) {
        throw std::logic_error("ArgCSR arrays contain storage outside group coverage");
    }

    std::vector<double> y(static_cast<std::size_t>(A.rows), 0.0);
    const std::size_t groupCount = A.groups.size();
    if (groupCount == 0) return y;

    const int threads = resolveThreadCount(numThreads);
    bool validMapping = true;
    bool validColumn = true;

    #pragma omp parallel num_threads(threads) reduction(&&:validMapping, validColumn) if(groupCount > 64)
    {
        std::vector<double> partialSum(blockSize, 0.0);

        #pragma omp for schedule(guided, 32)
        for (std::size_t g = 0; g < groupCount; ++g) {
            if (!validMapping || !validColumn) continue;

            const auto& group = A.groups[g];
            const std::size_t storageOffset = static_cast<std::size_t>(group.offset);
            const std::size_t mappingOffset = static_cast<std::size_t>(group.firstRow);
            const std::size_t chunkSize = static_cast<std::size_t>(group.chunkSize);

            int mappedChunks = 0;
            for (int i = 0; i < group.size; ++i) {
                const int count = A.threadsMapping[mappingOffset + static_cast<std::size_t>(i)];
                if (count <= 0 || mappedChunks > A.blockSize - count) {
                    validMapping = false;
                    break;
                }
                mappedChunks += count;
            }
            if (!validMapping) continue;

            for (int t = 0; t < mappedChunks; ++t) {
                double sum = 0.0;
                for (std::size_t i = 0; i < chunkSize; ++i) {
                    const std::size_t pos = storageOffset + static_cast<std::size_t>(t) + i * blockSize;
                    const int col = A.columns[pos];
                    if (col == -1) break;
                    if (col < 0 || col >= A.cols) {
                        validColumn = false;
                        break;
                    }
                    sum += A.values[pos] * x[static_cast<std::size_t>(col)];
                }
                if (!validColumn) break;
                partialSum[static_cast<std::size_t>(t)] = sum;
            }
            if (!validColumn) continue;

            int chunkPrefix = 0;
            for (int i = 0; i < group.size; ++i) {
                const int count = A.threadsMapping[mappingOffset + static_cast<std::size_t>(i)];
                double sum = 0.0;
                for (int c = 0; c < count; ++c) {
                    sum += partialSum[static_cast<std::size_t>(chunkPrefix + c)];
                }
                y[mappingOffset + static_cast<std::size_t>(i)] = sum;
                chunkPrefix += count;
            }
        }
    }

    if (!validMapping) {
        throw std::logic_error("ArgCSR threadsMapping has invalid chunk counts for a group");
    }
    if (!validColumn) {
        throw std::logic_error("ArgCSR contains an invalid column index");
    }

    return y;
}

}  // namespace argcsr_omp
