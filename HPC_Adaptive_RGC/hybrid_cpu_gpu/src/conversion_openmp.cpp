#include "hybrid_cpu_gpu/conversion_openmp.hpp"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hybrid_cpu_gpu {
namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int resolveThreads(int requested) {
    return requested > 0 ? requested : omp_get_max_threads();
}

namespace detail {

int requiredChunks(int nnzInRow, int assignedThreads) {
    return nnzInRow == 0 ? 0 : (nnzInRow + assignedThreads - 1) / assignedThreads;
}

std::vector<int> calculateThreadMapping(const argcsr::CSRMatrix& csr,
                                        const argcsr::ArgCSRGroupInfo& group,
                                        int blockSize) {
    std::vector<int> mapping(static_cast<std::size_t>(group.size), 1);
    std::vector<int> lengths(static_cast<std::size_t>(group.size));
    for (int i = 0; i < group.size; ++i) {
        const int row = group.firstRow + i;
        lengths[static_cast<std::size_t>(i)] =
            csr.row_pointers[static_cast<std::size_t>(row) + 1] -
            csr.row_pointers[static_cast<std::size_t>(row)];
    }
    int used = group.size;
    if (used > blockSize) throw std::logic_error("group exceeds block size");
    while (used < blockSize) {
        int best = 0;
        for (int i = 1; i < group.size; ++i) {
            if (requiredChunks(lengths[static_cast<std::size_t>(i)],
                               mapping[static_cast<std::size_t>(i)]) >
                requiredChunks(lengths[static_cast<std::size_t>(best)],
                               mapping[static_cast<std::size_t>(best)])) {
                best = i;
            }
        }
        const int before = requiredChunks(lengths[static_cast<std::size_t>(best)],
                                            mapping[static_cast<std::size_t>(best)]);
        if (before <= 1) break;
        ++mapping[static_cast<std::size_t>(best)];
        const int after = requiredChunks(lengths[static_cast<std::size_t>(best)],
                                         mapping[static_cast<std::size_t>(best)]);
        if (after == before) {
            --mapping[static_cast<std::size_t>(best)];
            break;
        }
        ++used;
    }
    return mapping;
}

int calculateChunkSize(const argcsr::CSRMatrix& csr, const argcsr::ArgCSRGroupInfo& group,
                       const std::vector<int>& mapping) {
    int chunk = 0;
    for (int i = 0; i < group.size; ++i) {
        const int row = group.firstRow + i;
        const int length = csr.row_pointers[static_cast<std::size_t>(row) + 1] -
                           csr.row_pointers[static_cast<std::size_t>(row)];
        chunk = std::max(chunk, requiredChunks(length, mapping[static_cast<std::size_t>(i)]));
    }
    return std::max(1, chunk);
}

struct GroupWork {
    std::vector<int> mapping;
    int chunkSize = 0;
    double mappingMs = 0.0;
    double chunkingMs = 0.0;
};

}  // namespace detail

}  // namespace

double gigaBytesPerSecond(std::size_t bytes, double milliseconds) {
    if (milliseconds <= 0.0 || bytes == 0) return 0.0;
    const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    const double seconds = milliseconds / 1000.0;
    return gib / seconds;
}

OpenMPConversionResult convertCsrToArgCsrOpenMP(const argcsr::CSRMatrix& csr,
                                                int desiredChunkSize,
                                                int blockSize,
                                                int ompThreads) {
    if (desiredChunkSize <= 0 || blockSize <= 0) {
        throw std::invalid_argument("desiredChunkSize and blockSize must be positive");
    }

    const auto totalStart = Clock::now();
    const int threads = resolveThreads(ompThreads);

    OpenMPConversionResult result;

    const auto groupStart = Clock::now();
    auto groups = argcsr::buildGroups(csr, desiredChunkSize, blockSize);
    const auto groupStop = Clock::now();
    result.timings.groupingMs = elapsedMs(groupStart, groupStop);

    const std::size_t groupCount = groups.size();
    std::vector<detail::GroupWork> work(groupCount);

    double mappingAccum = 0.0;
    double chunkingAccum = 0.0;

#pragma omp parallel for schedule(dynamic) num_threads(threads) reduction(+:mappingAccum, chunkingAccum)
    for (std::ptrdiff_t g = 0; g < static_cast<std::ptrdiff_t>(groupCount); ++g) {
        const auto mapStart = Clock::now();
        work[static_cast<std::size_t>(g)].mapping =
            detail::calculateThreadMapping(csr, groups[static_cast<std::size_t>(g)], blockSize);
        const auto mapStop = Clock::now();
        const auto chunkStart = Clock::now();
        work[static_cast<std::size_t>(g)].chunkSize =
            detail::calculateChunkSize(csr, groups[static_cast<std::size_t>(g)],
                                       work[static_cast<std::size_t>(g)].mapping);
        const auto chunkStop = Clock::now();
        work[static_cast<std::size_t>(g)].mappingMs = elapsedMs(mapStart, mapStop);
        work[static_cast<std::size_t>(g)].chunkingMs = elapsedMs(chunkStart, chunkStop);
        mappingAccum += work[static_cast<std::size_t>(g)].mappingMs;
        chunkingAccum += work[static_cast<std::size_t>(g)].chunkingMs;
    }
    result.timings.mappingMs = mappingAccum;
    result.timings.chunkingMs = chunkingAccum;

    long long totalStorage = 0;
    for (std::size_t g = 0; g < groupCount; ++g) {
        groups[g].chunkSize = work[g].chunkSize;
        if (totalStorage > std::numeric_limits<int>::max()) {
            throw std::overflow_error("ArgCSR storage offset exceeds int range");
        }
        groups[g].offset = static_cast<int>(totalStorage);
        totalStorage += static_cast<long long>(groups[g].chunkSize) * blockSize;
        if (totalStorage > std::numeric_limits<int>::max()) {
            throw std::overflow_error("ArgCSR storage exceeds int range");
        }
    }

    argcsr::ArgCSRMatrix out{csr.rows, csr.cols, csr.nnz, blockSize, desiredChunkSize,
                             std::move(groups), {}, {}, {}};
    out.values.assign(static_cast<std::size_t>(totalStorage), 0.0);
    out.columns.assign(static_cast<std::size_t>(totalStorage), -1);

    for (std::size_t g = 0; g < groupCount; ++g) {
        const auto& mapping = work[g].mapping;
        out.threadsMapping.insert(out.threadsMapping.end(), mapping.begin(), mapping.end());
    }

    const auto storageStart = Clock::now();

#pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (std::ptrdiff_t g = 0; g < static_cast<std::ptrdiff_t>(groupCount); ++g) {
        const auto& group = out.groups[static_cast<std::size_t>(g)];
        const auto& mapping = work[static_cast<std::size_t>(g)].mapping;
        const std::size_t base = static_cast<std::size_t>(group.offset);
        std::vector<int> prefix = argcsr::exclusivePrefixSum(mapping);

        for (int i = 0; i < group.size; ++i) {
            const int row = group.firstRow + i;
            const int begin = csr.row_pointers[static_cast<std::size_t>(row)];
            const int end = csr.row_pointers[static_cast<std::size_t>(row) + 1];
            const int chunks = mapping[static_cast<std::size_t>(i)];
            for (int k = 0; k < end - begin; ++k) {
                const int chunk = k / group.chunkSize;
                if (chunk >= chunks) throw std::logic_error("row data exceeds assigned chunks");
                const int within = k % group.chunkSize;
                const std::size_t dest = base + static_cast<std::size_t>(within) * blockSize +
                                         static_cast<std::size_t>(prefix[static_cast<std::size_t>(i)]) +
                                         static_cast<std::size_t>(chunk);
                out.values[dest] = csr.values[static_cast<std::size_t>(begin + k)];
                out.columns[dest] = csr.column_indices[static_cast<std::size_t>(begin + k)];
            }
        }
    }

    const auto storageStop = Clock::now();
    result.timings.storageMs = elapsedMs(storageStart, storageStop);

    const auto metaStart = Clock::now();
    result.rowGroup.assign(static_cast<std::size_t>(csr.rows), 0);
    result.rowPrefix.assign(static_cast<std::size_t>(csr.rows), 0);
    for (std::size_t g = 0; g < groupCount; ++g) {
        const auto& group = out.groups[g];
        int laneBase = 0;
        for (int i = 0; i < group.size; ++i) {
            const int row = group.firstRow + i;
            result.rowGroup[static_cast<std::size_t>(row)] = static_cast<int>(g);
            result.rowPrefix[static_cast<std::size_t>(row)] = laneBase;
            laneBase += out.threadsMapping[static_cast<std::size_t>(row)];
        }
    }
    const auto metaStop = Clock::now();
    result.timings.metadataMs = elapsedMs(metaStart, metaStop);

    result.timings.totalMs = elapsedMs(totalStart, Clock::now());
    result.matrix = std::move(out);
    return result;
}

}  // namespace hybrid_cpu_gpu
