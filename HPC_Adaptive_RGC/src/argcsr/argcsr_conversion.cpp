#include "argcsr/argcsr_types.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace argcsr {
std::vector<int> calculateThreadMapping(const CSRMatrix&, const ArgCSRGroupInfo&, int);
int calculateChunkSize(const CSRMatrix&, const ArgCSRGroupInfo&, const std::vector<int>&);

ArgCSRMatrix convertToArgCSR(const CSRMatrix& csr, int desiredChunkSize, int blockSize,
                             ConversionTimings* timings) {
    using Clock = std::chrono::steady_clock;
    if (timings) *timings = {};
    const auto groupingStart = Clock::now();
    auto groups = buildGroups(csr, desiredChunkSize, blockSize);
    const auto groupingStop = Clock::now();
    if (timings) timings->grouping = groupingStop - groupingStart;
    
    // Construct the output matrix with the same dimensions and non-zero count as the input CSR matrix, along with the specified block size and desired chunk size.
    // {} {} {} => values, columns, threadsMapping.
    ArgCSRMatrix out{csr.rows, csr.cols, csr.nnz, blockSize, desiredChunkSize,
                     std::move(groups), {}, {}, {}};
    
    // Stores prefixes for each group
    std::vector<std::vector<int>> prefixes;
    long long totalSize = 0; // Total size of the output matrix
    
    for (auto& group : out.groups) {
        const auto mappingStart = Clock::now();
        const auto mapping = calculateThreadMapping(csr, group, blockSize);
        const auto prefix = exclusivePrefixSum(mapping); // calculate prefix from the mappings obtained
        const auto mappingStop = Clock::now();
        if (timings) timings->mapping += mappingStop - mappingStart;
        const auto chunkingStart = Clock::now();
        group.chunkSize = calculateChunkSize(csr, group, mapping);
        const auto chunkingStop = Clock::now();
        if (timings) timings->chunking += chunkingStop - chunkingStart;
        
        if (totalSize > std::numeric_limits<int>::max()) throw std::overflow_error("ArgCSR storage offset exceeds int range");
        
        group.offset = static_cast<int>(totalSize); // Offset is the start location of a group
        totalSize += static_cast<long long>(group.chunkSize) * blockSize; // Maximum storage of argcsr
        
        if (totalSize > std::numeric_limits<int>::max()) throw std::overflow_error("ArgCSR storage exceeds int range");
        // Store threadmapping and prefix for each group
        out.threadsMapping.insert(out.threadsMapping.end(), mapping.begin(), mapping.end());
        prefixes.push_back(prefix); // for each group it is stored
    
    }

    const auto storageStart = Clock::now();
    out.values.assign(static_cast<std::size_t>(totalSize), 0.0); // size of the array is totalsize
    out.columns.assign(static_cast<std::size_t>(totalSize), -1);
    
    // Actual Conversion happens
    for (std::size_t g = 0; g < out.groups.size(); ++g) {
        const auto& group = out.groups[g];
        const auto& prefix = prefixes[g]; // Gets the prefix of that group alone
        const std::size_t base = static_cast<std::size_t>(group.offset); // Get the offset of the group
        

        for (int i = 0; i < group.size; ++i) {
            const int row = group.firstRow + i; // Get the row
            const int begin = csr.row_pointers[row], end = csr.row_pointers[row + 1]; // Get the begin and end of the row
            const int chunks = out.threadsMapping[static_cast<std::size_t>(row)]; // Get the number of threads assigned to that row
            
            for (int k = 0; k < end - begin; ++k) {
                const int chunk = k / group.chunkSize;
                if (chunk >= chunks) throw std::logic_error("row data exceeds assigned chunks");
                
                // find its position within the chunk
                const int within = k % group.chunkSize;
                const std::size_t dest = base + static_cast<std::size_t>(within) * blockSize + prefix[i] + chunk;
                out.values[dest] = csr.values[begin + k];
                out.columns[dest] = csr.column_indices[begin + k];
            }
        }
    }
    const auto storageStop = Clock::now();
    if (timings) timings->storage = storageStop - storageStart;
    
    // Structural and ownership-preserving validation: scan each row's assigned chunks
    // and compare its ordered (value,column) sequence with the source CSR row.
    int expectedFirst = 0; // Tracks which row next should start
    for (const auto& group : out.groups) { // Validate every group
        
        if (group.firstRow != expectedFirst || group.size <= 0 || group.firstRow < 0 ||
            group.firstRow + group.size > csr.rows || group.chunkSize <= 0 || group.offset < 0)
            throw std::logic_error("invalid group structure or row coverage");
        
        const int start = group.firstRow; // Group boundaries are defined
        const int stop = start + group.size;
        int mappedThreads = 0;
        
        for (int row = start; row < stop; ++row) {
            const int threads = out.threadsMapping[row];
            if (threads < 1) throw std::logic_error("row has no assigned thread");
            
            mappedThreads += threads; // counts the assqigned threads for the group
            const int rowOffset = row - start;
            const auto prefix = exclusivePrefixSum(std::vector<int>(out.threadsMapping.begin() + start, out.threadsMapping.begin() + row));
            (void)prefix;
            int found = 0; // Count of non-zero entries found in the output for this row
            const int rowBasePrefix = [&]() {
                int s = 0; for (int r = start; r < row; ++r) s += out.threadsMapping[r]; return s;
            }();
            
            // Assigned storage for a row
            for (int k = 0; k < group.chunkSize * threads; ++k) {
                // Reconstruct the dest value
                const std::size_t pos = static_cast<std::size_t>(group.offset) + static_cast<std::size_t>(k % group.chunkSize) * blockSize + rowBasePrefix + k / group.chunkSize;
                if (out.columns[pos] == -1) continue;
                
                const int source = csr.row_pointers[row] + found;
                // verify column and value match the source CSR row
                if (source >= csr.row_pointers[row + 1] || out.columns[pos] != csr.column_indices[source] || out.values[pos] != csr.values[source])
                    throw std::logic_error("converted row data mismatch");
                ++found;
            }
            
            if (found != csr.row_pointers[row + 1] - csr.row_pointers[row]) throw std::logic_error("converted row lost nonzero entries");
            (void)rowOffset;
        }

        if (mappedThreads > blockSize) throw std::logic_error("thread mapping exceeds block size");
        expectedFirst = stop;
    }

    if (expectedFirst != csr.rows) throw std::logic_error("groups do not cover all rows");
    if (csr.rows == 0 && !out.groups.empty()) throw std::logic_error("zero-row matrix must have no groups");
    return out;
} 

void printArgCSR(const ArgCSRMatrix& m) {
    std::cout << "ArgCSR Matrix\nRows: " << m.rows << "\nCols: " << m.cols << "\nNNZ: " << m.nnz
              << "\nBlock size: " << m.blockSize << "\nDesired chunk size: " << m.desiredChunkSize
              << "\nNumber of groups: " << m.groups.size() << '\n';
    for (std::size_t i = 0; i < m.groups.size(); ++i) {
        const auto& g = m.groups[i];
        std::cout << "Group " << i << ":\n    firstRow: " << g.firstRow << "\n    size: " << g.size
                  << "\n    offset: " << g.offset << "\n    chunkSize: " << g.chunkSize << '\n';
    }
    std::cout << "Threads mapping:\n";
    for (int x : m.threadsMapping) std::cout << x << ' ';
    std::cout << "\nAdaptive values:\n";
    for (double x : m.values) std::cout << x << ' ';
    std::cout << "\nAdaptive columns:\n";
    for (int x : m.columns) std::cout << x << ' ';
    std::cout << '\n';
}
}
