#include "argcsr/argcsr_types.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace argcsr {
namespace {
int required(int n, int t) { return n == 0 ? 0 : (n + t - 1) / t; }
}

// Internal conversion helper: distribute only threads that strictly reduce the
// current maximum chunk filling. Equal maxima are resolved by first row order.
std::vector<int> calculateThreadMapping(const CSRMatrix& csr, const ArgCSRGroupInfo& group, int blockSize) {
    
    // Initially assign one thread to each row in the group
    std::vector<int> mapping(static_cast<std::size_t>(group.size), 1);
    // To store the number of non zero elements in each row per group
    std::vector<int> lengths(static_cast<std::size_t>(group.size));

    // Calculate the number of non zero elements in each row of the group
    for (int i = 0; i < group.size; ++i) {
        const int row = group.firstRow + i;
        lengths[i] = csr.row_pointers[row + 1] - csr.row_pointers[row];
    }
    int used = group.size;
    if (used > blockSize) throw std::logic_error("group exceeds block size");
    // blocksize is the number of threads available for this group, so we can assign more threads to rows with more non zero elements
    while (used < blockSize) {
        int best = 0; // Imagine the row 0 needs more threads
        
        // Check which needs more threads and assign that to best
        for (int i = 1; i < group.size; ++i)
            if (required(lengths[i], mapping[i]) > required(lengths[best], mapping[best])) best = i;
        
        // Store the current maximum chunk size
        const int before = required(lengths[best], mapping[best]);
        if (before <= 1) break;
        ++mapping[best]; // Give that row one more thread
        const int after = required(lengths[best], mapping[best]); // Calulate the new workload for that row
        if (after == before) { --mapping[best]; break; } // If the new workload does not help
        ++used; // Increase the number of threads used
    }
    return mapping;
}

int calculateChunkSize(const CSRMatrix& csr, const ArgCSRGroupInfo& group, const std::vector<int>& mapping) {
    int chunk = 0;
    for (int i = 0; i < group.size; ++i) {
        const int row = group.firstRow + i;
        const int length = csr.row_pointers[row + 1] - csr.row_pointers[row]; // no:of non zero elements in the row
        chunk = std::max(chunk, required(length, mapping[i]));
    }
    return std::max(1, chunk); // Ensure that the chunk size is at least 1
}
}
