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
    std::vector<int> mapping(static_cast<std::size_t>(group.size), 1);
    std::vector<int> lengths(static_cast<std::size_t>(group.size));
    for (int i = 0; i < group.size; ++i) {
        const int row = group.firstRow + i;
        lengths[i] = csr.row_pointers[row + 1] - csr.row_pointers[row];
    }
    int used = group.size;
    if (used > blockSize) throw std::logic_error("group exceeds block size");
    while (used < blockSize) {
        int best = 0;
        for (int i = 1; i < group.size; ++i)
            if (required(lengths[i], mapping[i]) > required(lengths[best], mapping[best])) best = i;
        const int before = required(lengths[best], mapping[best]);
        if (before <= 1) break;
        ++mapping[best];
        const int after = required(lengths[best], mapping[best]);
        if (after == before) { --mapping[best]; break; }
        ++used;
    }
    return mapping;
}

int calculateChunkSize(const CSRMatrix& csr, const ArgCSRGroupInfo& group, const std::vector<int>& mapping) {
    int chunk = 0;
    for (int i = 0; i < group.size; ++i) {
        const int row = group.firstRow + i;
        const int length = csr.row_pointers[row + 1] - csr.row_pointers[row];
        chunk = std::max(chunk, required(length, mapping[i]));
    }
    return std::max(1, chunk);
}
}
