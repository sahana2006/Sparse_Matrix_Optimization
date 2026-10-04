#include "argcsr/argcsr_types.hpp"

#include <limits>
#include <stdexcept>

namespace argcsr {
std::vector<ArgCSRGroupInfo> buildGroups(const CSRMatrix& csr, int desiredChunkSize, int blockSize) {
    csr.validate();
    if (desiredChunkSize <= 0 || blockSize <= 0)
        throw std::invalid_argument("desiredChunkSize and blockSize must be positive");
    const long long nnzLimit = static_cast<long long>(desiredChunkSize) * blockSize;
    std::vector<ArgCSRGroupInfo> groups;
    int first = 0;
    long long count = 0;
    int size = 0;
    for (int row = 0; row < csr.rows; ++row) {
        const long long rowNnz = csr.row_pointers[row + 1] - csr.row_pointers[row];
        if (size > 0 && (count + rowNnz > nnzLimit || size + 1 > blockSize)) {
            groups.push_back({first, size, 0, 0});
            first = row;
            size = 0;
            count = 0;
        }
        ++size;
        count += rowNnz;
    }
    if (size > 0) groups.push_back({first, size, 0, 0});
    return groups;
}
}
