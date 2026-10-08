#include "validation/argcsr_validation.hpp"

#include "argcsr/argcsr_spmv.hpp"
#include "csr/csr_spmv.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace argcsr {
namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::logic_error(message);
}
}

void validateArgCSRStructure(const ArgCSRMatrix& A) {
    require(A.rows >= 0 && A.cols >= 0 && A.nnz >= 0 && A.blockSize > 0,
            "ArgCSR dimensions, NNZ, or block size is invalid");
    require(A.values.size() == A.columns.size(), "ArgCSR values and columns lengths differ");
    std::vector<bool> covered(static_cast<std::size_t>(A.rows), false);
    std::size_t mappingOffset = 0, storageOffset = 0;
    int expectedRow = 0;
    for (std::size_t gi = 0; gi < A.groups.size(); ++gi) {
        const auto& g = A.groups[gi];
        require(g.firstRow == expectedRow, "groups are unordered, overlapping, or leave a row gap");
        if (gi == 0) require(g.firstRow == 0, "first group does not start at row zero");
        require(g.size > 0 && g.chunkSize > 0 && g.offset >= 0,
                "group size, chunk size, or offset is invalid");
        require(g.size <= A.rows - g.firstRow, "group extends beyond matrix rows");
        require(static_cast<std::size_t>(g.offset) == storageOffset,
                "group offset is not the expected contiguous storage offset");
        const std::size_t rows = static_cast<std::size_t>(g.size);
        require(mappingOffset <= A.threadsMapping.size() && rows <= A.threadsMapping.size() - mappingOffset,
                "threadsMapping does not cover group rows");
        const std::size_t block = static_cast<std::size_t>(A.blockSize);
        const std::size_t chunks = static_cast<std::size_t>(g.chunkSize);
        require(chunks <= A.values.size() / block, "group storage extent exceeds arrays");
        const std::size_t extent = chunks * block;
        require(storageOffset <= A.values.size() && extent <= A.values.size() - storageOffset,
                "group storage access exceeds arrays");
        int mapped = 0;
        for (int i = 0; i < g.size; ++i) {
            const int count = A.threadsMapping[mappingOffset + static_cast<std::size_t>(i)];
            require(count > 0 && count <= A.blockSize - mapped,
                    "threadsMapping is invalid or exceeds group blockSize");
            mapped += count;
            covered[static_cast<std::size_t>(g.firstRow + i)] = true;
        }
        for (std::size_t pos = storageOffset; pos < storageOffset + extent; ++pos) {
            const int col = A.columns[pos];
            require(col == -1 || (col >= 0 && col < A.cols), "storage contains an invalid column index");
        }
        expectedRow = g.firstRow + g.size;
        mappingOffset += rows;
        storageOffset += extent;
    }
    require(expectedRow == A.rows, "final group does not end at rows");
    require(mappingOffset == A.threadsMapping.size(), "threadsMapping has entries outside groups");
    require(storageOffset == A.values.size(), "arrays contain storage outside groups");
    for (bool row : covered) require(row, "a matrix row is not covered");
}

void validateArgCSRNnzPreservation(const CSRMatrix& csr, const ArgCSRMatrix& A) {
    require(csr.rows == A.rows && csr.cols == A.cols, "CSR and ArgCSR dimensions differ");
    validateArgCSRStructure(A);
    std::size_t actualNnz = 0;
    for (int col : A.columns) if (col != -1) ++actualNnz;
    require(actualNnz == static_cast<std::size_t>(csr.nnz), "CSR NNZ differs from actual ArgCSR NNZ");
    require(A.nnz == csr.nnz, "CSR and ArgCSR declared NNZ differ");

    std::size_t mappingOffset = 0;
    for (const auto& g : A.groups) {
        int rowChunkPrefix = 0;
        for (int r = 0; r < g.size; ++r) {
            const int row = g.firstRow + r;
            const int count = A.threadsMapping[mappingOffset + static_cast<std::size_t>(r)];
            std::size_t source = static_cast<std::size_t>(csr.row_pointers[static_cast<std::size_t>(row)]);
            const std::size_t sourceEnd = static_cast<std::size_t>(csr.row_pointers[static_cast<std::size_t>(row) + 1]);
            for (int c = 0; c < count; ++c) {
                for (int i = 0; i < g.chunkSize; ++i) {
                    const std::size_t p = static_cast<std::size_t>(g.offset) +
                        static_cast<std::size_t>(i) * static_cast<std::size_t>(A.blockSize) +
                        static_cast<std::size_t>(rowChunkPrefix + c);
                    if (A.columns[p] == -1) continue;
                    require(source < sourceEnd, "ArgCSR contains extra or duplicate row entries");
                    require(A.columns[p] == csr.column_indices[source] && A.values[p] == csr.values[source],
                            "ArgCSR row entries do not match original CSR order/content");
                    ++source;
                }
            }
            require(source == sourceEnd, "ArgCSR lost one or more original CSR entries");
            rowChunkPrefix += count;
        }
        mappingOffset += static_cast<std::size_t>(g.size);
    }
}

void validateArgCSR(const CSRMatrix& csr, const ArgCSRMatrix& adaptive,
                    const std::vector<double>& x, double absoluteTolerance, double relativeTolerance) {
    require(absoluteTolerance >= 0.0 && relativeTolerance >= 0.0,
            "SpMV tolerances must be non-negative");
    validateArgCSRNnzPreservation(csr, adaptive);
    std::vector<double> expected;
    spmvCSRSerial(csr, x, expected);
    const auto actual = argcsr_spmv(adaptive, x);
    require(expected.size() == actual.size(), "CSR and ArgCSR SpMV output lengths differ");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double difference = std::abs(expected[i] - actual[i]);
        const double tolerance = absoluteTolerance + relativeTolerance * std::abs(expected[i]);
        require(difference <= tolerance, "CSR and ArgCSR SpMV differ at row " + std::to_string(i));
    }
}

}  // namespace argcsr
