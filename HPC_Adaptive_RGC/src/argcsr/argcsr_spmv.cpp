#include "argcsr/argcsr_spmv.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace argcsr {

std::vector<double> argcsr_spmv(const ArgCSRMatrix& A, const std::vector<double>& x) {

    // Validating input columns
    if (A.rows < 0 || A.cols < 0 || A.blockSize <= 0 || A.nnz < 0)
        throw std::invalid_argument("ArgCSR dimensions, NNZ, and block size must be valid");
    // x vector should contain as many values as the number of columns in the matrix 
    if (x.size() != static_cast<std::size_t>(A.cols))
        throw std::invalid_argument("ArgCSR SpMV input vector length must equal the matrix column count");
    
    if (A.values.size() != A.columns.size())
        throw std::logic_error("ArgCSR values and columns arrays have different lengths");

    std::vector<double> y(static_cast<std::size_t>(A.rows), 0.0); // output vector
    std::vector<bool> covered(static_cast<std::size_t>(A.rows), false); // to track which rows have been covered by groups for validation
    std::size_t mappingOffset = 0; // To track threadMapping
    std::size_t expectedStorageOffset = 0; // To track values and columns arrays offset used in the groups

    for (const auto& group : A.groups) { // Iterate over each group in the ArgCSR matrix
        
        if (group.firstRow < 0 || group.size <= 0 || group.firstRow > A.rows ||
            group.size > A.rows - group.firstRow)
            throw std::logic_error("ArgCSR group has invalid row coverage");
        if (group.offset < 0 || group.chunkSize <= 0)
            throw std::logic_error("ArgCSR group has invalid offset or chunk size");
        if (static_cast<std::size_t>(group.offset) != expectedStorageOffset)
            throw std::logic_error("ArgCSR group offsets are not contiguous or correctly ordered");

        const std::size_t rowsInGroup = static_cast<std::size_t>(group.size); // Number of rows covered by the group
        // Validate that the threadsMapping has enough entries for the rows in the group
        if (mappingOffset > A.threadsMapping.size() || rowsInGroup > A.threadsMapping.size() - mappingOffset)
            throw std::logic_error("ArgCSR threadsMapping does not cover a group's rows");

        const std::size_t chunkSize = static_cast<std::size_t>(group.chunkSize);
        const std::size_t blockSize = static_cast<std::size_t>(A.blockSize);
        
        if (chunkSize > (A.values.size() / blockSize))
            throw std::logic_error("ArgCSR group storage extent exceeds its arrays");
        
        const std::size_t groupStorage = chunkSize * blockSize; // Total storage used by the group in the values and columns arrays
        if (expectedStorageOffset > A.values.size() || groupStorage > A.values.size() - expectedStorageOffset)
            throw std::logic_error("ArgCSR group storage extent exceeds its arrays");

        int mappedChunks = 0;
        for (int i = 0; i < group.size; ++i) {
            const int count = A.threadsMapping[mappingOffset + static_cast<std::size_t>(i)]; // How many chunks assigned to each row
            if (count <= 0 || mappedChunks > A.blockSize - count)
                throw std::logic_error("ArgCSR threadsMapping has invalid chunk counts for a group");
            mappedChunks += count;
        }
        if (mappedChunks > A.blockSize)
            throw std::logic_error("ArgCSR threadsMapping exceeds the group's block size");

        
        std::vector<double> partialSum(blockSize, 0.0);
        for (int t = 0; t < mappedChunks; ++t) { // Iterate over each chunk assigned to the group
            double sum = 0.0;
            for (int i = 0; i < group.chunkSize; ++i) { // Iterate over each non-zero element in the chunk
                const std::size_t pos = expectedStorageOffset + static_cast<std::size_t>(t) + static_cast<std::size_t>(i) * blockSize;
                const int column = A.columns[pos];
                if (column == -1) break;
                if (column < 0 || column >= A.cols)
                    throw std::logic_error("ArgCSR contains an invalid column index");
                sum += A.values[pos] * x[static_cast<std::size_t>(column)];
            }
            partialSum[static_cast<std::size_t>(t)] = sum;
        }

        // Accumulate the partial sums into the output vector y for each row in the group
        int chunkPrefix = 0;
        for (int i = 0; i < group.size; ++i) {
            const std::size_t row = static_cast<std::size_t>(group.firstRow + i); // The actual row index in the output vector
            if (covered[row]) throw std::logic_error("ArgCSR groups cover a row more than once");
            covered[row] = true;
            const int count = A.threadsMapping[mappingOffset + static_cast<std::size_t>(i)]; // How many chunks assigned to this row
            double sum = 0.0;
            for (int c = 0; c < count; ++c)
                sum += partialSum[static_cast<std::size_t>(chunkPrefix + c)];
            y[row] = sum;
            chunkPrefix += count;
        }
        mappingOffset += rowsInGroup; // If current group had 4 groups, then next group will start from 4th index in threadsMapping
        expectedStorageOffset += groupStorage; // If current group had 4 chunks, then next group will start from 4th index in values and columns arrays
    }

    if (mappingOffset != A.threadsMapping.size())
        throw std::logic_error("ArgCSR threadsMapping contains entries outside group coverage");
    if (expectedStorageOffset != A.values.size())
        throw std::logic_error("ArgCSR arrays contain storage outside group coverage");
    for (bool rowCovered : covered)
        if (!rowCovered) throw std::logic_error("ArgCSR groups do not cover all matrix rows");
    return y;
}

}  // namespace argcsr
