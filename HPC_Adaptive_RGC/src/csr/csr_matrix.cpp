#include "csr/csr_matrix.hpp"

#include <algorithm>
#include <stdexcept>
#include <tuple>

namespace argcsr {

// Constructor for CSRMatrix that initializes the matrix with given dimensions and data
CSRMatrix::CSRMatrix(int row_count, int column_count, std::vector<double> nonzero_values,
                     std::vector<int> columns, std::vector<int> pointers)
    : rows(row_count),
      cols(column_count),
      nnz(static_cast<int>(nonzero_values.size())), // static_cast to ensure nnz is an int
      values(std::move(nonzero_values)), // Moves the internal memory of the vector to avoid copying
      column_indices(std::move(columns)),
      row_pointers(std::move(pointers)) {
    validate();
}

void CSRMatrix::validate() const {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("CSR dimensions must be non-negative");
    }
    if (values.size() != column_indices.size()) {
        throw std::invalid_argument("CSR values and column indices must have identical size");
    }
    if (values.size() != static_cast<std::size_t>(nnz)) {
        throw std::invalid_argument("CSR NNZ does not match the values and column arrays");
    }
    if (row_pointers.size() != static_cast<std::size_t>(rows) + 1) {
        throw std::invalid_argument("CSR row pointers size must equal rows + 1");
    }
    if (row_pointers.empty() || row_pointers.front() != 0) {
        throw std::invalid_argument("CSR row pointers must start at zero");
    }
    for (std::size_t i = 1; i < row_pointers.size(); ++i) {
        if (row_pointers[i] < row_pointers[i - 1]) {
            throw std::invalid_argument("CSR row pointers must be monotonically non-decreasing");
        }
    }
    if (row_pointers.back() != nnz) {
        throw std::invalid_argument("CSR final row pointer must equal NNZ");
    }
    for (int column : column_indices) {
        if (column < 0 || column >= cols) {
            throw std::invalid_argument("CSR column index is outside [0, cols)");
        }
    }
}

// COO to CSR conversion method
CSRMatrix CSRMatrix::fromCoordinates(int rows, int cols,
                                     const std::vector<int>& row_indices,
                                     const std::vector<int>& columns,
                                     const std::vector<double>& coordinate_values) {
    if (row_indices.size() != columns.size() || columns.size() != coordinate_values.size()) {
        throw std::invalid_argument("Coordinate row, column, and value arrays must have identical size");
    }
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("CSR dimensions must be non-negative");
    }

    // (row, column, value) tuples for sorting
    std::vector<std::tuple<int, int, double>> entries;
    entries.reserve(coordinate_values.size());
    for (std::size_t i = 0; i < coordinate_values.size(); ++i) {
        if (row_indices[i] < 0 || row_indices[i] >= rows || columns[i] < 0 || columns[i] >= cols) {
            throw std::invalid_argument("Coordinate index is outside the matrix dimensions");
        }
        entries.emplace_back(row_indices[i], columns[i], coordinate_values[i]);
    }

    // Sort entries by row and then by column
    std::stable_sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
        return std::get<0>(a) < std::get<0>(b) ||
               (std::get<0>(a) == std::get<0>(b) && std::get<1>(a) < std::get<1>(b));
    });

    std::vector<double> values;
    std::vector<int> sorted_columns;
    std::vector<int> pointers(static_cast<std::size_t>(rows) + 1, 0);
    values.reserve(entries.size()); // reserve : for efficiency, to avoid multiple reallocations
    sorted_columns.reserve(entries.size());
    for (const auto& entry : entries) {
        ++pointers[static_cast<std::size_t>(std::get<0>(entry)) + 1];
        sorted_columns.push_back(std::get<1>(entry));
        values.push_back(std::get<2>(entry));
    }

    // Convert counts to cumulative row pointers
    for (int row = 0; row < rows; ++row) {
        pointers[static_cast<std::size_t>(row) + 1] += pointers[static_cast<std::size_t>(row)];
    }
    return CSRMatrix(rows, cols, std::move(values), std::move(sorted_columns), std::move(pointers));
}

}  // namespace argcsr
