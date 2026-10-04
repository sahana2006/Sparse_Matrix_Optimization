#pragma once // Include this header file only once during compilation

#include <vector>

namespace argcsr {

class CSRMatrix {
public:
    int rows;
    int cols;
    int nnz; // Number of non-zero elements
    std::vector<double> values;
    std::vector<int> column_indices;
    std::vector<int> row_pointers;

    CSRMatrix(int rows, int cols, std::vector<double> values,
              std::vector<int> column_indices, std::vector<int> row_pointers);

    // Static method to create a CSRMatrix from coordinate format (COO)
    static CSRMatrix fromCoordinates(int rows, int cols,
                                     const std::vector<int>& row_indices,
                                     const std::vector<int>& column_indices,
                                     const std::vector<double>& values);

    void validate() const;
};

}  // namespace argcsr
