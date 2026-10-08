#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream> // Writing binary files
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
enum class Pattern { Uniform, Irregular }; // Creates two possible matrices 

// Defines what matrix we want to generate
struct MatrixSpec {
    std::string name; // U1, I1 [UNIFORM, IRREGULAR]
    std::int32_t rows;
    std::int32_t cols;
    std::int32_t targetAverageNnzPerRow; // Average number of non zero entries per row
    Pattern pattern; // Uniform or Irregular
    std::uint32_t seed; // Same seed gives the same matrix, different seeds give different matrices
};

struct Matrix {
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    std::vector<std::int32_t> rowPtr; // CSR row pointer array, size = rows + 1
    std::vector<std::int32_t> columns;
    std::vector<double> values;
};

struct RowStats {
    std::int32_t minNnz = 0;
    std::int32_t maxNnz = 0;
    double averageNnz = 0.0;
};

RowStats validate(const MatrixSpec& spec, const Matrix& matrix,
                  const std::vector<std::int32_t>& expectedCounts);

// Generates a sorted list of unique column indices for a row, given the number of columns and the number of non-zero entries (count).
void generateUniqueSortedColumns(std::int32_t cols, std::int32_t count,
                                 std::mt19937_64& rng,
                                 std::unordered_set<std::int32_t>& selected,
                                 std::vector<std::int32_t>& result) {

    if (count < 0 || count > cols) throw std::runtime_error("row NNZ exceeds column count");
    selected.clear();
    result.clear();
    result.reserve(static_cast<std::size_t>(count)); // Reserve space for count column indices
    
    // Floyd's sampling algorithm selects count distinct indices in O(count) space/time.
    for (std::int32_t j = cols - count; j < cols; ++j) {
        
        std::uniform_int_distribution<std::int32_t> pick(0, j); // Randomly pick an index from 0 to j
        const std::int32_t candidate = pick(rng); // Randomly select a candidate index
        const auto inserted = selected.insert(candidate); // Try to insert the candidate into the selected set
        result.push_back(inserted.second ? candidate : j); // If the candidate was already selected, use j instead
        if (!inserted.second) selected.insert(j); // Ensure j is also in the selected set if candidate was already present
    }
    std::sort(result.begin(), result.end());
}


// How many non zero elements should each row contain
std::vector<std::int32_t> makeRowCounts(const MatrixSpec& spec, std::mt19937_64& rng) {

    // Create a vector of counts initialized to the target average NNZ per row
    std::vector<std::int32_t> counts(static_cast<std::size_t>(spec.rows), spec.targetAverageNnzPerRow);
    if (spec.pattern == Pattern::Uniform) return counts; // Done

    const std::int32_t lowRows = spec.rows * 80 / 100; // 80% of rows will have low NNZ
    const std::int32_t mediumRows = spec.rows * 18 / 100; // 18% of rows will have medium NNZ
    const std::int32_t highRows = spec.rows - lowRows - mediumRows; // Remaining 2% of rows will have high NNZ
    
    if (highRows == 0) throw std::runtime_error("irregular pattern requires at least one high-NNZ row");
    
    constexpr std::int32_t lowNnz = 3;
    constexpr std::int32_t mediumNnz = 20;
    
    // Preserving the target average NNZ per row while having a mix of low, medium, and high NNZ rows
    const std::int64_t targetNnz = static_cast<std::int64_t>(spec.rows) * spec.targetAverageNnzPerRow;
    // How many nnz are left for high nnz rows
    const std::int64_t highTotal = targetNnz - static_cast<std::int64_t>(lowRows) * lowNnz -
                                   static_cast<std::int64_t>(mediumRows) * mediumNnz;
    
    if (highTotal <= 0 || highTotal % highRows != 0)
        throw std::runtime_error("irregular row profile cannot preserve the requested target NNZ");
    const std::int64_t highNnz64 = highTotal / highRows;
    
    if (highNnz64 > spec.cols || highNnz64 > std::numeric_limits<std::int32_t>::max())
        throw std::runtime_error("irregular high-NNZ row exceeds the matrix column count");
    const auto highNnz = static_cast<std::int32_t>(highNnz64);
    
    std::fill(counts.begin(), counts.begin() + lowRows, lowNnz); // First 80% gets 3
    std::fill(counts.begin() + lowRows, counts.begin() + lowRows + mediumRows, mediumNnz); // Next 18% gets 20
    std::fill(counts.begin() + lowRows + mediumRows, counts.end(), highNnz); // 2%
    std::shuffle(counts.begin(), counts.end(), rng); // shuffle it otherwise first 8000 rows will have 3 nnz and so on....
    return counts;
}

Matrix generate(const MatrixSpec& spec) {
    if (spec.rows <= 0 || spec.cols <= 0 || spec.targetAverageNnzPerRow <= 0)
        throw std::runtime_error("matrix dimensions and target average must be positive");
    std::mt19937_64 rng(spec.seed);
    
    const auto rowCounts = makeRowCounts(spec, rng);
    Matrix matrix;
    matrix.rows = spec.rows;
    matrix.cols = spec.cols;
    matrix.rowPtr.reserve(static_cast<std::size_t>(spec.rows) + 1); // CSR row pointer array has size rows + 1
    matrix.rowPtr.push_back(0);
    std::int64_t totalNnz = 0;
    
    // Loop through every row and calculate the total number of non-zero entries (NNZ) in the matrix
    for (auto count : rowCounts) {
        totalNnz += count;
        if (totalNnz > std::numeric_limits<std::int32_t>::max())
            throw std::runtime_error("matrix NNZ exceeds binary format int32 capacity");
        matrix.rowPtr.push_back(static_cast<std::int32_t>(totalNnz)); // Add the cumulative NNZ count to the row pointer array
    }
    matrix.columns.reserve(static_cast<std::size_t>(totalNnz));
    matrix.values.reserve(static_cast<std::size_t>(totalNnz));
    std::unordered_set<std::int32_t> selected;
    selected.reserve(256);
    std::vector<std::int32_t> rowColumns;
    // loop through every row to generate column and value entries for the matrix
    for (std::int32_t row = 0; row < spec.rows; ++row) {
        const auto count = rowCounts[static_cast<std::size_t>(row)];
        generateUniqueSortedColumns(spec.cols, count, rng, selected, rowColumns);
        matrix.columns.insert(matrix.columns.end(), rowColumns.begin(), rowColumns.end());
        for (std::int32_t i = 0; i < count; ++i) { // Generate a random value for each non-zero entry in the row
            std::int32_t raw = 0;
            while (raw == 0) raw = static_cast<std::int32_t>(rng() % 2000001ULL) - 1000000;
            matrix.values.push_back(static_cast<double>(raw) / 1000000.0);
        }
    }
    validate(spec, matrix, rowCounts);
    return matrix;
}

RowStats validate(const MatrixSpec& spec, const Matrix& matrix,
                  const std::vector<std::int32_t>& expectedCounts) {
    if (matrix.rows <= 0 || matrix.cols <= 0) throw std::runtime_error("generated dimensions are invalid");
    if (matrix.rowPtr.size() != static_cast<std::size_t>(matrix.rows) + 1)
        throw std::runtime_error("row_ptr size is not rows + 1");
    if (matrix.rowPtr.front() != 0) throw std::runtime_error("row_ptr[0] is not zero");
    if (matrix.rowPtr.back() < 0 || static_cast<std::size_t>(matrix.rowPtr.back()) != matrix.columns.size() ||
        matrix.columns.size() != matrix.values.size())
        throw std::runtime_error("final row pointer does not match the actual NNZ arrays");
    RowStats stats{std::numeric_limits<std::int32_t>::max(), 0, 0.0};
    for (std::int32_t row = 0; row < matrix.rows; ++row) {
        const auto begin = matrix.rowPtr[static_cast<std::size_t>(row)];
        const auto end = matrix.rowPtr[static_cast<std::size_t>(row) + 1];
        if (begin < 0 || end < begin || static_cast<std::size_t>(end) > matrix.columns.size())
            throw std::runtime_error("row pointers are not monotonic or are outside storage");
        const auto rowNnz = end - begin;
        if (rowNnz != expectedCounts[static_cast<std::size_t>(row)])
            throw std::runtime_error("row pointer count differs from generated row NNZ");
        stats.minNnz = std::min(stats.minNnz, rowNnz);
        stats.maxNnz = std::max(stats.maxNnz, rowNnz);
        for (std::int32_t p = begin; p < end; ++p) {
            const auto col = matrix.columns[static_cast<std::size_t>(p)];
            if (col < 0 || col >= matrix.cols) throw std::runtime_error("column index is out of bounds");
            if (p > begin && matrix.columns[static_cast<std::size_t>(p - 1)] >= col)
                throw std::runtime_error("row columns are not sorted unique");
            const double value = matrix.values[static_cast<std::size_t>(p)];
            if (!std::isfinite(value) || value == 0.0) throw std::runtime_error("matrix contains a zero or invalid value");
        }
    }
    const auto nnz = matrix.rowPtr.back();
    stats.averageNnz = static_cast<double>(nnz) / matrix.rows;
    if (std::abs(stats.averageNnz - spec.targetAverageNnzPerRow) >
        std::max(1.0, spec.targetAverageNnzPerRow * 0.20))
        throw std::runtime_error("average NNZ per row is not reasonably close to target");
    if (spec.pattern == Pattern::Irregular &&
        !(stats.minNnz < stats.averageNnz && stats.maxNnz >= stats.averageNnz * 5.0))
        throw std::runtime_error("irregular row NNZ distribution lacks meaningful variation");
    return stats;
}

void writeBinary(const std::filesystem::path& path, const Matrix& matrix) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open output file: " + path.string());
    const std::int32_t nnz = static_cast<std::int32_t>(matrix.columns.size());
    out.write(reinterpret_cast<const char*>(&matrix.rows), sizeof(std::int32_t));
    out.write(reinterpret_cast<const char*>(&matrix.cols), sizeof(std::int32_t));
    out.write(reinterpret_cast<const char*>(&nnz), sizeof(std::int32_t));
    out.write(reinterpret_cast<const char*>(matrix.rowPtr.data()),
              static_cast<std::streamsize>(matrix.rowPtr.size() * sizeof(std::int32_t)));
    out.write(reinterpret_cast<const char*>(matrix.columns.data()),
              static_cast<std::streamsize>(matrix.columns.size() * sizeof(std::int32_t)));
    out.write(reinterpret_cast<const char*>(matrix.values.data()),
              static_cast<std::streamsize>(matrix.values.size() * sizeof(double)));
    if (!out) throw std::runtime_error("failed while writing output file: " + path.string());
}

}  // namespace

int main() {
    try {
        const MatrixSpec specs[] = {
            {"U1", 10000, 10000, 10, Pattern::Uniform, 0xA101u},
            {"U2", 50000, 50000, 10, Pattern::Uniform, 0xA102u},
            {"U3", 100000, 100000, 10, Pattern::Uniform, 0xA103u},
            {"U4", 500000, 500000, 10, Pattern::Uniform, 0xA104u},
            {"I1", 10000, 10000, 10, Pattern::Irregular, 0xB201u},
            {"I2", 50000, 50000, 10, Pattern::Irregular, 0xB202u},
            {"I3", 100000, 100000, 10, Pattern::Irregular, 0xB203u},
            {"I4", 500000, 500000, 10, Pattern::Irregular, 0xB204u}
        };
        const std::filesystem::path outputDir = "benchmarks/matrices";
        std::filesystem::create_directories(outputDir);
        std::filesystem::create_directories("benchmarks/results");
        for (const auto& spec : specs) {
            std::cout << "Generating " << spec.name << "...\n";
            const Matrix matrix = generate(spec);
            std::vector<std::int32_t> actualCounts;
            actualCounts.reserve(static_cast<std::size_t>(matrix.rows));
            for (std::int32_t row = 0; row < matrix.rows; ++row)
                actualCounts.push_back(matrix.rowPtr[static_cast<std::size_t>(row) + 1] -
                                       matrix.rowPtr[static_cast<std::size_t>(row)]);
            const auto stats = validate(spec, matrix, actualCounts);
            const auto path = outputDir / (spec.name + ".csrbin");
            writeBinary(path, matrix);
            std::cout << "  pattern = " << (spec.pattern == Pattern::Uniform ? "uniform" : "irregular") << '\n'
                      << "  rows = " << matrix.rows << '\n'
                      << "  cols = " << matrix.cols << '\n'
                      << "  nnz = " << matrix.rowPtr.back() << '\n'
                      << "  avg_nnz_per_row = " << stats.averageNnz << '\n'
                      << "  min_nnz_per_row = " << stats.minNnz << '\n'
                      << "  max_nnz_per_row = " << stats.maxNnz << '\n'
                      << "  output = " << path.generic_string() << '\n';
        }
        std::cout << "All benchmark matrices generated successfully.\n";
    } catch (const std::exception& e) {
        std::cerr << "Matrix generation failed: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
