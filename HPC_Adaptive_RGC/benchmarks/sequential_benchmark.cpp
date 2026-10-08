#include "argcsr/argcsr_spmv.hpp"
#include "argcsr/argcsr_types.hpp"
#include "csr/csr_spmv.hpp"
#include "validation/argcsr_validation.hpp"

#include <algorithm>
#include <chrono> // Measuring execution time
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream> // Reading .csrbin and writing .csv files
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr int CONVERSION_RUNS = 3; // Convert CSR -> ArgCSR 3 times 
constexpr int SPMV_RUNS = 10; // Run SPMV 10 times
constexpr int DESIRED_CHUNK_SIZE = 2;
constexpr int BLOCK_SIZE = 4;
constexpr double ABS_TOLERANCE = 1e-9;
constexpr double REL_TOLERANCE = 1e-9;
using Clock = std::chrono::steady_clock;

struct BenchmarkRow {
    std::string name;
    std::string pattern;
    int rows;
    int cols;
    int nnz;
    double averageNnz;
    double conversionMs;
    double spmvMs;
    double totalMs;
    double spmvPercent;
    double conversionPercent;
};

struct TimingSummary {
    double minimum;
    double median;
    double maximum;
};

template <class T>
void readExact(std::ifstream& input, std::vector<T>& data, const std::string& path) {
    if (data.empty()) return;
    
    // Number of bytes to read
    const auto bytes = data.size() * sizeof(T);
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::runtime_error("binary section is too large: " + path);
    // Read the exact number of bytes into the vector
    input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
    if (!input) throw std::runtime_error("truncated binary matrix: " + path);
}

argcsr::CSRMatrix loadCsrBinary(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary); // Opens file in binary mode
    if (!input) throw std::runtime_error("cannot open matrix: " + path.string());
    
    std::int32_t rows = 0, cols = 0, nnz = 0; 
    input.read(reinterpret_cast<char*>(&rows), sizeof(rows)); // Read the number of rows from the binary file
    input.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    input.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));
    
    if (!input) throw std::runtime_error("matrix header is truncated: " + path.string());
    if (rows <= 0 || cols <= 0 || nnz < 0)
        throw std::runtime_error("matrix dimensions or NNZ are invalid: " + path.string());

    // Allocate vectors for row pointers, column indices, and values
    std::vector<std::int32_t> rowPtr(static_cast<std::size_t>(rows) + 1);
    std::vector<std::int32_t> columns(static_cast<std::size_t>(nnz));
    std::vector<double> values(static_cast<std::size_t>(nnz));
    
    // Now the .csrbin file is rowptr[], col[] and values[] in RAM
    readExact(input, rowPtr, path.string());
    readExact(input, columns, path.string());
    readExact(input, values, path.string());
    
    if (input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("matrix file contains trailing bytes: " + path.string());
    if (rowPtr.front() != 0 || rowPtr.back() != nnz)
        throw std::runtime_error("row pointer endpoints do not match header: " + path.string());
    
    // Check and validate the matrix and iterate through each row
    for (std::int32_t row = 0; row < rows; ++row) {
        const auto begin = rowPtr[static_cast<std::size_t>(row)];
        const auto end = rowPtr[static_cast<std::size_t>(row) + 1];
        
        if (begin < 0 || end < begin || end > nnz)
            throw std::runtime_error("row pointers are invalid: " + path.string());
        
        // end - begin is the row's nnz, and we check each column index and value for validity
        for (std::int32_t i = begin; i < end; ++i) { 
            const auto col = columns[static_cast<std::size_t>(i)];
            if (col < 0 || col >= cols)
                throw std::runtime_error("column index out of bounds: " + path.string());
            if (i > begin && columns[static_cast<std::size_t>(i - 1)] >= col) // Check if the column indices are sorted and unique
                throw std::runtime_error("row columns are unsorted or duplicated: " + path.string());
            const double value = values[static_cast<std::size_t>(i)];
            if (!std::isfinite(value) || value == 0.0)
                throw std::runtime_error("matrix has zero or invalid value: " + path.string());
        }
    }
    // matrix constructed
    return argcsr::CSRMatrix(rows, cols, std::move(values), std::move(columns), std::move(rowPtr));
}

// Creates x vector
std::vector<double> makeInputVector(int cols) {
    std::vector<double> x(static_cast<std::size_t>(cols)); // Create a vector of size 'cols' to hold the input values
    for (int i = 0; i < cols; ++i)
        x[static_cast<std::size_t>(i)] = static_cast<double>(i % 100 + 1) / 100.0; // Fill the vector with values from 0.01 to 1.00 in a repeating pattern
    return x;
}

// Finds min, median and max from samples
TimingSummary summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    const double median = samples.size() % 2 == 0
        ? (samples[middle - 1] + samples[middle]) / 2.0
        : samples[middle];
    return {samples.front(), median, samples.back()};
}

// Duration to milliseconds
double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

// Sees if ArgCSR and CSR produces the same result
void verifyCorrectness(const std::string& name, const std::vector<double>& csr,
                       const std::vector<double>& adaptive) {
    if (csr.size() != adaptive.size()) throw std::runtime_error(name + ": SpMV output sizes differ");
    double maxAbsoluteError = 0.0;
    double maxRelativeError = 0.0;
    bool valid = true;
    for (std::size_t i = 0; i < csr.size(); ++i) {
        const double error = std::abs(csr[i] - adaptive[i]); // Finding the error between the both values
        const double relative = csr[i] == 0.0
            ? (error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity())
            : error / std::abs(csr[i]);
        maxAbsoluteError = std::max(maxAbsoluteError, error);
        maxRelativeError = std::max(maxRelativeError, relative);
        if (!std::isfinite(csr[i]) || !std::isfinite(adaptive[i]) ||
            error > ABS_TOLERANCE + REL_TOLERANCE * std::abs(csr[i])) valid = false;
    }
    if (!valid) {
        std::cerr << "Correctness failure for " << name
                  << ": maximum absolute error = " << maxAbsoluteError
                  << ", maximum relative error = " << maxRelativeError << '\n';
        throw std::runtime_error("CSR and Adaptive RGC SpMV results differ for " + name);
    }
}

BenchmarkRow runOne(const std::string& name, const std::filesystem::path& matrixDir) {
    const auto csr = loadCsrBinary(matrixDir / (name + ".csrbin"));
    const auto x = makeInputVector(csr.cols);
    const std::string pattern = name.front() == 'U' ? "uniform" : "irregular";

    std::vector<double> conversionSamples;
    conversionSamples.reserve(CONVERSION_RUNS);
    argcsr::ArgCSRMatrix adaptive{};
    for (int run = 0; run < CONVERSION_RUNS; ++run) {
        const auto start = Clock::now();
        auto candidate = argcsr::convertToArgCSR(csr, DESIRED_CHUNK_SIZE, BLOCK_SIZE);
        const auto stop = Clock::now();
        conversionSamples.push_back(milliseconds(stop - start));
        if (run + 1 == CONVERSION_RUNS) adaptive = std::move(candidate);
    }
    const auto conversion = summarize(std::move(conversionSamples));

    // Warmup and correctness checks run outside all measured intervals.
    auto warmup = argcsr::argcsr_spmv(adaptive, x);
    (void)warmup;
    std::vector<double> csrY;
    argcsr::spmvCSRSerial(csr, x, csrY);
    const auto adaptiveY = argcsr::argcsr_spmv(adaptive, x);
    verifyCorrectness(name, csrY, adaptiveY);

    std::vector<double> spmvSamples;
    spmvSamples.reserve(SPMV_RUNS);
    for (int run = 0; run < SPMV_RUNS; ++run) {
        const auto start = Clock::now();
        auto y = argcsr::argcsr_spmv(adaptive, x);
        const auto stop = Clock::now();
        spmvSamples.push_back(milliseconds(stop - start));
        (void)y;
    }
    const auto spmv = summarize(std::move(spmvSamples));
    const double total = conversion.median + spmv.median;
    const double spmvPercent = total > 0.0 ? 100.0 * spmv.median / total : 0.0;
    const double conversionPercent = total > 0.0 ? 100.0 * conversion.median / total : 0.0;
    const double averageNnz = static_cast<double>(csr.nnz) / csr.rows;

    std::cout << "==================================================\n"
              << "Matrix: " << name << '\n'
              << "Pattern: " << pattern << '\n'
              << "Rows: " << csr.rows << '\n'
              << "Cols: " << csr.cols << '\n'
              << "NNZ: " << csr.nnz << '\n'
              << "Average NNZ/row: " << averageNnz << "\n\n"
              << "Conversion (min / median / max): " << conversion.minimum << " / "
              << conversion.median << " / " << conversion.maximum << " ms\n\n"
              << "Adaptive RGC SpMV (min / median / max): " << spmv.minimum << " / "
              << spmv.median << " / " << spmv.maximum << " ms\n\n"
              << "Total: " << total << " ms\n"
              << "SpMV contribution: " << spmvPercent << "%\n"
              << "Conversion contribution: " << conversionPercent << "%\n"
              << "Correctness: PASS\n"
              << "==================================================\n";
    return {name, pattern, csr.rows, csr.cols, csr.nnz, averageNnz,
            conversion.median, spmv.median, total, spmvPercent, conversionPercent};
}

void writeCsv(const std::filesystem::path& path, const std::vector<BenchmarkRow>& rows) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open results CSV: " + path.string());
    output << "matrix,pattern,rows,cols,nnz,avg_nnz_per_row,conversion_ms,spmv_ms,total_ms,spmv_percent,conversion_percent\n";
    output << std::setprecision(12);
    for (const auto& row : rows) {
        output << row.name << ',' << row.pattern << ',' << row.rows << ',' << row.cols << ','
               << row.nnz << ',' << row.averageNnz << ',' << row.conversionMs << ','
               << row.spmvMs << ',' << row.totalMs << ',' << row.spmvPercent << ','
               << row.conversionPercent << '\n';
    }
    if (!output) throw std::runtime_error("failed writing results CSV: " + path.string());
}
}  // namespace

int main() {
    try {
        const std::filesystem::path matrixDir = "benchmarks/matrices";
        const std::filesystem::path resultsPath = "benchmarks/results/sequential_results.csv";
        std::filesystem::create_directories(resultsPath.parent_path());
        const std::vector<std::string> names = {"U1", "U2", "U3", "U4", "I1", "I2", "I3", "I4"};
        std::vector<BenchmarkRow> results;
        results.reserve(names.size());
        for (const auto& name : names) results.push_back(runOne(name, matrixDir));
        writeCsv(resultsPath, results);
        std::cout << "Sequential benchmark completed successfully.\n"
                  << "Results written to:\n" << resultsPath.generic_string() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Sequential benchmark failed: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
