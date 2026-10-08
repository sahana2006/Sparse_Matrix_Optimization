#include "argcsr/argcsr_types.hpp"
#include "validation/argcsr_validation.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr int RUNS = 3;
constexpr int DESIRED_CHUNK_SIZE = 2;
constexpr int BLOCK_SIZE = 4;

struct Row {
    std::string matrix, pattern;
    int rows, cols, nnz;
    double grouping, chunking, mapping, storage, total, unprofiled;
    double groupingPercent, chunkingPercent, mappingPercent, storagePercent, unprofiledPercent;
    std::string dominant;
    double dominantPercent;
};

template <class T>
void readExact(std::ifstream& in, std::vector<T>& data, const std::string& path) {
    if (data.empty()) return;
    const auto bytes = data.size() * sizeof(T);
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::runtime_error("binary section is too large: " + path);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("truncated binary matrix: " + path);
}

argcsr::CSRMatrix loadCsrBinary(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open matrix: " + path.string());
    std::int32_t rows = 0, cols = 0, nnz = 0;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    in.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));
    if (!in || rows <= 0 || cols <= 0 || nnz < 0)
        throw std::runtime_error("invalid matrix header: " + path.string());
    std::vector<std::int32_t> ptr(static_cast<std::size_t>(rows) + 1);
    std::vector<std::int32_t> col(static_cast<std::size_t>(nnz));
    std::vector<double> val(static_cast<std::size_t>(nnz));
    readExact(in, ptr, path.string());
    readExact(in, col, path.string());
    readExact(in, val, path.string());
    return argcsr::CSRMatrix(rows, cols, std::move(val), std::move(col), std::move(ptr));
}

double ms(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const auto n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

Row profile(const std::string& name, const std::filesystem::path& dir) {
    const auto csr = loadCsrBinary(dir / (name + ".csrbin"));
    std::vector<double> grouping, chunking, mapping, storage, totalSamples;
    grouping.reserve(RUNS); chunking.reserve(RUNS); mapping.reserve(RUNS); storage.reserve(RUNS);
    totalSamples.reserve(RUNS);
    argcsr::ArgCSRMatrix output{};
    for (int run = 0; run < RUNS; ++run) {
        argcsr::ConversionTimings t;
        const auto conversionStart = Clock::now();
        auto candidate = argcsr::convertToArgCSR(csr, DESIRED_CHUNK_SIZE, BLOCK_SIZE, &t);
        const auto conversionStop = Clock::now();
        totalSamples.push_back(ms(conversionStop - conversionStart));
        grouping.push_back(ms(t.grouping));
        chunking.push_back(ms(t.chunking));
        mapping.push_back(ms(t.mapping));
        storage.push_back(ms(t.storage));
        if (run == RUNS - 1) output = std::move(candidate);
    }
    argcsr::validateArgCSRNnzPreservation(csr, output);
    const double g = median(std::move(grouping)), c = median(std::move(chunking));
    const double m = median(std::move(mapping)), s = median(std::move(storage));
    const double total = median(std::move(totalSamples));
    const double stageSum = g + c + m + s;
    const double unprofiled = std::max(0.0, total - stageSum);
    const double gp = total ? 100.0 * g / total : 0.0;
    const double cp = total ? 100.0 * c / total : 0.0;
    const double mp = total ? 100.0 * m / total : 0.0;
    const double sp = total ? 100.0 * s / total : 0.0;
    const double up = total ? 100.0 * unprofiled / total : 0.0;
    const double parts[] = {gp, cp, mp, sp};
    const char* labels[] = {"Grouping", "Chunking", "Mapping", "Storage conversion"};
    int dominant = 0;
    for (int i = 1; i < 4; ++i) if (parts[i] > parts[dominant]) dominant = i;
    Row row{name, name[0] == 'U' ? "uniform" : "irregular", csr.rows, csr.cols, csr.nnz,
            g, c, m, s, total, unprofiled, gp, cp, mp, sp, up, labels[dominant], parts[dominant]};
    std::cout << "========================================\nMatrix: " << name << "\nPattern: " << row.pattern
              << "\n\nGrouping: " << g << " ms\nChunking: " << c << " ms\nMapping: " << m
              << " ms\nStorage conversion: " << s << " ms\n\nTotal conversion: " << total
              << " ms\nUnprofiled conversion work (mostly validation): " << unprofiled
              << " ms\nDominant conversion stage: " << row.dominant
              << "\nDominant stage percentage: " << row.dominantPercent << "%\n========================================\n";
    return row;
}

void writeCsv(const std::filesystem::path& path, const std::vector<Row>& rows) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open results CSV: " + path.string());
    out << "matrix,pattern,rows,cols,nnz,grouping_ms,chunking_ms,mapping_ms,storage_ms,total_conversion_ms,unprofiled_conversion_ms,grouping_percent,chunking_percent,mapping_percent,storage_percent,unprofiled_percent\n";
    out << std::setprecision(12);
    for (const auto& r : rows)
        out << r.matrix << ',' << r.pattern << ',' << r.rows << ',' << r.cols << ',' << r.nnz << ','
            << r.grouping << ',' << r.chunking << ',' << r.mapping << ',' << r.storage << ',' << r.total << ','
            << r.unprofiled << ',' << r.groupingPercent << ',' << r.chunkingPercent << ',' << r.mappingPercent << ','
            << r.storagePercent << ',' << r.unprofiledPercent << '\n';
}
}

int main() {
    try {
        const std::filesystem::path dir = "benchmarks/matrices";
        const std::filesystem::path csv = "benchmarks/results/conversion_profile.csv";
        std::filesystem::create_directories(csv.parent_path());
        const std::vector<std::string> names = {"U1", "U2", "U3", "U4", "I1", "I2", "I3", "I4"};
        std::vector<Row> rows;
        for (const auto& name : names) rows.push_back(profile(name, dir));
        writeCsv(csv, rows);
        std::cout << "\nConversion profile summary (median ms)\n"
                  << "Matrix   Grouping   Chunking   Mapping   Storage   Total   Dominant stage (%)\n";
        for (const auto& r : rows)
            std::cout << std::left << std::setw(8) << r.matrix << std::right << std::setw(11) << r.grouping
                      << std::setw(11) << r.chunking << std::setw(10) << r.mapping << std::setw(10) << r.storage
                      << std::setw(10) << r.total << "   " << r.dominant << " (" << r.dominantPercent << "%)\n";
        std::cout << "Results written to " << csv.generic_string() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Conversion profiling failed: " << e.what() << '\n';
        return 1;
    }
}
