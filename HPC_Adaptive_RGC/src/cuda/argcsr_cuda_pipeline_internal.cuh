#pragma once

#include "cuda/argcsr_device.hpp"

#include <vector>

namespace argcsr::cuda_detail {

void launchRowStatistics(const int* rowPointers, int rows, int* rowLengths);
std::vector<ArgCSRGroupInfo> makeGroupsFromRowLengths(const std::vector<int>& rowLengths,
                                                      int desiredChunkSize, int blockSize);

void launchLaneMapping(const int* rowLengths, ArgCSRGroupInfo* groups, int groupCount,
                       int blockSize, int* rowMapping);
void launchChunkSizing(const int* rowLengths, const int* rowMapping,
                       ArgCSRGroupInfo* groups, int groupCount);
void launchMappingPrefixes(const ArgCSRGroupInfo* groups, int groupCount,
                           const int* rowMapping, int* rowPrefix);
void launchRowGroups(const ArgCSRGroupInfo* groups, int groupCount, int rows, int* rowGroup);

void launchStorageScatter(const int* rowPointers, const double* csrValues,
                          const int* csrColumns, int rows, int nnz,
                          const ArgCSRGroupInfo* groups, const int* rowGroup,
                          const int* rowMapping, const int* rowPrefix, int blockSize,
                          double* values, int* columns, int* storageError);

void launchAdaptiveSpmv(ArgCSRDeviceView matrix, const int* rowGroup,
                        const int* rowPrefix, const double* x, double* y);

} // namespace argcsr::cuda_detail
