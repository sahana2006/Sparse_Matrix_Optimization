CXX       := g++
NVCC      := nvcc
CUDA_ARCH := sm_89
CXXFLAGS  := -std=c++17 -O3 -fopenmp -Wall -Wextra -I../include -I../openmp -I.
NVCCFLAGS := -std=c++17 -O3 -arch=$(CUDA_ARCH) -I../include -I../openmp -I. -Xcompiler "-fopenmp -Wall -Wextra"
BUILD_DIR := ../build-linux/hybrid
OMP_OBJ   := ../build-linux/openmp/argcsr_openmp.o
CPU_LIB   := ../build-linux/cpu/libargcsr.a
CUDA_OBJS := ../build-linux/cuda/argcsr_device.o \
             ../build-linux/cuda/argcsr_cuda_pipeline.o \
             ../build-linux/cuda/argcsr_cuda_grouping.o \
             ../build-linux/cuda/argcsr_cuda_mapping.o \
             ../build-linux/cuda/argcsr_cuda_storage.o \
             ../build-linux/cuda/argcsr_cuda_spmv.o

.PHONY: all test clean

all: $(BUILD_DIR)/argcsr_hybrid.o $(BUILD_DIR)/test_argcsr_hybrid

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(OMP_OBJ): ../openmp/argcsr_openmp.cpp ../openmp/argcsr_openmp.hpp
	$(MAKE) -C ../openmp all

$(BUILD_DIR)/argcsr_hybrid.o: argcsr_hybrid.cu argcsr_hybrid.hpp ../openmp/argcsr_openmp.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -dc argcsr_hybrid.cu -o $@

$(BUILD_DIR)/test_argcsr_hybrid: test_argcsr_hybrid.cpp $(BUILD_DIR)/argcsr_hybrid.o $(OMP_OBJ) $(CUDA_OBJS) $(CPU_LIB) | $(BUILD_DIR)
	$(NVCC) -std=c++17 -O3 -arch=$(CUDA_ARCH) -I../include -I../openmp -I. -Xcompiler -fopenmp \
		test_argcsr_hybrid.cpp $(BUILD_DIR)/argcsr_hybrid.o $(OMP_OBJ) $(CUDA_OBJS) $(CPU_LIB) -o $@

test: all
	cd .. && ./build-linux/hybrid/test_argcsr_hybrid

clean:
	rm -rf $(BUILD_DIR)
