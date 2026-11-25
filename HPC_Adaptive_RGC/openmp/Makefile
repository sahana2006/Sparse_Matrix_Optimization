CXX      := g++
CXXFLAGS := -std=c++17 -O3 -fopenmp -Wall -Wextra -I../include -I.
BUILD_DIR := ../build-linux/openmp
CPU_LIB   := ../build-linux/cpu/libargcsr.a

.PHONY: all test clean

all: $(BUILD_DIR)/argcsr_openmp.o $(BUILD_DIR)/test_argcsr_openmp

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/argcsr_openmp.o: argcsr_openmp.cpp argcsr_openmp.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c argcsr_openmp.cpp -o $@

$(BUILD_DIR)/test_argcsr_openmp: test_argcsr_openmp.cpp $(BUILD_DIR)/argcsr_openmp.o $(CPU_LIB) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) test_argcsr_openmp.cpp $(BUILD_DIR)/argcsr_openmp.o $(CPU_LIB) -o $@

test: all
	cd .. && ./build-linux/openmp/test_argcsr_openmp

clean:
	rm -rf $(BUILD_DIR)
