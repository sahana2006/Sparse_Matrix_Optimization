#include "csr/csr_matrix.hpp"
#include "argcsr/argcsr_types.hpp"

int main() {
    const argcsr::CSRMatrix A(4, 4, {10, 2, 3, 4, 5, 7, 8},
                              {0, 3, 1, 0, 2, 2, 3}, {0, 2, 3, 5, 7});
    const auto adaptive = argcsr::convertToArgCSR(A, 2, 4);
    argcsr::printArgCSR(adaptive);
    return 0;
}
