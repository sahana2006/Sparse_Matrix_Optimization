#include "argcsr/argcsr_types.hpp"

#include <algorithm>
#include <stdexcept>

namespace argcsr {
std::vector<int> exclusivePrefixSum(const std::vector<int>& values) {
    std::vector<int> prefix(values.size());
    long long sum = 0;
    
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] < 0 || sum > 2147483647LL) throw std::overflow_error("prefix sum exceeds int range");
        
        prefix[i] = static_cast<int>(sum);
        sum += values[i];
    
    }
    
    if (sum > 2147483647LL) throw std::overflow_error("prefix sum exceeds int range");
    return prefix;
}
}
