#pragma once
// Shared types and constants for the NIST SP 800-22 frequency tests.

#include <cstdint>
#include <string>

struct TestResult {
    std::string test_name;     // e.g. "Frequency Test (Monobit)"
    uint64_t    sequence_len;  // n — number of bits tested
    double      p_value;       // computed P-value
    bool        passed;        // p_value >= ALPHA
    double      time_ms;       // wall-clock execution time
};

inline constexpr double ALPHA         = 0.01;   // NIST significance threshold
inline constexpr int    DEFAULT_BLOCK = 1024;   // default block size M
inline constexpr int    WGROUP_SIZE   = 256;    // power of 2, required for binary tree reduction
