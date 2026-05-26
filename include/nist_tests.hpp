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

// Forward declarations — implemented in their respective .cpp files.
// run_frequency_monobit : §2.1
// run_frequency_block   : §2.2
// run_runs_test         : §2.3
// run_longest_run       : §2.4
