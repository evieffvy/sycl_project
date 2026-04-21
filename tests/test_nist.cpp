// Unit tests for the SYCL Monobit and Block Frequency kernels,
// plus sanity checks on the igamc implementation.
//
// Reference vectors are taken from NIST SP 800-22 Rev. 1a, Appendix B.

#include "nist_tests.hpp"
#include "math_utils.hpp"

#include <sycl/sycl.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

TestResult run_frequency_monobit(sycl::queue&, const uint8_t*, uint64_t);
TestResult run_frequency_block  (sycl::queue&, const uint8_t*, uint64_t, int);

namespace {

int g_pass = 0;
int g_fail = 0;

void check(const std::string& name, bool cond, const std::string& detail = "")
{
    if (cond) {
        std::cout << "  [PASS] " << name << "\n";
        ++g_pass;
    } else {
        std::cout << "  [FAIL] " << name;
        if (!detail.empty()) std::cout << "  (" << detail << ")";
        std::cout << "\n";
        ++g_fail;
    }
}

void check_near(const std::string& name, double got, double expected, double tol = 1e-3)
{
    check(name, std::fabs(got - expected) < tol,
          "got=" + std::to_string(got) + " expected=" + std::to_string(expected));
}

std::vector<uint8_t> str_to_bits(const std::string& s)
{
    std::vector<uint8_t> bits;
    bits.reserve(s.size());
    for (char c : s) {
        if      (c == '0') bits.push_back(0);
        else if (c == '1') bits.push_back(1);
    }
    return bits;
}

// Upload host bits to device, run `fn`, then free the device memory.
template <typename Fn>
TestResult with_device_bits(sycl::queue& q, const std::vector<uint8_t>& host_bits, Fn fn)
{
    const uint64_t n = host_bits.size();
    uint8_t* d = sycl::malloc_device<uint8_t>(n, q);
    q.memcpy(d, host_bits.data(), n).wait();
    TestResult r = fn(d, n);
    sycl::free(d, q);
    return r;
}

// ─── Test cases ─────────────────────────────────────────────────

void test_igamc()
{
    std::cout << "\n== igamc (incomplete gamma) ==\n";
    check_near("igamc(1, 0) = 1.0",    igamc(1.0, 0.0), 1.0);
    check_near("igamc(1, 1) ≈ 0.3679", igamc(1.0, 1.0), std::exp(-1.0));
    check_near("igamc(2, 1) ≈ 0.7358", igamc(2.0, 1.0), 0.7358, 1e-3);
    check_near("igamc(0.5, 0.5)",      igamc(0.5, 0.5), 0.4795, 1e-3);
}

void test_monobit_reference(sycl::queue& q)
{
    std::cout << "\n== Monobit Test — NIST reference vector ==\n";
    const auto bits = str_to_bits("1011010101");
    const auto r = with_device_bits(q, bits, [&](const uint8_t* d, uint64_t n) {
        return run_frequency_monobit(q, d, n);
    });
    check_near("P-value ≈ 0.527089", r.p_value, 0.527089, 5e-4);
    check("Result = PASS",      r.passed);
    check("sequence_len = 10",  r.sequence_len == 10);
}

void test_monobit_all_zero(sycl::queue& q)
{
    std::cout << "\n== Monobit Test — all zeros (biased) ==\n";
    const std::vector<uint8_t> bits(1000, 0);
    const auto r = with_device_bits(q, bits, [&](const uint8_t* d, uint64_t n) {
        return run_frequency_monobit(q, d, n);
    });
    check("P-value < ALPHA", r.p_value < ALPHA);
    check("Result = FAIL",   !r.passed);
}

void test_monobit_alternating(sycl::queue& q)
{
    std::cout << "\n== Monobit Test — alternating 01 (n=1000) ==\n";
    std::vector<uint8_t> bits(1000);
    for (int i = 0; i < 1000; ++i) bits[i] = i % 2;
    const auto r = with_device_bits(q, bits, [&](const uint8_t* d, uint64_t n) {
        return run_frequency_monobit(q, d, n);
    });
    check("Result = PASS (balanced)", r.passed);
}

void test_block_reference(sycl::queue& q)
{
    std::cout << "\n== Block Frequency Test — NIST reference vector ==\n";
    const auto bits = str_to_bits("0110011010");
    const auto r = with_device_bits(q, bits, [&](const uint8_t* d, uint64_t n) {
        return run_frequency_block(q, d, n, 3);
    });
    check_near("P-value ≈ 0.801252", r.p_value, 0.801252, 5e-4);
    check("Result = PASS", r.passed);
}

void test_block_all_ones(sycl::queue& q)
{
    std::cout << "\n== Block Frequency Test — all ones (biased) ==\n";
    const std::vector<uint8_t> bits(10000, 1);
    const auto r = with_device_bits(q, bits, [&](const uint8_t* d, uint64_t n) {
        return run_frequency_block(q, d, n, 100);
    });
    check("P-value < ALPHA", r.p_value < ALPHA);
    check("Result = FAIL",   !r.passed);
}

void test_large_timing(sycl::queue& q)
{
    std::cout << "\n== Large input (100K bits) — smoke/timing ==\n";
    std::vector<uint8_t> bits(100'000);
    for (size_t i = 0; i < bits.size(); ++i) bits[i] = i % 2;

    uint8_t* d = sycl::malloc_device<uint8_t>(bits.size(), q);
    q.memcpy(d, bits.data(), bits.size()).wait();

    const auto r1 = run_frequency_monobit(q, d, bits.size());
    const auto r2 = run_frequency_block  (q, d, bits.size(), 128);
    sycl::free(d, q);

    check("Monobit finishes in < 5s",   r1.time_ms < 5000.0);
    check("BlockFreq finishes in < 5s", r2.time_ms < 5000.0);
    std::cout << "  Monobit  : " << r1.time_ms << " ms\n";
    std::cout << "  BlockFreq: " << r2.time_ms << " ms\n";
}

}  // namespace

int main()
{
    std::cout << "===================================\n"
                 " NIST SYCL — Unit Test Suite\n"
                 "===================================\n";

    sycl::queue q;
    try       { q = sycl::queue(sycl::gpu_selector_v); }
    catch (...) { q = sycl::queue(sycl::cpu_selector_v); }
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    test_igamc();
    test_monobit_reference  (q);
    test_monobit_all_zero   (q);
    test_monobit_alternating(q);
    test_block_reference    (q);
    test_block_all_ones     (q);
    test_large_timing       (q);

    std::cout << "\n===================================\n"
              << " Results: " << g_pass << " passed, " << g_fail << " failed\n"
              << "===================================\n";

    return (g_fail == 0) ? 0 : 1;
}
