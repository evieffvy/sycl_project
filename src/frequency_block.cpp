// Frequency Test within a Block — NIST SP 800-22 §2.2.
//
// The sequence is divided into N = ⌊n/M⌋ non-overlapping blocks.
// Per-block proportions π_i = ones_i / M give χ² = 4M Σ (π_i − 0.5)²,
// and the P-value is igamc(N/2, χ²/2).
//
// Parallelization (two-pass):
//   Pass 1 (BlockCountKernel): each work-item counts ones in one block.
//   Pass 2 (BlockChiSqKernel): sycl::reduction sums (π_i − 0.5)² in double.

#include "nist_tests.hpp"
#include "math_utils.hpp"

#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>

class BlockCountKernel;
class BlockChiSqKernel;

// `d_bits` must be a device USM pointer, pre-uploaded by the caller.
TestResult run_frequency_block(sycl::queue& q,
                               const uint8_t* d_bits,
                               uint64_t       n,
                               int            M)
{
    if (n == 0) throw std::invalid_argument("run_frequency_block: n must be > 0");
    if (M <= 0) throw std::invalid_argument("run_frequency_block: M must be > 0");

    const uint64_t N = n / static_cast<uint64_t>(M);
    if (N < 1) throw std::invalid_argument("run_frequency_block: sequence too short");

    const auto wall_start = std::chrono::high_resolution_clock::now();

    uint32_t* d_ones  = sycl::malloc_device<uint32_t>(N, q);
    double*   d_total = sycl::malloc_device<double>(1, q);
    q.fill(d_total, 0.0, 1).wait();

    // ── Pass 1: per-block ones count ────────────────────────────────
    auto e_count = q.submit([&](sycl::handler& h) {
        const int block_M = M;
        h.parallel_for<BlockCountKernel>(
            sycl::range<1>(N),
            [=](sycl::id<1> idx) {
                const uint64_t start = idx[0] * static_cast<uint64_t>(block_M);
                uint32_t ones = 0;
                for (int j = 0; j < block_M; ++j)
                    ones += d_bits[start + j];
                d_ones[idx[0]] = ones;
            });
    });

    // ── Pass 2: χ² accumulation via SYCL reduction (double precision) ─
    q.submit([&](sycl::handler& h) {
        h.depends_on(e_count);
        const double M_d = static_cast<double>(M);
        auto reduction = sycl::reduction(d_total, sycl::plus<double>());

        h.parallel_for<BlockChiSqKernel>(
            sycl::range<1>(N), reduction,
            [=](sycl::id<1> idx, auto& sum) {
                const double pi_i = static_cast<double>(d_ones[idx[0]]) / M_d;
                const double diff = pi_i - 0.5;
                sum += diff * diff;
            });
    });

    q.wait_and_throw();

    double dev_sum = 0.0;
    q.memcpy(&dev_sum, d_total, sizeof(double)).wait();
    const double chi_sq = 4.0 * static_cast<double>(M) * dev_sum;

    sycl::free(d_ones,  q);
    sycl::free(d_total, q);

    const auto wall_end = std::chrono::high_resolution_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    const double p_value = igamc(static_cast<double>(N) / 2.0, chi_sq / 2.0);

    return TestResult{
        /* test_name    */ "Frequency Test within a Block (M=" + std::to_string(M) + ")",
        /* sequence_len */ n,
        /* p_value      */ p_value,
        /* passed       */ p_value >= ALPHA,
        /* time_ms      */ elapsed_ms,
    };
}
