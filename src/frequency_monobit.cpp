// Frequency (Monobit) Test — NIST SP 800-22 §2.1.
//
// Computes S_n = Σ (2·bit_i − 1), then the P-value erfc(|S_n|/√(2n)).
// Parallelization: each work-group reduces its slice through local-memory
// tree reduction; the K per-group partial sums are combined via a single
// atomic add into a device-side global accumulator.

#include "nist_tests.hpp"

#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>

class MonobitReductionKernel;

// `d_bits` must be a device USM pointer, pre-uploaded by the caller.
TestResult run_frequency_monobit(sycl::queue& q,
                                 const uint8_t* d_bits,
                                 uint64_t       n)
{
    if (n == 0)
        throw std::invalid_argument("run_frequency_monobit: n must be > 0");

    const auto wall_start = std::chrono::high_resolution_clock::now();

    int64_t* d_sum = sycl::malloc_device<int64_t>(1, q);
    q.fill(d_sum, int64_t{0}, 1).wait();

    const size_t local_sz  = WGROUP_SIZE;
    const size_t global_sz = ((n + local_sz - 1) / local_sz) * local_sz;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int64_t, 1> local_sum(local_sz, h);
        const uint64_t n_cap = n;

        h.parallel_for<MonobitReductionKernel>(
            sycl::nd_range<1>(global_sz, local_sz),
            [=](sycl::nd_item<1> item) {
                const size_t gid = item.get_global_id(0);
                const size_t lid = item.get_local_id(0);

                // Map bit {0,1} → value {−1,+1}. Padding lanes contribute 0.
                int64_t val = 0;
                if (gid < n_cap)
                    val = (d_bits[gid] == 1) ? 1LL : -1LL;
                local_sum[lid] = val;
                item.barrier(sycl::access::fence_space::local_space);

                // In-place tree reduction over local memory.
                for (size_t stride = local_sz / 2; stride > 0; stride >>= 1) {
                    if (lid < stride)
                        local_sum[lid] += local_sum[lid + stride];
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Relaxed ordering is safe: intra-group sync already happened
                // via barriers, and the host only reads `d_sum` after wait().
                if (lid == 0) {
                    sycl::atomic_ref<int64_t,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomic_sum(d_sum[0]);
                    atomic_sum.fetch_add(local_sum[0]);
                }
            });
    });

    q.wait_and_throw();

    int64_t Sn = 0;
    q.memcpy(&Sn, d_sum, sizeof(int64_t)).wait();
    sycl::free(d_sum, q);

    const auto wall_end = std::chrono::high_resolution_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    const double s_obs   = std::fabs(static_cast<double>(Sn)) / std::sqrt(static_cast<double>(n));
    const double p_value = std::erfc(s_obs / std::sqrt(2.0));

    return TestResult{
        /* test_name    */ "Frequency Test (Monobit)",
        /* sequence_len */ n,
        /* p_value      */ p_value,
        /* passed       */ p_value >= ALPHA,
        /* time_ms      */ elapsed_ms,
    };
}
