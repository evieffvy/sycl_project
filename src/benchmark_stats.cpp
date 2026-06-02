// Supplementary benchmark tool: produces the three pieces of evidence a
// thesis committee typically asks for on top of single-run speedups.
//
//   PART A  Repeated-run timing variance (mean ± std, CV%) per device.
//   PART B  Host→device transfer cost, measured separately from kernel time.
//   PART C  Positive control: biased / structured inputs must FAIL the tests,
//           confirming the kernels actually detect non-randomness.
//
// It reuses the exact kernels from the thesis (run_frequency_monobit /
// run_frequency_block) so the numbers are directly comparable to Chapters 3–5.
//
// Build target: benchmark_stats  (see CMakeLists.txt)
// Usage:
//   ./benchmark_stats [input_file] [block_M] [repeats]
//     input_file : optional .bin/.txt bit sequence for PARTS A and B.
//                  If omitted, a fair random stream is generated in-memory.
//     block_M    : block size for the Block Frequency test (default 1024).
//     repeats    : measured runs per configuration (default 30).

#include "nist_tests.hpp"
#include "data_loader.hpp"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

TestResult run_frequency_monobit(sycl::queue&, const uint8_t*, uint64_t);
TestResult run_frequency_block  (sycl::queue&, const uint8_t*, uint64_t, int);

namespace {

struct Stats {
    double mean, stddev, vmin, vmax, cv_pct;
};

// Sample statistics (n-1 denominator) over a list of measurements.
Stats summarize(const std::vector<double>& xs)
{
    Stats s{0, 0, 0, 0, 0};
    if (xs.empty()) return s;
    double sum = 0.0;
    s.vmin = s.vmax = xs.front();
    for (double x : xs) {
        sum += x;
        s.vmin = std::min(s.vmin, x);
        s.vmax = std::max(s.vmax, x);
    }
    s.mean = sum / xs.size();
    if (xs.size() > 1) {
        double acc = 0.0;
        for (double x : xs) acc += (x - s.mean) * (x - s.mean);
        s.stddev = std::sqrt(acc / (xs.size() - 1));
    }
    s.cv_pct = (s.mean > 0) ? 100.0 * s.stddev / s.mean : 0.0;
    return s;
}

void print_stat_row(const std::string& label, const Stats& s)
{
    std::cout << std::fixed << std::setprecision(4) << std::left
              << "  " << std::setw(26) << label
              << "mean " << std::setw(11) << s.mean
              << "± "    << std::setw(11) << s.stddev
              << "(CV "  << std::setprecision(2) << s.cv_pct << "%)"
              << std::setprecision(4)
              << "  [min " << s.vmin << ", max " << s.vmax << "]\n";
}

// One full A+B sweep on a single device, already holding uploaded d_bits.
void measure_device(const std::string& dev_label,
                    sycl::queue& q,
                    const std::vector<uint8_t>& host_bits,
                    uint64_t n,
                    int block_M,
                    int repeats)
{
    std::cout << "\n──────────────────────────────────────────────────────────────\n";
    std::cout << dev_label << " : " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "  n = " << n << " bits, block_M = " << block_M
              << ", repeats = " << repeats << "\n";

    uint8_t* d_bits = sycl::malloc_device<uint8_t>(n, q);
    if (!d_bits) { std::cerr << "  allocation failed\n"; return; }

    // ── PART B: host→device transfer (measured on its own) ──────────
    std::vector<double> xfer_ms;
    for (int r = 0; r < repeats; ++r) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        q.memcpy(d_bits, host_bits.data(), n).wait();
        const auto t1 = std::chrono::high_resolution_clock::now();
        xfer_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    const Stats xs = summarize(xfer_ms);
    const double mb       = static_cast<double>(n) / (1024.0 * 1024.0);
    const double gbps_eff = (xs.mean > 0) ? (static_cast<double>(n) / (xs.mean / 1000.0)) / 1e9 : 0.0;

    std::cout << "\n  [B] Host→device transfer (" << std::fixed << std::setprecision(1)
              << mb << " MB, one byte per bit)\n";
    print_stat_row("transfer time (ms)", xs);
    std::cout << "    effective link bandwidth : " << std::setprecision(2) << gbps_eff << " GB/s\n";

    // ── PART A: repeated kernel-time variance ───────────────────────
    // One untimed warm-up per test (JIT + first-touch), matching Ch.3–5.
    run_frequency_monobit(q, d_bits, n);
    run_frequency_block  (q, d_bits, n, block_M);

    std::vector<double> mono_ms, block_ms;
    double mono_p = 0.0, block_p = 0.0;
    for (int r = 0; r < repeats; ++r) {
        const auto m = run_frequency_monobit(q, d_bits, n);
        mono_ms.push_back(m.time_ms); mono_p = m.p_value;
    }
    for (int r = 0; r < repeats; ++r) {
        const auto b = run_frequency_block(q, d_bits, n, block_M);
        block_ms.push_back(b.time_ms); block_p = b.p_value;
    }
    const Stats ms_mono  = summarize(mono_ms);
    const Stats ms_block = summarize(block_ms);

    std::cout << "\n  [A] Kernel time over " << repeats << " runs (P-value is constant per input)\n";
    print_stat_row("Monobit (ms)", ms_mono);
    std::cout << "    Monobit P-value = " << std::setprecision(6) << mono_p << "\n";
    print_stat_row("Block Frequency (ms)", ms_block);
    std::cout << "    Block P-value   = " << std::setprecision(6) << block_p << "\n";

    // End-to-end = one transfer + one kernel (transfer excluded from Ch.3–5).
    std::cout << "\n  [B+A] End-to-end estimate (transfer + kernel, mean)\n";
    std::cout << std::setprecision(4)
              << "    Monobit         : " << (xs.mean + ms_mono.mean)  << " ms"
              << "  (kernel " << ms_mono.mean  << " + transfer " << xs.mean << ")\n";
    std::cout << "    Block Frequency : " << (xs.mean + ms_block.mean) << " ms"
              << "  (kernel " << ms_block.mean << " + transfer " << xs.mean << ")\n";

    // CSV for plotting.
    std::cout << "\nCSV_STATS,device,n,test,mean_ms,std_ms,cv_pct,min_ms,max_ms,xfer_mean_ms,xfer_std_ms\n";
    std::cout << "CSV_STATS," << dev_label << "," << n << ",Monobit,"
              << ms_mono.mean << "," << ms_mono.stddev << "," << ms_mono.cv_pct << ","
              << ms_mono.vmin << "," << ms_mono.vmax << "," << xs.mean << "," << xs.stddev << "\n";
    std::cout << "CSV_STATS," << dev_label << "," << n << ",BlockFreq,"
              << ms_block.mean << "," << ms_block.stddev << "," << ms_block.cv_pct << ","
              << ms_block.vmin << "," << ms_block.vmax << "," << xs.mean << "," << xs.stddev << "\n";

    sycl::free(d_bits, q);
}

// ── PART C generators ───────────────────────────────────────────────
// Bernoulli(p) stream: P(bit = 1) = p_one.
std::vector<uint8_t> gen_biased(uint64_t n, double p_one, uint64_t seed)
{
    std::vector<uint8_t> v(n);
    std::mt19937_64 rng(seed);
    std::bernoulli_distribution coin(p_one);
    for (uint64_t i = 0; i < n; ++i) v[i] = coin(rng) ? 1 : 0;
    return v;
}

// Alternating all-ones / all-zeros blocks of size M. Globally balanced
// (Monobit should PASS) but each block is maximally skewed (Block Frequency
// should FAIL) — the complementarity claim in thesis §2.1.3.
std::vector<uint8_t> gen_blocky(uint64_t n, int M)
{
    std::vector<uint8_t> v(n);
    for (uint64_t i = 0; i < n; ++i)
        v[i] = ((i / static_cast<uint64_t>(M)) % 2 == 0) ? 1 : 0;
    return v;
}

void run_one_control(sycl::queue& q, const std::string& desc,
                     const std::vector<uint8_t>& bits, uint64_t n, int block_M,
                     const std::string& expectation)
{
    uint8_t* d = sycl::malloc_device<uint8_t>(n, q);
    q.memcpy(d, bits.data(), n).wait();
    const auto m = run_frequency_monobit(q, d, n);
    const auto b = run_frequency_block(q, d, n, block_M);
    sycl::free(d, q);

    std::cout << std::left << std::fixed << std::setprecision(6)
              << "  " << std::setw(34) << desc
              << "Mono P=" << std::setw(11) << m.p_value
              << (m.passed ? "PASS  " : "FAIL  ")
              << "Block P=" << std::setw(11) << b.p_value
              << (b.passed ? "PASS  " : "FAIL  ")
              << "| expect: " << expectation << "\n";
}

void positive_control(sycl::queue& q, int block_M)
{
    const uint64_t n = 10'000'000;   // 10 M bits — quick but well past NIST minimums
    const uint64_t seed = 12345;
    std::cout << "\n══════════════════════════════════════════════════════════════\n";
    std::cout << "PART C  Positive control (detection power), n = " << n << " bits\n";
    std::cout << "  α = " << ALPHA << " ; a flagged sequence has P < α.\n\n";

    run_one_control(q, "fair p=0.50 (negative control)", gen_biased(n, 0.50, seed),  n, block_M, "both PASS");
    run_one_control(q, "biased p=0.51",                  gen_biased(n, 0.51, seed),  n, block_M, "FAIL (small bias, large n)");
    run_one_control(q, "biased p=0.52",                  gen_biased(n, 0.52, seed),  n, block_M, "FAIL");
    run_one_control(q, "biased p=0.55",                  gen_biased(n, 0.55, seed),  n, block_M, "FAIL strongly");
    run_one_control(q, "blocky (global-balanced)",       gen_blocky(n, block_M),     n, block_M, "Mono PASS, Block FAIL");
}

}  // namespace

int main(int argc, char** argv)
{
    const std::string input_file = (argc >= 2) ? argv[1] : "";
    const int block_M = (argc >= 3) ? std::stoi(argv[2]) : DEFAULT_BLOCK;
    const int repeats = (argc >= 4) ? std::stoi(argv[3]) : 30;

    // Load real input if given; otherwise synthesize a fair random stream.
    std::vector<uint8_t> host_bits;
    uint64_t n = 0;
    if (!input_file.empty()) {
        try {
            LoadResult d = load_bits(input_file, 0);
            host_bits = std::move(d.bits);
            n = d.n;
        } catch (const std::exception& e) {
            std::cerr << "Error loading " << input_file << ": " << e.what() << "\n";
            return 1;
        }
        std::cout << "Loaded " << n << " bits from " << input_file << "\n";
    } else {
        n = 100'000'000;
        std::cout << "No input file given — generating " << n << " fair random bits.\n";
        host_bits = gen_biased(n, 0.5, 2024);
    }

    sycl::queue q_cpu;
    bool have_cpu = true;
    try { q_cpu = sycl::queue(sycl::cpu_selector_v); }
    catch (...) { have_cpu = false; std::cerr << "Note: no SYCL CPU device.\n"; }

    sycl::queue q_gpu;
    bool have_gpu = true;
    try { q_gpu = sycl::queue(sycl::gpu_selector_v); }
    catch (...) { have_gpu = false; std::cerr << "Note: no SYCL GPU device.\n"; }

    std::cout << "\n══════════════════════════════════════════════════════════════\n";
    std::cout << "PARTS A + B  Timing variance and transfer cost\n";
    if (have_cpu) measure_device("CPU", q_cpu, host_bits, n, block_M, repeats);
    if (have_gpu) measure_device("GPU", q_gpu, host_bits, n, block_M, repeats);

    // Positive control on the fastest available device.
    if (have_gpu)      positive_control(q_gpu, block_M);
    else if (have_cpu) positive_control(q_cpu, block_M);

    std::cout << "\nDone.\n";
    return 0;
}
