// Benchmark driver: runs the SYCL tests and the official NIST STS 2.1.2
// serial reference on the same input and reports speedup + P-value match.
//
// The NIST STS install location can be set via the NIST_STS_PATH env var;
// it falls back to a hardcoded default otherwise.

#include "nist_tests.hpp"
#include "data_loader.hpp"

#include <sycl/sycl.hpp>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

TestResult run_frequency_monobit(sycl::queue&, const uint8_t*, uint64_t);
TestResult run_frequency_block  (sycl::queue&, const uint8_t*, uint64_t, int);

namespace {

constexpr double P_MATCH_TOL = 1e-5;
const std::string NIST_STS_FALLBACK = "/home/evie/sts-2.1.2/sts-2.1.2";

std::string nist_sts_path()
{
    const char* env = std::getenv("NIST_STS_PATH");
    return (env && *env) ? std::string(env) : NIST_STS_FALLBACK;
}

struct NistResult {
    double time_ms;
    double freq_p;
    double block_p;
    bool   ok;
};

// Writes the scripted input consumed by the NIST STS `assess` binary to `tmp_path`.
// Selection sequence: individual tests, input file, only tests 1 and 2, block size M, etc.
bool write_nist_input_script(const std::string& tmp_path,
                             const std::string& data_path,
                             int block_M)
{
    FILE* f = std::fopen(tmp_path.c_str(), "w");
    if (!f) return false;
    std::fprintf(f,
        "0\n%s\n0\n"                                    // run mode, input file
        "1\n1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n" // tests 1+2 on, 3-15 off
        "1\n"                                           // apply parameters
        "%d\n"                                          // block size M
        "0\n1\n0\n",                                    // termination sequence
        data_path.c_str(), block_M);
    std::fclose(f);
    return true;
}

NistResult run_nist_sts_official(const std::string& data_path, uint64_t n, int block_M)
{
    NistResult res{0, 0, 0, false};

    const std::string tmp = "/tmp/nist_input.txt";
    if (!write_nist_input_script(tmp, data_path, block_M))
        return res;

    const auto t0 = std::chrono::high_resolution_clock::now();
    const std::string cmd = "cd " + nist_sts_path() + " && ./assess " +
                            std::to_string(n) + " < " + tmp + " > /dev/null 2>&1";
    (void)std::system(cmd.c_str());  // NIST STS returns 1 even on success
    const auto t1 = std::chrono::high_resolution_clock::now();
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const std::string freq_file  = nist_sts_path() + "/experiments/AlgorithmTesting/Frequency/results.txt";
    const std::string block_file = nist_sts_path() + "/experiments/AlgorithmTesting/BlockFrequency/results.txt";

    bool freq_ok = false, block_ok = false;
    if (FILE* ff = std::fopen(freq_file.c_str(), "r")) {
        freq_ok = (std::fscanf(ff, "%lf", &res.freq_p) == 1);
        std::fclose(ff);
    }
    if (FILE* bf = std::fopen(block_file.c_str(), "r")) {
        block_ok = (std::fscanf(bf, "%lf", &res.block_p) == 1);
        std::fclose(bf);
    }

    res.ok = freq_ok && block_ok;
    return res;
}

void print_comparison(const std::string& name,
                      uint64_t n,
                      double nist_p,  double sycl_p,
                      double nist_ms, double sycl_ms)
{
    const double speedup = (nist_ms > 0 && sycl_ms > 0) ? nist_ms / sycl_ms : 0.0;
    const bool   match   = std::fabs(nist_p - sycl_p) < P_MATCH_TOL;

    std::cout << std::fixed << std::setprecision(6) << std::left;
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  " << std::setw(60) << name << "║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Bits (n)     : " << std::setw(45) << n << "║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║                   NIST STS 2.1.2 (Serial)    SYCL (Parallel) ║\n";
    std::cout << "║  P-value      :   " << std::setw(23) << nist_p
              << "    " << std::setw(16) << sycl_p << "║\n";
    std::cout << "║  Result       :   "
              << std::setw(23) << (nist_p >= ALPHA ? "PASS" : "FAIL")
              << "    " << std::setw(16) << (sycl_p >= ALPHA ? "PASS" : "FAIL") << "║\n";
    std::cout << "║  Time (ms)    :   " << std::setw(23) << nist_ms
              << "    " << std::setw(16) << sycl_ms << "║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Speedup      :   " << std::setw(40) << speedup << " x ║\n";
    std::cout << "║  P-value      :   " << std::setw(42) << (match ? "match" : "not match") << " ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file> [block_size]\n";
        return 1;
    }

    char resolved[PATH_MAX];
    if (!realpath(argv[1], resolved)) {
        std::cerr << "Error: cannot resolve path: " << argv[1] << "\n";
        return 1;
    }
    const std::string input_file = resolved;
    const int block_M = (argc >= 3) ? std::stoi(argv[2]) : DEFAULT_BLOCK;

    LoadResult data;
    try { data = load_bits(input_file, 0); }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    const uint64_t n = data.n;

    sycl::queue q;
    try { q = sycl::queue(sycl::gpu_selector_v); }
    catch (...) { q = sycl::queue(sycl::cpu_selector_v); }

    std::cout << "\nDevice  : " << q.get_device().get_info<sycl::info::device::name>() << "\n"
              << "Bits (n): " << n << "\n"
              << "Block M : " << block_M << "\n";

    // Upload once, reuse across warm-up and measurement runs.
    uint8_t* d_bits = sycl::malloc_device<uint8_t>(n, q);
    if (!d_bits) {
        std::cerr << "Error: device memory allocation failed.\n";
        return 1;
    }
    q.memcpy(d_bits, data.bits.data(), n).wait();
    std::cout << "Data uploaded to device (" << (n / 1024 / 1024) << " MB)\n";

    std::cout << "\nRunning NIST STS 2.1.2 (official serial)...\n";
    const NistResult nist = run_nist_sts_official(input_file, n, block_M);
    if (!nist.ok) {
        std::cerr << "Error: NIST STS failed. Check path: " << nist_sts_path() << "\n";
        sycl::free(d_bits, q);
        return 1;
    }
    std::cout << "NIST STS complete. Time: " << nist.time_ms << " ms\n";

    // Warm-up pass (JIT, first-touch allocation) followed by the measured pass.
    run_frequency_monobit(q, d_bits, n);
    run_frequency_block  (q, d_bits, n, block_M);
    const auto r_mono  = run_frequency_monobit(q, d_bits, n);
    const auto r_block = run_frequency_block  (q, d_bits, n, block_M);

    // NIST STS runs both tests in a single process; split the wall time 50/50
    // as a rough attribution.
    const double nist_freq_ms  = nist.time_ms * 0.5;
    const double nist_block_ms = nist.time_ms * 0.5;

    print_comparison("Frequency Test (Monobit)", n,
                     nist.freq_p, r_mono.p_value,
                     nist_freq_ms, r_mono.time_ms);

    print_comparison("Frequency Test within a Block (M=" + std::to_string(block_M) + ")", n,
                     nist.block_p, r_block.p_value,
                     nist_block_ms, r_block.time_ms);

    sycl::free(d_bits, q);

    // CSV summary line for downstream plotting.
    std::cout << "\nTest,N,NIST_STS(ms),SYCL_Parallel(ms),Speedup,NIST_P,SYCL_P,P_match\n";

    const auto fmt_match = [](double a, double b) {
        return std::fabs(a - b) < P_MATCH_TOL ? "match" : "not match";
    };

    std::cout << "Monobit,"        << n << ","
              << nist_freq_ms      << "," << r_mono.time_ms  << ","
              << nist_freq_ms / r_mono.time_ms  << ","
              << nist.freq_p       << "," << r_mono.p_value  << ","
              << fmt_match(nist.freq_p, r_mono.p_value) << "\n";

    std::cout << "BlockFreq(M=" << block_M << ")," << n << ","
              << nist_block_ms     << "," << r_block.time_ms << ","
              << nist_block_ms / r_block.time_ms << ","
              << nist.block_p      << "," << r_block.p_value << ","
              << fmt_match(nist.block_p, r_block.p_value) << "\n";

    return 0;
}
