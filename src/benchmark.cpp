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
                      double nist_p,    double cpu_p,    double gpu_p,
                      double nist_ms,   double cpu_ms,   double gpu_ms)
{
    const double cpu_speedup = (nist_ms > 0 && cpu_ms > 0) ? nist_ms / cpu_ms : 0.0;
    const double gpu_speedup = (nist_ms > 0 && gpu_ms > 0) ? nist_ms / gpu_ms : 0.0;
    const bool   cpu_match   = std::fabs(nist_p - cpu_p) < P_MATCH_TOL;
    const bool   gpu_match   = std::fabs(nist_p - gpu_p) < P_MATCH_TOL;

    std::cout << std::fixed << std::setprecision(6) << std::left;
    std::cout << "\n╔════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  " << std::setw(78) << name << "║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Bits (n)     : " << std::setw(63) << n << "║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║                   NIST STS (Serial)     SYCL CPU            SYCL GPU           ║\n";
    std::cout << "║  P-value      :   " << std::setw(22) << nist_p
              << "    " << std::setw(16) << cpu_p
              << "    " << std::setw(15) << gpu_p << "║\n";
    std::cout << "║  Result       :   " << std::setw(22) << (nist_p >= ALPHA ? "PASS" : "FAIL")
              << "    " << std::setw(16) << (cpu_p  >= ALPHA ? "PASS" : "FAIL")
              << "    " << std::setw(15) << (gpu_p  >= ALPHA ? "PASS" : "FAIL") << "║\n";
    std::cout << "║  Time (ms)    :   " << std::setw(22) << nist_ms
              << "    " << std::setw(16) << cpu_ms
              << "    " << std::setw(15) << gpu_ms << "║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Speedup vs serial (CPU)  :   " << std::setw(46) << cpu_speedup << " x ║\n";
    std::cout << "║  Speedup vs serial (GPU)  :   " << std::setw(46) << gpu_speedup << " x ║\n";
    std::cout << "║  P-value match (CPU)      :   " << std::setw(48) << (cpu_match ? "match" : "not match") << " ║\n";
    std::cout << "║  P-value match (GPU)      :   " << std::setw(48) << (gpu_match ? "match" : "not match") << " ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════╝\n";
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

    // Build both queues. GPU is optional — if it fails we still run the CPU path.
    sycl::queue q_cpu;
    try { q_cpu = sycl::queue(sycl::cpu_selector_v); }
    catch (const sycl::exception& e) {
        std::cerr << "Error: no SYCL CPU device available: " << e.what() << "\n";
        return 1;
    }

    bool have_gpu = true;
    sycl::queue q_gpu;
    try { q_gpu = sycl::queue(sycl::gpu_selector_v); }
    catch (...) {
        std::cerr << "Note: no SYCL GPU device available — running CPU only.\n";
        have_gpu = false;
    }

    std::cout << "\nCPU device : " << q_cpu.get_device().get_info<sycl::info::device::name>() << "\n";
    if (have_gpu)
        std::cout << "GPU device : " << q_gpu.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Bits (n)   : " << n << "\n"
              << "Block M    : " << block_M << "\n";

    // Allocate device memory per queue; both pull from the same host buffer.
    uint8_t* d_bits_cpu = sycl::malloc_device<uint8_t>(n, q_cpu);
    uint8_t* d_bits_gpu = have_gpu ? sycl::malloc_device<uint8_t>(n, q_gpu) : nullptr;
    if (!d_bits_cpu || (have_gpu && !d_bits_gpu)) {
        std::cerr << "Error: device memory allocation failed.\n";
        if (d_bits_cpu) sycl::free(d_bits_cpu, q_cpu);
        if (d_bits_gpu) sycl::free(d_bits_gpu, q_gpu);
        return 1;
    }
    q_cpu.memcpy(d_bits_cpu, data.bits.data(), n).wait();
    if (have_gpu) q_gpu.memcpy(d_bits_gpu, data.bits.data(), n).wait();
    std::cout << "Data uploaded (" << (n / 1024 / 1024) << " MB)\n";

    std::cout << "\nRunning NIST STS 2.1.2 (official serial)...\n";
    const NistResult nist = run_nist_sts_official(input_file, n, block_M);
    if (!nist.ok) {
        std::cerr << "Error: NIST STS failed. Check path: " << nist_sts_path() << "\n";
        sycl::free(d_bits_cpu, q_cpu);
        if (d_bits_gpu) sycl::free(d_bits_gpu, q_gpu);
        return 1;
    }
    std::cout << "NIST STS complete. Time: " << nist.time_ms << " ms\n";

    // Warm-up + measured pass on CPU.
    std::cout << "\nRunning SYCL CPU...\n";
    run_frequency_monobit(q_cpu, d_bits_cpu, n);
    run_frequency_block  (q_cpu, d_bits_cpu, n, block_M);
    const auto cpu_mono  = run_frequency_monobit(q_cpu, d_bits_cpu, n);
    const auto cpu_block = run_frequency_block  (q_cpu, d_bits_cpu, n, block_M);
    std::cout << "SYCL CPU complete.\n";

    // Warm-up + measured pass on GPU (if present).
    TestResult gpu_mono{}, gpu_block{};
    if (have_gpu) {
        std::cout << "\nRunning SYCL GPU...\n";
        run_frequency_monobit(q_gpu, d_bits_gpu, n);
        run_frequency_block  (q_gpu, d_bits_gpu, n, block_M);
        gpu_mono  = run_frequency_monobit(q_gpu, d_bits_gpu, n);
        gpu_block = run_frequency_block  (q_gpu, d_bits_gpu, n, block_M);
        std::cout << "SYCL GPU complete.\n";
    }

    // NIST STS runs both tests in a single process; split the wall time 50/50
    // as a rough attribution.
    const double nist_freq_ms  = nist.time_ms * 0.5;
    const double nist_block_ms = nist.time_ms * 0.5;

    print_comparison("Frequency Test (Monobit)", n,
                     nist.freq_p,  cpu_mono.p_value,  have_gpu ? gpu_mono.p_value  : 0.0,
                     nist_freq_ms, cpu_mono.time_ms,  have_gpu ? gpu_mono.time_ms  : 0.0);

    print_comparison("Frequency Test within a Block (M=" + std::to_string(block_M) + ")", n,
                     nist.block_p,  cpu_block.p_value, have_gpu ? gpu_block.p_value : 0.0,
                     nist_block_ms, cpu_block.time_ms, have_gpu ? gpu_block.time_ms : 0.0);

    sycl::free(d_bits_cpu, q_cpu);
    if (d_bits_gpu) sycl::free(d_bits_gpu, q_gpu);

    // CSV summary lines for downstream plotting.
    std::cout << "\nTest,N,NIST_STS(ms),SYCL_CPU(ms),SYCL_GPU(ms),"
                 "CPU_Speedup,GPU_Speedup,NIST_P,CPU_P,GPU_P,CPU_match,GPU_match\n";

    const auto fmt_match = [](double a, double b) {
        return std::fabs(a - b) < P_MATCH_TOL ? "match" : "not match";
    };

    auto emit_row = [&](const std::string& label, double nist_ms,
                        const TestResult& cpu_r, const TestResult& gpu_r, double nist_p) {
        const double gpu_ms = have_gpu ? gpu_r.time_ms : 0.0;
        const double gpu_p  = have_gpu ? gpu_r.p_value : 0.0;
        std::cout << label << "," << n << ","
                  << nist_ms << "," << cpu_r.time_ms << "," << gpu_ms << ","
                  << (cpu_r.time_ms > 0 ? nist_ms / cpu_r.time_ms : 0.0) << ","
                  << (gpu_ms       > 0 ? nist_ms / gpu_ms         : 0.0) << ","
                  << nist_p << "," << cpu_r.p_value << "," << gpu_p << ","
                  << fmt_match(nist_p, cpu_r.p_value) << ","
                  << (have_gpu ? fmt_match(nist_p, gpu_p) : "n/a") << "\n";
    };

    emit_row("Monobit",                                    nist_freq_ms,  cpu_mono,  gpu_mono,  nist.freq_p);
    emit_row("BlockFreq(M=" + std::to_string(block_M) + ")", nist_block_ms, cpu_block, gpu_block, nist.block_p);

    return 0;
}
