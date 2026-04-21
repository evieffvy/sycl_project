// CLI entry point for the SYCL-based NIST SP 800-22 frequency tests.
//
// Usage: nist_sycl [OPTIONS] <input_file>
//   -n <bits>    Number of bits to test         (default: all)
//   -M <int>     Block size M                   (default: 1024)
//   -d cpu|gpu   Force device selection         (default: prefer GPU)
//   -v           Verbose: device info + timing
//   -csv         Emit results as CSV
//   -h           Show help

#include "nist_tests.hpp"
#include "data_loader.hpp"

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

// Implemented in frequency_monobit.cpp / frequency_block.cpp
TestResult run_frequency_monobit(sycl::queue& q, const uint8_t* bits, uint64_t n);
TestResult run_frequency_block  (sycl::queue& q, const uint8_t* bits, uint64_t n, int M);

// ─── CLI options ─────────────────────────────────────────────────────────
struct Options {
    std::string input_file;
    uint64_t    max_bits = 0;                // 0 means "load everything"
    int         block_M  = DEFAULT_BLOCK;
    bool        force_cpu = false;
    bool        verbose   = false;
    bool        csv_mode  = false;
};

static void print_usage(const char* prog)
{
    std::cerr <<
        "Usage: " << prog << " [OPTIONS] <input_file>\n\n"
        "Options:\n"
        "  -n <bits>   Number of bits to test (default: all)\n"
        "  -M <int>    Block size for block-frequency test (default: 1024)\n"
        "  -d cpu|gpu  Force device (default: prefer GPU)\n"
        "  -v          Verbose output including timing\n"
        "  -csv        Output results in CSV format\n"
        "  -h          Show this help\n\n"
        "Input file formats:\n"
        "  Raw binary  : each byte encodes 8 bits (MSB-first)\n"
        "  ASCII bits  : each character is '0' or '1'\n\n"
        "NIST recommendation: n >= 100, M >= 20, N = n/M >= 100\n";
}

// Returns false and prints usage on error.
static bool parse_args(int argc, char** argv, Options& opt)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h") { print_usage(argv[0]); std::exit(0); }
        else if (arg == "-v")   { opt.verbose  = true; }
        else if (arg == "-csv") { opt.csv_mode = true; }
        else if (arg == "-n" && i + 1 < argc) { opt.max_bits = std::stoull(argv[++i]); }
        else if (arg == "-M" && i + 1 < argc) { opt.block_M  = std::stoi  (argv[++i]); }
        else if (arg == "-d" && i + 1 < argc) {
            std::string dev = argv[++i];
            if      (dev == "cpu") opt.force_cpu = true;
            else if (dev == "gpu") opt.force_cpu = false;
            else { std::cerr << "Unknown device: " << dev << "\n"; return false; }
        }
        else if (arg[0] != '-') { opt.input_file = arg; }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return false;
        }
    }

    if (opt.input_file.empty()) {
        std::cerr << "Error: no input file specified.\n";
        print_usage(argv[0]);
        return false;
    }
    return true;
}

// ─── SYCL device setup ───────────────────────────────────────────────────
static sycl::queue make_queue(bool force_cpu)
{
    const auto props = sycl::property_list{sycl::property::queue::enable_profiling{}};

    if (force_cpu)
        return sycl::queue(sycl::cpu_selector_v, props);

    try {
        return sycl::queue(sycl::gpu_selector_v, props);
    } catch (const sycl::exception&) {
        std::cerr << "No GPU found, falling back to CPU.\n";
        return sycl::queue(sycl::cpu_selector_v, props);
    }
}

static void print_device_info(const sycl::queue& q)
{
    auto dev = q.get_device();
    std::cout << "Device : " << dev.get_info<sycl::info::device::name>()    << "\n"
              << "Vendor : " << dev.get_info<sycl::info::device::vendor>()  << "\n"
              << "Version: " << dev.get_info<sycl::info::device::version>() << "\n\n";
}

// ─── Result formatting ──────────────────────────────────────────────────
static double throughput_mbps(const TestResult& r)
{
    if (r.time_ms <= 0) return 0.0;
    return static_cast<double>(r.sequence_len) / (r.time_ms / 1000.0) / 1e6;
}

static void print_result(const TestResult& r, bool verbose)
{
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "┌─────────────────────────────────────────────────┐\n";
    std::cout << "│  " << std::left << std::setw(47) << r.test_name << "│\n";
    std::cout << "├─────────────────────────────────────────────────┤\n";
    std::cout << "│  Bits (n)   : " << std::setw(34) << r.sequence_len << "│\n";
    std::cout << "│  P-value    : " << std::setw(34) << r.p_value      << "│\n";
    std::cout << "│  Result     : " << (r.passed ? "PASS" : "FAIL")
              << std::setw(30) << " " << "│\n";

    if (verbose) {
        char tput_buf[64];
        std::snprintf(tput_buf, sizeof(tput_buf), "%.6f Mbit/s", throughput_mbps(r));
        std::cout << "│  Time (ms)  : " << std::setw(34) << r.time_ms  << "│\n";
        std::cout << "│  Throughput : " << std::setw(34) << tput_buf   << "│\n";
    }
    std::cout << "└─────────────────────────────────────────────────┘\n\n";
}

static void print_csv_row(const TestResult& r)
{
    std::cout << r.test_name        << ","
              << r.sequence_len     << ","
              << std::fixed << std::setprecision(8) << r.p_value << ","
              << (r.passed ? "PASS" : "FAIL") << ","
              << r.time_ms           << ","
              << throughput_mbps(r)  << "\n";
}

static void report(const TestResult& r, const Options& opt)
{
    if (opt.csv_mode) print_csv_row(r);
    else              print_result (r, opt.verbose);
}

// ─── Main ────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    Options opt;
    if (!parse_args(argc, argv, opt))
        return 1;

    sycl::queue q;
    try {
        q = make_queue(opt.force_cpu);
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL device selection failed: " << e.what() << "\n";
        return 1;
    }

    if (opt.verbose)
        print_device_info(q);

    // Load bits into host memory
    if (opt.verbose)
        std::cout << "Loading " << opt.input_file << " ...\n";

    LoadResult data;
    try {
        data = load_bits(opt.input_file, opt.max_bits);
    } catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << "\n";
        return 1;
    }

    if (opt.verbose)
        std::cout << "Loaded " << data.n << " bits ("
                  << (data.format == FileFormat::RAW_BINARY ? "binary" : "ASCII")
                  << " format)\n\n";

    if (data.n < 100) {
        std::cerr << "Error: need at least 100 bits (got " << data.n << ").\n";
        return 1;
    }

    // Upload once, reuse across both tests
    const uint64_t n = data.n;
    uint8_t* d_bits = sycl::malloc_device<uint8_t>(n, q);
    if (!d_bits) {
        std::cerr << "Error: device memory allocation failed.\n";
        return 1;
    }
    q.memcpy(d_bits, data.bits.data(), n).wait();

    if (opt.csv_mode)
        std::cout << "Test,N,P-value,Result,Time(ms),Throughput(Mbit/s)\n";

    int exit_code = 0;
    try {
        report(run_frequency_monobit(q, d_bits, n),          opt);
        report(run_frequency_block  (q, d_bits, n, opt.block_M), opt);
    } catch (const std::exception& e) {
        std::cerr << "Test execution error: " << e.what() << "\n";
        exit_code = 1;
    }

    sycl::free(d_bits, q);
    return exit_code;
}
