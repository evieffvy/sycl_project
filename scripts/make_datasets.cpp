// make_datasets.cpp — standalone generator for the test bit datasets
// ---------------------------------------------------------------------------
// This is a standalone example, independent of the main project code.
// It produces the files input_{1m,10m,50m,100m}.bin that the benchmark uses.
//
// Idea (matches the methodology described in the thesis):
//   - The data is a *pseudo-random bit sequence* generated here, not data
//     collected from an external source.
//   - It uses the Mersenne Twister 64-bit engine (std::mt19937_64), the same
//     PRNG as the main code.
//   - A fixed seed is used, so re-running always yields the same data
//     (reproducible).
//   - Each bit is a fair coin: P(bit = 1) = 0.5, so the sequence is expected
//     to PASS the randomness tests (used to verify correctness vs NIST STS).
//
// Output format: raw binary, packed 8 bits per byte (MSB-first).
//   This matches exactly what data_loader.cpp reads in RAW_BINARY mode.
//   (100 M bits = 12.5 MB instead of 100 MB if stored as ASCII.)
//
// Build (no SYCL/oneAPI needed — a plain g++/clang++ works):
//   g++ -O3 -std=c++17 scripts/make_datasets.cpp -o make_datasets
//
// Run:
//   ./make_datasets            # write all four sizes to the current dir
//   ./make_datasets ./data     # write into ./data
//
// Verify the data actually passes the tests:
//   ./build/nist_sycl -v input_1m.bin
// ---------------------------------------------------------------------------

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

// Fixed seed — same value as benchmark_stats.cpp, so the bit stream matches.
constexpr uint64_t SEED = 2024;

// The sizes used in the thesis (units are "number of bits").
struct Dataset {
    const char* name;   // output file name
    uint64_t    bits;   // number of bits
};

constexpr Dataset DATASETS[] = {
    {"input_1m.bin",     1'000'000ULL},
    {"input_10m.bin",   10'000'000ULL},
    {"input_50m.bin",   50'000'000ULL},
    {"input_100m.bin", 100'000'000ULL},
};

// Generate n random bits and pack them into a file as bytes (MSB-first).
// Returns true on success.
bool write_dataset(const std::string& path, uint64_t n_bits)
{
    std::mt19937_64           rng(SEED);   // same PRNG as the main code
    std::bernoulli_distribution coin(0.5); // fair coin: P(1) = 0.5

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "  error: cannot open file for writing: %s\n", path.c_str());
        return false;
    }

    // Round the bit count up to a whole number of bytes (8 bits/byte);
    // any extra padding bits are still drawn from the same random stream.
    const uint64_t n_bytes = (n_bits + 7) / 8;

    constexpr size_t WRITE_BUF = 64 * 1024;  // write in 64 KB chunks for speed
    std::vector<uint8_t> buf;
    buf.reserve(WRITE_BUF);

    for (uint64_t b = 0; b < n_bytes; ++b) {
        uint8_t byte = 0;
        for (int bit = 7; bit >= 0; --bit)        // MSB-first
            byte |= static_cast<uint8_t>(coin(rng) ? 1 : 0) << bit;

        buf.push_back(byte);
        if (buf.size() == WRITE_BUF) {
            out.write(reinterpret_cast<const char*>(buf.data()),
                      static_cast<std::streamsize>(buf.size()));
            buf.clear();
        }
    }
    if (!buf.empty())
        out.write(reinterpret_cast<const char*>(buf.data()),
                  static_cast<std::streamsize>(buf.size()));

    return static_cast<bool>(out);
}

}  // namespace

int main(int argc, char** argv)
{
    // First argument (if given) is the destination directory.
    std::string dir = (argc >= 2) ? argv[1] : ".";
    if (!dir.empty() && dir.back() != '/') dir += '/';

    std::printf("Generating random bit datasets with std::mt19937_64 "
                "(seed = %llu, P(1) = 0.5)\n\n",
                static_cast<unsigned long long>(SEED));

    for (const Dataset& d : DATASETS) {
        const std::string path = dir + d.name;
        std::printf("  %-16s : %12llu bits (%.1f MB) ... ",
                    d.name,
                    static_cast<unsigned long long>(d.bits),
                    static_cast<double>(d.bits) / 8.0 / (1024.0 * 1024.0));
        std::fflush(stdout);

        if (write_dataset(path, d.bits))
            std::printf("done\n");
        else
            std::printf("failed\n");
    }

    std::printf("\nFinished. Verify the data passes the tests with:\n");
    std::printf("  ./build/nist_sycl -v %sinput_1m.bin\n", dir.c_str());
    return 0;
}
