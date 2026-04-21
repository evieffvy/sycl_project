// File loader implementation.
//
// Format is detected by the first byte of the file:
//   '0' (0x30) or '1' (0x31) → treated as ASCII bit stream
//   anything else            → treated as raw binary (MSB-first bits)

#include "data_loader.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace {

constexpr size_t READ_BUF_BYTES = 64 * 1024;  // 64 KB — fits L1 on most CPUs

// Read ASCII '0'/'1' chars into result.bits, skipping any other byte
// (whitespace, newlines, etc.).
void load_ascii(std::ifstream& f, uint64_t max_bits, LoadResult& out)
{
    out.bits.reserve(max_bits);
    char buf[READ_BUF_BYTES];

    while (f && out.n < max_bits) {
        const uint64_t want = std::min<uint64_t>(READ_BUF_BYTES, max_bits - out.n);
        f.read(buf, static_cast<std::streamsize>(want));
        const std::streamsize got = f.gcount();

        for (std::streamsize i = 0; i < got && out.n < max_bits; ++i) {
            if      (buf[i] == '0') { out.bits.push_back(0); ++out.n; }
            else if (buf[i] == '1') { out.bits.push_back(1); ++out.n; }
            // else: silently skip
        }
    }
}

// Unpack raw bytes into one uint8_t per bit (MSB-first).
void load_binary(std::ifstream& f, uint64_t max_bits, LoadResult& out)
{
    out.bits.reserve(max_bits);
    uint8_t buf[READ_BUF_BYTES];

    while (f && out.n < max_bits) {
        f.read(reinterpret_cast<char*>(buf), READ_BUF_BYTES);
        const std::streamsize bytes = f.gcount();

        for (std::streamsize b = 0; b < bytes && out.n < max_bits; ++b) {
            for (int bit = 7; bit >= 0 && out.n < max_bits; --bit) {
                out.bits.push_back((buf[b] >> bit) & 1);
                ++out.n;
            }
        }
    }
}

}  // namespace

FileFormat detect_format(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    uint8_t first = 0;
    f.read(reinterpret_cast<char*>(&first), 1);
    return (first == '0' || first == '1')
         ? FileFormat::ASCII_BITS
         : FileFormat::RAW_BINARY;
}

LoadResult load_bits(const std::string& path, uint64_t max_bits)
{
    const FileFormat fmt = detect_format(path);

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    f.seekg(0, std::ios::end);
    const uint64_t file_bytes = static_cast<uint64_t>(f.tellg());
    f.seekg(0, std::ios::beg);

    LoadResult result{};
    result.format = fmt;

    // Cap `max_bits` to the file's natural capacity.
    const uint64_t capacity = (fmt == FileFormat::ASCII_BITS)
                            ? file_bytes           // worst-case: every byte is a bit
                            : file_bytes * 8;      // 8 bits per byte
    const uint64_t target = (max_bits > 0 && max_bits < capacity) ? max_bits : capacity;

    if (fmt == FileFormat::ASCII_BITS) load_ascii (f, target, result);
    else                               load_binary(f, target, result);

    return result;
}
