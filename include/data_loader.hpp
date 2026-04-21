#pragma once
// File loader for binary bit sequences.
// Format is auto-detected: ASCII ('0'/'1' chars) vs raw binary (each byte = 8 bits, MSB-first).

#include <cstdint>
#include <string>
#include <vector>

enum class FileFormat {
    RAW_BINARY,
    ASCII_BITS,
};

struct LoadResult {
    std::vector<uint8_t> bits;   // each element is 0 or 1
    uint64_t             n;      // number of valid bits
    FileFormat           format;
};

FileFormat detect_format(const std::string& path);

// Loads up to `max_bits` from `path`. Pass 0 to load the entire file.
LoadResult load_bits(const std::string& path, uint64_t max_bits = 0);
