# SYCL-based Parallel Frequency Tests for Randomness Assessment

Parallel implementation of NIST SP 800-22 frequency randomness tests using SYCL/DPC++, targeting CPU and GPU via Intel oneAPI or AdaptiveCpp.

## Tests Implemented

| Test | NIST SP 800-22 | Description |
|------|---------------|-------------|
| Frequency (Monobit) | Section 2.1 | Checks proportion of 1s and 0s in the sequence |
| Frequency within a Block | Section 2.2 | Checks proportion of 1s within fixed-size blocks |

## Requirements

- **Compiler**: Intel DPC++ (`icpx`) or AdaptiveCpp (`acpp`)
- **Runtime**: Intel oneAPI Base Toolkit or AdaptiveCpp
- **Hardware**: Intel CPU/GPU, NVIDIA GPU (via Codeplay plugin), or AMD GPU
- **CMake**: 3.20+
- **C++ Standard**: C++17

## Build

```bash
# Intel oneAPI
source /opt/intel/oneapi/setvars.sh
cmake -DCMAKE_CXX_COMPILER=icpx -B build
cmake --build build

# AdaptiveCpp
cmake -DCMAKE_CXX_COMPILER=acpp -B build
cmake --build build
```

To target different hardware, edit `SYCL_FLAGS` in `CMakeLists.txt`:
```
spir64               # Intel CPU/GPU (generic)
nvptx64-nvidia-cuda  # NVIDIA GPU
amdgcn-amd-amdhsa    # AMD GPU
```

## Usage

```bash
./build/nist_sycl [OPTIONS] <input_file>

Options:
  -n <bits>   Number of bits to test (default: all)
  -M <int>    Block size for block-frequency test (default: 1024)
  -d cpu|gpu  Force device (default: prefer GPU)
  -v          Verbose output with timing and device info
  -csv        Output results in CSV format
  -h          Show help
```

### Input Formats

- **Raw binary**: each byte = 8 bits (MSB-first)
- **ASCII bits**: each character is `'0'` or `'1'`

### Example

```bash
# Run both tests on a binary file
./build/nist_sycl -v random.bin

# Run with custom block size on GPU
./build/nist_sycl -d gpu -M 2048 random.bin

# Export results as CSV
./build/nist_sycl -csv random.bin > results.csv
```

### Sample Output

```
┌─────────────────────────────────────────────────┐
│  Frequency Test (Monobit)                       │
├─────────────────────────────────────────────────┤
│  Bits (n)   : 1000000                           │
│  P-value    : 0.318435                          │
│  Result     : PASS                              │
└─────────────────────────────────────────────────┘
```

> NIST recommendation: n ≥ 100, M ≥ 20, N = n/M ≥ 100

## Project Structure

```
.
├── CMakeLists.txt
├── include/
│   ├── nist_tests.hpp      # TestResult struct, constants
│   ├── data_loader.hpp     # Bit file loader interface
│   └── math_utils.hpp      # igamc (incomplete gamma)
├── src/
│   ├── main.cpp            # CLI entry point, SYCL device setup
│   ├── frequency_monobit.cpp  # Monobit test kernel
│   ├── frequency_block.cpp    # Block frequency test kernels
│   ├── data_loader.cpp     # Binary/ASCII bit file loading
│   ├── math_utils.cpp      # Math utilities
│   └── benchmark.cpp       # Benchmarking tool
└── tests/
    └── test_nist.cpp       # Unit tests
```

## References

- [NIST SP 800-22 Rev. 1a](https://csrc.nist.gov/publications/detail/sp/800-22/rev-1a/final) — Statistical Test Suite for Random Bit Generators
- [Intel oneAPI DPC++ Documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)
- [SYCL 2020 Specification](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html)
