cmake_minimum_required(VERSION 3.20)
project(nist_sycl VERSION 1.0 LANGUAGES CXX)

# ── C++ standard ───────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── SYCL compiler & target backends ────────────────────────────
#
# Intel oneAPI:
#     source /opt/intel/oneapi/setvars.sh
#     cmake -DCMAKE_CXX_COMPILER=icpx -B build
#
# AdaptiveCpp (open source SYCL):
#     cmake -DCMAKE_CXX_COMPILER=acpp -B build
#
# Change `-fsycl-targets` to match your hardware:
#     spir64                – Intel CPU/GPU (generic SPIR-V)
#     spir64_x86_64         – Intel CPU only
#     nvptx64-nvidia-cuda   – NVIDIA GPU (Codeplay plugin)
#     amdgcn-amd-amdhsa     – AMD GPU
set(SYCL_FLAGS
    -fsycl
    -fsycl-targets=nvptx64-nvidia-cuda
)

# ── Common options for all targets ─────────────────────────────
function(configure_sycl_target target)
    target_include_directories(${target} PRIVATE include)
    target_compile_options    (${target} PRIVATE ${SYCL_FLAGS} -O3)
    target_link_options       (${target} PRIVATE ${SYCL_FLAGS})
    target_link_libraries     (${target} PRIVATE m)
endfunction()

# ── Main executable ────────────────────────────────────────────
add_executable(nist_sycl
    src/main.cpp
    src/frequency_monobit.cpp
    src/frequency_block.cpp
    src/math_utils.cpp
    src/data_loader.cpp
)
configure_sycl_target(nist_sycl)

# ── Unit tests ─────────────────────────────────────────────────
add_executable(test_nist
    tests/test_nist.cpp
    src/frequency_monobit.cpp
    src/frequency_block.cpp
    src/math_utils.cpp
)
configure_sycl_target(test_nist)

# ── Benchmark vs. NIST STS 2.1.2 ───────────────────────────────
add_executable(benchmark
    src/benchmark.cpp
    src/frequency_monobit.cpp
    src/frequency_block.cpp
    src/math_utils.cpp
    src/data_loader.cpp
)
configure_sycl_target(benchmark)

message(STATUS "Build configured. Example invocation:")
message(STATUS "  source /opt/intel/oneapi/setvars.sh")
message(STATUS "  cmake -DCMAKE_CXX_COMPILER=icpx -B build && cmake --build build")
