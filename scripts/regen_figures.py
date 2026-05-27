#!/usr/bin/env python3
"""Regenerate all data-dependent figures used in Thesis-lastone.docx.

Data source: latest ./build/benchmark output (kernel-only times).

All figures now show **both SYCL CPU and SYCL GPU** so the thesis
does not read as a GPU-only project.

  fig3-2_monobit_perf.png       Figure 3-2 — Monobit (a) speedup bars CPU+GPU, (b) throughput lines CPU+GPU
  fig4-2_blockfreq_perf.png     Figure 4-2 — Block Frequency (a) speedup bars CPU+GPU, (b) throughput lines CPU+GPU
  fig5-7_throughput_linear.png  Figure 5-7 — Throughput vs Input Size (linear) — 4 lines (Mono/Block, CPU/GPU)
  fig5-2_speedup_comparison.png Figure 5-8 — Speedup vs Input Size — 4 lines
  fig5-3_time_compare.png       Figure 5-9 — Execution Time NIST vs SYCL CPU vs SYCL GPU (log, 3-way bars)
  fig5-4_speedup_trend.png      Figure 5-10 — Speedup scaling — 4 lines
  fig5-5_throughput_loglog.png  Figure 5-11 — Throughput log-log — 4 lines
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ------------------------------------------------------------------
# Canonical data — single ./build/benchmark run on
#   GPU : NVIDIA RTX 4080 SUPER (Codeplay oneAPI for NVIDIA 2024.0)
#   CPU : 13th Gen Intel Core i9-13900KF (Intel OpenCL 3.0)
# ------------------------------------------------------------------
N_BITS  = [1_000_000, 10_000_000, 50_000_000, 100_000_000]
N_LABEL = ['1 M', '10 M', '50 M', '100 M']

NIST_MS       = [19.579407, 136.741281, 676.716762, 1390.100334]
MONO_GPU_MS   = [0.148906,  0.218383,   0.571479,   1.052322]
BLOCK_GPU_MS  = [0.066654,  0.153620,   0.356922,   0.622099]
MONO_CPU_MS   = [0.301174,  1.861750,   7.705076,   14.705045]
BLOCK_CPU_MS  = [0.434392,  0.617269,   2.214528,   3.893991]

def speedup(times):
    return [NIST_MS[i] / times[i] for i in range(4)]
def gbps(times):
    return [N_BITS[i] / times[i] / 1e6 for i in range(4)]
def mbps(times):
    return [N_BITS[i] / times[i] / 1e3 for i in range(4)]

MONO_GPU_SPD,  BLOCK_GPU_SPD  = speedup(MONO_GPU_MS),  speedup(BLOCK_GPU_MS)
MONO_CPU_SPD,  BLOCK_CPU_SPD  = speedup(MONO_CPU_MS),  speedup(BLOCK_CPU_MS)
MONO_GPU_THR,  BLOCK_GPU_THR  = gbps(MONO_GPU_MS),     gbps(BLOCK_GPU_MS)
MONO_CPU_THR,  BLOCK_CPU_THR  = gbps(MONO_CPU_MS),     gbps(BLOCK_CPU_MS)
MONO_GPU_MBPS, BLOCK_GPU_MBPS = mbps(MONO_GPU_MS),     mbps(BLOCK_GPU_MS)
MONO_CPU_MBPS, BLOCK_CPU_MBPS = mbps(MONO_CPU_MS),     mbps(BLOCK_CPU_MS)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Style
# ------------------------------------------------------------------
plt.rcParams.update({
    'font.size':       11,
    'axes.titlesize':  13,
    'axes.labelsize':  11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi':      130,
    'savefig.dpi':     150,
    'savefig.bbox':    'tight',
})

# Colors
C_GPU_MONO  = '#3B82F6'  # blue
C_CPU_MONO  = '#93C5FD'  # light blue
C_GPU_BLOCK = '#16A34A'  # green
C_CPU_BLOCK = '#86EFAC'  # light green
C_GPU_RED   = '#DC2626'  # used in 4-2 for Block
C_CPU_RED   = '#FCA5A5'
C_GRAY      = '#9CA3AF'  # NIST serial baseline

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')

def fmt_int(x, _pos):
    return f'{int(x):,}'

# ==================================================================
# Figure 3-2: Monobit performance — (a) grouped speedup bars CPU+GPU
#                                  (b) two throughput lines CPU+GPU
# ==================================================================
def fig_3_2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(4); w = 0.38

    # (a) Grouped speedup bars: GPU + CPU
    b1 = ax1.bar(x - w/2, MONO_GPU_SPD, width=w, color=C_GPU_MONO, label='SYCL GPU (RTX 4080S)')
    b2 = ax1.bar(x + w/2, MONO_CPU_SPD, width=w, color=C_CPU_MONO, label='SYCL CPU (i9-13900KF)')
    ax1.set_title('(a) Speedup — Frequency (Monobit) Test', fontweight='bold')
    ax1.set_ylabel('Speedup over NIST STS 2.1.2 Serial (×)')
    ax1.set_xlabel('Input size (bits)')
    ax1.set_xticks(x); ax1.set_xticklabels(['1M','10M','50M','100M'])
    ax1.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax1.set_ylim(0, max(MONO_GPU_SPD) * 1.25)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left', frameon=True, fontsize=9)
    for b, v in zip(b1, MONO_GPU_SPD):
        ax1.text(b.get_x() + b.get_width()/2, v + max(MONO_GPU_SPD)*0.02,
                 f'{v:,.0f}×', ha='center', va='bottom', fontsize=8.5, color=C_GPU_MONO)
    for b, v in zip(b2, MONO_CPU_SPD):
        ax1.text(b.get_x() + b.get_width()/2, v + max(MONO_GPU_SPD)*0.02,
                 f'{v:.0f}×', ha='center', va='bottom', fontsize=8.5, color='#1D4ED8')

    # (b) Throughput lines: GPU + CPU
    ax2.plot(x, MONO_GPU_THR, 'o-', color=C_GPU_MONO, linewidth=2.2, markersize=8, label='SYCL GPU')
    ax2.plot(x, MONO_CPU_THR, 's--', color='#1D4ED8', linewidth=2, markersize=7, label='SYCL CPU')
    ax2.set_title('(b) Throughput — Frequency (Monobit) Test', fontweight='bold')
    ax2.set_ylabel('Throughput (Gbit/s)')
    ax2.set_xlabel('Input size (bits)')
    ax2.set_xticks(x); ax2.set_xticklabels(['1M','10M','50M','100M'])
    ax2.set_ylim(0, max(MONO_GPU_THR) * 1.18)
    ax2.grid(linestyle='--', alpha=0.4)
    ax2.legend(loc='upper left', frameon=True, fontsize=9)
    for xi, v in zip(x, MONO_GPU_THR):
        ax2.annotate(f'{v:.1f}', xy=(xi, v), xytext=(0, 8),
                     textcoords='offset points', ha='center', fontsize=9, color=C_GPU_MONO)
    for xi, v in zip(x, MONO_CPU_THR):
        ax2.annotate(f'{v:.1f}', xy=(xi, v), xytext=(0, -14),
                     textcoords='offset points', ha='center', fontsize=9, color='#1D4ED8')
    save(fig, 'fig3-2_monobit_perf.png')

# ==================================================================
# Figure 4-2: Block Frequency performance — same layout as 3-2 (green)
# ==================================================================
def fig_4_2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(4); w = 0.38

    b1 = ax1.bar(x - w/2, BLOCK_GPU_SPD, width=w, color=C_GPU_BLOCK, label='SYCL GPU (RTX 4080S)')
    b2 = ax1.bar(x + w/2, BLOCK_CPU_SPD, width=w, color=C_CPU_BLOCK, label='SYCL CPU (i9-13900KF)')
    ax1.set_title('(a) Speedup — Block Frequency Test', fontweight='bold')
    ax1.set_ylabel('Speedup over NIST STS 2.1.2 Serial (×)')
    ax1.set_xlabel('Input size (bits)')
    ax1.set_xticks(x); ax1.set_xticklabels(['1M','10M','50M','100M'])
    ax1.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax1.set_ylim(0, max(BLOCK_GPU_SPD) * 1.25)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left', frameon=True, fontsize=9)
    for b, v in zip(b1, BLOCK_GPU_SPD):
        ax1.text(b.get_x() + b.get_width()/2, v + max(BLOCK_GPU_SPD)*0.02,
                 f'{v:,.0f}×', ha='center', va='bottom', fontsize=8.5, color=C_GPU_BLOCK)
    for b, v in zip(b2, BLOCK_CPU_SPD):
        ax1.text(b.get_x() + b.get_width()/2, v + max(BLOCK_GPU_SPD)*0.02,
                 f'{v:.0f}×', ha='center', va='bottom', fontsize=8.5, color='#15803D')

    ax2.plot(x, BLOCK_GPU_THR, 's-',  color=C_GPU_BLOCK, linewidth=2.2, markersize=8, label='SYCL GPU')
    ax2.plot(x, BLOCK_CPU_THR, 'D--', color='#15803D',   linewidth=2,   markersize=7, label='SYCL CPU')
    ax2.set_title('(b) Throughput — Block Frequency Test', fontweight='bold')
    ax2.set_ylabel('Throughput (Gbit/s)')
    ax2.set_xlabel('Input size (bits)')
    ax2.set_xticks(x); ax2.set_xticklabels(['1M','10M','50M','100M'])
    ax2.set_ylim(0, max(BLOCK_GPU_THR) * 1.18)
    ax2.grid(linestyle='--', alpha=0.4)
    ax2.legend(loc='upper left', frameon=True, fontsize=9)
    for xi, v in zip(x, BLOCK_GPU_THR):
        ax2.annotate(f'{v:.1f}', xy=(xi, v), xytext=(0, 8),
                     textcoords='offset points', ha='center', fontsize=9, color=C_GPU_BLOCK)
    for xi, v in zip(x, BLOCK_CPU_THR):
        ax2.annotate(f'{v:.1f}', xy=(xi, v), xytext=(0, -14),
                     textcoords='offset points', ha='center', fontsize=9, color='#15803D')
    save(fig, 'fig4-2_blockfreq_perf.png')

# ==================================================================
# Figure 5-7: Throughput vs Input Size — linear (Mbit/s), 4 lines
# ==================================================================
def fig_5_7():
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(4)
    ax.plot(x, BLOCK_GPU_MBPS, 's-',  color=C_GPU_RED,   linewidth=2.4, markersize=9, label='Block Frequency — SYCL GPU')
    ax.plot(x, MONO_GPU_MBPS,  'o-',  color=C_GPU_MONO,  linewidth=2.4, markersize=9, label='Monobit — SYCL GPU')
    ax.plot(x, BLOCK_CPU_MBPS, 'D--', color=C_CPU_RED,   linewidth=2,   markersize=7, label='Block Frequency — SYCL CPU')
    ax.plot(x, MONO_CPU_MBPS,  '^--', color=C_CPU_MONO,  linewidth=2,   markersize=7, label='Monobit — SYCL CPU')
    ax.set_xticks(x); ax.set_xticklabels(N_LABEL)
    ax.set_xlabel('Input Size (bits)')
    ax.set_ylabel('Throughput (Mbit/s)')
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax.grid(linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', frameon=True, fontsize=9.5)
    ax.set_ylim(0, max(BLOCK_GPU_MBPS) * 1.15)
    for xi, v in zip(x, BLOCK_GPU_MBPS):
        ax.annotate(f'{v:,.0f}', xy=(xi, v), xytext=(8, 6),
                    textcoords='offset points', fontsize=9, color=C_GPU_RED)
    for xi, v in zip(x, MONO_GPU_MBPS):
        ax.annotate(f'{v:,.0f}', xy=(xi, v), xytext=(8, -4),
                    textcoords='offset points', fontsize=9, color=C_GPU_MONO)
    save(fig, 'fig5-7_throughput_linear.png')

# ==================================================================
# Figure 5-8 (filename keeps fig5-2_speedup_comparison.png): 4 lines
# ==================================================================
def fig_5_8():
    fig, ax = plt.subplots(figsize=(9, 5.3))
    x = np.arange(4)
    ax.plot(x, BLOCK_GPU_SPD, 's-',  color=C_GPU_BLOCK, linewidth=2.4, markersize=9, label='Block Frequency — SYCL GPU')
    ax.plot(x, MONO_GPU_SPD,  'o-',  color=C_GPU_MONO,  linewidth=2.4, markersize=9, label='Monobit — SYCL GPU')
    ax.plot(x, BLOCK_CPU_SPD, 'D--', color='#15803D',   linewidth=2,   markersize=7, label='Block Frequency — SYCL CPU')
    ax.plot(x, MONO_CPU_SPD,  '^--', color='#1D4ED8',   linewidth=2,   markersize=7, label='Monobit — SYCL CPU')
    ax.set_title('Speedup over Serial Baseline — SYCL GPU vs SYCL CPU', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(['1M','10M','50M','100M'])
    ax.set_xlabel('Input size (bits)')
    ax.set_ylabel('Speedup over NIST STS 2.1.2 Serial (×)')
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax.grid(linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', frameon=True, fontsize=9.5)
    ax.set_ylim(0, max(BLOCK_GPU_SPD) * 1.18)
    for xi, v in zip(x, BLOCK_GPU_SPD):
        ax.annotate(f'{v:,.0f}×', xy=(xi, v), xytext=(8, 6),
                    textcoords='offset points', fontsize=9, color=C_GPU_BLOCK)
    for xi, v in zip(x, MONO_GPU_SPD):
        ax.annotate(f'{v:,.0f}×', xy=(xi, v), xytext=(8, -4),
                    textcoords='offset points', fontsize=9, color=C_GPU_MONO)
    save(fig, 'fig5-2_speedup_comparison.png')

# ==================================================================
# Figure 5-9 (filename keeps fig5-3_time_compare.png):
#   3-way grouped bars NIST | SYCL CPU | SYCL GPU on log axis
# ==================================================================
def fig_5_9():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
    x = np.arange(4); w = 0.26

    fig.suptitle('Figure 5-9  Execution Time: NIST STS Serial vs. SYCL CPU vs. SYCL GPU',
                 fontsize=14, y=1.02)

    for ax, title, cpu_ms, gpu_ms in [
        (ax1, 'Monobit Test',                 MONO_CPU_MS,  MONO_GPU_MS),
        (ax2, 'Block Frequency Test (M=1024)', BLOCK_CPU_MS, BLOCK_GPU_MS),
    ]:
        b0 = ax.bar(x - w, NIST_MS, width=w, color=C_GRAY,       label='NIST STS 2.1.2 (Serial)')
        b1 = ax.bar(x,     cpu_ms,  width=w, color=C_CPU_MONO,   label='SYCL CPU (i9-13900KF)')
        b2 = ax.bar(x + w, gpu_ms,  width=w, color=C_GPU_BLOCK,  label='SYCL GPU (RTX 4080S)')
        ax.set_yscale('log')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Input Size (bits)')
        ax.set_ylabel('Execution Time (ms, log scale)')
        ax.set_xticks(x); ax.set_xticklabels(N_LABEL)
        ax.grid(axis='y', which='both', linestyle='--', alpha=0.4)
        ax.legend(loc='upper left', fontsize=9)
        for b, v in zip(b0, NIST_MS):
            ax.text(b.get_x() + b.get_width()/2, v * 1.18, f'{v:,.1f}',
                    ha='center', fontsize=8, color='#555')
        for b, v in zip(b1, cpu_ms):
            ax.text(b.get_x() + b.get_width()/2, v * 1.18, f'{v:.2f}',
                    ha='center', fontsize=8, color='#1D4ED8')
        for b, v in zip(b2, gpu_ms):
            ax.text(b.get_x() + b.get_width()/2, v * 1.18, f'{v:.3f}',
                    ha='center', fontsize=8, color=C_GPU_BLOCK)
        ax.set_ylim(0.04, max(NIST_MS) * 6)
    save(fig, 'fig5-3_time_compare.png')

# ==================================================================
# Figure 5-10 (filename keeps fig5-4_speedup_trend.png): 4 lines
# ==================================================================
def fig_5_10():
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(4)
    ax.plot(x, BLOCK_GPU_SPD, 's-',  color=C_GPU_RED,   linewidth=2.4, markersize=9, label='Block Frequency — SYCL GPU')
    ax.plot(x, MONO_GPU_SPD,  'o-',  color=C_GPU_MONO,  linewidth=2.4, markersize=9, label='Monobit — SYCL GPU')
    ax.plot(x, BLOCK_CPU_SPD, 'D--', color=C_CPU_RED,   linewidth=2,   markersize=7, label='Block Frequency — SYCL CPU')
    ax.plot(x, MONO_CPU_SPD,  '^--', color=C_CPU_MONO,  linewidth=2,   markersize=7, label='Monobit — SYCL CPU')
    ax.set_title("Figure 5-10  Speedup Scaling Trend (Amdahl's-law behaviour) — CPU & GPU", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(N_LABEL)
    ax.set_xlabel('Input Size (bits)')
    ax.set_ylabel('Speedup (×)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{int(v):,}×'))
    ax.grid(linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', fontsize=9.5)
    ax.set_ylim(0, max(BLOCK_GPU_SPD) * 1.18)
    for xi, v in zip(x, BLOCK_GPU_SPD):
        ax.annotate(f'{v:,.1f}×', xy=(xi, v), xytext=(8, 6),
                    textcoords='offset points', fontsize=9, color=C_GPU_RED)
    for xi, v in zip(x, MONO_GPU_SPD):
        ax.annotate(f'{v:,.1f}×', xy=(xi, v), xytext=(8, -4),
                    textcoords='offset points', fontsize=9, color=C_GPU_MONO)
    save(fig, 'fig5-4_speedup_trend.png')

# ==================================================================
# Figure 5-11 (filename keeps fig5-5_throughput_loglog.png): 4 lines
# ==================================================================
def fig_5_11():
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(N_BITS, BLOCK_GPU_MBPS, 's-',  color=C_GPU_RED,  linewidth=2.4, markersize=9, label='Block Frequency — SYCL GPU')
    ax.plot(N_BITS, MONO_GPU_MBPS,  'o-',  color=C_GPU_MONO, linewidth=2.4, markersize=9, label='Monobit — SYCL GPU')
    ax.plot(N_BITS, BLOCK_CPU_MBPS, 'D--', color=C_CPU_RED,  linewidth=2,   markersize=7, label='Block Frequency — SYCL CPU')
    ax.plot(N_BITS, MONO_CPU_MBPS,  '^--', color=C_CPU_MONO, linewidth=2,   markersize=7, label='Monobit — SYCL CPU')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title('Figure 5-11  Throughput vs. Input Size (log-log) — CPU & GPU', fontsize=13)
    ax.set_xlabel('Input Size (bits)')
    ax.set_ylabel('Throughput (Mbit/s, log scale)')
    ax.grid(which='both', linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', fontsize=9.5)
    ax.set_xticks(N_BITS); ax.set_xticklabels(N_LABEL)
    for xi, v in zip(N_BITS, BLOCK_GPU_MBPS):
        ax.annotate(f'{v:,.0f}', xy=(xi, v), xytext=(8, 6),
                    textcoords='offset points', fontsize=9, color=C_GPU_RED)
    for xi, v in zip(N_BITS, MONO_GPU_MBPS):
        ax.annotate(f'{v:,.0f}', xy=(xi, v), xytext=(8, -10),
                    textcoords='offset points', fontsize=9, color=C_GPU_MONO)
    save(fig, 'fig5-5_throughput_loglog.png')

# ------------------------------------------------------------------
if __name__ == '__main__':
    fig_3_2()
    fig_4_2()
    fig_5_7()
    fig_5_8()
    fig_5_9()
    fig_5_10()
    fig_5_11()
    print('\nDone. Open docs/figures/*.png to inspect.')
