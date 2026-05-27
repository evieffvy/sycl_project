#!/usr/bin/env python3
"""Regenerate all data-dependent figures used in Thesis-lastone.docx.

Source data: latest benchmark.cpp output (kernel-only timing).
Output: docs/figures/*.png

Generated figures (filename ↔ in-thesis caption):
  fig3-2_monobit_perf.png        Figure 3-2 — Monobit (a) speedup bars, (b) throughput line
  fig4-2_blockfreq_perf.png      Figure 4-2 — Block Frequency (a) speedup bars, (b) throughput line
  fig5-7_throughput_linear.png   Figure 5-7 — Throughput vs Input Size (linear)
  fig5-2_speedup_comparison.png  Figure 5-8 — Speedup vs Input Size  [overwrites existing]
  fig5-3_time_compare.png        Figure 5-9 — Execution Time NIST vs SYCL (log)  [overwrites existing]
  fig5-4_speedup_trend.png       Figure 5-10 — Speedup scaling trend  [overwrites existing]
  fig5-5_throughput_loglog.png   Figure 5-11 — Throughput vs Input Size (log-log)  [overwrites existing]
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ------------------------------------------------------------------
# Canonical data — from `./build/benchmark` (kernel-only times)
# ------------------------------------------------------------------
N_BITS  = [1_000_000, 10_000_000, 50_000_000, 100_000_000]
N_LABEL = ['1 M', '10 M', '50 M', '100 M']

NIST_MS       = [19.579407, 136.741281, 676.716762, 1390.100334]
MONO_GPU_MS   = [0.148906,  0.218383,   0.571479,   1.052322]
BLOCK_GPU_MS  = [0.066654,  0.153620,   0.356922,   0.622099]
MONO_CPU_MS   = [0.301174,  1.861750,   7.705076,   14.705045]
BLOCK_CPU_MS  = [0.434392,  0.617269,   2.214528,   3.893991]

# Derived
MONO_GPU_SPD  = [NIST_MS[i] / MONO_GPU_MS[i]  for i in range(4)]
BLOCK_GPU_SPD = [NIST_MS[i] / BLOCK_GPU_MS[i] for i in range(4)]

# Throughput (Gbit/s) = n_bits / time_ms / 1e6
def gbps(time_ms_list):
    return [N_BITS[i] / time_ms_list[i] / 1e6 for i in range(4)]
# Throughput (Mbit/s)
def mbps(time_ms_list):
    return [N_BITS[i] / time_ms_list[i] / 1e3 for i in range(4)]

MONO_GPU_THR  = gbps(MONO_GPU_MS)    # Gbit/s
BLOCK_GPU_THR = gbps(BLOCK_GPU_MS)
MONO_GPU_MBPS  = mbps(MONO_GPU_MS)   # Mbit/s
BLOCK_GPU_MBPS = mbps(BLOCK_GPU_MS)

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

C_BLUE  = '#3B82F6'   # Monobit
C_GREEN = '#22A06B'   # Block Frequency (in 5-x figures it was green)
C_RED   = '#DC2626'   # Block Frequency (in 4-2 figure)
C_GRAY  = '#9CA3AF'   # Serial baseline

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')

def fmt_int(x, _pos):
    return f'{int(x):,}'

# ------------------------------------------------------------------
# Figure 3-2: Monobit (a) speedup bar (b) throughput line
# ------------------------------------------------------------------
def fig_3_2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(4)

    # (a) Speedup
    bars = ax1.bar(x, MONO_GPU_SPD, color=C_BLUE, width=0.55)
    ax1.set_title('(a) Speedup — Frequency (Monobit) Test', fontweight='bold')
    ax1.set_ylabel('Speedup over NIST STS 2.1.2 Serial (×)')
    ax1.set_xlabel('Input size (bits)')
    ax1.set_xticks(x); ax1.set_xticklabels(['1M','10M','50M','100M'])
    ax1.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax1.set_ylim(0, max(MONO_GPU_SPD) * 1.18)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    for b, v in zip(bars, MONO_GPU_SPD):
        ax1.text(b.get_x() + b.get_width()/2, v + max(MONO_GPU_SPD)*0.02,
                 f'{v:,.0f}×', ha='center', va='bottom', fontsize=10)

    # (b) Throughput
    ax2.plot(x, MONO_GPU_THR, 'o-', color=C_BLUE, linewidth=2.2, markersize=8)
    ax2.fill_between(x, 0, MONO_GPU_THR, color=C_BLUE, alpha=0.10)
    ax2.set_title('(b) Throughput — Frequency (Monobit) Test', fontweight='bold')
    ax2.set_ylabel('Throughput (Gbit/s)')
    ax2.set_xlabel('Input size (bits)')
    ax2.set_xticks(x); ax2.set_xticklabels(['1M','10M','50M','100M'])
    ax2.set_ylim(0, max(MONO_GPU_THR) * 1.18)
    ax2.grid(linestyle='--', alpha=0.4)
    for xi, v in zip(x, MONO_GPU_THR):
        ax2.annotate(f'{v:.1f}', xy=(xi, v), xytext=(0, 8),
                     textcoords='offset points', ha='center', fontsize=10, color=C_BLUE)
    save(fig, 'fig3-2_monobit_perf.png')

# ------------------------------------------------------------------
# Figure 4-2: Block Frequency (a) speedup bar (b) throughput line
# ------------------------------------------------------------------
def fig_4_2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(4)

    bars = ax1.bar(x, BLOCK_GPU_SPD, color=C_GREEN, width=0.55)
    ax1.set_title('(a) Speedup — Block Frequency Test', fontweight='bold')
    ax1.set_ylabel('Speedup over NIST STS 2.1.2 Serial (×)')
    ax1.set_xlabel('Input size (bits)')
    ax1.set_xticks(x); ax1.set_xticklabels(['1M','10M','50M','100M'])
    ax1.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax1.set_ylim(0, max(BLOCK_GPU_SPD) * 1.18)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    for b, v in zip(bars, BLOCK_GPU_SPD):
        ax1.text(b.get_x() + b.get_width()/2, v + max(BLOCK_GPU_SPD)*0.02,
                 f'{v:,.0f}×', ha='center', va='bottom', fontsize=10)

    ax2.plot(x, BLOCK_GPU_THR, 's-', color=C_GREEN, linewidth=2.2, markersize=8)
    ax2.fill_between(x, 0, BLOCK_GPU_THR, color=C_GREEN, alpha=0.10)
    ax2.set_title('(b) Throughput — Block Frequency Test', fontweight='bold')
    ax2.set_ylabel('Throughput (Gbit/s)')
    ax2.set_xlabel('Input size (bits)')
    ax2.set_xticks(x); ax2.set_xticklabels(['1M','10M','50M','100M'])
    ax2.set_ylim(0, max(BLOCK_GPU_THR) * 1.18)
    ax2.grid(linestyle='--', alpha=0.4)
    for xi, v in zip(x, BLOCK_GPU_THR):
        ax2.annotate(f'{v:.1f}', xy=(xi, v), xytext=(0, 8),
                     textcoords='offset points', ha='center', fontsize=10, color=C_GREEN)
    save(fig, 'fig4-2_blockfreq_perf.png')

# ------------------------------------------------------------------
# Figure 5-7: Throughput vs Input Size — linear (Mbit/s)
# ------------------------------------------------------------------
def fig_5_7():
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = np.arange(4)
    ax.plot(x, MONO_GPU_MBPS,  'o-', color=C_BLUE,  linewidth=2.4, markersize=9,
            label='Monobit Test')
    ax.plot(x, BLOCK_GPU_MBPS, 's-', color=C_RED,   linewidth=2.4, markersize=9,
            label='Block Frequency Test (M=1024)')
    ax.set_xticks(x); ax.set_xticklabels(N_LABEL)
    ax.set_xlabel('Input Size (bits)')
    ax.set_ylabel('Throughput (Mbit/s)')
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax.grid(linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', frameon=True)
    for xi, v in zip(x, MONO_GPU_MBPS):
        ax.annotate(f'{v:,.0f}', xy=(xi, v), xytext=(8, -4),
                    textcoords='offset points', fontsize=10, color=C_BLUE)
    for xi, v in zip(x, BLOCK_GPU_MBPS):
        ax.annotate(f'{v:,.0f}', xy=(xi, v), xytext=(8, 6),
                    textcoords='offset points', fontsize=10, color=C_RED)
    ax.set_ylim(0, max(BLOCK_GPU_MBPS) * 1.15)
    save(fig, 'fig5-7_throughput_linear.png')

# ------------------------------------------------------------------
# Figure 5-8 (filename keeps fig5-2_speedup_comparison.png)
# ------------------------------------------------------------------
def fig_5_8():
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(4)
    ax.plot(x, MONO_GPU_SPD,  'o-', color=C_BLUE,  linewidth=2.4, markersize=9,
            label='Frequency (Monobit) Test')
    ax.plot(x, BLOCK_GPU_SPD, 's-', color=C_GREEN, linewidth=2.4, markersize=9,
            label='Block Frequency Test (M=1024)')
    ax.set_title('GPU Speedup over Serial Baseline', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(['1M','10M','50M','100M'])
    ax.set_xlabel('Input size (bits)')
    ax.set_ylabel('Speedup over NIST STS 2.1.2 Serial (×)')
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax.grid(linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', frameon=True)
    ymax = max(BLOCK_GPU_SPD) * 1.18
    ax.set_ylim(0, ymax)
    for xi, v in zip(x, MONO_GPU_SPD):
        ax.annotate(f'{v:,.0f}×', xy=(xi, v), xytext=(8, -4),
                    textcoords='offset points', fontsize=9, color=C_BLUE)
    for xi, v in zip(x, BLOCK_GPU_SPD):
        ax.annotate(f'{v:,.0f}×', xy=(xi, v), xytext=(8, 6),
                    textcoords='offset points', fontsize=9, color=C_GREEN)
    save(fig, 'fig5-2_speedup_comparison.png')

# ------------------------------------------------------------------
# Figure 5-9 (filename keeps fig5-3_time_compare.png)
# ------------------------------------------------------------------
def fig_5_9():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    x = np.arange(4); w = 0.36

    fig.suptitle('Figure 5-9  Execution Time: NIST STS Serial vs. SYCL Parallel',
                 fontsize=14, y=1.02)

    for ax, title, sycl_ms, sycl_color in [
        (ax1, 'Monobit Test',                MONO_GPU_MS,  C_GREEN),
        (ax2, 'Block Frequency Test (M=1024)', BLOCK_GPU_MS, C_GREEN),
    ]:
        b1 = ax.bar(x - w/2, NIST_MS, width=w, color=C_GRAY, label='NIST STS 2.1.2 (Serial)')
        b2 = ax.bar(x + w/2, sycl_ms, width=w, color=sycl_color,  label='SYCL Parallel (RTX 4080S)')
        ax.set_yscale('log')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Input Size (bits)')
        ax.set_ylabel('Execution Time (ms, log scale)')
        ax.set_xticks(x); ax.set_xticklabels(N_LABEL)
        ax.grid(axis='y', which='both', linestyle='--', alpha=0.4)
        ax.legend(loc='upper left')
        for b, v in zip(b1, NIST_MS):
            ax.text(b.get_x() + b.get_width()/2, v * 1.18, f'{v:,.1f}',
                    ha='center', fontsize=9, color='#555')
        for b, v in zip(b2, sycl_ms):
            ax.text(b.get_x() + b.get_width()/2, v * 1.18, f'{v:.3f}',
                    ha='center', fontsize=9, color=sycl_color)
        ax.set_ylim(0.04, max(NIST_MS) * 4)
    save(fig, 'fig5-3_time_compare.png')

# ------------------------------------------------------------------
# Figure 5-10 (filename keeps fig5-4_speedup_trend.png)
# ------------------------------------------------------------------
def fig_5_10():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(4)
    ax.plot(x, MONO_GPU_SPD,  'o-', color=C_BLUE, linewidth=2.4, markersize=10, label='Monobit')
    ax.plot(x, BLOCK_GPU_SPD, 's-', color=C_RED,  linewidth=2.4, markersize=10, label='Block Frequency (M=1024)')
    ax.set_title("Figure 5-10  Speedup Scaling Trend (Amdahl's-law behaviour)", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(N_LABEL)
    ax.set_xlabel('Input Size (bits)')
    ax.set_ylabel('Speedup (×)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{int(v):,}×'))
    ax.grid(linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', frameon=True)
    ax.set_ylim(0, max(BLOCK_GPU_SPD) * 1.15)
    for xi, v in zip(x, MONO_GPU_SPD):
        ax.annotate(f'{v:,.1f}×', xy=(xi, v), xytext=(8, -4),
                    textcoords='offset points', fontsize=10, color=C_BLUE)
    for xi, v in zip(x, BLOCK_GPU_SPD):
        ax.annotate(f'{v:,.1f}×', xy=(xi, v), xytext=(8, 6),
                    textcoords='offset points', fontsize=10, color=C_RED)
    save(fig, 'fig5-4_speedup_trend.png')

# ------------------------------------------------------------------
# Figure 5-11 (filename keeps fig5-5_throughput_loglog.png)
# ------------------------------------------------------------------
def fig_5_11():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(N_BITS, MONO_GPU_MBPS,  'o-', color=C_BLUE, linewidth=2.4, markersize=10,
            label='Monobit Test')
    ax.plot(N_BITS, BLOCK_GPU_MBPS, 's-', color=C_RED,  linewidth=2.4, markersize=10,
            label='Block Frequency Test (M=1024)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title('Figure 5-11  Throughput vs. Input Size (log-log)', fontsize=13)
    ax.set_xlabel('Input Size (bits)')
    ax.set_ylabel('Throughput (Mbit/s, log scale)')
    ax.grid(which='both', linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', frameon=True)
    ax.set_xticks(N_BITS); ax.set_xticklabels(N_LABEL)
    for xi, v in zip(N_BITS, MONO_GPU_MBPS):
        ax.annotate(f'{v:,.0f}', xy=(xi, v), xytext=(8, -10),
                    textcoords='offset points', fontsize=10, color=C_BLUE)
    for xi, v in zip(N_BITS, BLOCK_GPU_MBPS):
        ax.annotate(f'{v:,.0f}', xy=(xi, v), xytext=(8, 6),
                    textcoords='offset points', fontsize=10, color=C_RED)
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
