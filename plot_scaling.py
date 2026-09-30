#!/usr/bin/env python3
"""plot_scaling.py - Exercise 4 graphs.

Reads sum_scaling.csv and pi_scaling.csv (written by run_scaling.sh) and
produces four PNG files:
    sum_time.png, sum_speedup.png, pi_time.png, pi_speedup.png

Usage: python3 plot_scaling.py
Needs: pip3 install matplotlib
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    procs, times, speedups = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            procs.append(int(row["processes"]))
            times.append(float(row["time"]))
            speedups.append(float(row["speedup"]))
    return procs, times, speedups


def plot(procs, values, title, ylabel, outfile, ideal=False):
    plt.figure(figsize=(7, 4.5))
    plt.plot(procs, values, marker="o", linewidth=2, label="measured")
    if ideal:
        plt.plot(procs, procs, linestyle="--", color="gray", label="ideal (linear)")
    plt.title(title)
    plt.xlabel("Number of processes")
    plt.ylabel(ylabel)
    plt.xticks(procs, [str(p) for p in procs])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print("wrote", outfile)


for csv_file, tag, name in [
    ("sum_scaling.csv", "sum", "Sum 1..10,000,000"),
    ("pi_scaling.csv", "pi", "Monte Carlo Pi, 10,000,000 darts"),
]:
    if not os.path.exists(csv_file):
        print("skipping", csv_file, "(not found - run ./run_scaling.sh first)")
        continue
    procs, times, speedups = load(csv_file)
    plot(procs, times, f"{name}: Time vs Number of Processors",
         "Time (s)", f"{tag}_time.png")
    plot(procs, speedups, f"{name}: Speedup",
         "Speedup (T1 / Tp)", f"{tag}_speedup.png", ideal=True)
