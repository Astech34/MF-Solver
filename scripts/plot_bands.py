"""Plot a band structure from a bands.csv written by save_bands().

Usage:
    python scripts/plot_bands.py runs/<run_name>/bands.csv [-o out.png]

If no output path is given, the plot is saved as bands.png next to the CSV.
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Phys. Rev. B style: single-column width, serif fonts, inward ticks on all sides.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "ytick.minor.visible": True,
    "ytick.minor.width": 0.4,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load_bands(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    i_coord = header.index("path_coord")
    i_label = header.index("label")
    band_cols = [i for i, h in enumerate(header) if h.startswith("band_")]

    k = np.array([float(r[i_coord]) for r in rows])
    bands = np.array([[float(r[i]) for i in band_cols] for r in rows])
    ticks = [(float(r[i_coord]), r[i_label]) for r in rows if r[i_label]]
    # Older CSVs have no mu column; fall back to 0 (plots E rather than E - mu)
    mu = float(rows[0][header.index("mu")]) if "mu" in header else 0.0
    return k, bands - mu, ticks


def main():
    parser = argparse.ArgumentParser(description="Plot bands.csv")
    parser.add_argument("csv", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    args = parser.parse_args()

    k, bands, ticks = load_bands(args.csv)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(k, bands, color="k", lw=1.0)

    tick_pos = [p for p, _ in ticks]
    for p in tick_pos[1:-1]:
        ax.axvline(p, color="k", lw=0.5)
    ax.set_xticks(tick_pos, [lbl for _, lbl in ticks])
    ax.tick_params(axis="x", length=0)

    ax.set_xlim(k[0], k[-1])
    ax.axhline(0.0, color="k", lw=0.5, ls="--")
    ax.set_ylabel(r"$(E-\mu)/t$")

    out = args.output or args.csv.with_name("bands.png")
    fig.savefig(out, dpi=300)
    print(f"Saved plot to {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
