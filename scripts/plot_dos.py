"""Plot a density of states from a dos.csv written by save_dos().

Usage:
    python scripts/plot_dos.py runs/<run_name>/dos.csv [-o out.png] [--xlim EMIN EMAX]

The x axis is E - mu (mu = chemical potential / Fermi energy stored in the CSV).
If no output path is given, the plot is saved as dos.png next to the CSV.
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
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load_dos(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    energy = np.array([float(r[header.index("energy")]) for r in rows])
    dos = np.array([float(r[header.index("dos")]) for r in rows])
    mu = float(rows[0][header.index("mu")]) if "mu" in header else 0.0
    return energy - mu, dos


def main():
    parser = argparse.ArgumentParser(description="Plot dos.csv")
    parser.add_argument("csv", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--xlim", type=float, nargs=2, metavar=("EMIN", "EMAX"),
                        help="energy window relative to mu")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    args = parser.parse_args()

    e, dos = load_dos(args.csv)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(e, dos, color="k", lw=1.0)
    ax.fill_between(e, dos, where=e <= 0.0, color="0.8", lw=0)  # occupied states (T = 0)
    ax.axvline(0.0, color="k", lw=0.5, ls="--")

    ax.set_xlim(*(args.xlim or (e[0], e[-1])))
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$(E-\mu)/t$")
    ax.set_ylabel(r"DOS (states / $t$ / unit cell)")

    out = args.output or args.csv.with_name("dos.png")
    fig.savefig(out, dpi=300)
    print(f"Saved plot to {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
