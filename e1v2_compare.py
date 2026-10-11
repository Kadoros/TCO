#!/usr/bin/env python3
"""Overlay several E1/E1v2 runs (regime 'full') across nu.

  python e1v2_compare.py runs/E1a_nu_sweep runs/V2a_stlsq ... --out runs/V2_compare

Writes compare_full.png (3 panels: tensor error, on-rule error, selection mass)
and compare_full.md (table). E1 (v1) results have no 'label' -> named 'lstsq (E1)'.
"""
import argparse
import json
import os

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# reference categorical palette, fixed order (never cycled)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def load(path):
    d = json.load(open(os.path.join(path, "results.json")))
    label = d.get("label") or "lstsq (E1)"
    rows = [r for r in d["results"] if r["regime"] == "full"]
    return label, sorted(rows, key=lambda r: r["nu"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="runs/V2_compare")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = [load(p) for p in a.runs]
    data = [(l, r) for l, r in data if r]
    if len(data) > len(CAT):
        raise SystemExit(f"max {len(CAT)} runs per figure (got {len(data)}); split into two calls")

    keys = [("rel_err_T", "Tensor error (all entries)", True),
            ("rel_err_T_on_rule", "Tensor error on selection rule p+q=k", True),
            ("selection_mass", "Fraction of |T| on selection rule", False)]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), facecolor=SURFACE)
    for ax, (key, title, logy) in zip(axes, keys):
        ax.set_facecolor(SURFACE)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(INK2); ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        for i, (label, rows) in enumerate(data):
            x = [r["nu"] for r in rows]
            y = [max(r["metrics"][key], 1e-16) for r in rows]
            ax.plot(x, y, color=CAT[i], lw=2, marker="o", ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1.03)
        ax.set_xlabel("viscosity nu (smaller = stronger truncation effect)", color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, labelcolor=INK2, loc="lower center",
               ncol=min(len(l), 4), bbox_to_anchor=(0.5, -0.12))
    fig.savefig(os.path.join(a.out, "compare_full.png"), dpi=160, facecolor=SURFACE, bbox_inches="tight")

    cols = ["rel_err_T", "rel_err_T_on_rule", "selection_mass", "rel_err_A_diag",
            "energy_violation", "test_rel_err_dzdt", "rollout_rel_err_final"]
    lines = ["| run | nu | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for label, rows in data:
        for r in rows:
            m = r["metrics"]
            lines.append(f"| {label} | {r['nu']} | " +
                         " | ".join(f"{m[c]:.2e}" if c in m else "-" for c in cols) + " |")
    open(os.path.join(a.out, "compare_full.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved", a.out)


if __name__ == "__main__":
    main()
