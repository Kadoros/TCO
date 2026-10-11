#!/usr/bin/env python3
"""Overlay several tco_pde.py runs of the SAME PDE (regime 'full').

  python tco_compare.py runs/M2_kdv_lstsq runs/M2_kdv_mask ... --out runs/M2_compare

Writes compare.png (4 panels) and compare.md (table incl. input-tensor and OOD
metrics when present).
"""
import argparse
import json
import os

import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def load(path, regime):
    d = json.load(open(os.path.join(path, "results.json")))
    rows = sorted([r for r in d["results"] if r["regime"] == regime], key=lambda r: r["param"])
    return d["pde"], d["label"], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--regime", default="full")
    ap.add_argument("--out", default="runs/compare")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = [load(p, a.regime) + (p,) for p in a.runs]
    data = [x for x in data if x[2]]
    # disambiguate duplicate labels with the run folder name
    labs = [x[1] for x in data]
    data = [(pde, (f"{lab} [{os.path.basename(os.path.normpath(p))}]" if labs.count(lab) > 1 else lab), rows)
            for pde, lab, rows, p in data]
    pdes = {x[0] for x in data}
    if len(pdes) > 1:
        raise SystemExit(f"runs mix PDEs {pdes}; compare one PDE at a time")
    if len(data) > len(CAT):
        raise SystemExit(f"max {len(CAT)} runs per figure")
    pde = pdes.pop()
    m0 = data[0][2][0]["metrics"]
    has_L = "L_rel_err" in m0
    is_pod = "eig_purity_mean" in m0
    if is_pod:   # tco_pod.py results: show basis purity instead of closure
        panels = [("T_rel_err", "Triad tensor error (in Fourier coords)", True),
                  ("T_rel_err_on_rule", "Triad tensor error on selection rule", True),
                  ("pod_purity_mean", "Basis purity: POD modes (Fourier = 1)", False),
                  ("eig_purity_mean", "Purity of learned-A eigenbasis", False)]
    else:
        panels = [("T_rel_err", "Triad tensor error (all entries)", True),
                  ("T_rel_err_on_rule", "Triad tensor error on selection rule", True),
                  ("closure.nu_t_rel_err", "Eddy-viscosity spectrum error (test)", True),
                  ("L_rel_err" if has_L else "rollout_short_rel_err_median5",
                   "Input tensor L error" if has_L else "Short-horizon rollout error (median of 5)", True)]

    def get(m, key):
        if key.startswith("closure."):
            return m.get("closure", {}).get(key.split(".", 1)[1])
        return m.get(key)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3), facecolor=SURFACE)
    for ax, (key, title, logy) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(INK2); ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        for i, (_, label, rows) in enumerate(data):
            x = [r["param"] for r in rows]
            y = [get(r["metrics"], key) for r in rows]
            y = [max(v, 1e-16) if (v is not None and v != float("inf")) else float("nan") for v in y]
            ax.plot(x, y, color=CAT[i], lw=2, marker="o", ms=6, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=label)
        ax.set_xscale("log")
        finite = [v for line in ax.get_lines() for v in line.get_ydata() if np.isfinite(v) and v > 0]
        if logy and finite:
            ax.set_yscale("log")
        elif not logy:
            ax.set_ylim(0, 1.03)
        ax.set_xlabel(f"{pde} parameter", color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, labelcolor=INK2, loc="lower center",
               ncol=min(len(l), 4), bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f"{pde} ({a.regime} data)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.savefig(os.path.join(a.out, "compare.png"), dpi=160, facecolor=SURFACE, bbox_inches="tight")

    cols = ["T_rel_err", "T_rel_err_on_rule", "T_selection_mass", "closure.nu_t_rel_err",
            "test_rel_err_dzdt", "rollout_rel_err_final_median5", "rollout_short_rel_err_median5",
            "spectrum_rel_err_median5"]
    if has_L:
        cols += ["L_rel_err", "L_rel_err_on_rule", "ood_test_rel_err_dzdt", "ood_rollout_rel_err_final_median5"]
    if is_pod:
        cols = ["T_rel_err", "T_rel_err_on_rule", "T_selection_mass", "A_rel_err", "span_mismatch",
                "pod_purity_mean", "eig_purity_mean", "test_rel_err_dzdt", "rollout_short_rel_err_median5",
                "spectrum_rel_err_median5"]
    lines = ["| run | param | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for _, label, rows in data:
        for r in rows:
            vals = []
            for c in cols:
                v = get(r["metrics"], c)
                vals.append("-" if v is None else f"{v:.2e}")
            lines.append(f"| {label} | {r['param']} | " + " | ".join(vals) + " |")
    open(os.path.join(a.out, "compare.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print("saved", a.out)


if __name__ == "__main__":
    main()
