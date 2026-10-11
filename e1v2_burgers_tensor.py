#!/usr/bin/env python3
"""
TCO E1 v2 — Can we keep the learned triad tensor clean when the data come from
a truncated (closure-contaminated) system?

Model on the first K Fourier modes (z = [c_{-K}..c_K], complex):
    dz_k/dt = sum_j A_kj z_j                      (linear)
            + sum_{p<=q}     T_k,pq  z_p z_q       (quadratic = triad tensor, interpreted)
            + sum_{p<=q<=r}  C_k,pqr z_p z_q z_r   (optional cubic CLOSURE channel)

Why a cubic closure: unresolved modes are (to leading order) slaved quadratically
to resolved ones, so their feedback through the triads is ~cubic in z. Giving it
its own channel should stop it from leaking into T.

Options (combine freely):
  --method stlsq        sparse regression with BLOCK-WISE thresholds
                        (linear / quadratic / cubic thresholded separately; fixes
                        E1f where small viscous terms were cut at small nu)
  --energy_constraint   exact linear constraint: the quadratic part conserves
                        energy for every real field, Re sum_k conj(z_k) T_k(z,z) = 0
                        (true Burgers triads satisfy this). Solved by KKT projection.
  --closure cubic       add the cubic channel C
  --mask_selection      symmetry PRIOR (translation invariance): only allow
                        A_kk, T_k,pq with p+q=k, C_k,pqr with p+q+r=k.
                        Not discovery any more — an upper-bound reference.

Needs e1_burgers_tensor.py in the same folder (solver, data, true tensor).

Outputs in --out: results.json, log.txt, fig_tensor_k{k}.png (if one nu), and
per-nu metrics. Use e1v2_compare.py to overlay several runs.
"""
import argparse
import json
import os
import time

import numpy as np

from e1_burgers_tensor import (simulate, true_operators, pair_index,
                               tensor_slice, SURFACE, INK, INK2, SEQ)


# ----------------------------------------------------------------------------
# Feature blocks
# ----------------------------------------------------------------------------
def triple_index(K):
    n = 2 * K + 1
    idx = [(a, b, c) for a in range(n) for b in range(a, n) for c in range(b, n)]
    if not idx:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
    a, b, c = map(np.array, zip(*idx))
    return a, b, c


class Blocks:
    def __init__(self, K, closure):
        self.K = K
        self.n = 2 * K + 1
        self.modes, self.P, self.Q = pair_index(K)
        self.closure = closure
        if closure == "cubic":
            self.A3, self.B3, self.C3 = triple_index(K)
        else:
            self.A3 = self.B3 = self.C3 = np.zeros(0, int)
        self.nl, self.nq, self.nc = self.n, self.P.size, self.A3.size
        self.F = self.nl + self.nq + self.nc
        self.sl = slice(0, self.nl)
        self.sq = slice(self.nl, self.nl + self.nq)
        self.sc = slice(self.nl + self.nq, self.F)

    def features(self, Z):
        parts = [Z, Z[:, self.P] * Z[:, self.Q]]
        if self.nc:
            parts.append(Z[:, self.A3] * Z[:, self.B3] * Z[:, self.C3])
        return np.concatenate(parts, axis=1)

    def allowed_mask(self):
        """(n, F) bool — translation-invariance selection rules."""
        m = self.modes
        allow = np.zeros((self.n, self.F), bool)
        for i, k in enumerate(m):
            allow[i, self.sl] = (m == k)
            allow[i, self.sq] = (m[self.P] + m[self.Q] == k)
            if self.nc:
                allow[i, self.sc] = (m[self.A3] + m[self.B3] + m[self.C3] == k)
        return allow


# ----------------------------------------------------------------------------
# Energy-conservation constraint on the quadratic block
# ----------------------------------------------------------------------------
def energy_constraints(bl):
    """Cubic form  sum_k sum_{p<=q} T_k,pq z_{-k} z_p z_q  must have zero real part
    for every real field (z_{-a} = conj z_a). Group terms by monomial multiset m;
    with alpha_m = sum of T entries mapping to m, the condition is
        alpha_m + conj(alpha_{mbar}) = 0,  mbar = negated multiset.
    Returns list of (entries, kind) where entries = list of (row_k, col_q, sign)
    and kind in {'re', 'im'} describing one real linear equation:
        sum sign * Re/Im(T[row, col]) = 0
    """
    m = bl.modes
    groups = {}
    for i, k in enumerate(m):
        for f, (pi, qi) in enumerate(zip(bl.P, bl.Q)):
            key = tuple(sorted((-k, m[pi], m[qi])))
            groups.setdefault(key, []).append((i, f))
    eqs, seen = [], set()
    for key, ent in groups.items():
        if key in seen:
            continue
        nkey = tuple(sorted(-x for x in key))
        seen.add(key); seen.add(nkey)
        ent_bar = groups.get(nkey, [])
        if nkey == key:
            eqs.append(([(i, f, 1.0) for i, f in ent], "re"))
        else:
            # Re(alpha_m) + Re(alpha_mbar) = 0 ; Im(alpha_m) - Im(alpha_mbar) = 0
            eqs.append(([(i, f, 1.0) for i, f in ent] + [(i, f, 1.0) for i, f in ent_bar], "re"))
            eqs.append(([(i, f, 1.0) for i, f in ent] + [(i, f, -1.0) for i, f in ent_bar], "im"))
    return eqs


def constraint_residual(T, eqs):
    r = []
    for ent, kind in eqs:
        v = sum(s * (T[i, f].real if kind == "re" else T[i, f].imag) for i, f, s in ent)
        r.append(v)
    return float(np.max(np.abs(r))) if r else 0.0


# ----------------------------------------------------------------------------
# Solver: per-output complex LS in real form, optional KKT projection
# ----------------------------------------------------------------------------
def real_form(G):
    return np.block([[G.real, -G.imag], [G.imag, G.real]])


def fit_v2(Z, dZ, bl, ridge, method, thresh, n_iter, energy, mask_sel):
    X = bl.features(Z)
    scale = np.sqrt(np.mean(np.abs(X) ** 2, axis=0)) + 1e-14
    Xs = X / scale
    G = Xs.conj().T @ Xs                       # (F, F)
    Bv = Xs.conj().T @ dZ                      # (F, n)
    n, F = bl.n, bl.F
    big = 1e8 * np.mean(np.real(np.diag(G)))
    allow = bl.allowed_mask() if mask_sel else np.ones((n, F), bool)
    pen = np.where(allow, ridge, big)          # (n, F)
    Rbase = real_form(G)
    eqs = energy_constraints(bl) if energy else []
    q0 = bl.sq.start

    def solve(pen):
        W = np.zeros((F, n), complex)
        Rinv = []
        for k in range(n):
            R = Rbase + np.diag(np.concatenate([pen[k], pen[k]]))
            Ri = np.linalg.inv(R)
            Rinv.append(Ri)
            x = Ri @ np.concatenate([Bv[:, k].real, Bv[:, k].imag])
            W[:, k] = x[:F] + 1j * x[F:]
        if eqs:
            W = project(W, Rinv)
        return W

    def project(W, Rinv):
        # constraint rows in scaled variables: T[k,f] = W[q0+f, k] / scale[q0+f]
        m = len(eqs)
        rows = []                              # per k: (m, 2F) sparse-ish dense
        Ck = [np.zeros((m, 2 * F)) for _ in range(n)]
        for r, (ent, kind) in enumerate(eqs):
            for i, f, s in ent:
                col = q0 + f
                off = 0 if kind == "re" else F
                Ck[i][r, off + col] += s / scale[col]
        Mmat = np.zeros((m, m))
        resid = np.zeros(m)
        RC = []
        for k in range(n):
            x = np.concatenate([W[:, k].real, W[:, k].imag])
            RCk = Rinv[k] @ Ck[k].T            # (2F, m)
            RC.append(RCk)
            Mmat += Ck[k] @ RCk
            resid += Ck[k] @ x
        lam = np.linalg.lstsq(Mmat, resid, rcond=1e-12)[0]
        for k in range(n):
            x = np.concatenate([W[:, k].real, W[:, k].imag]) - RC[k] @ lam
            W[:, k] = x[:F] + 1j * x[F:]
        return W

    W = solve(pen)
    if method == "stlsq":
        for _ in range(n_iter):
            mag = np.abs(W / scale[:, None])   # coefficient size (NOT contribution:
                                               # true triads have O(|k|) coefficients but
                                               # very different feature rms)
            cut = np.zeros_like(mag, bool)
            for s in (bl.sl, bl.sq, bl.sc):
                if s.stop - s.start == 0:
                    continue
                blk = mag[s, :]
                ref = np.max(blk, axis=0, keepdims=True) + 1e-30
                cut[s, :] = blk < thresh * ref
            pen_new = pen.copy()
            pen_new[cut.T] = big
            if np.array_equal(pen_new, pen):
                break
            pen = pen_new
            W = solve(pen)
    W = W / scale[:, None]
    A = W[bl.sl, :].T
    T = W[bl.sq, :].T
    C = W[bl.sc, :].T if bl.nc else np.zeros((n, 0), complex)
    return A, T, C, eqs


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def rhs_model(z, A, T, C, bl):
    out = A @ z + T @ (z[bl.P] * z[bl.Q])
    if bl.nc:
        out = out + C @ (z[bl.A3] * z[bl.B3] * z[bl.C3])
    return out


def metrics_v2(A_l, T_l, C_l, A_t, T_t, bl, Zte, dZte, eqs, traj_id, dt_save):
    m_ = bl.modes
    sel = (m_[bl.P][None, :] + m_[bl.Q][None, :] == m_[:, None])
    absT = np.abs(T_l)
    out = {
        "rel_err_T": float(np.linalg.norm(T_l - T_t) / np.linalg.norm(T_t)),
        "rel_err_T_on_rule": float(np.linalg.norm((T_l - T_t)[sel]) / np.linalg.norm(T_t[sel])),
        "selection_mass": float(absT[sel].sum() / (absT.sum() + 1e-14)),
        "max_off_rule_ratio": float(absT[~sel].max() / (absT[sel].max() + 1e-14)) if (~sel).any() else 0.0,
        "rel_err_A_diag": float(np.linalg.norm(np.diag(A_l) - np.diag(A_t)) / (np.linalg.norm(np.diag(A_t)) + 1e-14)),
        "offdiag_A_ratio": float(np.linalg.norm(A_l - np.diag(np.diag(A_l))) / (np.linalg.norm(np.diag(A_l)) + 1e-14)),
        "n_nonzero_T": int(np.sum(absT > 1e-8 * absT.max())),
        "n_nonzero_C": int(np.sum(np.abs(C_l) > 1e-8 * (np.abs(C_l).max() + 1e-30))) if bl.nc else 0,
    }
    Nq = (Zte[:, bl.P] * Zte[:, bl.Q]) @ T_l.T
    num = np.abs(np.real(np.sum(np.conj(Zte) * Nq, axis=1)))
    den = np.linalg.norm(Zte, axis=1) * np.linalg.norm(Nq, axis=1) + 1e-14
    out["energy_violation"] = float(np.mean(num / den))
    if eqs:
        out["energy_constraint_residual"] = constraint_residual(T_l, eqs)
    pred = Zte @ A_l.T + Nq
    if bl.nc:
        Nc = (Zte[:, bl.A3] * Zte[:, bl.B3] * Zte[:, bl.C3]) @ C_l.T
        pred = pred + Nc
        out["cubic_to_quad_ratio"] = float(np.linalg.norm(Nc) / (np.linalg.norm(Nq) + 1e-14))
    out["test_rel_err_dzdt"] = float(np.linalg.norm(pred - dZte) / np.linalg.norm(dZte))
    # rollout on first test trajectory
    Zr = Zte[traj_id == 0]
    z = Zr[0].copy()
    h = dt_save / 10
    err = 0.0
    for t in range(1, Zr.shape[0]):
        for _ in range(10):
            k1 = rhs_model(z, A_l, T_l, C_l, bl); k2 = rhs_model(z + h / 2 * k1, A_l, T_l, C_l, bl)
            k3 = rhs_model(z + h / 2 * k2, A_l, T_l, C_l, bl); k4 = rhs_model(z + h * k3, A_l, T_l, C_l, bl)
            z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(z)) or np.linalg.norm(z) > 1e6:
            err = float("inf"); break
        err = float(np.linalg.norm(z - Zr[t]) / (np.linalg.norm(Zr[t]) + 1e-14))
    out["rollout_rel_err_final"] = err
    return out


def fig_tensor_v2(T_t, T_l, K, k_show, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    cmap.set_bad(SURFACE)
    S_t, S_l = tensor_slice(T_t, K, k_show), tensor_slice(T_l, K, k_show)
    vmax = np.nanmax(S_t)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), facecolor=SURFACE)
    modes = np.arange(-K, K + 1)
    for ax, (t, S) in zip(axes, [("True (Galerkin)", S_t), (title, S_l)]):
        im = ax.imshow(S, cmap=cmap, vmin=0, vmax=vmax, origin="lower",
                       extent=[-K - .5, K + .5, -K - .5, K + .5])
        ax.plot(modes, k_show - modes, color=INK2, lw=0.8, ls="--")
        ax.set_xlim(-K - .5, K + .5); ax.set_ylim(-K - .5, K + .5)
        ax.set_title(t, color=INK, fontsize=10)
        ax.set_xlabel("q", color=INK2, fontsize=9); ax.set_ylabel("p", color=INK2, fontsize=9)
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label(f"|T[k={k_show}, p, q]|", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8); cb.outline.set_visible(False)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--kmax_full", type=int, default=128)
    ap.add_argument("--nus", type=float, nargs="+", default=[0.5, 0.2, 0.1, 0.05, 0.02])
    ap.add_argument("--regimes", nargs="+", default=["full"])
    ap.add_argument("--n_traj", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--t_final", type=float, default=2.0)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--amp", type=float, default=1.0)
    ap.add_argument("--mean_amp", type=float, default=0.5)
    ap.add_argument("--kmax_ic", type=int, default=None)
    ap.add_argument("--ridge", type=float, default=1e-10)
    ap.add_argument("--method", choices=["lstsq", "stlsq"], default="lstsq")
    ap.add_argument("--stlsq_thresh", type=float, default=0.02)
    ap.add_argument("--stlsq_iter", type=int, default=8)
    ap.add_argument("--energy_constraint", action="store_true")
    ap.add_argument("--closure", choices=["none", "cubic"], default="none")
    ap.add_argument("--mask_selection", action="store_true")
    ap.add_argument("--label", default=None, help="name used in comparison plots")
    ap.add_argument("--k_show", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="runs/e1v2")
    a = ap.parse_args()
    if a.quick:
        a.n_traj, a.n_test, a.t_final, a.nus, a.kmax_full = 40, 10, 0.5, [0.2, 0.05], 48
    kmax_ic = a.kmax_ic or a.K
    label = a.label or "+".join(
        [a.method] + (["energy"] if a.energy_constraint else []) +
        (["cubic"] if a.closure == "cubic" else []) + (["mask"] if a.mask_selection else []))
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "log.txt"), "w")

    def log(*m):
        s = " ".join(str(x) for x in m)
        print(s, flush=True); logf.write(s + "\n")

    log("args:", json.dumps(vars(a)))
    bl = Blocks(a.K, a.closure)
    log(f"features per output: linear {bl.nl}, quadratic {bl.nq}, cubic {bl.nc}")
    # sanity: the TRUE tensor must satisfy the energy constraint
    _, T_chk = true_operators(a.K, 0.1)
    log(f"sanity: energy-constraint residual of TRUE tensor = {constraint_residual(T_chk, energy_constraints(bl)):.2e}")

    results = []
    last = None
    for nu in a.nus:
        A_t, T_t = true_operators(a.K, nu)
        for regime in a.regimes:
            t0 = time.time()
            kmax_solver = a.K if regime == "closed" else a.kmax_full
            Z, dZ, _ = simulate(nu, kmax_solver, a.K, a.n_traj, a.t_final, a.dt, a.save_every,
                                np.random.default_rng(a.seed), a.amp, kmax_ic, a.mean_amp)
            Zte, dZte, tid = simulate(nu, kmax_solver, a.K, a.n_test, a.t_final, a.dt, a.save_every,
                                      np.random.default_rng(a.seed + 10_000), a.amp, kmax_ic, a.mean_amp)
            t1 = time.time()
            A_l, T_l, C_l, eqs = fit_v2(Z, dZ, bl, a.ridge, a.method, a.stlsq_thresh, a.stlsq_iter,
                                        a.energy_constraint, a.mask_selection)
            m = metrics_v2(A_l, T_l, C_l, A_t, T_t, bl, Zte, dZte, eqs, tid, a.dt * a.save_every)
            m["time_sim_s"], m["time_fit_s"] = round(t1 - t0, 1), round(time.time() - t1, 1)
            extra = f" cubic/quad={m['cubic_to_quad_ratio']:.2e}" if "cubic_to_quad_ratio" in m else ""
            extra += f" cons_res={m['energy_constraint_residual']:.1e}" if "energy_constraint_residual" in m else ""
            log(f"[{label} nu={nu:<5} {regime:<6}] relT={m['rel_err_T']:.2e} rule={m['rel_err_T_on_rule']:.2e} "
                f"sel={m['selection_mass']:.3f} Adiag={m['rel_err_A_diag']:.2e} Eviol={m['energy_violation']:.1e} "
                f"dz={m['test_rel_err_dzdt']:.2e} roll={m['rollout_rel_err_final']:.2e}{extra} "
                f"(sim {m['time_sim_s']}s fit {m['time_fit_s']}s)")
            results.append({"nu": nu, "regime": regime, "label": label, "metrics": m})
            last = (nu, regime, T_l, T_t)
            np.savez_compressed(os.path.join(a.out, f"tensors_nu{nu}_{regime}.npz"),
                                A=A_l, T=T_l, C=C_l, A_true=A_t, T_true=T_t)

    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump({"args": vars(a), "label": label, "results": results}, f, indent=2)
    nu, regime, T_l, T_t = last
    fig_tensor_v2(T_t, T_l, a.K, a.k_show, f"{label}, {regime}, nu={nu}",
                  os.path.join(a.out, f"fig_tensor_k{a.k_show}_nu{nu}_{regime}.png"))
    log("saved to", a.out)


if __name__ == "__main__":
    main()
