#!/usr/bin/env python3
"""
TCO M6a — learn the coupling tensor in a DATA-DRIVEN basis (POD), with no
Fourier basis and no symmetry prior given to the model.

Pipeline
  1. data = field values on a sensor grid (n_grid points) and their time
     derivatives. No Fourier coefficients are given.
  2. basis: POD of the training snapshots (top r = 2K+1 modes), or the real
     Fourier basis (--basis fourier) as a same-data reference.
  3. learn  da/dt = A a + T(a,a) [+ C(a,a,a)]  in the chosen coordinates by
     least squares (no mask, no energy constraint: nothing about the PDE or
     its symmetry is assumed).
  4. evaluation: the learned vector field is mapped to REAL Fourier
     coordinates b (b = Q a, Q = R^T Phi) and its linear and quadratic parts
     are extracted exactly (even/odd split + polarization), then compared with
     the exact Galerkin truth. This comparison is invariant to the rotation
     freedom inside degenerate POD pairs.
  5. intrinsic POD analysis: does POD find Fourier by itself?
       span_mismatch  ||Q^T Q - I|| / sqrt(r)       (0 = same subspace as Fourier |k|<=K)
       purity_j       fraction of mode j's energy in a single wavenumber
       pair_ratio     sigma_{2j-1}/sigma_{2j} for the cos/sin pairs (1 = degenerate pair)

Real Fourier coordinates: u = b_0 + sum_k (b_ck sqrt2 cos kx + b_sk sqrt2 sin kx),
orthonormal for <f,g> = mean(f g). Selection rule in real coordinates:
output wavenumber k_out in {k_p + k_q, |k_p - k_q|}.

Needs tco_pde.py in the same folder. PDEs: burgers, kdv, ks, fisher.
  python tco_pod.py --pde burgers --quick --regimes closed full --out runs/M6_smoke
"""
import argparse
import json
import os
import time

import numpy as np

from tco_pde import (PDE, Solver, Blocks, fit, model_rhs, bandlimited_field, DEFAULTS,
                     _style, SURFACE, INK, INK2, CAT, SEQ)


# ----------------------------------------------------------------------------
def simulate_grid(pde, kmax_solver, n_grid, n_traj, t_final, dt, save_every, rng, amp, mean_amp, kmax_ic):
    sol = Solver(pde, kmax_solver)
    nk = sol.M // 2 + 1
    ng = n_grid // 2 + 1
    U, Ut, tid = [], [], []
    n_steps = int(round(t_final / dt))

    def to_grid(c):
        cc = np.zeros(ng, complex)
        m = min(ng - 1, nk)
        cc[:m] = c[:m]                       # sensor field = band-limited projection
        return np.fft.irfft(cc * n_grid, n=n_grid)

    for tr in range(n_traj):
        c = amp * bandlimited_field(rng, min(kmax_ic, kmax_solver), nk)
        c[0] = rng.uniform(0.2, 0.6) if pde.name == "fisher" else rng.uniform(-mean_amp, mean_amp)
        c = c * sol.mask
        for s in range(n_steps + 1):
            if s % save_every == 0:
                U.append(to_grid(c)); Ut.append(to_grid(sol.rhs(c))); tid.append(tr)
            if s < n_steps:
                c = sol.step(c, dt)
                if not np.all(np.isfinite(c)) or np.max(np.abs(c)) > 1e6:
                    raise RuntimeError(f"solver blew up ({pde.name}, {pde.param})")
    return np.array(U), np.array(Ut), np.array(tid)


def real_fourier_basis(n_grid, K):
    x = 2 * np.pi * np.arange(n_grid) / n_grid
    cols, kl = [np.ones(n_grid)], [0]
    for k in range(1, K + 1):
        cols += [np.sqrt(2) * np.cos(k * x), np.sqrt(2) * np.sin(k * x)]; kl += [k, k]
    return np.array(cols).T, np.array(kl)


def pod_basis(U, r):
    _, sv, Vh = np.linalg.svd(U, full_matrices=False)
    Phi = Vh[:r].T * np.sqrt(U.shape[1])      # orthonormal for mean inner product
    return Phi, sv[:r]


# ---- conversions between real Fourier coords b and complex modes z ----------
def b_to_z(b, K):
    z = np.zeros(b.shape[:-1] + (2 * K + 1,), complex)
    z[..., K] = b[..., 0]
    for k in range(1, K + 1):
        ck = (b[..., 2 * k - 1] - 1j * b[..., 2 * k]) / np.sqrt(2)
        z[..., K + k] = ck; z[..., K - k] = np.conj(ck)
    return z


def z_to_b(z, K):
    b = np.zeros(z.shape[:-1] + (2 * K + 1,))
    b[..., 0] = z[..., K].real
    for k in range(1, K + 1):
        b[..., 2 * k - 1] = np.sqrt(2) * z[..., K + k].real
        b[..., 2 * k] = -np.sqrt(2) * z[..., K + k].imag
    return b


def extract_linear_quadratic(f, r):
    """Exact linear and canonical quadratic parts of a vector field f (batch -> batch)
    containing linear, quadratic and cubic terms."""
    I = np.eye(r)
    Lin = lambda e: (f(e * I) - f(-e * I)) / (2 * e)      # = A + e^2 C-terms
    A = ((4 * Lin(5e-3) - Lin(1e-2)) / 3).T              # Richardson removes cubic
    Tq = lambda V: 0.5 * (f(V) + f(-V))                    # even part = quadratic, exact
    P, Q = np.triu_indices(r)
    single = Tq(I)                                          # (r, r): row p = T(e_p, e_p)
    V = I[P] + I[Q]
    pair = Tq(V) - single[P] - single[Q]
    T = np.where((P == Q)[:, None], single[P], pair).T    # (r, n_pairs)
    return A, T, P, Q


def rollout_matrix_if(a0, f, A, Aref, dt_save, n_sub=10):
    """IF-RK4 with the full learned linear operator (via its eigendecomposition)."""
    lam, Vv = np.linalg.eig(A)
    Vi = np.linalg.inv(Vv)
    h = dt_save / n_sub
    E, E2 = np.exp(lam * h), np.exp(lam * h / 2)
    Nw = lambda w: Vi @ (f((Vv @ w).real[None, :])[0] - A @ (Vv @ w).real)
    w = Vi @ a0.astype(complex)
    nT = Aref.shape[0]
    traj = [a0.copy()]
    for t in range(1, nT):
        for _ in range(n_sub):
            a_ = h * Nw(w); b_ = h * Nw(E2 * (w + a_ / 2)); c_ = h * Nw(E2 * w + b_ / 2); d_ = h * Nw(E * w + E2 * c_)
            w = E * w + (E * a_ + 2 * E2 * (b_ + c_) + d_) / 6.0
        x = (Vv @ w).real
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e6:
            return float("inf"), float("inf"), float("inf")
        traj.append(x)
    traj = np.array(traj)
    rel = lambda t: float(np.linalg.norm(traj[t] - Aref[t]) / (np.linalg.norm(Aref[t]) + 1e-14))
    half = nT // 2
    Sm, Sr = np.mean(traj[half:] ** 2, axis=0), np.mean(Aref[half:] ** 2, axis=0)
    return rel(nT - 1), rel(max(1, (nT - 1) // 10)), float(np.linalg.norm(Sm - Sr) / (np.linalg.norm(Sr) + 1e-30))


def fig_pod_modes(Phi, sv, n_grid, path, title, n_show=8):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    spec = np.abs(np.fft.rfft(Phi, axis=0)) ** 2            # (n_grid/2+1, r)
    spec = spec / spec.sum(axis=0, keepdims=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    kmax_show = min(spec.shape[0], 16)
    im = ax.imshow(spec[:kmax_show], aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=1)
    ax.set_xlabel("POD mode index (by energy)", color=INK2, fontsize=9)
    ax.set_ylabel("Fourier wavenumber |k|", color=INK2, fontsize=9)
    ax.set_title("Fourier content of each POD mode", color=INK, fontsize=10, loc="left")
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label("energy fraction", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8); cb.outline.set_visible(False)
    ax = axes[1]; _style(ax)
    x = np.arange(n_grid) * 2 * np.pi / n_grid
    for j in range(min(n_show, 4)):
        ax.plot(x, Phi[:, j], color=CAT[j], lw=1.8, label=f"POD {j}")
    ax.set_xlabel("x", color=INK2, fontsize=9)
    ax.set_title("Leading POD modes", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", choices=["burgers", "kdv", "ks", "fisher"], required=True)
    ap.add_argument("--basis", choices=["pod", "fourier"], default="pod")
    ap.add_argument("--params", type=float, nargs="+", default=None)
    ap.add_argument("--regimes", nargs="+", default=["closed", "full"])
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n_grid", type=int, default=64)
    ap.add_argument("--kmax_full", type=int, default=128)
    ap.add_argument("--n_traj", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--t_final", type=float, default=None)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--save_every", type=int, default=None)
    ap.add_argument("--amp", type=float, default=None)
    ap.add_argument("--mean_amp", type=float, default=None)
    ap.add_argument("--kmax_ic", type=int, default=None)
    ap.add_argument("--ridge", type=float, default=1e-10)
    ap.add_argument("--method", choices=["lstsq", "stlsq"], default="lstsq")
    ap.add_argument("--stlsq_thresh", type=float, default=0.02)
    ap.add_argument("--stlsq_iter", type=int, default=8)
    ap.add_argument("--closure", choices=["none", "cubic"], default="none")
    ap.add_argument("--label", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dp, dT, ddt, dsv, damp, dmean = DEFAULTS[a.pde]
    params = a.params or dp
    t_final, dt = a.t_final or dT, a.dt or ddt
    save_every = a.save_every or dsv
    amp = a.amp if a.amp is not None else damp
    mean_amp = a.mean_amp if a.mean_amp is not None else dmean
    kmax_ic = a.kmax_ic or a.K
    if a.quick:
        a.n_traj, a.n_test, a.kmax_full = 40, 10, 48
        t_final, params = t_final / 4, params[:2]
    if 2 * a.K + 1 > a.n_grid // 2:
        raise SystemExit("n_grid too small for K")
    label = a.label or "+".join([a.basis, a.method] + (["cubic"] if a.closure == "cubic" else []))
    out = a.out or f"runs/M6_{a.pde}_{label}"
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "w")

    def log(*m):
        s = " ".join(str(x) for x in m); print(s, flush=True); logf.write(s + "\n")

    log("args:", json.dumps({**vars(a), "params": params, "t_final": t_final}))
    r = 2 * a.K + 1
    R, kl = real_fourier_basis(a.n_grid, a.K)
    bl = Blocks(a.K, 0, a.closure, False)                 # n = r coordinates
    blF = Blocks(a.K, 0, "none", False)                   # Fourier truth
    results = []
    for prm in params:
        pde = PDE(a.pde, prm)
        A_z, T_z, _ = blF.truth(pde)
        f_true_b = lambda B: z_to_b(model_rhs(b_to_z(B, a.K), None,
                                              {"A": A_z, "T": T_z, "C": np.zeros((r, 0)), "L": np.zeros((r, 0))}, blF), a.K)
        A_tb, T_tb, P, Q = extract_linear_quadratic(f_true_b, r)
        rule = np.zeros_like(T_tb, bool)
        for o in range(r):
            rule[o] = (kl[o] == kl[P] + kl[Q]) | (kl[o] == np.abs(kl[P] - kl[Q]))
        for regime in a.regimes:
            t0 = time.time()
            ks = a.K if regime == "closed" else a.kmax_full
            sim = lambda n, seed: simulate_grid(pde, ks, a.n_grid, n, t_final, dt, save_every,
                                                np.random.default_rng(seed), amp, mean_amp, kmax_ic)
            U, Ut, _ = sim(a.n_traj, a.seed)
            Ute, Utte, tid = sim(a.n_test, a.seed + 10_000)
            t1 = time.time()
            if a.basis == "pod":
                Phi, sv = pod_basis(U, r)
            else:
                Phi, sv = R.copy(), np.full(r, np.nan)
            proj = lambda X: X @ Phi / a.n_grid
            Atr, dAtr, Ate, dAte = proj(U), proj(Ut), proj(Ute), proj(Utte)
            ops, _ = fit(Atr.astype(complex), dAtr.astype(complex), None, bl, a.ridge, a.method,
                         a.stlsq_thresh, a.stlsq_iter, False, False)
            imag = max(float(np.abs(v.imag).max()) if v.size else 0.0 for v in ops.values())
            ops = {k: v.real for k, v in ops.items()}
            f_pod = lambda X: model_rhs(X, None, ops, bl).real
            # ---- map to real Fourier coordinates and compare with truth
            Qm = R.T @ Phi / a.n_grid                       # b = Qm a
            Qp = np.linalg.pinv(Qm)
            f_b = lambda B: (Qm @ f_pod(B @ Qp.T).T).T
            A_lb, T_lb, _, _ = extract_linear_quadratic(f_b, r)
            absT = np.abs(T_lb)
            m = {
                "T_rel_err": float(np.linalg.norm(T_lb - T_tb) / np.linalg.norm(T_tb)),
                "T_rel_err_on_rule": float(np.linalg.norm((T_lb - T_tb)[rule]) / np.linalg.norm(T_tb[rule])),
                "T_selection_mass": float(absT[rule].sum() / (absT.sum() + 1e-30)),
                "A_rel_err": float(np.linalg.norm(A_lb - A_tb) / (np.linalg.norm(A_tb) + 1e-30)),
                "span_mismatch": float(np.linalg.norm(Qm.T @ Qm - np.eye(r)) / np.sqrt(r)),
                "fit_imag_max": imag,
            }
            # ---- intrinsic POD analysis
            spec = np.abs(np.fft.rfft(Phi, axis=0)) ** 2
            spec = spec / spec.sum(axis=0, keepdims=True)
            m["pod_purity"] = spec.max(axis=0).tolist()
            m["pod_dominant_k"] = spec.argmax(axis=0).tolist()
            m["pod_purity_mean"] = float(np.mean(spec.max(axis=0)))
            m["pod_dominant_k_within_K"] = int(np.sum(spec.argmax(axis=0) <= a.K))
            # ---- gauge fixing WITHOUT prior: eigenbasis of the learned linear operator.
            # If the system is translation invariant, A commutes with shifts and its
            # eigenvectors are Fourier modes -> physical coordinates recovered from the
            # learned model alone.
            lam, Vv = np.linalg.eig(ops["A"])
            G = Phi @ Vv                                         # eigen-functions on the grid
            sp = np.abs(np.fft.fft(G, axis=0)) ** 2               # two-sided spectrum
            ng = a.n_grid
            kk = np.minimum(np.arange(ng), ng - np.arange(ng))   # |k| of each FFT bin
            spk = np.zeros((ng // 2 + 1, r))
            for j in range(ng // 2 + 1):
                spk[j] = sp[kk == j].sum(axis=0)
            spk = spk / spk.sum(axis=0, keepdims=True)
            m["eig_purity"] = spk.max(axis=0).tolist()
            m["eig_dominant_k"] = spk.argmax(axis=0).tolist()
            m["eig_purity_mean"] = float(np.mean(spk.max(axis=0)))
            m["eig_values"] = [[float(v.real), float(v.imag)] for v in lam]
            if a.basis == "pod":
                m["pod_singular_values"] = sv.tolist()
                m["pod_pair_ratio"] = [float(sv[i] / sv[i + 1]) for i in range(1, r - 1, 2)]
            # ---- prediction in learned coordinates
            pred = f_pod(Ate)
            m["test_rel_err_dzdt"] = float(np.linalg.norm(pred - dAte) / np.linalg.norm(dAte))
            errs = []
            for tr in np.unique(tid)[:5]:
                Aref = Ate[tid == tr]
                errs.append(rollout_matrix_if(Aref[0], f_pod, ops["A"], Aref, dt * save_every))
            errs = np.array(errs)
            m["rollout_rel_err_final_median5"] = float(np.median(errs[:, 0]))
            m["rollout_short_rel_err_median5"] = float(np.median(errs[:, 1]))
            m["spectrum_rel_err_median5"] = float(np.median(errs[:, 2]))
            m["time_sim_s"], m["time_fit_s"] = round(t1 - t0, 1), round(time.time() - t1, 1)
            log(f"[{a.pde} {label} p={prm:<5} {regime:<6}] Trel={m['T_rel_err']:.2e} Trule={m['T_rel_err_on_rule']:.2e} "
                f"Tsel={m['T_selection_mass']:.3f} Arel={m['A_rel_err']:.2e} span={m['span_mismatch']:.1e} "
                f"purity={m['pod_purity_mean']:.3f} eig_purity={m['eig_purity_mean']:.3f} dz={m['test_rel_err_dzdt']:.2e} "
                f"roll_short={m['rollout_short_rel_err_median5']:.1e} spec={m['spectrum_rel_err_median5']:.1e} "
                f"(sim {m['time_sim_s']}s fit {m['time_fit_s']}s)")
            results.append({"pde": a.pde, "param": prm, "regime": regime, "label": label, "metrics": m})
            np.savez_compressed(os.path.join(out, f"pod_{prm}_{regime}.npz"), Phi=Phi, sv=sv,
                                A_learned_b=A_lb, T_learned_b=T_lb, A_true_b=A_tb, T_true_b=T_tb, Qm=Qm)
            if a.basis == "pod":
                fig_pod_modes(Phi, sv, a.n_grid, os.path.join(out, f"fig_pod_modes_{prm}_{regime}.png"),
                              f"{a.pde}, param={prm}, {regime}: POD basis learned from sensor data")

    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({"args": vars(a), "pde": a.pde, "label": label, "results": results}, f, indent=2)
    log("saved to", out)


if __name__ == "__main__":
    main()
