#!/usr/bin/env python3
"""
TCO stage 2 — 2D incompressible Navier–Stokes (vorticity form, periodic, Kolmogorov
forcing). Learn the triad tensor of the truncated Fourier dynamics from data and
compare with the exact one.

  w_t + u . grad(w) = nu lap(w) - alpha w + f,   u = (psi_y, -psi_x),  -lap(psi) = w
  f = F sin(n_f y)  (Kolmogorov forcing)

Exact modal dynamics (modes k = (kx, ky) != 0, w = sum w_k e^{i k.x}):
  dw_k/dt = -(nu |k|^2 + alpha) w_k + f_k + sum_{p+q=k} c(p,q) w_p w_q
  canonical unordered pair coefficient:  T[k,{p,q}] = -(p x q) (1/|p|^2 - 1/|q|^2),
  p x q = p_x q_y - p_y q_x   (zero for p = q)

Learned model on the square |kx|,|ky| <= K (n = (2K+1)^2 - 1 modes):
  masked (default): per output k only diag-linear, bias and pairs with p+q=k
                    (translation invariance; ~n/2 features per output)
  --dense          : all linear, bias, all pairs (n^2/2 features per output) — only
                    feasible for small K; shows why structure is needed to scale.

Closure analysis (shell-averaged eddy viscosity), short-horizon rollout and
enstrophy-spectrum statistics are computed on test data.

Needs numpy, matplotlib, torch (GPU used if available).
  python tco_ns2d.py --quick --out runs/NS_smoke
"""
import argparse
import json
import os
import time

import numpy as np
import torch

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


# ============================================================================
# Pseudo-spectral solver (torch, batched over trajectories)
# ============================================================================
class NS2D:
    def __init__(self, N, kmax, nu, alpha, F, nf, device):
        self.N, self.dev = N, device
        k = torch.fft.fftfreq(N, d=1.0 / N).to(device)
        self.KX, self.KY = torch.meshgrid(k, k, indexing="ij")       # w[ix, iy]
        self.K2 = self.KX ** 2 + self.KY ** 2
        self.K2inv = torch.where(self.K2 > 0, 1.0 / self.K2, torch.zeros_like(self.K2))
        self.mask = ((self.KX.abs() <= kmax) & (self.KY.abs() <= kmax)).to(torch.float64)
        self.L = (-nu * self.K2 - alpha).to(torch.complex128)
        self.f = torch.zeros(N, N, dtype=torch.complex128, device=device)
        iy = nf % N
        self.f[0, iy] = F / 2j
        self.f[0, (-nf) % N] = -F / 2j

    def nonlinear(self, wh):
        ikx, iky = 1j * self.KX, 1j * self.KY
        psih = wh * self.K2inv
        u = torch.fft.ifft2(iky * psih).real * self.N ** 2
        v = torch.fft.ifft2(-ikx * psih).real * self.N ** 2
        wx = torch.fft.ifft2(ikx * wh).real * self.N ** 2
        wy = torch.fft.ifft2(iky * wh).real * self.N ** 2
        adv = torch.fft.fft2(u * wx + v * wy) / self.N ** 2
        return (-adv + self.f) * self.mask

    def rhs(self, wh):
        return self.L * wh + self.nonlinear(wh)

    def step(self, wh, dt):
        E, E2 = torch.exp(self.L * dt), torch.exp(self.L * dt / 2)
        a = dt * self.nonlinear(wh)
        b = dt * self.nonlinear(E2 * (wh + a / 2))
        c = dt * self.nonlinear(E2 * wh + b / 2)
        d = dt * self.nonlinear(E * wh + E2 * c)
        return E * wh + (E * a + 2 * E2 * (b + c) + d) / 6.0


def random_ic(B, N, k0, amp, gen, device):
    k = torch.fft.fftfreq(N, d=1.0 / N)
    KX, KY = torch.meshgrid(k, k, indexing="ij")
    kk = torch.sqrt(KX ** 2 + KY ** 2)
    spec = torch.where((kk > 0) & (kk <= k0), kk ** -1.0, torch.zeros_like(kk))
    noise = torch.randn(B, N, N, generator=gen) + 1j * torch.randn(B, N, N, generator=gen)
    w = torch.fft.ifft2(noise * spec).real                      # real field
    w = w / w.abs().amax(dim=(1, 2), keepdim=True) * amp
    return (torch.fft.fft2(w) / N ** 2).to(torch.complex128).to(device)


# ============================================================================
# Mode bookkeeping
# ============================================================================
class Modes:
    def __init__(self, K):
        self.K = K
        kk = [(kx, ky) for kx in range(-K, K + 1) for ky in range(-K, K + 1) if (kx, ky) != (0, 0)]
        self.k = np.array(kk)                                   # (n, 2)
        self.n = len(kk)
        self.index = {tuple(v): i for i, v in enumerate(kk)}
        self.k2 = (self.k ** 2).sum(1).astype(float)

    def gather(self, Wfull, N):
        """(B, N, N) full spectra -> (B, n) on the K-square."""
        ix = self.k[:, 0] % N
        iy = self.k[:, 1] % N
        return Wfull[:, ix, iy]

    def allowed_pairs(self, dense):
        """Per output r: (P, Q) index arrays (p <= q by index)."""
        out = []
        if dense:
            P, Q = np.triu_indices(self.n)
            for _ in range(self.n):
                out.append((P, Q))
            return out
        for r in range(self.n):
            kr = self.k[r]
            P, Q = [], []
            for i in range(self.n):
                q = tuple(kr - self.k[i])
                j = self.index.get(q)
                if j is not None and i <= j:
                    P.append(i); Q.append(j)
            out.append((np.array(P, int), np.array(Q, int)))
        return out

    def truth(self, nu, alpha, fvec, pairs):
        A = -(nu * self.k2 + alpha)
        T = []
        for r, (P, Q) in enumerate(pairs):
            p, q = self.k[P], self.k[Q]
            cross = p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]
            coef = -cross * (1.0 / self.k2[P] - 1.0 / self.k2[Q])
            ok = (p + q == self.k[r]).all(1)
            T.append(np.where(ok, coef, 0.0).astype(complex))
        return A.astype(complex), fvec, T


# ============================================================================
# Fit per output (scaled lstsq, SVD-based), model RHS, metrics
# ============================================================================
def fit(Z, dZ, modes, pairs, dense):
    n = modes.n
    A = np.zeros((n, n), complex) if dense else np.zeros(n, complex)
    b = np.zeros(n, complex)
    T = []
    one = np.ones((Z.shape[0], 1), complex)
    for r in range(n):
        P, Q = pairs[r]
        lin = Z if dense else Z[:, r:r + 1]
        X = np.concatenate([lin, one, Z[:, P] * Z[:, Q]], axis=1)
        sc = np.sqrt(np.mean(np.abs(X) ** 2, axis=0)) + 1e-14
        w = np.linalg.lstsq(X / sc, dZ[:, r], rcond=1e-12)[0] / sc
        nl = lin.shape[1]
        if dense:
            A[r] = w[:nl]
        else:
            A[r] = w[0]
        b[r] = w[nl]
        T.append(w[nl + 1:])
    return A, b, T


def model_rhs(Z, A, b, T, pairs, dense):
    out = (Z @ A.T) if dense else Z * A[None, :]
    out = out + b[None, :]
    for r, (P, Q) in enumerate(pairs):
        if P.size:
            out[:, r] += (Z[:, P] * Z[:, Q]) @ T[r]
    return out


def shell_spectrum(Z, k2):
    shells = np.rint(np.sqrt(k2)).astype(int)
    E = np.mean(np.abs(Z) ** 2, axis=0)
    ks = np.unique(shells)
    return ks, np.array([E[shells == s].sum() for s in ks])


def rollout(z0, A, b, T, pairs, dense, nT, dt_save, n_sub=10):
    D = A if not dense else np.diag(A)
    f = lambda z: model_rhs(z[None, :], A, b, T, pairs, dense)[0]
    N = lambda z: f(z) - D * z
    h = dt_save / n_sub
    E, E2 = np.exp(D * h), np.exp(D * h / 2)
    z, traj = z0.copy(), [z0.copy()]
    for _ in range(1, nT):
        for _ in range(n_sub):
            a = h * N(z); bb = h * N(E2 * (z + a / 2)); c = h * N(E2 * z + bb / 2); d = h * N(E * z + E2 * c)
            z = E * z + (E * a + 2 * E2 * (bb + c) + d) / 6.0
        if not np.all(np.isfinite(z)) or np.linalg.norm(z) > 1e6:
            return None
        traj.append(z.copy())
    return np.array(traj)


# ============================================================================
def simulate(nu, alpha, F, nf, N, kmax, modes, n_traj, t_spin, t_rec, dt, save_every, amp, seed, device):
    sol = NS2D(N, kmax, nu, alpha, F, nf, device)
    gen = torch.Generator().manual_seed(seed)
    wh = random_ic(n_traj, N, 4, amp, gen, device) * sol.mask
    for _ in range(int(round(t_spin / dt))):
        wh = sol.step(wh, dt)
    Zs, dZs = [], []
    n_rec = int(round(t_rec / dt))
    for s in range(n_rec + 1):
        if s % save_every == 0:
            Zs.append(modes.gather(wh, N).cpu().numpy())
            dZs.append(modes.gather(sol.rhs(wh), N).cpu().numpy())
        if s < n_rec:
            wh = sol.step(wh, dt)
        if not torch.isfinite(wh.real).all():
            raise RuntimeError("solver blew up; lower --dt")
    Z = np.stack(Zs, 1)            # (B, T, n)
    dZ = np.stack(dZs, 1)
    return Z, dZ


def fig_closure(ks, ex, mo, title, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, lw=0.6); ax.tick_params(colors=INK2, labelsize=9)
    ax.plot(ks, ex, color=CAT[0], lw=2.5, marker="o", ms=8, markeredgecolor=SURFACE, label="exact closure (test)")
    ax.plot(ks, mo, color=CAT[1], lw=2, ls="--", marker="s", ms=5, markeredgecolor=SURFACE, label="implied by learned model")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("shell |k|", color=INK2, fontsize=9); ax.set_ylabel("eddy viscosity nu_t(|k|)", color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left"); ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)


def fig_tensor(modes, pairs, Tl, Tt, r, title, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ); cmap.set_bad(SURFACE)
    K = modes.K
    def grid(T):
        G = np.full((2 * K + 1, 2 * K + 1), np.nan)
        P, Q = pairs[r]
        for c, (i, j) in enumerate(zip(P, Q)):
            for idx in (i, j):
                px, py = modes.k[idx]
                G[px + K, py + K] = abs(T[c])
        return G
    Gt, Gl = grid(Tt[r]), grid(Tl[r])
    vmax = np.nanmax(Gt) if np.isfinite(np.nanmax(Gt)) and np.nanmax(Gt) > 0 else 1
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), facecolor=SURFACE)
    for ax, (t, G) in zip(axes, [("True", Gt), (title, Gl)]):
        im = ax.imshow(G.T, origin="lower", cmap=cmap, vmin=0, vmax=vmax, extent=[-K - .5, K + .5, -K - .5, K + .5])
        ax.set_title(t, color=INK, fontsize=10); ax.tick_params(colors=INK2, labelsize=8)
        ax.set_xlabel("p_x", color=INK2, fontsize=9); ax.set_ylabel("p_y", color=INK2, fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.85); cb.outline.set_visible(False)
    cb.set_label(f"|T[k={tuple(modes.k[r])}; p, k-p]|", color=INK2, fontsize=9); cb.ax.tick_params(colors=INK2, labelsize=8)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--nus", type=float, nargs="+", default=[0.02, 0.01, 0.005])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--F", type=float, default=1.0)
    ap.add_argument("--nf", type=int, default=4)
    ap.add_argument("--regimes", nargs="+", default=["closed", "full"])
    ap.add_argument("--N_full", type=int, default=64)
    ap.add_argument("--n_traj", type=int, default=64)
    ap.add_argument("--n_test", type=int, default=8)
    ap.add_argument("--t_spin", type=float, default=10.0)
    ap.add_argument("--t_rec", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--amp", type=float, default=2.0)
    ap.add_argument("--dense", action="store_true", help="no symmetry mask (only for small K)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.quick:
        a.n_traj, a.n_test, a.t_spin, a.t_rec, a.nus, a.N_full = 16, 4, 2.0, 2.0, a.nus[:2], 48
    if a.nf > a.K:
        raise SystemExit(f"forcing wavenumber nf={a.nf} must be <= K={a.K} (use --nf to lower it)")
    torch.set_default_dtype(torch.float64)
    modes = Modes(a.K)
    pairs = modes.allowed_pairs(a.dense)
    nfeat = (modes.n if a.dense else 1) + 1 + pairs[0][0].size
    if a.dense and pairs[0][0].size > 5000:
        raise SystemExit(f"--dense needs {pairs[0][0].size} quadratic features per output at K={a.K}; use K<=4")
    label = "dense" if a.dense else "mask"
    out = a.out or f"runs/NS_K{a.K}_{label}"
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "w")

    def log(*m):
        s = " ".join(str(x) for x in m); print(s, flush=True); logf.write(s + "\n")

    dense_pairs = modes.n * (modes.n + 1) // 2
    log("args:", json.dumps(vars(a)))
    log(f"modes n={modes.n}; features/output: {'dense' if a.dense else 'masked'} ~{nfeat} "
        f"(dense would need {modes.n + 1 + dense_pairs}); device={a.device}")
    results = []
    for nu in a.nus:
        for regime in a.regimes:
            t0 = time.time()
            N = max(32, 4 * a.K) if regime == "closed" else a.N_full
            kmax = a.K if regime == "closed" else N // 3
            sim = lambda n, seed: simulate(nu, a.alpha, a.F, a.nf, N, kmax, modes, n, a.t_spin, a.t_rec,
                                           a.dt, a.save_every, a.amp, seed, a.device)
            Ztr, dZtr = sim(a.n_traj, a.seed)
            Zte, dZte = sim(a.n_test, a.seed + 10_000)
            t1 = time.time()
            fvec = np.zeros(modes.n, complex)
            for r, kv in enumerate(modes.k):
                if kv[0] == 0 and kv[1] == a.nf:
                    fvec[r] = a.F / 2j
                if kv[0] == 0 and kv[1] == -a.nf:
                    fvec[r] = -a.F / 2j
            A_t, b_t, T_t = modes.truth(nu, a.alpha, fvec, pairs)
            Z, dZ = Ztr.reshape(-1, modes.n), dZtr.reshape(-1, modes.n)
            A_l, b_l, T_l = fit(Z, dZ, modes, pairs, a.dense)
            A_ld = np.diag(A_l) if a.dense else A_l
            Tl_all, Tt_all = np.concatenate(T_l), np.concatenate(T_t)
            onrule = np.concatenate([(modes.k[P] + modes.k[Q] == modes.k[r]).all(1) for r, (P, Q) in enumerate(pairs)])
            m = {
                "T_rel_err": float(np.linalg.norm(Tl_all - Tt_all) / np.linalg.norm(Tt_all)),
                "T_rel_err_on_rule": float(np.linalg.norm((Tl_all - Tt_all)[onrule]) / np.linalg.norm(Tt_all[onrule])),
                "T_selection_mass": float(np.abs(Tl_all[onrule]).sum() / (np.abs(Tl_all).sum() + 1e-30)),
                "A_diag_rel_err": float(np.linalg.norm(A_ld - A_t) / np.linalg.norm(A_t)),
                "forcing_rel_err": float(np.linalg.norm(b_l - b_t) / np.linalg.norm(b_t)) if np.linalg.norm(b_t) > 0 else None,
                "n_features_per_output": int(nfeat), "n_samples": int(Z.shape[0]),
            }
            # scale-resolved tensor error: only triads with |k|,|p|,|q| <= cut
            kmag = np.sqrt(modes.k2)
            by_cut = {}
            for cut in range(1, a.K + 1):
                num = den = 0.0
                for r, (P, Q) in enumerate(pairs):
                    if kmag[r] <= cut and P.size:
                        ok = (kmag[P] <= cut) & (kmag[Q] <= cut)
                        num += float(np.sum(np.abs(T_l[r][ok] - T_t[r][ok]) ** 2))
                        den += float(np.sum(np.abs(T_t[r][ok]) ** 2))
                by_cut[str(cut)] = float(np.sqrt(num / den)) if den > 0 else None
            m["T_rel_err_by_cut"] = by_cut
            Zt2, dZt2 = Zte.reshape(-1, modes.n), dZte.reshape(-1, modes.n)
            Atd = np.diag(A_t) if a.dense else A_t
            pred = model_rhs(Zt2, A_l, b_l, T_l, pairs, a.dense)
            gal = model_rhs(Zt2, Atd, b_t, T_t, pairs, a.dense)
            m["test_rel_err_dzdt"] = float(np.linalg.norm(pred - dZt2) / np.linalg.norm(dZt2))
            m["closure_rel_size"] = float(np.linalg.norm(dZt2 - gal) / np.linalg.norm(dZt2))
            # shell-averaged eddy viscosity
            E = np.mean(np.abs(Zt2) ** 2, axis=0) + 1e-30
            Pex = np.mean(np.real(np.conj(Zt2) * (dZt2 - gal)), axis=0) / E
            Pmo = np.mean(np.real(np.conj(Zt2) * (pred - gal)), axis=0) / E
            shells = np.rint(np.sqrt(modes.k2)).astype(int)
            ks = np.unique(shells)
            nut_ex = np.array([-(Pex[shells == s] / modes.k2[shells == s]).mean() for s in ks])
            nut_mo = np.array([-(Pmo[shells == s] / modes.k2[shells == s]).mean() for s in ks])
            m["closure"] = {"shell": ks.tolist(), "nu_t_exact": nut_ex.tolist(), "nu_t_model": nut_mo.tolist(),
                            "nu_t_rel_err": float(np.linalg.norm(nut_mo - nut_ex) / (np.linalg.norm(nut_ex) + 1e-30))}
            # rollouts on test trajectories: short-horizon and enstrophy-spectrum stats
            nT = Zte.shape[1]
            dts = a.dt * a.save_every
            sh, sp = [], []
            for b_ in range(min(3, a.n_test)):
                tr = rollout(Zte[b_, 0], A_l, b_l, T_l, pairs, a.dense, nT, dts)
                if tr is None:
                    sh.append(np.inf); sp.append(np.inf); continue
                t_s = max(1, (nT - 1) // 10)
                sh.append(float(np.linalg.norm(tr[t_s] - Zte[b_, t_s]) / np.linalg.norm(Zte[b_, t_s])))
                _, Sm = shell_spectrum(tr[nT // 2:], modes.k2)
                _, Sr = shell_spectrum(Zte[b_, nT // 2:], modes.k2)
                sp.append(float(np.linalg.norm(Sm - Sr) / np.linalg.norm(Sr)))
            m["rollout_short_rel_err_median"] = float(np.median(sh))
            m["spectrum_rel_err_median"] = float(np.median(sp))
            m["time_sim_s"], m["time_fit_s"] = round(t1 - t0, 1), round(time.time() - t1, 1)
            log(f"[NS {label} nu={nu:<6} {regime:<6}] Trel={m['T_rel_err']:.2e} Trule={m['T_rel_err_on_rule']:.2e} "
                f"Tsel={m['T_selection_mass']:.3f} T(|k|<=K/2)={(m['T_rel_err_by_cut'][str(max(1, a.K // 2))] or float('nan')):.1e} Adiag={m['A_diag_rel_err']:.2e} forcing={(m['forcing_rel_err'] if m['forcing_rel_err'] is not None else float('nan')):.1e} "
                f"dz={m['test_rel_err_dzdt']:.2e} closure={m['closure_rel_size']:.1e} nu_t_err={m['closure']['nu_t_rel_err']:.1e} "
                f"roll_short={m['rollout_short_rel_err_median']:.1e} spec={m['spectrum_rel_err_median']:.1e} "
                f"(sim {m['time_sim_s']}s fit {m['time_fit_s']}s)")
            results.append({"nu": nu, "regime": regime, "label": label, "metrics": m})
            r_show = modes.index[(1, 2)] if (1, 2) in modes.index else 0
            fig_tensor(modes, pairs, T_l, T_t, r_show, f"learned ({label}, {regime}, nu={nu})",
                       os.path.join(out, f"fig_tensor_nu{nu}_{regime}.png"))
            if regime == "full":
                fig_closure(ks, nut_ex, nut_mo, f"2D NS, {label}, nu={nu}", os.path.join(out, f"fig_closure_nu{nu}.png"))
            np.savez_compressed(os.path.join(out, f"ops_nu{nu}_{regime}.npz"), A=A_l, b=b_l,
                                T=np.array(T_l, dtype=object), T_true=np.array(T_t, dtype=object), allow_pickle=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({"args": vars(a), "pde": "ns2d", "label": label, "results": results}, f, indent=2)
    log("saved to", out)


if __name__ == "__main__":
    main()
