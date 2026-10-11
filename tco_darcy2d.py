#!/usr/bin/env python3
"""
TCO stage 2 — 2D periodic Darcy: input FUNCTIONS (a(x), f(x)) -> solution u(x).

  -div( a(x) grad u ) = f(x)  on the 2-torus,  mean(u) = mean(f) = 0
  a(x) = exp( sigma * g(x) ),  g a smooth Gaussian random field (max|g| = 1)

Fourier (k != 0):  sum_{i+j=k} (k . j) a_i u_j = f_k
-> the operator (a, f) -> u is NONLINEAR in a (u = A(a)^{-1} f), but its structure is
   a fixed 3rd-order tensor L_k,j,i contracted with the input modes a_i:
       A(a)_{kj} = sum_i L_{k j i} a_i ,   L_true = (k . j) on the selection rule i + j = k

Learned (masked by the selection rule): u modes |kx|,|ky| <= K (k != 0), a modes |i| <= Ka.
For each output k, solve the linear least-squares problem  sum L_kji a_i u_j = f_k  over
all samples. f varies per sample, otherwise rows with f_k = 0 make L = 0 the answer.

Regimes
  closed : a truncated to Ka modes and u solved with K modes -> learned L must be exact
  full   : a has all modes, u solved with Kf (> K) modes -> truncation acts like a
           closure; the learned L is an EFFECTIVE (homogenised) operator
Prediction: given (a, f), solve the learned K-mode system. Baseline: classical Galerkin
with the TRUE tensor but truncated a and u ("naive truncation").

  python tco_darcy2d.py --quick --out runs/Darcy_smoke
"""
import argparse
import json
import os
import time

import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def square(K, zero=False):
    return np.array([(x, y) for x in range(-K, K + 1) for y in range(-K, K + 1) if zero or (x, y) != (0, 0)])


def grf(rng, N, tau, alpha):
    k = np.fft.fftfreq(N, 1.0 / N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    spec = (KX ** 2 + KY ** 2 + tau ** 2) ** (-alpha / 2)
    spec[0, 0] = 0
    g = np.fft.ifft2((rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) * spec).real
    return g / np.abs(g).max()


def coeffs(field_hat, N, modes):
    return field_hat[modes[:, 0] % N, modes[:, 1] % N]


def galerkin_matrix(a_hat, N, umodes, a_lookup_K=None):
    """A[k, j] = (k . j) a_{k-j}  for k, j in umodes. a_hat: full FFT (normalised).
    If a_lookup_K is given, a_{k-j} is set to 0 when |k-j|_inf > a_lookup_K (truncated input)."""
    d = umodes[:, None, :] - umodes[None, :, :]
    ah = a_hat[d[..., 0] % N, d[..., 1] % N]
    if a_lookup_K is not None:
        ah = np.where(np.abs(d).max(-1) <= a_lookup_K, ah, 0)
    kj = (umodes[:, None, :] * umodes[None, :, :]).sum(-1)
    return kj * ah


def make_sample(rng, N, sigma, tau, alpha, Kf, K, Ka, umodes_f, umodes, amodes, fK, truncate_a):
    g = grf(rng, N, tau, alpha)
    a = np.exp(sigma * g)
    a_hat = np.fft.fft2(a) / N ** 2
    # random forcing band-limited to fK, mean zero
    fm = square(fK)
    f_hat_vec = (rng.standard_normal(len(fm)) + 1j * rng.standard_normal(len(fm))) / (1 + (fm ** 2).sum(1))
    f_full = np.zeros((N, N), complex)
    f_full[fm[:, 0] % N, fm[:, 1] % N] = f_hat_vec
    f_full = 0.5 * (f_full + np.conj(np.roll(np.flip(f_full, (0, 1)), 1, (0, 1))))   # Hermitian -> real f
    if truncate_a:      # closed regime: a has only |i| <= Ka modes, solve with K modes
        A = galerkin_matrix(a_hat, N, umodes, a_lookup_K=Ka)
        rhs = coeffs(f_full, N, umodes)
        u = np.linalg.solve(A, rhs)
        uK = u
    else:               # full: true a, solve with Kf modes, project to K
        A = galerkin_matrix(a_hat, N, umodes_f)
        rhs = coeffs(f_full, N, umodes_f)
        u = np.linalg.solve(A, rhs)
        idx = {tuple(v): i for i, v in enumerate(umodes_f)}
        uK = u[[idx[tuple(v)] for v in umodes]]
    return coeffs(a_hat, N, amodes), uK, coeffs(f_full, N, umodes), a_hat


class Rule:
    def __init__(self, umodes, amodes):
        self.u, self.a = umodes, amodes
        uidx = {tuple(v): i for i, v in enumerate(umodes)}
        self.pairs = []               # per output k: arrays (I, J) with a_I u_J, I + J = k
        for r, k in enumerate(umodes):
            I, J = [], []
            for i, av in enumerate(amodes):
                j = uidx.get(tuple(k - av))
                if j is not None:
                    I.append(i); J.append(j)
            self.pairs.append((np.array(I, int), np.array(J, int)))

    def truth(self):
        return [(self.u[r] * self.u[J]).sum(1).astype(complex) for r, (I, J) in enumerate(self.pairs)]

    def matrix(self, L, av):
        """A(a)_{kj} = sum_i L_kji a_i."""
        n = len(self.u)
        A = np.zeros((n, n), complex)
        for r, (I, J) in enumerate(self.pairs):
            np.add.at(A[r], J, L[r] * av[I])
        return A


def fit(Av, U, Fv, rule):
    L = []
    for r, (I, J) in enumerate(rule.pairs):
        X = Av[:, I] * U[:, J]
        sc = np.sqrt(np.mean(np.abs(X) ** 2, axis=0)) + 1e-14
        L.append(np.linalg.lstsq(X / sc, Fv[:, r], rcond=1e-12)[0] / sc)
    return L


def refine_output_error(L_init, rule, Av, U, Fv, steps, lr, device, log=None):
    """Gradient descent on the SOLUTION error  ||A(a)^{-1} f - u||^2 / ||u||^2, starting from
    the least-squares tensor. The structure (selection rule) is kept; only its values move.
    This is the nonlinear (in L) training an operator-learning model would do."""
    import torch
    R = np.concatenate([np.full(len(I), r) for r, (I, J) in enumerate(rule.pairs)])
    Ii = np.concatenate([I for I, J in rule.pairs]); Jj = np.concatenate([J for I, J in rule.pairs])
    th0 = np.concatenate(L_init)
    dev = torch.device(device)
    re = torch.tensor(th0.real, device=dev, requires_grad=True)
    im = torch.tensor(th0.imag, device=dev, requires_grad=True)
    A_ = torch.tensor(Av, device=dev); U_ = torch.tensor(U, device=dev); F_ = torch.tensor(Fv, device=dev)
    Rt, It, Jt = (torch.tensor(x, device=dev) for x in (R, Ii, Jj))
    n = len(rule.u)
    opt = torch.optim.Adam([re, im], lr=lr)
    def loss_fn():
        th = torch.complex(re, im)
        M = torch.zeros(A_.shape[0], n, n, dtype=torch.complex128, device=dev)
        M.index_put_((torch.arange(A_.shape[0], device=dev)[:, None].expand(-1, len(Rt)),
                      Rt[None, :].expand(A_.shape[0], -1), Jt[None, :].expand(A_.shape[0], -1)),
                     th[None, :] * A_[:, It], accumulate=True)
        up = torch.linalg.solve(M, F_.unsqueeze(-1)).squeeze(-1)
        return ((up - U_).abs() ** 2).sum(1).div((U_.abs() ** 2).sum(1)).mean()
    for t in range(steps):
        opt.zero_grad(); l = loss_fn(); l.backward(); opt.step()
        if log and (t % max(1, steps // 5) == 0 or t == steps - 1):
            log(f"    refine step {t}: train solution error^2 = {l.item():.3e}")
    th = (re + 1j * im).detach().cpu().numpy()
    out, s0 = [], 0
    for I, J in rule.pairs:
        out.append(th[s0:s0 + len(I)]); s0 += len(I)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--Ka", type=int, default=3)
    ap.add_argument("--Kf", type=int, default=12, help="'full' solver resolution")
    ap.add_argument("--fK", type=int, default=None,
                    help="forcing bandwidth (default K). Must be >= K: an output mode with f_k = 0 in "
                         "every sample makes L = 0 the least-squares answer (not identifiable)")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.3, 0.6, 1.0])
    ap.add_argument("--sigma_ood_factor", type=float, default=1.5)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--regimes", nargs="+", default=["closed", "full"])
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=50)
    ap.add_argument("--refine_steps", type=int, default=0,
                    help="gradient-descent steps on the solution error after least squares (torch)")
    ap.add_argument("--refine_lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.quick:
        a.n_train, a.n_test, a.Kf, a.sigmas = 120, 15, 9, a.sigmas[:2]
    a.fK = a.fK or a.K
    if a.fK < a.K or 2 * a.Kf >= a.N // 2:
        raise SystemExit("need fK >= K and N > 4 Kf")
    out = a.out or ("runs/Darcy2d_refine" if a.refine_steps else "runs/Darcy2d")
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "w")

    def log(*m):
        s = " ".join(str(x) for x in m); print(s, flush=True); logf.write(s + "\n")

    umodes, umodes_f, amodes = square(a.K), square(a.Kf), square(a.Ka, zero=True)
    rule = Rule(umodes, amodes)
    L_t = rule.truth()
    nfeat = max(len(I) for I, _ in rule.pairs)
    log("args:", json.dumps(vars(a)))
    log(f"u modes {len(umodes)}, a modes {len(amodes)}, features/output <= {nfeat} "
        f"(dense would need {len(umodes) * len(amodes)})")
    results = []
    for sigma in a.sigmas:
        for regime in a.regimes:
            t0 = time.time()
            trunc = regime == "closed"

            def data(n, seed, sig):
                rng = np.random.default_rng(seed)
                S = [make_sample(rng, a.N, sig, a.tau, a.alpha, a.Kf, a.K, a.Ka, umodes_f, umodes, amodes, a.fK, trunc)
                     for _ in range(n)]
                return (np.array([s[0] for s in S]), np.array([s[1] for s in S]), np.array([s[2] for s in S]),
                        [s[3] for s in S])
            Atr, Utr, Ftr, _ = data(a.n_train, a.seed, sigma)
            t1 = time.time()
            L_l = fit(Atr, Utr, Ftr, rule)
            if a.refine_steps > 0:
                import torch
                dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
                L_l = refine_output_error(L_l, rule, Atr, Utr, Ftr, a.refine_steps, a.refine_lr, dev, log)
            Ll, Lt = np.concatenate(L_l), np.concatenate(L_t)
            m = {"L_rel_err": float(np.linalg.norm(Ll - Lt) / np.linalg.norm(Lt)), "n_features_max": int(nfeat)}
            kmag = np.sqrt((umodes ** 2).sum(1))
            by = {}
            for cut in range(1, a.K + 1):
                num = den = 0.0
                for r, (I, J) in enumerate(rule.pairs):
                    if kmag[r] <= cut and J.size:
                        ok = kmag[J] <= cut
                        num += float(np.sum(np.abs(L_l[r][ok] - L_t[r][ok]) ** 2)); den += float(np.sum(np.abs(L_t[r][ok]) ** 2))
                by[str(cut)] = float(np.sqrt(num / den)) if den > 0 else None
            m["L_rel_err_by_cut"] = by
            for tag, sig, seed in [("test", sigma, a.seed + 10_000), ("ood", sigma * a.sigma_ood_factor, a.seed + 20_000)]:
                Ate, Ute, Fte, ahats = data(a.n_test, seed, sig)
                e_l, e_g = [], []
                for s in range(a.n_test):
                    u_l = np.linalg.solve(rule.matrix(L_l, Ate[s]), Fte[s])
                    A_naive = galerkin_matrix(ahats[s], a.N, umodes, a_lookup_K=a.Ka)
                    u_g = np.linalg.solve(A_naive, Fte[s])
                    e_l.append(np.linalg.norm(u_l - Ute[s]) / np.linalg.norm(Ute[s]))
                    e_g.append(np.linalg.norm(u_g - Ute[s]) / np.linalg.norm(Ute[s]))
                m[f"{tag}_u_rel_err_learned"] = float(np.median(e_l))
                m[f"{tag}_u_rel_err_naive_truncation"] = float(np.median(e_g))
            m["time_data_s"], m["time_fit_s"] = round(t1 - t0, 1), round(time.time() - t1, 1)
            log(f"[Darcy sigma={sigma:<4} {regime:<6}] Lrel={m['L_rel_err']:.2e} L(|k|<=K/2)={(by[str(max(1, a.K // 2))] or float('nan')):.1e} "
                f"u_test: learned={m['test_u_rel_err_learned']:.2e} naive={m['test_u_rel_err_naive_truncation']:.2e} | "
                f"u_OOD(sigma x{a.sigma_ood_factor}): learned={m['ood_u_rel_err_learned']:.2e} naive={m['ood_u_rel_err_naive_truncation']:.2e} "
                f"({m['time_data_s']}s + {m['time_fit_s']}s)")
            results.append({"sigma": sigma, "regime": regime, "metrics": m})
            np.savez_compressed(os.path.join(out, f"L_sigma{sigma}_{regime}.npz"),
                                L=np.array(L_l, dtype=object), L_true=np.array(L_t, dtype=object), allow_pickle=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({"args": vars(a), "pde": "darcy2d", "results": results}, f, indent=2)
    log("saved to", out)


if __name__ == "__main__":
    main()
