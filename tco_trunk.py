#!/usr/bin/env python3
"""
TCO M6c — learn the basis with a neural TRUNK network (PyTorch), jointly with the
coupling tensor, then fix the gauge and read the tensor.

Model
  trunk  phi(x) = MLP([cos x, sin x]) in R^r      (only "the domain is periodic" is assumed)
  basis  Phi = QR-orthonormalised trunk outputs on the sensor grid (mean inner product)
  coeffs a = <u, Phi>,  da = <u_t, Phi>
  dynamics da/dt = A a + T(a,a)   — for a GIVEN basis, (A, T) is the closed-form least-squares
                                    solution (variable projection): it is optimal at every step
  loss = reconstruction(u ≈ a Phi) + beta * normalised dynamics residual
         beta = 0  -> only representation (should land on the POD subspace)
         beta > 0  -> also asks for a basis in which the dynamics close (dynamics-aware basis)
After training: (A, T) are refit by least squares in the learned basis and evaluated exactly
like tco_pod.py (map to real Fourier coordinates and compare with the truth; purity of the
basis and of the eigenbasis of the learned A). A POD baseline on the same data is included.

Needs tco_pde.py and tco_pod.py in the same folder, and torch (GPU used if available).
  python tco_trunk.py --pde burgers --quick --out runs/M6c_smoke
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from tco_pde import PDE, Blocks, fit, model_rhs, DEFAULTS
from tco_pod import (simulate_grid, real_fourier_basis, pod_basis, extract_linear_quadratic,
                     rollout_matrix_if, b_to_z, z_to_b, fig_pod_modes)


class Trunk(torch.nn.Module):
    def __init__(self, r, width, depth):
        super().__init__()
        layers, d = [], 2
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]; d = width
        layers += [torch.nn.Linear(d, r)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(torch.stack([torch.cos(x), torch.sin(x)], -1))


def quad_features(a, P, Q):
    return torch.cat([a, a[:, P] * a[:, Q]], 1)


def train_trunk(U, Ut, r, steps, lr, beta, width, depth, ridge, seed, device, log):
    torch.manual_seed(seed)
    ng = U.shape[1]
    x = torch.linspace(0, 2 * np.pi, ng + 1, dtype=torch.float64, device=device)[:-1]
    net = Trunk(r, width, depth).to(device).double()
    U_, Ut_ = torch.tensor(U, device=device), torch.tensor(Ut, device=device)
    P, Q = np.triu_indices(r)
    P, Q = torch.tensor(P, device=device), torch.tensor(Q, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist = []
    for t in range(steps):
        Qm, _ = torch.linalg.qr(net(x))
        Phi = Qm * np.sqrt(ng)
        a, da = U_ @ Phi / ng, Ut_ @ Phi / ng
        rec = ((U_ - a @ Phi.T) ** 2).sum() / (U_ ** 2).sum()
        if beta > 0:
            X = quad_features(a, P, Q)
            sc = X.pow(2).mean(0).sqrt() + 1e-12
            Xs = X / sc
            G = Xs.T @ Xs + ridge * torch.eye(Xs.shape[1], dtype=X.dtype, device=device)
            W = torch.linalg.solve(G, Xs.T @ da)
            dyn = ((Xs @ W - da) ** 2).sum() / (da ** 2).sum()
        else:
            dyn = torch.zeros((), dtype=U_.dtype, device=device)
        loss = rec + beta * dyn
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if t % max(1, steps // 10) == 0 or t == steps - 1:
            hist.append([t, float(rec), float(dyn)])
            log(f"    step {t}: recon {float(rec):.3e}  dyn {float(dyn):.3e}")
    with torch.no_grad():
        Qm, _ = torch.linalg.qr(net(x))
    return (Qm * np.sqrt(ng)).cpu().numpy(), hist


def evaluate_basis(Phi, U, Ut, Ute, Utte, tid, bl, r, K, n_grid, R, kl, A_tb, T_tb, P, Q, rule, dt_save, method):
    proj = lambda X: X @ Phi / n_grid
    Atr, dAtr, Ate, dAte = proj(U), proj(Ut), proj(Ute), proj(Utte)
    ops, _ = fit(Atr.astype(complex), dAtr.astype(complex), None, bl, 1e-10, method, 0.02, 8, False, False)
    ops = {k: v.real for k, v in ops.items()}
    f_pod = lambda X: model_rhs(X, None, ops, bl).real
    Qm = R.T @ Phi / n_grid
    Qp = np.linalg.pinv(Qm)
    f_b = lambda B: (Qm @ f_pod(B @ Qp.T).T).T
    A_lb, T_lb, _, _ = extract_linear_quadratic(f_b, r)
    absT = np.abs(T_lb)
    m = {"T_rel_err": float(np.linalg.norm(T_lb - T_tb) / np.linalg.norm(T_tb)),
         "T_rel_err_on_rule": float(np.linalg.norm((T_lb - T_tb)[rule]) / np.linalg.norm(T_tb[rule])),
         "T_selection_mass": float(absT[rule].sum() / (absT.sum() + 1e-30)),
         "A_rel_err": float(np.linalg.norm(A_lb - A_tb) / (np.linalg.norm(A_tb) + 1e-30)),
         "span_mismatch": float(np.linalg.norm(Qm.T @ Qm - np.eye(r)) / np.sqrt(r)),
         "recon_rel_err_test": float(np.linalg.norm(Ute - Ate @ Phi.T) / np.linalg.norm(Ute))}
    spec = np.abs(np.fft.rfft(Phi, axis=0)) ** 2
    spec = spec / spec.sum(0, keepdims=True)
    m["basis_purity_mean"] = float(spec.max(0).mean())
    lam, Vv = np.linalg.eig(ops["A"])
    G = Phi @ Vv
    sp = np.abs(np.fft.fft(G, axis=0)) ** 2
    kk = np.minimum(np.arange(n_grid), n_grid - np.arange(n_grid))
    spk = np.array([sp[kk == j].sum(0) for j in range(n_grid // 2 + 1)])
    spk = spk / spk.sum(0, keepdims=True)
    m["eig_purity_mean"] = float(spk.max(0).mean())
    m["test_rel_err_dzdt"] = float(np.linalg.norm(f_pod(Ate) - dAte) / np.linalg.norm(dAte))
    errs = np.array([rollout_matrix_if(Ate[tid == tr][0], f_pod, ops["A"], Ate[tid == tr], dt_save)
                     for tr in np.unique(tid)[:5]])
    m["rollout_short_rel_err_median5"] = float(np.median(errs[:, 1]))
    m["spectrum_rel_err_median5"] = float(np.median(errs[:, 2]))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", choices=["burgers", "kdv", "ks", "fisher"], required=True)
    ap.add_argument("--params", type=float, nargs="+", default=None)
    ap.add_argument("--regimes", nargs="+", default=["closed", "full"])
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n_grid", type=int, default=64)
    ap.add_argument("--kmax_full", type=int, default=128)
    ap.add_argument("--n_traj", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--ridge", type=float, default=1e-8)
    ap.add_argument("--method", choices=["lstsq", "stlsq"], default="lstsq")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dp, dT, ddt, dsv, damp, dmean = DEFAULTS[a.pde]
    params, t_final = a.params or dp, dT
    if a.quick:
        a.n_traj, a.n_test, a.kmax_full, a.steps = 40, 10, 48, 600
        t_final, params = t_final / 4, params[:1]
    torch.set_default_dtype(torch.float64)
    out = a.out or f"runs/M6c_{a.pde}_beta{a.beta}"
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "w")

    def log(*m):
        s = " ".join(str(x) for x in m); print(s, flush=True); logf.write(s + "\n")

    log("args:", json.dumps({**vars(a), "params": params}))
    r = 2 * a.K + 1
    R, kl = real_fourier_basis(a.n_grid, a.K)
    bl, blF = Blocks(a.K, 0, "none", False), Blocks(a.K, 0, "none", False)
    results = []
    for prm in params:
        pde = PDE(a.pde, prm)
        A_z, T_z, _ = blF.truth(pde)
        f_true_b = lambda B: z_to_b(model_rhs(b_to_z(B, a.K), None, {"A": A_z, "T": T_z, "C": np.zeros((r, 0)),
                                                                     "L": np.zeros((r, 0))}, blF), a.K)
        A_tb, T_tb, P, Q = extract_linear_quadratic(f_true_b, r)
        rule = np.zeros_like(T_tb, bool)
        for o in range(r):
            rule[o] = (kl[o] == kl[P] + kl[Q]) | (kl[o] == np.abs(kl[P] - kl[Q]))
        for regime in a.regimes:
            t0 = time.time()
            ks = a.K if regime == "closed" else a.kmax_full
            sim = lambda n, seed: simulate_grid(pde, ks, a.n_grid, n, t_final, ddt, dsv, np.random.default_rng(seed),
                                                damp, dmean, a.K)
            U, Ut, _ = sim(a.n_traj, a.seed)
            Ute, Utte, tid = sim(a.n_test, a.seed + 10_000)
            log(f"[{a.pde} p={prm} {regime}] training trunk (beta={a.beta}, {a.steps} steps, {a.device})")
            t1 = time.time()
            Phi_t, hist = train_trunk(U, Ut, r, a.steps, a.lr, a.beta, a.width, a.depth, a.ridge, a.seed, a.device, log)
            t2 = time.time()
            args_e = (U, Ut, Ute, Utte, tid, bl, r, a.K, a.n_grid, R, kl, A_tb, T_tb, P, Q, rule, ddt * dsv, a.method)
            m_trunk = evaluate_basis(Phi_t, *args_e)
            Phi_p, _ = pod_basis(U, r)
            m_pod = evaluate_basis(Phi_p, *args_e)
            for name, mm in [("trunk", m_trunk), ("pod", m_pod)]:
                log(f"[{a.pde} p={prm:<5} {regime:<6} {name:<5}] Trel={mm['T_rel_err']:.2e} Trule={mm['T_rel_err_on_rule']:.2e} "
                    f"Tsel={mm['T_selection_mass']:.3f} Arel={mm['A_rel_err']:.2e} span={mm['span_mismatch']:.1e} "
                    f"recon={mm['recon_rel_err_test']:.1e} purity={mm['basis_purity_mean']:.3f} eig_purity={mm['eig_purity_mean']:.3f} "
                    f"dz={mm['test_rel_err_dzdt']:.2e} roll_short={mm['rollout_short_rel_err_median5']:.1e} "
                    f"spec={mm['spectrum_rel_err_median5']:.1e}")
            results.append({"pde": a.pde, "param": prm, "regime": regime, "beta": a.beta,
                            "trunk": m_trunk, "pod": m_pod, "train_history": hist,
                            "time_sim_s": round(t1 - t0, 1), "time_train_s": round(t2 - t1, 1)})
            np.savez_compressed(os.path.join(out, f"trunk_{prm}_{regime}.npz"), Phi_trunk=Phi_t, Phi_pod=Phi_p)
            fig_pod_modes(Phi_t, np.full(r, np.nan), a.n_grid, os.path.join(out, f"fig_trunk_modes_{prm}_{regime}.png"),
                          f"{a.pde}, param={prm}, {regime}: basis learned by neural trunk (beta={a.beta})")
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({"args": vars(a), "results": results}, f, indent=2)
    log("saved to", out)


if __name__ == "__main__":
    main()
