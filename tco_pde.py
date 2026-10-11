#!/usr/bin/env python3
"""
TCO multi-PDE framework — learn interpretable coupling tensors from data and
compare them with the exact Galerkin tensors, for several 1D periodic PDEs.

State: Fourier modes z = [c_{-K}..c_K] (complex), u(x) = sum_k c_k e^{ikx}.
Learned model (blocks are switched on/off by options):
    dz_k/dt = sum_j     A_kj  z_j                    linear
            + sum_{p<=q} T_k,pq z_p z_q              quadratic triads  (interpreted)
            + sum_{p<=q<=r} C_k,pqr z_p z_q z_r      cubic closure     (--closure cubic)
            + sum_{i,j} L_k,ji v_i z_j               input x state     (PDE 'varvisc' only)
where v_i are the Fourier modes of an INPUT FUNCTION (here the viscosity field nu(x)).

PDEs (--pde), parameter swept with --params:
  burgers   u_t + u u_x = nu u_xx                 param = nu
  kdv       u_t + u u_x + delta u_xxx = 0         param = delta   (energy conserving)
  ks        u_t + u u_x + u_xx + nu u_xxxx = 0    param = nu      (chaotic for small nu)
  fisher    u_t = D u_xx + u - u^2                param = D       (non-derivative nonlinearity)
  allencahn u_t = eps u_xx + u - u^3              param = eps     (cubic physics -> 4th-order tensor C;
                                                                    needs --closure cubic)
  varvisc   u_t + u u_x = (nu(x) u_x)_x           param = nu0 = mean of nu(x)
            nu(x) = nu_m (1 + eps * xi(x)), nu_m = nu0 * U(0.5, 1.5) per trajectory,
            xi random, band-limited (|i|<=Kv), max|xi| = 1
            -> E4: input FUNCTION nu(x) enters as a tensor index, L_k,j,i = -k j  (i+j=k)

Exact truths (canonical, pairs p<=q):
  quadratic N_k = g(k) (u^2)_k  ->  T[k,(p,q)] = g(k) * (2 if p<q else 1) if p+q=k
     burgers/kdv/ks/varvisc: g = -i k / 2 ;  fisher: g = -1
  linear diag A_kk: burgers -nu k^2 ; kdv +i delta k^3 ; ks k^2 - nu k^4 ;
                    fisher 1 - D k^2 ; varvisc 0 (all diffusion is in L)

Closure analysis (automatic, every run, on TEST data):
  exact closure      = dZ_data - GalerkinRHS_true(Z)        (what truncated modes do)
  model closure      = ModelRHS(Z) - GalerkinRHS_true(Z)    (what the model implies)
  energy transfer    Pi(k) = Re<conj(z_k) closure_k> / <|z_k|^2>
  eddy viscosity     nu_t(k) = -Pi(k) / k^2   (k >= 1)
  -> fig_closure_{param}.png, and 'closure' entry in results.json

Only numpy + matplotlib.
  python tco_pde.py --pde burgers --quick --out runs/smoke_burgers
  python tco_pde.py --pde varvisc --mask_selection --closure cubic --out runs/E4
"""
import argparse
import json
import os
import time

import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


# ============================================================================
# PDE registry
# ============================================================================
class PDE:
    def __init__(self, name, param):
        self.name, self.param = name, param
        self.has_input = name == "varvisc"
        self.cubic = name == "allencahn"           # physics itself is cubic (4th-order tensor)
        self.degree = 3 if self.cubic else 2
        self.energy_conserving_quad = name not in ("fisher", "allencahn")

    def lin(self, k):
        """Diagonal linear symbol on signed or unsigned wavenumbers k (float array)."""
        p, n = self.param, self.name
        k = np.asarray(k, float)
        if n == "burgers":
            return (-p * k ** 2).astype(complex)
        if n == "kdv":
            return 1j * p * k ** 3
        if n == "ks":
            return (k ** 2 - p * k ** 4).astype(complex)
        if n == "fisher":
            return (1.0 - p * k ** 2).astype(complex)
        if n == "varvisc":                      # true A = 0; solver uses mean visc in IF
            return np.zeros_like(k, complex)
        if n == "allencahn":
            return (1.0 - p * k ** 2).astype(complex)
        raise ValueError(n)

    def g(self, k):
        """Quadratic coefficient: N_k = g(k) (u^2)_k."""
        k = np.asarray(k, float)
        if self.name == "fisher":
            return -np.ones_like(k, complex)
        if self.name == "allencahn":
            return np.zeros_like(k, complex)
        return -0.5j * k


DEFAULTS = {
    #           params                      t_final dt     save  amp  mean_amp
    "burgers": ([0.5, 0.2, 0.1, 0.05, 0.02], 2.0, 1e-3, 20, 1.0, 0.5),
    "kdv":     ([0.1, 0.05, 0.02],           2.0, 5e-4, 40, 1.0, 0.5),
    "ks":      ([0.1, 0.05, 0.02],           5.0, 1e-3, 50, 1.0, 0.5),
    "fisher":  ([0.1, 0.05, 0.02],           2.0, 1e-3, 20, 0.3, 0.0),
    "varvisc": ([0.2, 0.1, 0.05],            2.0, 1e-3, 20, 1.0, 0.5),
    "allencahn": ([0.1, 0.05, 0.02],         2.0, 1e-3, 20, 1.0, 0.3),
}


# ============================================================================
# Spectral solver (exactly dealiased Galerkin truncation at |k| <= kmax)
# ============================================================================
class Solver:
    def __init__(self, pde, kmax, Kv=0):
        self.pde, self.kmax = pde, kmax
        # dealiasing: a degree-d product of fields with |k|<=kmax is exact on M >= (d+1) kmax + 1
        self.M = int(2 ** np.ceil(np.log2((pde.degree + 1) * max(kmax, Kv) + 2)))
        self.k = np.arange(self.M // 2 + 1).astype(float)
        self.mask = (self.k <= kmax).astype(float)
        self.L = pde.lin(self.k)
        self.gk = pde.g(self.k)
        self.nu_hat = None                     # varvisc: rfft coeffs of nu'(x) = nu - nu0

    def set_input(self, nu_hat_full, nu0):
        """nu_hat_full: rfft-normalised coefficients of nu(x) (length M//2+1)."""
        self.L = -nu0 * self.k ** 2 + 0j
        nh = nu_hat_full.copy()
        nh[0] -= nu0
        self.nu_hat = nh

    def grid(self, c):
        return np.fft.irfft(c * self.M, n=self.M)

    def spec(self, f):
        return np.fft.rfft(f) / self.M

    def nonlinear(self, c):
        u = self.grid(c)
        N = self.gk * self.spec(u * u)
        if self.pde.cubic:
            N = N - self.spec(u * u * u)
        if self.nu_hat is not None:
            ux = self.grid(1j * self.k * c)
            nup = self.grid(self.nu_hat)
            N = N + 1j * self.k * self.spec(nup * ux)
        return N * self.mask

    def rhs(self, c):
        return self.L * c + self.nonlinear(c)

    def step(self, c, dt):
        E, E2 = np.exp(self.L * dt), np.exp(self.L * dt / 2)
        a = dt * self.nonlinear(c)
        b = dt * self.nonlinear(E2 * (c + a / 2))
        d = dt * self.nonlinear(E2 * c + b / 2)
        e = dt * self.nonlinear(E * c + E2 * d)
        return E * c + (E * a + 2 * E2 * (b + d) + e) / 6.0


def bandlimited_field(rng, kmax_f, nk, decay=1.0):
    c = np.zeros(nk, complex)
    ks = np.arange(1, kmax_f + 1)
    c[1:kmax_f + 1] = rng.uniform(0.5, 1.5, ks.size) / ks ** decay * np.exp(1j * rng.uniform(0, 2 * np.pi, ks.size))
    M = 2 * (nk - 1)
    u = np.fft.irfft(c * M, n=M)
    return c / (np.max(np.abs(u)) + 1e-12)


def to_state(c, K):
    pos = c[:K + 1]
    return np.concatenate([np.conj(pos[1:][::-1]), pos])


def simulate(pde, kmax_solver, K, Kv, n_traj, t_final, dt, save_every, rng, amp, mean_amp,
             kmax_ic, eps):
    sol = Solver(pde, kmax_solver, Kv)
    nk = sol.M // 2 + 1
    Zs, dZs, Vs, tid = [], [], [], []
    n_steps = int(round(t_final / dt))
    for tr in range(n_traj):
        if pde.name == "fisher":
            c = amp * bandlimited_field(rng, min(kmax_ic, kmax_solver), nk)
            c[0] = rng.uniform(0.2, 0.6)
        else:
            c = amp * bandlimited_field(rng, min(kmax_ic, kmax_solver), nk)
            c[0] = rng.uniform(-mean_amp, mean_amp)
        c = c * sol.mask
        v = None
        if pde.has_input:
            # the MEAN viscosity must also vary across trajectories, otherwise the
            # feature nu_0 * z_k is collinear with the linear feature z_k
            nu_mean = pde.param * rng.uniform(0.5, 1.5)
            xi = bandlimited_field(rng, Kv, nk, decay=0.5)
            nu_hat = nu_mean * eps * xi
            nu_hat[0] = nu_mean
            sol.set_input(nu_hat, nu_mean)
            v = to_state(nu_hat, Kv)
        for s in range(n_steps + 1):
            if s % save_every == 0:
                Zs.append(to_state(c, K)); dZs.append(to_state(sol.rhs(c), K)); tid.append(tr)
                if v is not None:
                    Vs.append(v)
            if s < n_steps:
                c = sol.step(c, dt)
                if not np.all(np.isfinite(c)) or np.max(np.abs(c)) > 1e6:
                    raise RuntimeError(f"solver blew up ({pde.name}, param={pde.param}); lower --dt")
    V = np.array(Vs) if Vs else None
    return np.array(Zs), np.array(dZs), V, np.array(tid)


# ============================================================================
# Feature blocks, truth, masks
# ============================================================================
class Blocks:
    def __init__(self, K, Kv, closure, has_input):
        self.K, self.n = K, 2 * K + 1
        self.m = np.arange(-K, K + 1)
        self.P, self.Q = np.triu_indices(self.n)
        n = self.n
        if closure == "cubic":
            t = [(a, b, c) for a in range(n) for b in range(a, n) for c in range(b, n)]
            self.A3, self.B3, self.C3 = map(np.array, zip(*t))
        else:
            self.A3 = self.B3 = self.C3 = np.zeros(0, int)
        self.Kv = Kv if has_input else 0
        self.mv = np.arange(-self.Kv, self.Kv + 1) if has_input else np.zeros(0, int)
        nv = self.mv.size
        self.IV, self.JZ = (np.repeat(np.arange(nv), n), np.tile(np.arange(n), nv)) if nv else (np.zeros(0, int),) * 2
        sizes = [n, self.P.size, self.A3.size, self.IV.size]
        st = np.cumsum([0] + sizes)
        self.sl, self.sq, self.sc, self.si = [slice(st[i], st[i + 1]) for i in range(4)]
        self.F = st[-1]

    def features(self, Z, V=None):
        parts = [Z, Z[:, self.P] * Z[:, self.Q]]
        if self.A3.size:
            parts.append(Z[:, self.A3] * Z[:, self.B3] * Z[:, self.C3])
        if self.IV.size:
            parts.append(V[:, self.IV] * Z[:, self.JZ])
        return np.concatenate(parts, axis=1)

    def allowed(self):
        m, n = self.m, self.n
        al = np.zeros((n, self.F), bool)
        for r, k in enumerate(m):
            al[r, self.sl] = m == k
            al[r, self.sq] = m[self.P] + m[self.Q] == k
            if self.A3.size:
                al[r, self.sc] = m[self.A3] + m[self.B3] + m[self.C3] == k
            if self.IV.size:
                al[r, self.si] = self.mv[self.IV] + m[self.JZ] == k
        return al

    def truth(self, pde):
        m, n = self.m, self.n
        A = np.diag(pde.lin(m))
        g = pde.g(m)
        T = np.zeros((n, self.P.size), complex)
        p, q = m[self.P], m[self.Q]
        for r, k in enumerate(m):
            s = p + q == k
            T[r, s] = g[r] * np.where(p[s] < q[s], 2.0, 1.0)
        Lt = np.zeros((n, self.IV.size), complex)
        if self.IV.size:
            i, j = self.mv[self.IV], m[self.JZ]
            for r, k in enumerate(m):
                s = i + j == k
                Lt[r, s] = -k * j[s]
        return A, T, Lt

    def truth_C(self, pde):
        """Exact cubic tensor (Allen-Cahn: N = -(u^3)_k), canonical sorted triples."""
        C = np.zeros((self.n, self.A3.size), complex)
        if pde.cubic and self.A3.size:
            m = self.m
            a, b, c = self.A3, self.B3, self.C3
            mult = np.where((a == b) & (b == c), 1.0, np.where((a == b) | (b == c) | (a == c), 3.0, 6.0))
            for r, k in enumerate(m):
                s = m[a] + m[b] + m[c] == k
                C[r, s] = -mult[s]
        return C


def energy_constraints(bl):
    m = bl.m
    groups = {}
    for r, k in enumerate(m):
        for f, (pi, qi) in enumerate(zip(bl.P, bl.Q)):
            groups.setdefault(tuple(sorted((-k, m[pi], m[qi]))), []).append((r, f))
    eqs, seen = [], set()
    for key, ent in groups.items():
        if key in seen:
            continue
        nkey = tuple(sorted(-x for x in key)); seen |= {key, nkey}
        bar = groups.get(nkey, [])
        if nkey == key:
            eqs.append(([(r, f, 1.0) for r, f in ent], "re"))
        else:
            eqs.append(([(r, f, 1.0) for r, f in ent] + [(r, f, 1.0) for r, f in bar], "re"))
            eqs.append(([(r, f, 1.0) for r, f in ent] + [(r, f, -1.0) for r, f in bar], "im"))
    return eqs


def cons_residual(T, eqs):
    if not eqs:
        return 0.0
    return float(max(abs(sum(s * (T[r, f].real if kd == "re" else T[r, f].imag) for r, f, s in ent))
                     for ent, kd in eqs))


# ============================================================================
# Fit: per-output complex LS in real form, masks/thresholds as penalties,
#      exact energy constraint on the quadratic block by KKT projection
# ============================================================================
def fit(Z, dZ, V, bl, ridge, method, thresh, n_iter, energy, mask_sel):
    X = bl.features(Z, V)
    scale = np.sqrt(np.mean(np.abs(X) ** 2, axis=0)) + 1e-14
    Xs = X / scale
    G = Xs.conj().T @ Xs
    Bv = Xs.conj().T @ dZ
    n, F = bl.n, bl.F
    big = 1e8 * np.mean(np.real(np.diag(G)))
    allow = bl.allowed() if mask_sel else np.ones((n, F), bool)
    pen = np.where(allow, ridge, big)
    Rb = np.block([[G.real, -G.imag], [G.imag, G.real]])
    eqs = energy_constraints(bl) if energy else []
    q0 = bl.sq.start

    def solve(pen):
        W = np.zeros((F, n), complex); Rinv = []
        for r in range(n):
            Ri = np.linalg.inv(Rb + np.diag(np.concatenate([pen[r], pen[r]])))
            Rinv.append(Ri)
            x = Ri @ np.concatenate([Bv[:, r].real, Bv[:, r].imag])
            W[:, r] = x[:F] + 1j * x[F:]
        if eqs:
            W = project(W, Rinv)
        return W

    def project(W, Rinv):
        mq = len(eqs)
        Ck = [np.zeros((mq, 2 * F)) for _ in range(n)]
        for e, (ent, kd) in enumerate(eqs):
            for r, f, s in ent:
                Ck[r][e, (0 if kd == "re" else F) + q0 + f] += s / scale[q0 + f]
        Mm, res, RC = np.zeros((mq, mq)), np.zeros(mq), []
        for r in range(n):
            x = np.concatenate([W[:, r].real, W[:, r].imag])
            RCr = Rinv[r] @ Ck[r].T; RC.append(RCr)
            Mm += Ck[r] @ RCr; res += Ck[r] @ x
        lam = np.linalg.lstsq(Mm, res, rcond=1e-12)[0]
        for r in range(n):
            x = np.concatenate([W[:, r].real, W[:, r].imag]) - RC[r] @ lam
            W[:, r] = x[:F] + 1j * x[F:]
        return W

    W = solve(pen)
    if method == "stlsq":
        for _ in range(n_iter):
            mag = np.abs(W / scale[:, None])
            cut = np.zeros_like(mag, bool)
            for s in (bl.sl, bl.sq, bl.sc, bl.si):
                if s.stop > s.start:
                    blk = mag[s, :]
                    cut[s, :] = blk < thresh * (blk.max(axis=0, keepdims=True) + 1e-30)
            pn = pen.copy(); pn[cut.T] = big
            if np.array_equal(pn, pen):
                break
            pen = pn; W = solve(pen)
    W = W / scale[:, None]
    return {"A": W[bl.sl].T, "T": W[bl.sq].T, "C": W[bl.sc].T, "L": W[bl.si].T}, eqs


def model_rhs(Zb, Vb, ops, bl):
    out = Zb @ ops["A"].T + (Zb[:, bl.P] * Zb[:, bl.Q]) @ ops["T"].T
    if bl.A3.size:
        out = out + (Zb[:, bl.A3] * Zb[:, bl.B3] * Zb[:, bl.C3]) @ ops["C"].T
    if bl.IV.size:
        out = out + (Vb[:, bl.IV] * Zb[:, bl.JZ]) @ ops["L"].T
    return out


# ============================================================================
# Metrics
# ============================================================================
def tensor_metrics(Tl, Tt, rule, prefix):
    a = np.abs(Tl)
    d = {f"{prefix}rel_err": float(np.linalg.norm(Tl - Tt) / (np.linalg.norm(Tt) + 1e-30)),
         f"{prefix}rel_err_on_rule": float(np.linalg.norm((Tl - Tt)[rule]) / (np.linalg.norm(Tt[rule]) + 1e-30)),
         f"{prefix}selection_mass": float(a[rule].sum() / (a.sum() + 1e-30))}
    return d


def closure_spectrum(Z, dZ, V, ops, truth_ops, bl):
    """Energy-transfer spectra (computed on the data passed in — use TEST data):
    exact closure vs model-implied closure, and the model closure split by block
    (where did the model put the effect of the unresolved modes?)."""
    gal = model_rhs(Z, V, truth_ops, bl)
    exact = dZ - gal
    model = model_rhs(Z, V, ops, bl) - gal
    zero = {k: np.zeros_like(v) for k, v in ops.items()}
    parts = {}
    for name in ["A", "T", "C", "L"]:
        if ops[name].size == 0:
            continue
        o = dict(zero); o[name] = ops[name] - truth_ops[name]
        parts[name] = model_rhs(Z, V, o, bl)
    E = np.mean(np.abs(Z) ** 2, axis=0) + 1e-30
    pos = bl.m >= 1
    k = bl.m[pos].astype(float)

    def spec(X):
        Pi = np.mean(np.real(np.conj(Z) * X), axis=0) / E
        return 0.5 * (Pi[pos] + Pi[::-1][pos])          # average +k / -k
    out = {"k": k.tolist()}
    for name, X in [("exact", exact), ("model", model)] + [(f"model_{n}", X) for n, X in parts.items()]:
        Pi = spec(X)
        out[f"Pi_{name}"] = Pi.tolist()
        out[f"nu_t_{name}"] = (-Pi / k ** 2).tolist()
    out["closure_rel_size"] = float(np.linalg.norm(exact) / (np.linalg.norm(dZ) + 1e-30))
    ex, mo = np.array(out["nu_t_exact"]), np.array(out["nu_t_model"])
    out["nu_t_rel_err"] = float(np.linalg.norm(mo - ex) / (np.linalg.norm(ex) + 1e-30))
    return out


def rollout(z0, v, ops, bl, Zref, dt_save, n_sub=10):
    """Integrating-factor RK4: the diagonal of the learned A is integrated exactly
    (stiff decay of high modes, e.g. KS), everything else explicitly.
    Returns (final_err, short_err, spectrum_err):
      final_err    relative error at the last saved time
      short_err    relative error after ~10% of the trajectory (meaningful for chaos)
      spectrum_err relative error of the time-averaged energy spectrum |z_k|^2 over
                   the second half (statistical skill; meaningful for chaos)"""
    vb = v[None, :] if v is not None else None
    D = np.diag(ops["A"]).copy()
    full = lambda z: model_rhs(z[None, :], vb, ops, bl)[0]
    N = lambda z: full(z) - D * z
    h = dt_save / n_sub
    E, E2 = np.exp(D * h), np.exp(D * h / 2)
    nT = Zref.shape[0]
    t_short = max(1, (nT - 1) // 10)
    z = z0.copy()
    traj = [z.copy()]
    for t in range(1, nT):
        for _ in range(n_sub):
            a = h * N(z); b = h * N(E2 * (z + a / 2)); c = h * N(E2 * z + b / 2); d = h * N(E * z + E2 * c)
            z = E * z + (E * a + 2 * E2 * (b + c) + d) / 6.0
        if not np.all(np.isfinite(z)) or np.linalg.norm(z) > 1e6:
            return float("inf"), float("inf"), float("inf")
        traj.append(z.copy())
    traj = np.array(traj)
    rel = lambda t: float(np.linalg.norm(traj[t] - Zref[t]) / (np.linalg.norm(Zref[t]) + 1e-14))
    half = nT // 2
    Sm = np.mean(np.abs(traj[half:]) ** 2, axis=0)
    Sr = np.mean(np.abs(Zref[half:]) ** 2, axis=0)
    spec = float(np.linalg.norm(Sm - Sr) / (np.linalg.norm(Sr) + 1e-30))
    return rel(nT - 1), rel(t_short), spec


def evaluate(ops, truth, bl, Zte, dZte, Vte, tid, eqs, dt_save, pde):
    A_t, T_t, L_t = truth
    m = bl.m
    rule_T = m[bl.P][None, :] + m[bl.Q][None, :] == m[:, None]
    out = tensor_metrics(ops["T"], T_t, rule_T, "T_")
    if pde.cubic:   # true quadratic tensor is zero: report learned T relative to the cubic physics
        C_t = bl.truth_C(pde)
        sp = float(np.linalg.norm(ops["T"]) / (np.linalg.norm(C_t) + 1e-30))
        out["T_rel_err"] = out["T_rel_err_on_rule"] = sp
        out["T_spurious_rel_to_C"] = sp
        m3 = bl.m
        rule_C = m3[bl.A3][None, :] + m3[bl.B3][None, :] + m3[bl.C3][None, :] == m3[:, None]
        out.update(tensor_metrics(ops["C"], C_t, rule_C, "C_"))
    dA, dAt = np.diag(ops["A"]), np.diag(A_t)
    if np.linalg.norm(dAt) > 0:
        out["A_diag_rel_err"] = float(np.linalg.norm(dA - dAt) / np.linalg.norm(dAt))
    else:  # varvisc: true A = 0; report size relative to mean diffusion
        out["A_diag_rel_to_visc"] = float(np.linalg.norm(dA) / np.linalg.norm(pde.param * m ** 2))
    out["A_offdiag_ratio"] = float(np.linalg.norm(ops["A"] - np.diag(dA)) / (np.linalg.norm(dA) + 1e-30))
    if bl.IV.size:
        rule_L = bl.mv[bl.IV][None, :] + m[bl.JZ][None, :] == m[:, None]
        out.update(tensor_metrics(ops["L"], L_t, rule_L, "L_"))
    Nq = (Zte[:, bl.P] * Zte[:, bl.Q]) @ ops["T"].T
    num = np.abs(np.real(np.sum(np.conj(Zte) * Nq, axis=1)))
    out["T_energy_violation"] = float(np.mean(num / (np.linalg.norm(Zte, axis=1) * np.linalg.norm(Nq, axis=1) + 1e-14)))
    if eqs:
        out["energy_constraint_residual"] = cons_residual(ops["T"], eqs)
    if bl.A3.size:
        Nc = (Zte[:, bl.A3] * Zte[:, bl.B3] * Zte[:, bl.C3]) @ ops["C"].T
        out["cubic_to_quad_ratio"] = float(np.linalg.norm(Nc) / (np.linalg.norm(Nq) + 1e-14))
    pred = model_rhs(Zte, Vte, ops, bl)
    out["test_rel_err_dzdt"] = float(np.linalg.norm(pred - dZte) / np.linalg.norm(dZte))
    errs = []
    for tr in np.unique(tid)[:5]:
        sel = tid == tr
        v = Vte[sel][0] if Vte is not None else None
        errs.append(rollout(Zte[sel][0], v, ops, bl, Zte[sel], dt_save))
    errs = np.array(errs)
    out["rollout_rel_err_final_median5"] = float(np.median(errs[:, 0]))
    out["rollout_short_rel_err_median5"] = float(np.median(errs[:, 1]))
    out["spectrum_rel_err_median5"] = float(np.median(errs[:, 2]))
    return out


# ============================================================================
# Figures
# ============================================================================
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK2); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)


def fig_closure(cl, title, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.2), facecolor=SURFACE)
    _style(ax)
    k = np.array(cl["k"])
    ax.plot(k, cl["nu_t_exact"], color=CAT[0], lw=2.5, marker="o", ms=8, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label="exact closure (test data)")
    ax.plot(k, cl["nu_t_model"], color=CAT[1], lw=2, ls="--", marker="s", ms=5, markeredgecolor=SURFACE,
            markeredgewidth=1, label="learned model, total")
    names = {"A": "  from linear A", "T": "  from quadratic T error", "C": "  from cubic C", "L": "  from input L error"}
    ci = 2
    for b in ["A", "T", "C", "L"]:
        key = f"nu_t_model_{b}"
        if key in cl:
            ax.plot(k, cl[key], color=CAT[ci], lw=1.5, ls=":", marker="o", ms=4, label=names[b]); ci += 1
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("mode k (cutoff K = %d)" % k.max(), color=INK2, fontsize=9)
    ax.set_ylabel("eddy viscosity  nu_t(k) = -Pi(k) / k^2", color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)


def fig_tensor(Tt, Tl, bl, k_show, title, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ); cmap.set_bad(SURFACE)
    K, n = bl.K, bl.n
    r = list(bl.m).index(k_show)

    def sl(T):
        S = np.full((n, n), np.nan)
        for c, (p, q) in enumerate(zip(bl.P, bl.Q)):
            S[p, q] = abs(T[r, c])
        return S
    St, Sl = sl(Tt), sl(Tl)
    vmax = np.nanmax(St) if np.nanmax(St) > 0 else np.nanmax(Sl)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), facecolor=SURFACE)
    for ax, (t, S) in zip(axes, [("True (Galerkin)", St), (title, Sl)]):
        im = ax.imshow(S, cmap=cmap, vmin=0, vmax=vmax, origin="lower", extent=[-K - .5, K + .5, -K - .5, K + .5])
        ax.plot(bl.m, k_show - bl.m, color=INK2, lw=0.8, ls="--")
        ax.set_xlim(-K - .5, K + .5); ax.set_ylim(-K - .5, K + .5)
        ax.set_title(t, color=INK, fontsize=10); ax.tick_params(colors=INK2, labelsize=8)
        ax.set_xlabel("q", color=INK2, fontsize=9); ax.set_ylabel("p", color=INK2, fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label(f"|T[k={k_show}, p, q]|", color=INK2, fontsize=9); cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_visible(False)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", choices=list(DEFAULTS), required=True)
    ap.add_argument("--params", type=float, nargs="+", default=None)
    ap.add_argument("--regimes", nargs="+", default=["full"])
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--Kv", type=int, default=3, help="varvisc: input bandwidth")
    ap.add_argument("--eps", type=float, default=0.5, help="varvisc: nu(x) variation amplitude (train)")
    ap.add_argument("--eps_ood", type=float, default=0.9, help="varvisc: amplitude for OOD test")
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
    ap.add_argument("--energy_constraint", action="store_true")
    ap.add_argument("--closure", choices=["none", "cubic"], default="none")
    ap.add_argument("--mask_selection", action="store_true")
    ap.add_argument("--label", default=None)
    ap.add_argument("--k_show", type=int, default=3)
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
    if a.energy_constraint and a.pde in ("fisher", "allencahn"):
        raise SystemExit("energy constraint does not apply to fisher/allencahn")
    if a.pde == "allencahn" and a.closure != "cubic":
        raise SystemExit("allencahn needs --closure cubic (its physics is the cubic tensor)")
    label = a.label or "+".join([a.method] + (["energy"] if a.energy_constraint else []) +
                                (["cubic"] if a.closure == "cubic" else []) + (["mask"] if a.mask_selection else []))
    out = a.out or f"runs/{a.pde}_{label}"
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "w")

    def log(*m):
        s = " ".join(str(x) for x in m); print(s, flush=True); logf.write(s + "\n")

    log("args:", json.dumps({**vars(a), "params": params, "t_final": t_final, "dt": dt,
                              "save_every": save_every, "amp": amp, "mean_amp": mean_amp}))
    pde0 = PDE(a.pde, params[0])
    bl = Blocks(a.K, a.Kv, a.closure, pde0.has_input)
    log(f"[{a.pde}] features/output: lin {bl.sl.stop - bl.sl.start}, quad {bl.sq.stop - bl.sq.start}, "
        f"cubic {bl.sc.stop - bl.sc.start}, input {bl.si.stop - bl.si.start}")
    if pde0.energy_conserving_quad:
        log(f"sanity: energy-constraint residual of TRUE T = {cons_residual(bl.truth(pde0)[1], energy_constraints(bl)):.2e}")

    results = []
    for prm in params:
        pde = PDE(a.pde, prm)
        truth = bl.truth(pde)
        truth_ops = {"A": truth[0], "T": truth[1], "C": bl.truth_C(pde), "L": truth[2]}
        for regime in a.regimes:
            t0 = time.time()
            ks = a.K if regime == "closed" else a.kmax_full
            sim = lambda n, seed, eps: simulate(pde, ks, a.K, a.Kv, n, t_final, dt, save_every,
                                                np.random.default_rng(seed), amp, mean_amp, kmax_ic, eps)
            Z, dZ, V, _ = sim(a.n_traj, a.seed, a.eps)
            Zte, dZte, Vte, tid = sim(a.n_test, a.seed + 10_000, a.eps)
            t1 = time.time()
            ops, eqs = fit(Z, dZ, V, bl, a.ridge, a.method, a.stlsq_thresh, a.stlsq_iter,
                           a.energy_constraint, a.mask_selection)
            m = evaluate(ops, truth, bl, Zte, dZte, Vte, tid, eqs, dt * save_every, pde)
            if pde.has_input:
                Zo, dZo, Vo, to = sim(a.n_test, a.seed + 20_000, a.eps_ood)
                mo = evaluate(ops, truth, bl, Zo, dZo, Vo, to, eqs, dt * save_every, pde)
                m["ood_test_rel_err_dzdt"] = mo["test_rel_err_dzdt"]
                m["ood_rollout_rel_err_final_median5"] = mo["rollout_rel_err_final_median5"]
            cl = closure_spectrum(Zte, dZte, Vte, ops, truth_ops, bl)
            m["closure"] = cl
            m["n_samples"], m["time_sim_s"], m["time_fit_s"] = int(Z.shape[0]), round(t1 - t0, 1), round(time.time() - t1, 1)
            np.savez_compressed(os.path.join(out, f"ops_{prm}_{regime}.npz"), **ops,
                                A_true=truth[0], T_true=truth[1], L_true=truth[2])
            extra = ""
            if "L_rel_err" in m:
                extra += f" Lrel={m['L_rel_err']:.2e} Lrule={m['L_rel_err_on_rule']:.2e} Lsel={m['L_selection_mass']:.3f}"
                extra += f" OOD_dz={m['ood_test_rel_err_dzdt']:.2e} OOD_roll={m['ood_rollout_rel_err_final_median5']:.2e}"
            if "cubic_to_quad_ratio" in m:
                extra += f" cub/quad={m['cubic_to_quad_ratio']:.1e}"
            if "C_rel_err" in m:
                extra += f" Crel={m['C_rel_err']:.2e} Crule={m['C_rel_err_on_rule']:.2e} Csel={m['C_selection_mass']:.3f}"
            Akey = "A_diag_rel_err" if "A_diag_rel_err" in m else "A_diag_rel_to_visc"
            log(f"[{a.pde} {label} p={prm:<5} {regime:<6}] Trel={m['T_rel_err']:.2e} Trule={m['T_rel_err_on_rule']:.2e} "
                f"Tsel={m['T_selection_mass']:.3f} {Akey}={m[Akey]:.2e} dz={m['test_rel_err_dzdt']:.2e} "
                f"roll={m['rollout_rel_err_final_median5']:.2e} roll_short={m['rollout_short_rel_err_median5']:.1e} spec={m['spectrum_rel_err_median5']:.1e} closure={cl['closure_rel_size']:.1e} nu_t_err={cl['nu_t_rel_err']:.1e}{extra} "
                f"(sim {m['time_sim_s']}s fit {m['time_fit_s']}s)")
            results.append({"pde": a.pde, "param": prm, "regime": regime, "label": label, "metrics": m})
            if regime == "full":
                fig_closure(cl, f"{a.pde}, {label}, param={prm}", os.path.join(out, f"fig_closure_{prm}.png"))
            fig_tensor(truth[1], ops["T"], bl, a.k_show, f"{label}, {regime}, p={prm}",
                       os.path.join(out, f"fig_tensor_{prm}_{regime}.png"))

    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({"args": vars(a), "pde": a.pde, "label": label, "results": results}, f, indent=2)
    log("saved to", out)


if __name__ == "__main__":
    main()
