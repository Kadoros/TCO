| regime | model | u_t err | u_t err in span | rollout | b_rel | A_rel | T_rel | T_var | T_var R2(nu) | params |
|---|---|---|---|---|---|---|---|---|---|---|
| closed | deeponet_joint | 2.01e-02 | 2.00e-02 | 8.13e-03 | 2.60e-02 | 2.48e-02 | 6.36e-02 | 5.25e-02 | 0.14 | 220962 |
| closed | galerkin (ref) | 3.79e-04 | 2.39e-04 | 6.76e-05 | 0 | 0 | 0 | 0 | - | - |
| full | deeponet_joint | 3.85e-01 | 1.33e-01 | 2.72e-02 | 2.47e-02 | 2.68e-02 | 7.19e-02 | 6.03e-02 | 0.16 | 220962 |
| full | galerkin (ref) | 3.85e-01 | 1.33e-01 | 2.29e-02 | 0 | 0 | 0 | 0 | - | - |

| regime | span u (basis) | span u_t (basis) | span u (POD) | span u_t (POD) | span u (Fourier) | span u_t (Fourier) | cond[1,c] median / max |
|---|---|---|---|---|---|---|---|
| closed | 1.05e-04 | 2.95e-04 | 1.38e-15 | 1.31e-15 | 1.05e-15 | 1.48e-15 | 2.1e+02 / 9.9e+03 |
| full | 4.70e-02 | 3.64e-01 | 4.70e-02 | 3.64e-01 | 4.70e-02 | 3.64e-01 | 2.1e+02 / 7.0e+03 |
