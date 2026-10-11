| regime | model | u_t err | u_t err in span | rollout | b_rel | A_rel | T_rel | T_var | T_var R2(nu) | params |
|---|---|---|---|---|---|---|---|---|---|---|
| closed | tco | 2.30e-02 | 2.30e-02 | 2.33e-02 | 4.06e-02 | 1.49e-02 | 1.71e-02 | 1.59e-02 | 0.16 | 911707 |
| closed | deeponet | 7.15e-02 | 7.15e-02 | 2.07e-02 | 3.88e-02 | 5.54e-02 | 2.30e-01 | 2.16e-01 | 0.19 | 185361 |
| closed | galerkin (ref) | 3.81e-04 | 2.34e-04 | 6.61e-05 | 0 | 0 | 0 | 0 | - | - |
| full | tco | 4.04e-01 | 1.86e-01 | 1.32e-01 | 2.37e-01 | 1.99e-01 | 3.17e-01 | 2.81e-01 | 0.10 | 911707 |
| full | deeponet | 3.87e-01 | 1.38e-01 | 4.70e-02 | 7.26e-02 | 1.41e-01 | 7.08e-01 | 6.31e-01 | 0.16 | 185361 |
| full | galerkin (ref) | 3.85e-01 | 1.34e-01 | 2.31e-02 | 0 | 0 | 0 | 0 | - | - |

| regime | span u (basis) | span u_t (basis) | span u (POD) | span u_t (POD) | span u (Fourier) | span u_t (Fourier) | cond[1,c] median / max |
|---|---|---|---|---|---|---|---|
| closed | 1.05e-04 | 3.01e-04 | 1.43e-15 | 1.07e-15 | 1.05e-15 | 1.48e-15 | 2.0e+02 / 3.4e+03 |
| full | 4.70e-02 | 3.65e-01 | 4.71e-02 | 3.64e-01 | 4.70e-02 | 3.64e-01 | 2.0e+02 / 3.7e+03 |
