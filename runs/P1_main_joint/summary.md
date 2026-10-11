| regime | model | u_t err | u_t err in span | rollout | b_rel | A_rel | T_rel | T_var | T_var R2(nu) | params |
|---|---|---|---|---|---|---|---|---|---|---|
| closed | deeponet_joint | 3.56e-02 | 3.55e-02 | 1.26e-02 | 3.82e-02 | 3.85e-02 | 9.82e-02 | 8.50e-02 | 0.16 | 220962 |
| closed | galerkin (ref) | 3.81e-04 | 2.34e-04 | 6.61e-05 | 0 | 0 | 0 | 0 | - | - |
| full | deeponet_joint | 4.02e-01 | 1.83e-01 | 6.85e-02 | 3.98e-02 | 4.16e-02 | 1.23e-01 | 1.05e-01 | 0.18 | 220962 |
| full | galerkin (ref) | 3.85e-01 | 1.34e-01 | 2.31e-02 | 0 | 0 | 0 | 0 | - | - |

| regime | span u (basis) | span u_t (basis) | span u (POD) | span u_t (POD) | span u (Fourier) | span u_t (Fourier) | cond[1,c] median / max |
|---|---|---|---|---|---|---|---|
| closed | 1.05e-04 | 3.01e-04 | 1.43e-15 | 1.07e-15 | 1.05e-15 | 1.48e-15 | 2.0e+02 / 3.4e+03 |
| full | 4.70e-02 | 3.65e-01 | 4.71e-02 | 3.64e-01 | 4.70e-02 | 3.64e-01 | 2.0e+02 / 3.7e+03 |
