| regime | model | u_t err | u_t err in span | rollout | b_rel | A_rel | T_rel | T_var | T_var R2(nu) | params |
|---|---|---|---|---|---|---|---|---|---|---|
| closed | tco | 2.80e-02 | 2.80e-02 | 2.41e-02 | 5.38e-02 | 1.90e-02 | 2.80e-02 | 2.84e-02 | 0.07 | 911707 |
| closed | deeponet | 4.70e-02 | 4.70e-02 | 1.75e-02 | 8.46e-02 | 7.91e-02 | 1.11e-01 | 9.46e-02 | 0.18 | 185361 |
| closed | galerkin (ref) | 3.81e-04 | 2.34e-04 | 6.61e-05 | 0 | 0 | 0 | 0 | - | - |
| full | tco | 4.13e-01 | 2.09e-01 | 1.31e-01 | 2.17e-01 | 1.42e-01 | 3.71e-01 | 3.40e-01 | 0.10 | 911707 |
| full | deeponet | 3.77e-01 | 1.02e-01 | 4.14e-02 | 1.42e-01 | 1.09e-01 | 2.61e-01 | 2.02e-01 | 0.14 | 185361 |
| full | galerkin (ref) | 3.85e-01 | 1.34e-01 | 2.31e-02 | 0 | 0 | 0 | 0 | - | - |

| regime | span u (basis) | span u_t (basis) | span u (POD) | span u_t (POD) | span u (Fourier) | span u_t (Fourier) | cond[1,c] median / max |
|---|---|---|---|---|---|---|---|
| closed | 1.05e-04 | 3.01e-04 | 1.43e-15 | 1.07e-15 | 1.05e-15 | 1.48e-15 | 2.0e+02 / 3.4e+03 |
| full | 4.70e-02 | 3.65e-01 | 4.71e-02 | 3.64e-01 | 4.70e-02 | 3.64e-01 | 2.0e+02 / 3.7e+03 |
