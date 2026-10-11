| regime | model | u_t err | u_t err in span | rollout | b_rel | A_rel | T_rel | T_var | T_var R2(nu) | params |
|---|---|---|---|---|---|---|---|---|---|---|
| closed | deeponet_joint | 6.82e-01 | 6.16e-01 | 4.87e-01 | 6.87e-01 | 9.54e-01 | 8.70e-01 | 3.67e-01 | 0.27 | 220962 |
| closed | opinf | 1.19e-14 | 1.18e-14 | 1.58e-08 | 3.61e-11 | 2.37e-11 | 6.93e-11 | 1.78e-11 | 0.26 | 17731 |
| closed | galerkin (ref) | 3.47e-15 | 3.13e-15 | 1.58e-08 | 0 | 0 | 0 | 0 | - | - |
| full | deeponet_joint | 7.31e-01 | 6.27e-01 | 4.62e-01 | 7.05e-01 | 9.45e-01 | 8.78e-01 | 3.73e-01 | 0.27 | 220962 |
| full | opinf | 5.94e-01 | 5.11e-01 | 6.37e-02 | 2.51e-01 | 1.69e-01 | 1.77e+00 | 9.54e-01 | 0.53 | 17731 |
| full | galerkin (ref) | 3.68e-01 | 1.10e-01 | 2.84e-02 | 0 | 0 | 0 | 0 | - | - |

| regime | span u (basis) | span u_t (basis) | span u (POD) | span u_t (POD) | span u (Fourier) | span u_t (Fourier) | cond[1,c] median / max |
|---|---|---|---|---|---|---|---|
| closed | 1.07e-15 | 1.50e-15 | 1.75e-15 | 1.31e-15 | 1.07e-15 | 1.50e-15 | 1.3e+01 / 2.9e+01 |
| full | 4.45e-02 | 3.53e-01 | 4.52e-02 | 3.54e-01 | 4.45e-02 | 3.53e-01 | 1.3e+01 / 2.9e+01 |
