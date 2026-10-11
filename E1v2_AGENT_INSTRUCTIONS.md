# E1 v2 실행 지시 (서버 에이전트용)

## 목표
E1에서 모드를 자른 데이터(full)로 학습하면 텐서가 오염됐다. 다음 방법 중 무엇이 텐서를 깨끗하게 지키는지 비교한다.
- 블록별 STLSQ (희소 회귀, E1f의 버그 수정판)
- 에너지 보존 제약 (이차항이 에너지를 만들지 않도록 정확히 강제)
- 평행이동 대칭 사전지식 (`--mask_selection`: 선택 규칙 $p+q=k$ 밖의 성분을 0으로 고정)
- 3차 closure 항 (`--closure cubic`)

## 준비
아래 파일 3개를 **E1 폴더와 같은 곳**에 둔다. v2는 `e1_burgers_tensor.py`에서 솔버를 가져다 쓴다.
- `e1_burgers_tensor.py` (E1 그대로, 수정 금지)
- `e1v2_burgers_tensor.py`
- `e1v2_compare.py`

```bash
python3 e1v2_burgers_tensor.py --quick --regimes closed full --out runs/V2_smoke
```
로그 첫 줄들에서 `sanity: energy-constraint residual of TRUE tensor = 0.00e+00`이 나와야 하고, closed 줄의 `relT`가 1e-7 이하여야 한다. 아니면 멈추고 보고한다.

## 실행할 실험
기본값은 nu 5개(0.5, 0.2, 0.1, 0.05, 0.02)에 regime은 full이다. 각 run은 `--out`이 달라서 병렬로 돌려도 된다. `--closure cubic`이 들어간 run은 특징 수가 1139개라 메모리를 약 1~2GB 쓰고, fit에 nu당 수 분이 걸릴 수 있다.

```bash
# V2a: 블록별 STLSQ (closed도 같이 — E1f의 nu=0.02 closed 이상이 고쳐졌는지 확인)
python3 e1v2_burgers_tensor.py --method stlsq --regimes closed full --out runs/V2a_stlsq

# V2b: 에너지 보존 제약
python3 e1v2_burgers_tensor.py --energy_constraint --out runs/V2b_energy

# V2c: 에너지 제약 + STLSQ
python3 e1v2_burgers_tensor.py --energy_constraint --method stlsq --out runs/V2c_energy_stlsq

# V2d: 3차 closure 단독
python3 e1v2_burgers_tensor.py --closure cubic --out runs/V2d_cubic

# V2e: 대칭 사전지식
python3 e1v2_burgers_tensor.py --mask_selection --out runs/V2e_mask

# V2f: 대칭 + 에너지
python3 e1v2_burgers_tensor.py --mask_selection --energy_constraint --out runs/V2f_mask_energy

# V2g: 대칭 + closure
python3 e1v2_burgers_tensor.py --mask_selection --closure cubic --out runs/V2g_mask_cubic

# V2h: 대칭 + closure + 에너지
python3 e1v2_burgers_tensor.py --mask_selection --closure cubic --energy_constraint --out runs/V2h_mask_cubic_energy
```

끝나면 비교 그림 두 장을 만든다. E1a(기존 결과)를 기준선으로 넣는다. 그림 한 장에 최대 8개 run까지 들어간다.
```bash
python3 e1v2_compare.py runs/E1a_nu_sweep runs/V2a_stlsq runs/V2b_energy runs/V2c_energy_stlsq runs/V2d_cubic --out runs/V2_compare_discovery
python3 e1v2_compare.py runs/E1a_nu_sweep runs/V2e_mask runs/V2f_mask_energy runs/V2g_mask_cubic runs/V2h_mask_cubic_energy --out runs/V2_compare_prior
```

## 추가로 확인할 것 (E1 결과 관련)
`runs/E1f_stlsq/results.json`에서 nu=0.02, regime=closed의 `rel_err_A_diag` 값을 알려준다.

## 지표
E1과 같고, 다음이 추가됐다.
| 키 | 뜻 |
|---|---|
| `energy_constraint_residual` | 에너지 제약 위반 크기 (제약을 건 run에서 1e-10 이하여야 정상) |
| `cubic_to_quad_ratio` | 3차 closure 항의 크기 / 이차항 크기 (closure가 얼마나 일하나) |
| `n_nonzero_T`, `n_nonzero_C` | 0이 아닌 성분 수 (희소성) |
| `time_sim_s`, `time_fit_s` | 데이터 생성 / 학습 시간 |

텐서는 run 폴더에 `tensors_nu{nu}_{regime}.npz`로 저장된다(A, T, C, A_true, T_true).

## 스모크 테스트에서 이미 본 것 (quick 설정이라 숫자는 참고만)
- 에너지 제약: 위반이 1e-15 수준으로 정확히 걸림. full 텐서 오차가 1.50 → 0.84 (nu=0.2)로 줄어듦
- 대칭 사전지식: full 텐서 오차가 1.50 → 0.015 (nu=0.2), 3.04 → 0.046 (nu=0.05)
- 대칭 + closure는 대칭만과 거의 같음 (0.017, 0.054)
- closure 단독: quick 설정에선 데이터(1,040개)가 특징 수(1,139개)보다 적어 발산. 데이터를 늘리니 closed는 맞지만 full은 여전히 오차 1.26
- 블록별 STLSQ: closed가 정확함(1e-13). full의 규칙 위 오차가 개선(0.135 → 0.039, nu=0.2)

## 보고할 것
1. run별 `results.json`, `log.txt`, 텐서 그림, 비교 그림 두 장과 `compare_full.md`
2. 핵심 표: run × nu별 `rel_err_T`, `rel_err_T_on_rule`, `selection_mass`, `rel_err_A_diag`, `energy_violation`, `rollout_rel_err_final`, `cubic_to_quad_ratio`(있으면)
3. E1f nu=0.02 closed의 `rel_err_A_diag`
4. 스모크 관찰과 **다른** 결과
5. 에러, 발산, 메모리 문제, 실행 시간

코드는 수정하지 말고, 에러가 나면 고치기 전에 보고한다.
