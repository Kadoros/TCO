# TCO 다중 PDE 실행 지시 (서버 에이전트용)

## 목표
데이터만으로 학습한 결합 텐서가 수학으로 구한 정답 텐서를 얼마나 복원하는지 PDE 5개에서 비교한다.
- **Burgers, KdV, Kuramoto–Sivashinsky(KS), Fisher–KPP**: 이차 삼체 텐서 $T$
- **varvisc** (가변 점성 Burgers, E4): **입력 함수** $\nu(x)$가 텐서 인덱스로 들어가는 입력 텐서 $L$. 처음 보는(분포 밖) 입력에서의 성능도 같이 잰다.

모든 run은 **closure 분석**을 자동으로 한다. 테스트 데이터에서 정확한 closure의 eddy viscosity 스펙트럼과, 학습된 모델이 암묵적으로 배운 스펙트럼을 비교하고 블록별로 나눠서 `fig_closure_{param}.png`로 그린다.

## 준비
아래 두 파일을 기존 폴더(E1, E1v2 결과가 있는 곳)에 둔다. 이 프레임워크는 다른 파일에 의존하지 않는다.
- `tco_pde.py`
- `tco_compare.py`

numpy와 matplotlib만 쓴다. 스모크 테스트:
```bash
for p in burgers kdv fisher varvisc; do python3 tco_pde.py --pde $p --quick --regimes closed full --out runs/M_smoke_$p; done
python3 tco_pde.py --pde ks --quick --K 12 --regimes closed full --out runs/M_smoke_ks
```
closed 줄의 `Trel`이 Burgers, KdV, KS에서 1e-6 이하여야 한다. Fisher와 varvisc는 quick 설정에서 데이터가 적어 1e-2 이하면 된다. 아니면 멈추고 보고한다.

## M0: 재현 확인 (먼저, 반드시)
프레임워크를 새로 짰으므로 V2 결과가 재현되는지 확인한다.
```bash
python3 tco_pde.py --pde burgers --mask_selection --out runs/M0_burgers_mask
python3 tco_pde.py --pde burgers --mask_selection --closure cubic --out runs/M0_burgers_mask_cubic
```
기대값 (`T_rel_err`, param 0.5 → 0.02):
- mask: 1.48e-3, 6.91e-3, 3.11e-2, 8.08e-2, 1.40e-1 (V2e)
- mask+cubic: 1.48e-3, 3.65e-3, 8.85e-3, 3.22e-2, 8.34e-2 (V2g)

상대 차이가 1%를 넘으면 **멈추고 보고**한다. 롤아웃 지표는 계산 방식이 바뀌어서(5개 궤적의 중앙값, 적분기 변경) V2와 달라도 된다.

## M1~M5: 본 실험
`--out`이 다르면 병렬로 돌려도 된다. `--closure cubic` run은 특징 수가 많아 메모리를 많이 쓴다. K=8이면 약 2GB, **KS(K=12)는 약 10GB 이상**이다. KS cubic run은 따로, 동시에 너무 많이 돌리지 않는다.

```bash
# M1 Burgers (M0 두 개 + 아래 둘)
python3 tco_pde.py --pde burgers --out runs/M1_burgers_lstsq
python3 tco_pde.py --pde burgers --mask_selection --closure cubic --energy_constraint --out runs/M1_burgers_mask_cubic_energy

# M2 KdV (param = delta: 0.1, 0.05, 0.02)
python3 tco_pde.py --pde kdv --regimes closed full --out runs/M2_kdv_lstsq
python3 tco_pde.py --pde kdv --mask_selection --out runs/M2_kdv_mask
python3 tco_pde.py --pde kdv --mask_selection --closure cubic --out runs/M2_kdv_mask_cubic
python3 tco_pde.py --pde kdv --mask_selection --closure cubic --energy_constraint --out runs/M2_kdv_mask_cubic_energy

# M3 KS (param = nu: 0.1, 0.05, 0.02), K=12 필수
python3 tco_pde.py --pde ks --K 12 --regimes closed full --out runs/M3_ks_lstsq
python3 tco_pde.py --pde ks --K 12 --mask_selection --out runs/M3_ks_mask
python3 tco_pde.py --pde ks --K 12 --mask_selection --closure cubic --out runs/M3_ks_mask_cubic
python3 tco_pde.py --pde ks --K 12 --mask_selection --closure cubic --energy_constraint --out runs/M3_ks_mask_cubic_energy

# M4 Fisher-KPP (param = D: 0.1, 0.05, 0.02). 에너지 제약은 적용 불가
python3 tco_pde.py --pde fisher --regimes closed full --out runs/M4_fisher_lstsq
python3 tco_pde.py --pde fisher --mask_selection --out runs/M4_fisher_mask
python3 tco_pde.py --pde fisher --mask_selection --closure cubic --out runs/M4_fisher_mask_cubic

# M5 E4 가변 점성 (param = 평균 점성 nu0: 0.2, 0.1, 0.05)
python3 tco_pde.py --pde varvisc --regimes closed full --out runs/M5_varvisc_lstsq
python3 tco_pde.py --pde varvisc --mask_selection --regimes closed full --out runs/M5_varvisc_mask
python3 tco_pde.py --pde varvisc --mask_selection --closure cubic --out runs/M5_varvisc_mask_cubic
python3 tco_pde.py --pde varvisc --mask_selection --closure cubic --energy_constraint --out runs/M5_varvisc_mask_cubic_energy
# M5-OOD: 학습 입력 변동 폭을 줄이고(0.3) 훨씬 큰 변동(0.9)에서 테스트
python3 tco_pde.py --pde varvisc --mask_selection --closure cubic --eps 0.3 --eps_ood 0.9 --out runs/M5_varvisc_ood_gap
python3 tco_pde.py --pde varvisc --eps 0.3 --eps_ood 0.9 --out runs/M5_varvisc_ood_gap_lstsq
```

비교 그림 (PDE별로 한 장):
```bash
python3 tco_compare.py runs/M1_burgers_lstsq runs/M0_burgers_mask runs/M0_burgers_mask_cubic runs/M1_burgers_mask_cubic_energy --out runs/M1_compare
python3 tco_compare.py runs/M2_kdv_lstsq runs/M2_kdv_mask runs/M2_kdv_mask_cubic runs/M2_kdv_mask_cubic_energy --out runs/M2_compare
python3 tco_compare.py runs/M3_ks_lstsq runs/M3_ks_mask runs/M3_ks_mask_cubic runs/M3_ks_mask_cubic_energy --out runs/M3_compare
python3 tco_compare.py runs/M4_fisher_lstsq runs/M4_fisher_mask runs/M4_fisher_mask_cubic --out runs/M4_compare
python3 tco_compare.py runs/M5_varvisc_lstsq runs/M5_varvisc_mask runs/M5_varvisc_mask_cubic runs/M5_varvisc_mask_cubic_energy runs/M5_varvisc_ood_gap runs/M5_varvisc_ood_gap_lstsq --out runs/M5_compare
```

## 지표 (results.json → metrics)
| 키 | 뜻 |
|---|---|
| `T_rel_err`, `T_rel_err_on_rule`, `T_selection_mass` | 삼체 텐서: 전체 오차, 선택 규칙 위 오차, 규칙 위 질량 비율 |
| `A_diag_rel_err` | 선형 대각 오차 (varvisc는 정답이 0이라 대신 `A_diag_rel_to_visc`) |
| `L_rel_err`, `L_rel_err_on_rule`, `L_selection_mass` | varvisc 입력 텐서 |
| `ood_test_rel_err_dzdt`, `ood_rollout_rel_err_final_median5` | varvisc 분포 밖 입력 성능 |
| `closure.nu_t_rel_err` | 테스트 데이터에서 eddy viscosity 스펙트럼 오차 (모델 vs 정확) |
| `closure.closure_rel_size` | closure 크기 / 전체 시간미분 크기. 이게 1e-3보다 작으면 `nu_t_rel_err`는 의미가 약하다 |
| `closure.nu_t_model_A/T/C/L` | 모델이 eddy viscosity를 어느 블록에 담았는지 |
| `rollout_rel_err_final_median5` | 테스트 궤적 5개 롤아웃 오차의 중앙값 |
| `energy_constraint_residual`, `cubic_to_quad_ratio` | V2와 같음 |

텐서는 `ops_{param}_{regime}.npz`에 저장된다.

## 이미 알려진 것 (스모크, 참고만)
- KS는 K=8이면 불안정 모드가 거의 다 차서 부족하다. K=12를 쓴다.
- KS nu=0.02는 카오스라 **정확한 모델도 롤아웃이 벌어진다**(closed에서도 약 2%). 이 영역의 롤아웃은 품질 지표가 아니라 예측 한계로 해석한다.
- Fisher는 해가 빨리 u=1로 가서 데이터 다양성이 낮다. closed에서도 1e-4~1e-2 수준일 수 있다(조건수 한계).
- varvisc는 평균 점성도 궤적마다 0.5~1.5배로 바뀐다. 이걸 고정하면 $\nu_0 z_k$ 와 $z_k$ 가 같은 방향이 되어 식별이 안 된다.

## 보고할 것
1. **M0 재현 결과**(기대값과 비교 표). 1% 넘게 다르면 거기서 멈추고 보고
2. PDE별 `compare.png`, `compare.md`, 대표 closure 그림(각 PDE의 lstsq와 mask+cubic, 가장 작은 param)
3. PDE × 방법 × param 핵심 표: `T_rel_err`, `T_rel_err_on_rule`, `closure.nu_t_rel_err`, `closure.closure_rel_size`, `rollout`, (varvisc는 `L_rel_err`, OOD 두 개)
4. 스모크 관찰과 다른 결과, 이상한 점
5. 에러, 발산, 메모리 최대치, 실행 시간

코드는 수정하지 말고, 에러가 나면 고치기 전에 보고한다.
