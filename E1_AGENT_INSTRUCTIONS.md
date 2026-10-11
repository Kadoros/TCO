# E1 실행 지시 (서버 에이전트용)

## 목표
1D 점성 Burgers 데이터만으로 3차 결합 텐서 $T_{kij}$ 를 학습했을 때, 해석적으로 아는 진짜 삼체 텐서(선택 규칙 $p+q=k$)를 복원하는지 측정한다. 스크립트는 `e1_burgers_tensor.py` 하나이고, numpy와 matplotlib만 쓴다(GPU 불필요).

## 환경
```bash
python3 -m pip install numpy matplotlib   # 없으면
python3 e1_burgers_tensor.py --quick --out runs/e1_smoke   # 약 15초 스모크 테스트
```
스모크 테스트의 closed 줄에서 `relT`가 1e-7 이하로 나와야 정상이다. 아니면 멈추고 보고한다.

## 실행할 실험 (순서대로)
각 실험은 `--out` 폴더가 다르니 병렬로 돌려도 된다. 기본 설정 1회는 대략 10~20분 걸린다.

```bash
# E1a: 기본 nu 스윕 (closed vs full)
python3 e1_burgers_tensor.py --out runs/E1a_nu_sweep

# E1b: 식별 가능성 — 초기조건 대역폭 (K=8)
for kic in 2 4 8; do
  python3 e1_burgers_tensor.py --nus 0.1 --kmax_ic $kic --out runs/E1b_kic$kic
done

# E1c: 노이즈 강건성 (closed만)
for nz in 0 1e-4 1e-3 1e-2; do
  python3 e1_burgers_tensor.py --nus 0.1 --regimes closed --noise $nz --out runs/E1c_noise$nz
done

# E1d: 조건수 문제 — ridge 스윕 (K=12, 점성 큼)
for rg in 1e-10 1e-6 1e-3; do
  python3 e1_burgers_tensor.py --nus 1.0 --K 12 --regimes full --ridge $rg --out runs/E1d_ridge$rg
done

# E1e: 모드 수 K
for K in 6 8 12; do
  python3 e1_burgers_tensor.py --nus 0.1 --K $K --kmax_ic $K --regimes full --out runs/E1e_K$K
done

# E1f: 희소 회귀 비교
python3 e1_burgers_tensor.py --method stlsq --out runs/E1f_stlsq
```

## 지표 설명 (results.json의 metrics)
| 키 | 뜻 | 좋은 방향 |
|---|---|---|
| `rel_err_T` | 학습 텐서와 정답 텐서의 전체 상대오차 | 0 |
| `rel_err_T_on_rule` | 선택 규칙 $p+q=k$ 위 성분만의 상대오차 (진짜 삼체 상호작용이 맞나) | 0 |
| `selection_mass` | $\lvert T\rvert$ 질량 중 선택 규칙 위의 비율 | 1 |
| `max_off_rule_ratio` | 규칙 밖 최대 성분 / 규칙 위 최대 성분 | 0 |
| `rel_err_A_diag` | 선형부 대각($-\nu k^2$) 상대오차 | 0 |
| `offdiag_A_ratio` | 선형부 비대각 크기 비율 | 0 |
| `energy_violation` | 학습한 이차항이 에너지를 만들거나 없애는 정도 (정답은 0) | 0 |
| `test_rel_err_dzdt` | 보류 데이터에서 시간미분 예측 오차 | 0 |
| `rollout_rel_err_final` | 보류 궤적 1개를 학습 모델로 굴렸을 때 마지막 시점 오차 (`inf`면 발산) | 0 |

## 스모크 테스트에서 이미 본 것 (참고, 이걸 재현/확장하는 게 목적)
- closed: `rel_err_T` 약 1e-8, `selection_mass` 1.000 → 데이터만으로 정답 텐서 거의 정확히 복원
- full: `rel_err_T` 1.5~3.0인데 `rel_err_T_on_rule`은 0.13~0.31 → 진짜 삼체는 대체로 맞고, 잘린 모드 영향(closure)이 규칙 밖으로 새어 들어감. 점성이 작을수록 심해짐
- 초기조건이 낮은 모드만 자극(`--kmax_ic 3`)하면 미분 예측은 잘 맞는데 텐서가 엉뚱함 → 식별 불가능성
- K=12, nu=1.0, 기본 ridge에서 `rel_err_T` 약 1e5 → 고모드 진폭이 너무 작아 조건수 붕괴
- 평균(k=0)을 0으로 고정하면 k=0이 들어가는 삼체가 식별 불가 → 그래서 `--mean_amp` 기본값 0.5

## 보고할 것
1. 실험별 `results.json`과 그림(`fig_tensor_k3.png`, `fig_nu_sweep.png`) 전부
2. 실험×(nu, regime)별 핵심 지표를 모은 표 하나: `rel_err_T`, `rel_err_T_on_rule`, `selection_mass`, `energy_violation`, `test_rel_err_dzdt`, `rollout_rel_err_final`
3. 스모크 테스트 관찰과 **다른** 결과가 나온 곳
4. 실행 중 에러, 발산(`solver blew up`, rollout `inf`), 실행 시간

해석이나 코드 수정은 하지 말고, 에러가 나면 수정 전에 보고한다.
