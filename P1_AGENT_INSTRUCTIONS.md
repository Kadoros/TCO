# 실험 1 (P1): 입력에 따라 바뀌는 텐서, 학습된 기저 위에서 채점 (서버 에이전트용)

## 무엇을 하나
- **PDE**: u_t = −u u_x + (ν(x) u_x)_x + f(x), 주기 경계
- **입력 함수**: a = (ν(x), f(x)). 입력마다 무작위로 뽑고, 입력 하나에 궤적 여러 개를 만든다.
- **배우는 것**: (a, u) → u_t

모든 모델은 같은 trunk 기저 Φ를 쓴다. Φ는 1단계에서 데이터만으로 학습한다.

| 모델 | 구조 |
|---|---|
| `tco` | branch MLP(a) → (b(a), A(a), T(a)). 이후 dc/dt = b + A c + T(c,c) |
| `deeponet` | MLP(a, u) → dc/dt. 표준 DeepONet branch이고 trunk는 같다 |
| `opinf` | 입력 계수에 대해 아핀인 parametric Operator Inference. 최소제곱으로 푼다 |
| `galerkin` | 같은 기저 위의 정확한 Galerkin 계. 채점 기준이고, 학습하지 않는다 |

**채점**: 모든 모델에 똑같은 추출기를 쓴다. 각 입력의 상태 평균 근처에서 (b, A, T)를 뽑고, **같은 기저 위에서 계산한 Galerkin 정답**과 비교한다. 정답을 모델의 기저 위에서 계산하므로 기저의 gauge와 무관하다. 학습에는 Galerkin이 쓰이지 않는다.

**regime 두 가지**
- `closed`: 기저 안의 Galerkin 계로 데이터를 만든다. 채점과 같은 스펙트럼 미분을 쓰므로 정답과의 차이가 ≈ 0이어야 한다.
- `full`: 고해상도 DNS(192 격자)로 데이터를 만든다. 정답과의 차이가 곧 closure다.

## 파일
`tco_param.py`를 기존 폴더에 둔다. `tco_pod.py`, `tco_trunk.py`가 같은 폴더에 있어야 한다. torch를 쓰고 GPU가 있으면 자동으로 쓴다.

## 스모크 테스트
```bash
python3 tco_param.py --quick --trunk fourier --models opinf --out runs/P1_smoke_fourier
python3 tco_param.py --quick --steps 300 --trunk_steps 200 --out runs/P1_smoke
```
통과 기준은 두 가지다.
1. 첫 줄의 closed `opinf`에서 `b`, `A`, `T`가 모두 1e-8 이하다. 정답 계산과 데이터 생성이 일관되는지 보는 기준이다.
2. 로그의 `extractor check`가 1e-8 이하다.

quick에서 TCO와 DeepONet 숫자는 의미가 없다.

## 실험
### P1-1 본 실험
```bash
python3 tco_param.py --out runs/P1_main
```
기본값: 학습 trunk, closed와 full, 학습 입력 512개 × 궤적 4개 × 상태 8개, 시험 입력 16개, step 20000, seed 0 1 2.

### P1-2 학습 입력 수 (가장 중요한 그림)
TCO의 T가 입력 수가 늘수록 정답에 가까워지고 입력과 무관해지는지 본다.
```bash
for n in 128 2048; do
  python3 tco_param.py --n_inputs $n --seeds 0 1 --out runs/P1_n$n
done
```
n=512는 P1-1을 그대로 쓴다. 2048이 메모리나 시간 문제로 안 되면 1024로 줄이고 보고한다.

### P1-3 기저를 바꿔도 결론이 같은가
```bash
python3 tco_param.py --trunk pod     --seeds 0 --out runs/P1_pod
python3 tco_param.py --trunk fourier --seeds 0 --out runs/P1_fourier
```

### P1-4 학습률 (공정성 확인, seed 0만)
```bash
for lr in 3e-4 3e-3; do
  python3 tco_param.py --lr $lr --seeds 0 --models tco deeponet --out runs/P1_lr$lr
done
```

## 내가 미리 확인한 것 (참고)
조건: Fourier 기저, closed, 학습 입력 512개, 3000 step, CPU, seed 0.

| 모델 | u_t 오차 | b 오차 | A 오차 | T 오차 | T_var |
|---|---|---|---|---|---|
| tco | 5.2% | 10% | 3.8% | 6.9% | 7.2% |
| deeponet | 14.7% | 25% | 36% | 35% | 24% |
| opinf | ~1e-14 | ~1e-11 | ~1e-11 | ~1e-10 | ~1e-11 |

- 학습 입력이 32개일 때 TCO는 학습 손실이 2e-6인데 시험 오차는 29%였다. branch가 입력별로 외워버린 것이다. 그래서 입력 수 실험(P1-2)이 핵심이다.
- OpInf는 closed에서 정확하다. 아핀 구조를 미리 알고 있어서다. full에서는 특징(1043개) 대비 샘플이 적으면 크게 나빠진다(quick에서 T 오차 1.8).
- 학습 trunk의 closed span 잔차는 1e-3 수준이다. 그래서 학습 trunk 위의 정답 비교도 그 수준에서 멈춘다.

## 보고할 것
1. 스모크 결과
2. P1-1 `summary.md` 두 표: 모델 표와 span·cond 표. seed 평균±표준편차
3. P1-2: n_inputs(128, 512, 2048)별 모델별 `T_rel`, `T_var`, `A_rel`, `b_rel`, `ut_err_inspan`
4. P1-3: 기저별 같은 표
5. P1-4: lr별 tco/deeponet 비교
6. 그림: `fig_T_slices_closed.png`, `fig_T_slices_full.png`, `fig_T_vs_nu_closed.png`, `fig_T_vs_nu_full.png` (P1-1과 P1_n2048)
7. 이상한 점: 발산(inf), 큰 seed 편차, `cond[1,c]` 최대값이 1e4를 넘는 경우, 데이터 생성 시간, GPU 사용 여부, 총 실행 시간

코드는 수정하지 말고, 에러가 나면 고치기 전에 보고한다.
