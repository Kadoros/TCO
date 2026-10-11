# M6a 실행 지시: 데이터로 배운 기저(POD) (서버 에이전트용)

## 목표
지금까지는 모델에 **푸리에 기저**를 줬다. M6a에서는 푸리에도, 대칭 조건도 주지 않는다. 확인할 것은 세 가지다.
1. 센서 격자 값(64점)만 보고 **POD 기저**를 뽑았을 때 그 기저가 스스로 푸리에 모드를 찾는가 (`pod_purity_mean`)
2. 그 기저에서 배운 텐서를 정답 텐서와 비교하면 얼마나 맞는가. 비교는 학습된 모델 전체를 푸리에 좌표로 옮겨서 한다(`T_rel_err` 등).
3. 학습된 선형부 $A$ 의 **고유기저**가 푸리에 모드와 일치하는가 (`eig_purity_mean`). 아무 사전지식 없이 물리 좌표를 되찾는 방법이다.

## 준비
아래 파일을 기존 폴더에 둔다. **`tco_pde.py`와 `tco_compare.py`는 새 버전으로 교체**한다(카오스용 지표와 범례 중복 수정이 추가됨, 기존 결과와 호환됨).
- `tco_pod.py` (새 파일, `tco_pde.py`에서 import)
- `tco_pde.py` (교체)
- `tco_compare.py` (교체)

스모크 테스트:
```bash
python3 tco_pod.py --pde burgers --quick --out runs/M6_smoke_pod
python3 tco_pod.py --pde burgers --basis fourier --quick --out runs/M6_smoke_fourier
```
두 run 모두 closed 줄의 `Trel`이 1e-6 이하여야 한다. 아니면 멈추고 보고한다.

## 실행할 실험
`--out`이 다르면 병렬로 돌려도 된다. 기본 regime은 closed와 full 둘 다다.

```bash
# M6a-1: POD vs 푸리에 (사전지식 없음, 같은 데이터)
for p in burgers kdv fisher; do
  python3 tco_pod.py --pde $p --basis pod     --out runs/M6a_${p}_pod
  python3 tco_pod.py --pde $p --basis fourier --out runs/M6a_${p}_fourier
done
python3 tco_pod.py --pde ks --K 12 --basis pod     --out runs/M6a_ks_pod
python3 tco_pod.py --pde ks --K 12 --basis fourier --out runs/M6a_ks_fourier

# M6a-2: POD + 희소 회귀
python3 tco_pod.py --pde burgers --basis pod --method stlsq --out runs/M6a_burgers_pod_stlsq
python3 tco_pod.py --pde ks --K 12 --basis pod --method stlsq --out runs/M6a_ks_pod_stlsq

# M6a-3: 데이터 양에 따른 POD 순도 (Burgers nu=0.2, closed)
for n in 25 50 100 200 400; do
  python3 tco_pod.py --pde burgers --params 0.2 --regimes closed --n_traj $n --out runs/M6a_datasize_$n
done

# (선택) M3b: KS를 새 카오스 지표로 다시 (메모리 약 12GB, 약 15분)
python3 tco_pde.py --pde ks --K 12 --out runs/M3b_ks_lstsq
python3 tco_pde.py --pde ks --K 12 --mask_selection --closure cubic --out runs/M3b_ks_mask_cubic
```

비교 (closure 패널은 POD run에 해당 지표가 없어 비어 있어도 정상):
```bash
for p in burgers kdv fisher ks; do
  python3 tco_compare.py runs/M6a_${p}_fourier runs/M6a_${p}_pod --out runs/M6a_compare_$p
done
```

## 지표 (`tco_pod.py`)
| 키 | 뜻 |
|---|---|
| `T_rel_err`, `T_rel_err_on_rule`, `T_selection_mass` | 학습 모델을 실수 푸리에 좌표로 옮긴 뒤의 이차 텐서 오차. 실수 좌표의 선택 규칙은 $k_{out} \in \{k_p+k_q, \lvert k_p-k_q\rvert\}$ |
| `A_rel_err` | 선형부 오차 (실수 푸리에 좌표, 전체 행렬) |
| `span_mismatch` | 기저가 만든 공간이 푸리에 $\lvert k\rvert \le K$ 공간과 얼마나 다른가 (0이면 같음) |
| `pod_purity_mean`, `pod_purity`, `pod_dominant_k` | POD 모드 각각이 한 파수에 몰린 정도 (1이면 순수한 푸리에 모드) |
| `pod_singular_values`, `pod_pair_ratio` | POD 특이값과 cos/sin 짝 비율 (1이면 완전히 겹친 짝) |
| `eig_purity_mean`, `eig_dominant_k`, `eig_values` | 학습된 $A$ 의 고유벡터의 순도와 고윳값 |
| `test_rel_err_dzdt` | 시간미분 예측 오차 |
| `rollout_short_rel_err_median5`, `spectrum_rel_err_median5`, `rollout_rel_err_final_median5` | 짧은 구간 롤아웃 오차, 에너지 스펙트럼 오차, 마지막 시점 오차 |

새 `tco_pde.py`에도 `rollout_short_rel_err_median5`와 `spectrum_rel_err_median5`가 추가됐다. 카오스(KS)에서는 마지막 시점 오차 대신 이 둘을 본다.

run 폴더에 `fig_pod_modes_{param}_{regime}.png`(POD 모드의 푸리에 성분)와 `pod_{param}_{regime}.npz`(기저, 텐서)가 저장된다.

## 스모크에서 이미 본 것 (참고만)
- closed에서 POD 기저로도 정답 텐서를 복원한다 (1e-7~1e-12). POD가 만든 공간은 푸리에 공간과 거의 같다.
- quick 설정에서 POD 순도는 0.65~0.86이다. Burgers 기본 크기에서는 0.96이었다. 데이터가 많을수록 POD가 푸리에에 가까워질 것으로 예상한다(M6a-3로 확인).
- 학습된 $A$ 의 고유기저 순도는 Burgers, KdV, Fisher에서 0.98~1.000이다. KS는 0.96~0.99인데, nu=0.05에서 $k=2$ 와 $k=4$ 의 선형 고윳값이 같아서(둘 다 3.2) 고유벡터가 섞일 수 있다. KS에서 순도가 1보다 낮은 모드가 어느 파수인지 보고한다.
- full에서 POD의 텐서 오차는 같은 데이터의 푸리에 기본 회귀와 비슷하다 (대칭 조건이 없으므로 예상된 결과).
- `--closure cubic`은 quick에서 데이터가 특징 수보다 적어 발산한다. 이번 실험에는 넣지 않았다.

## 보고할 것
1. 스모크 결과
2. PDE별 POD vs 푸리에 표 (param × regime): `T_rel_err`, `T_rel_err_on_rule`, `T_selection_mass`, `A_rel_err`, `span_mismatch`, `pod_purity_mean`, `eig_purity_mean`, `test_rel_err_dzdt`, `rollout_short`, `spectrum`
3. M6a-3 표: n_traj별 `pod_purity_mean`, `eig_purity_mean`, `T_rel_err`
4. KS의 `eig_purity`가 1보다 낮은 모드의 `eig_dominant_k`와 `eig_values`
5. 대표 그림: 각 PDE의 `fig_pod_modes`(가장 작은 param, closed와 full)
6. (선택 run을 했다면) M3b의 `rollout_short`, `spectrum`
7. 스모크 관찰과 다른 결과, 에러, 메모리, 실행 시간

코드는 수정하지 말고, 에러가 나면 고치기 전에 보고한다.
