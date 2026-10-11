# P1 추가: 기본 DeepONet(trunk+branch 같이 학습) 비교 모델

## 바뀐 것
`tco_param.py`를 **새 버전으로 교체**한다. 기존 결과와 호환되고, 모델 `deeponet_joint`가 하나 추가됐다.

- **`deeponet`** (기존): 얼린 공용 trunk 위의 DeepONet branch. TCO와 branch만 다른 통제 비교다.
- **`deeponet_joint`** (새): 진짜 기본 DeepONet. trunk와 branch를 처음부터 같이, 필드 u_t 손실로 학습한다.
  - 채점할 때는 학습된 trunk를 QR로 직교화하고, **그 기저 위의 Galerkin 정답**과 비교한다.
  - 로그의 `basis 'deeponet_joint sN'` 줄에 그 trunk의 span 잔차(u, u_t)가 나온다. 이 trunk는 u_t를 표현하려고 배운 것이라 u를 잘 못 담을 수 있다. 이 값을 꼭 같이 보고한다.
  - `direct u_t`는 투영 없이 모델이 그대로 예측한 u_t 오차다.

기본 `--models`가 이제 `tco deeponet deeponet_joint opinf`다.

## 스모크
```bash
python3 tco_param.py --quick --trunk fourier --models deeponet_joint opinf --steps 800 --out runs/P1_smoke_joint
```
통과 기준: 에러 없이 끝나고, `basis 'deeponet_joint s0'` 줄의 `extractor check`가 1e-8 이하다.

## 실행
P1-1~P1-4가 이미 돌고 있으면 **멈추지 말고** 끝까지 둔다. 끝난 뒤 아래만 추가로 돌린다.
```bash
python3 tco_param.py --models deeponet_joint --out runs/P1_main_joint
python3 tco_param.py --models deeponet_joint --n_inputs 2048 --seeds 0 1 --out runs/P1_n2048_joint
```
아직 시작 전이면 새 버전으로 P1-1부터 그대로 돌리면 된다(`deeponet_joint`가 자동 포함).

## 보고
P1 표에 `deeponet_joint` 줄을 추가한다. 따로 적을 것: `direct u_t`, 자기 trunk의 span 잔차(u, u_t), `T_rel`, `T_var`.
