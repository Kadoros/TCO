# 2단계(어려운 PDE) + M6c(학습 트렁크) 실행 지시 (서버 에이전트용)

## 파일
기존 폴더에 둔다. `tco_pde.py`는 **교체**한다(Allen–Cahn 추가, 기존 결과와 호환).
- `tco_pde.py` (교체: `--pde allencahn` 추가)
- `tco_ns2d.py` (새: 2D Navier–Stokes, PyTorch 솔버)
- `tco_darcy2d.py` (새: 2D 주기 Darcy, 입력 함수 a(x), f(x) → 해 u(x))
- `tco_trunk.py` (새: M6c 신경망 트렁크, `tco_pod.py`와 `tco_pde.py`에서 import)
- `tco_pod.py`, `tco_compare.py`는 M6a 때 것 그대로

필요 패키지: numpy, matplotlib, **torch**(GPU가 있으면 자동으로 씀). `python3 -c "import torch; print(torch.cuda.is_available())"`로 확인한다.

## 스모크 테스트
```bash
python3 tco_pde.py --pde allencahn --closure cubic --mask_selection --n_traj 60 --n_test 10 --params 0.05 --kmax_full 48 --regimes closed full --out runs/S2_smoke_ac
python3 tco_ns2d.py --quick --K 4 --out runs/S2_smoke_ns
python3 tco_darcy2d.py --quick --out runs/S2_smoke_darcy
python3 tco_trunk.py --pde burgers --quick --beta 1 --out runs/S2_smoke_trunk
```
통과 기준: 각 스모크의 **closed** 줄에서 Allen–Cahn `Crel`, NS `Trel`, Darcy `Lrel`이 1e-8 이하. 트렁크는 closed `trunk` 줄의 `Trel`이 1e-3 이하, `span`이 1e-3 이하. 아니면 멈추고 보고한다.

## 실험
### S2-NS: 2D Navier–Stokes (Kolmogorov 강제 n_f=4, 감쇠 α=0.1, ν = 0.02, 0.01, 0.005)
```bash
python3 tco_ns2d.py --K 6 --out runs/S2_ns_K6_mask
python3 tco_ns2d.py --K 8 --out runs/S2_ns_K8_mask
# 대칭 조건 유무 비교 (dense는 특징 수가 n^2/2라 작은 K에서만 가능)
python3 tco_ns2d.py --K 4 --nf 2 --n_traj 128 --out runs/S2_ns_K4_mask
python3 tco_ns2d.py --K 4 --nf 2 --n_traj 128 --dense --out runs/S2_ns_K4_dense
```
dense K=4는 출력당 특징이 약 3,300개라 CPU 최소제곱이 오래 걸릴 수 있다(수십 분). 메모리가 부족하면 `--n_traj 96`으로 줄이고 보고한다.

### S2-Darcy: 입력 함수 (a, f) → u
```bash
python3 tco_darcy2d.py --out runs/S2_darcy_smooth
python3 tco_darcy2d.py --alpha 2 --tau 2 --regimes full --out runs/S2_darcy_rough
# 해의 오차를 직접 줄이는 경사하강법 다듬기 (GPU 권장)
python3 tco_darcy2d.py --alpha 2 --tau 2 --regimes full --refine_steps 3000 --refine_lr 3e-3 --out runs/S2_darcy_rough_refine
```

### S2-AC: Allen–Cahn (3차 물리 = 4차 텐서)
```bash
python3 tco_pde.py --pde allencahn --closure cubic --mask_selection --regimes closed full --out runs/S2_ac_mask
python3 tco_pde.py --pde allencahn --closure cubic --mask_selection --amp 0.3 --regimes closed full --out runs/S2_ac_mask_amp03
python3 tco_pde.py --pde allencahn --closure cubic --regimes closed full --out runs/S2_ac_nomask
```

### M6c: 신경망 트렁크
```bash
for b in 0 1 10; do
  python3 tco_trunk.py --pde burgers --beta $b --out runs/M6c_burgers_beta$b
done
python3 tco_trunk.py --pde kdv --beta 1 --out runs/M6c_kdv_beta1
python3 tco_trunk.py --pde ks --K 12 --beta 1 --out runs/M6c_ks_beta1
```

## 지표
- **tco_ns2d.py**: `T_rel_err`(선택 규칙 위 전체), `T_rel_err_by_cut`(|k|,|p|,|q| ≤ cut인 삼체만, **스케일별**), `A_diag_rel_err`, `forcing_rel_err`(강제항 복원), `closure_rel_size`, `closure.nu_t_*`(껍질 평균 eddy viscosity), `rollout_short_rel_err_median`, `spectrum_rel_err_median`(엔스트로피 스펙트럼)
- **tco_darcy2d.py**: `L_rel_err`, `L_rel_err_by_cut`, `test_u_rel_err_learned` vs `test_u_rel_err_naive_truncation`(참값 물리를 그냥 잘랐을 때), `ood_*`(입력 대비 σ×1.5)
- **Allen–Cahn**: `C_rel_err`, `C_rel_err_on_rule`, `C_selection_mass`(4차 텐서), `T_spurious_rel_to_C`(정답은 0인 이차항이 얼마나 생겼나)
- **tco_trunk.py**: `trunk`와 `pod`(같은 데이터) 각각 `T_rel_err`, `span_mismatch`, `recon_rel_err_test`, `basis_purity_mean`, `eig_purity_mean`, 롤아웃 지표

## 스모크에서 이미 본 것 (참고만)
- **NS**: closed는 정확하다(1e-10~1e-14). full은 closure가 매우 크다(K=4에서 46~51%, K=6 13%, K=8 4%). 전체 텐서 오차는 크지만(K=8 quick 7.4), **큰 스케일 삼체(|k| ≤ 4)는 0.15%**로 복원된다. 오차는 차단선 근처에 몰린다. 그래서 `T_rel_err_by_cut`을 꼭 본다.
- **NS dense vs mask (K=2, quick)**: closed는 둘 다 정확하다. full에서 dense는 텐서 오차 42에 롤아웃이 발산했고, mask는 0.69에 롤아웃 1.8%였다.
- **Darcy**: closed는 정확하다(1e-13). full에서는 **학습 연산자가 단순 절단보다 1.2~1.6배 나빴다**. 최소제곱이 "방정식 잔차"를 줄일 뿐 "해 오차"를 줄이지 않아서로 추정한다. 경사하강법 다듬기는 CPU 300번에 해 오차 3.89% → 3.43%로 내려가는 중이었다(단순 절단은 2.41%). 3000번에서 단순 절단을 넘는지가 관심사다.
- **Allen–Cahn**: mask+cubic은 closed가 정확하다(1e-11). full은 C 오차 0.17~0.44다. **mask 없이는 closed에서도 실패한다**(SVD로 풀어도 학습 데이터는 1e-12로 맞추지만 C 오차가 122배). 해가 ±1로 포화되면 u³ ≈ u가 되어 식별이 안 되는 것으로 보인다.
- **M6c**: 트렁크가 POD/푸리에 공간을 다시 찾는다(span 1e-5~1e-4). closed T 오차는 2e-5(트렁크 근사 정확도 한계), 고유기저 순도는 1.000. 주기 1D에서는 POD와 거의 같고, beta에 따른 차이도 작았다.

## 보고할 것
1. 스모크 결과
2. NS 표: K × ν × regime별 `T_rel_err`, `T_rel_err_by_cut`(2, 4, K), `A_diag_rel_err`, `forcing_rel_err`, `closure_rel_size`, `closure.nu_t_rel_err`, `rollout_short`, `spectrum`. K=4 mask vs dense 비교. dense 실행 시간과 메모리
3. Darcy 표: σ × regime별 `L_rel_err`, `L_rel_err_by_cut`(K/2), learned vs naive(test, OOD). refine 학습 곡선과 최종 비교
4. Allen–Cahn 표: 세 run의 `C_rel_err`, `C_selection_mass`, `T_spurious_rel_to_C`, `test_rel_err_dzdt`, `rollout_short`
5. M6c 표: β별 trunk vs pod 지표, 트렁크 모드 그림(`fig_trunk_modes_*`)
6. 대표 그림: NS의 `fig_tensor_*`, `fig_closure_*`(가장 작은 ν)
7. 스모크 관찰과 다른 결과, 에러, GPU 사용 여부, 메모리, 실행 시간

코드는 수정하지 말고, 에러가 나면 고치기 전에 보고한다.
