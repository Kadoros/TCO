# TCO 실행 묶음 (2026-10-10)

이 폴더의 파일을 서버의 기존 작업 폴더(E1, M 시리즈 결과가 있는 곳)에 **덮어써서** 넣는다.

| 파일 | 역할 |
|---|---|
| `tco_pde.py` | 1D 프레임워크 (최신: 카오스 지표 + Allen–Cahn 포함) |
| `tco_pod.py` | M6a: POD 기저 |
| `tco_trunk.py` | M6c: 신경망 트렁크 (`tco_pod.py`, `tco_pde.py` 필요) |
| `tco_ns2d.py` | 2단계: 2D Navier–Stokes (torch) |
| `tco_darcy2d.py` | 2단계: 2D Darcy (torch는 `--refine_steps` 쓸 때만) |
| `tco_compare.py` | 비교 그림 (최신: 범례 중복 처리, POD 패널) |

실행 순서 (둘은 서로 독립이라 동시에 돌려도 된다):
1. `M6a_AGENT_INSTRUCTIONS.md` — POD 기저
2. `S2_M6c_AGENT_INSTRUCTIONS.md` — 어려운 PDE + 신경망 트렁크

필요 패키지: numpy, matplotlib, torch (GPU 자동 사용).
