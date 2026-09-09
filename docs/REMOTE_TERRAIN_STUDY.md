# 지형 스터디 인수인계 — 다른 머신에서 실행하기

이 문서 하나로 끝나도록 썼습니다. **0장부터 순서대로** 하시면 됩니다.

무엇을 하는 실험인가. locomotion teacher를 두 개 새로 학습시키고 (하나는 지면 마찰을
환경마다 `[0.3, 1.0]`에서 뽑고, 하나는 최대 `0.4` 기울기의 경사면), 그 위에서 Table I의
파이프라인 전체를 다시 돌립니다.

왜 하는가. 논문의 핵심 주장은 *"디딤발이 base를 고정 좌표계에 묶으므로 관절 엔코더
히스토리가 privileged 상태를 결정한다"*인데, 발표된 숫자는 전부 마찰이 `(1.0, 1.0)`으로
고정된 평지에서 나왔습니다. 그 묶임이 가장 강한 조건입니다. **마찰**은 접촉이
미끄러지면 어떻게 되는지를, **경사**는 접촉이 묶이는 그 지면이 중력과 어긋나면 어떻게
되는지를 묻습니다. 후자가 더 날카롭습니다 — 관절각은 **지면 기준** 자세를 주는데
`projected_gravity` 타깃은 **중력 기준** 자세를 요구하고, 평지에서만 그 둘이 일치합니다.

---

## 0. 환경 확인 (제일 먼저)

그쪽 머신에 이미 `jose` 환경이 있다고 들었습니다. **버전이 맞는지부터 확인해 주세요.**
논문 결과를 낸 이 머신의 환경을 레퍼런스로 같이 보냅니다.

```bash
cd <JOSE 체크아웃>
conda activate jose            # 또는 그쪽에서 쓰는 이름
python docs/check_env.py
```

`docs/reference_env_jose.txt`와 대조해서 패키지별로 맞는지, 그리고 버전표로는 못 잡는
것들 — JOSE가 import 되는지, 새 지형 task id 4개가 등록되는지, 495차원 관측 레이아웃이
그대로인지, 로봇 에셋이 있는지, 워크트리가 깨끗한지 — 까지 봅니다.

기대 출력:

```
  ok    jose imports
  ok    terrain task ids register  4 ids
  ok    observation layout frozen  policy 495, target 9
  ok    robot assets present  usd/ 81 MB
  ok    git worktree clean  HEAD <sha>

PASS -- this environment matches the reference closely enough to train on.
```

### 무엇이 blocking이고 무엇이 아닌가

| 항목 | 레퍼런스 | 다르면 |
|---|---|---|
| isaacsim | 5.1.0.0 | **중단.** 시뮬레이터가 다르면 teacher가 배우는 게 달라집니다 |
| isaaclab / _tasks / _rl / _assets | 0.54.4 / 0.11.16 / 0.5.2 / 0.2.4 | **중단.** 환경 구성과 매니저 동작이 바뀝니다 |
| torch | 2.7.0+cu128 | **중단** |
| rsl-rl-lib | 5.0.1 | **중단.** PPO teacher를 싣고 돌리는 쪽입니다 |
| skrl | 2.1.0 | **중단.** (`skrl_compat.py`가 `>=2,<3`을 강제합니다) |
| numpy, gymnasium, scipy, trimesh 등 | 표 참조 | 기록만. 치명적이지 않습니다 |

체커가 blocking을 잡으면 exit 1로 끝납니다.

### GPU가 다른 건 괜찮습니다

레퍼런스는 **RTX 4070 (12 GB)**, 그쪽은 **RTX 4090 (24 GB)**입니다. 문제 없습니다.

이 저장소는 애초에 머신 간 비트 재현성을 주장하지 않습니다 —
`torch.use_deterministic_algorithms`는 어디서도 호출되지 않고
`cudnn.deterministic`은 `train_ppo_walk.py:110`에서 `False`입니다. 같은 GPU·같은 스택
안에서만 성립하고, 실제로 그 안에서는 입증돼 있습니다 (walk의 `lstm_w25_all`이 21시간 뒤
다른 digest에서 `10.939131736755371`을 자릿수까지 재현).

이 스터디에는 영향이 없습니다. **teacher 두 개를 그쪽에서 처음부터 학습시키고, 모든
방법이 자기 옆에서 학습된 그 teacher와 비교되기 때문**입니다. 다만 한 가지는 기억해
주세요 — **그쪽에서 나온 teacher는 이 머신이 만들었을 teacher와 다릅니다.** 같은 시드라도
그렇습니다. 그러니 평지 teacher 숫자와의 차이를 곧바로 "지형 효과"로 읽으면 안 됩니다.

**설정은 바꾸지 마세요.** 24 GB가 있다고 `--num_envs`를 올리면 배치 구성이 달라져서
평지 결과와 나란히 놓을 수 없게 됩니다. 4096 그대로 두시면 됩니다. VRAM 여유는 학습이
더 안정적으로 돌아가는 쪽으로만 쓰입니다.

---

## 1. 무엇을 받아야 하나

```
git clone <remote>/JOSE.git && git checkout <commit>   # 브랜치 끝이 아니라 그 커밋
usd/         81 MB   로봇 에셋, 패키지 상대경로로 해석됨
motions/    3.8 MB   locomotion엔 불필요하지만 import 안전용
```

teacher 체크포인트는 **보내지 않습니다.** 이 스터디는 자기 teacher를 직접 학습합니다.

Python 소스에는 절대경로가 하나도 없습니다 (grep으로 확인). `logs/jose_g1/`의 셸
드라이버들은 대부분 절대경로를 박고 있지만, **이 스터디용 스크립트 두 개는 환경변수를
받도록 새로 썼습니다.**

실행은 `python -m jose.<모듈>` 또는 저장소 루트에서 `python <스크립트>.py`로 하세요.
옛 드라이버에 보이는 `python -m JOSE.<모듈>`(대문자)은 체크아웃 디렉터리 이름이 `JOSE`이고
그 부모가 작업 디렉터리일 때만 우연히 동작합니다.

`run_friction_sweep.py`는 워크트리가 더러우면 실행을 거부합니다. 시작 전에 커밋해서
매니페스트의 `git_head`가 실제로 돌아간 코드를 가리키게 하세요.

---

## 2. 실행

```bash
export JOSE_PY=/path/to/envs/jose/bin/python

VARIANT=slope    bash logs/jose_g1/run_terrain_study.sh
VARIANT=friction bash logs/jose_g1/run_terrain_study.sh
```

변형마다 4단계이고 각 단계에서 재개됩니다. 이미 있는 teacher 체크포인트는 재학습하지
않고 재사용하며, 스터디 러너 둘 다 기본이 `--resume`입니다.

| 단계 | 내용 | RTX 4070 실측 | 4090 예상 |
|---|---|---|---|
| 1 | teacher, 4096 env로 5000 iteration | 2시간 33분 | 대략 1시간 20분~1시간 40분 |
| 2 | teacher 게이트 — sanity check 6개 | 수 분 | 수 분 |
| 3 | `run_method_comparison.py`, 4개 암 × 3시드 | 약 5시간 15분 | 대략 3시간 |
| 4 | `run_set_baseline.py`, 3시드 | 약 50분 | 대략 30분 |

4070 기준 변형당 약 9시간, 둘 다면 18시간입니다. 4090 숫자는 **추정치**입니다 — 실측이
아니라 이 워크로드의 통상적인 세대 차이로 잡은 값이니 그대로 믿지는 마세요. 팬아웃은
없습니다. GPU 하나짜리 큐입니다.

경사 변형은 첫 실행 때 씬 빌드에 몇 분이 더 붙습니다. 지형 타일 200개를 생성하고
쿠킹하기 때문입니다. `use_cache=True`라서 한 번만 내면 되고, 이건 생성기 시드를
`seed=42`로 고정했기 때문에 안전합니다.

### 2단계는 보고서가 아니라 게이트입니다

teacher가 명령을 따르지 않으면 그 아래 모든 숫자가 무의미해지므로 스크립트가 **멈춥니다.**
가정이 아닙니다 — 상류 레시피가 "명령을 무시하는 정지 정책"으로 수렴한 전례가 있고,
그래서 이 저장소의 보상 세트가 상류와 다릅니다. 게이트가 실패하면 답은 iteration을 더
돌리거나 보상 가중치를 보는 것이지, 그냥 진행하는 게 아닙니다.

머신이 처음이면 스모크부터:

```bash
$JOSE_PY train_ppo_walk.py --task Isaac-G1-PPO-Walk-Slope-JOSE-v0 \
    --headless --num_envs 64 --max_iterations 5
```

스모크가 남긴 체크포인트를 본 실행이 teacher로 재사용하는 사고는 막아뒀습니다 —
드라이버가 iteration 번호를 보고 예산에 못 미치면 무시합니다.

---

## 3. 결과 보내기

```bash
VARIANT=all bash logs/jose_g1/collect_terrain_results.sh
```

tarball 하나가 나옵니다 — 결과·리포트·매니페스트, teacher 체크포인트 두 개, 드라이버
로그, git head와 설치 버전이 든 `PROVENANCE.txt`, 그리고 `SHA256SUMS`.

`logs/`는 gitignore라 git으로는 결과가 이동하지 않습니다. **`logs/`를 통째로 복사하지
마세요** — 약 69 GB이고 거의 전부가 라운드별 추정기 체크포인트와 데이터셋 캐시이며,
번들에 든 것만으로 전부 재생성됩니다.

```bash
rsync -avP ~/jose_terrain_all_<stamp>.tar.gz <user>@<host>:~/
# 받는 쪽에서, 분석 전에 반드시
tar -xzf jose_terrain_all_<stamp>.tar.gz
cd jose_terrain_all_<stamp> && sha256sum -c SHA256SUMS
```

---

## 4. 도착 후 (이쪽에서 할 일)

1. `PROVENANCE.txt` 확인 — `git_dirty`가 `no`, 버전이 0장 표와 일치.
2. teacher 게이트 통과 확인 — `driver_logs/teacher_eval.json`, sanity check 6개 전부.
3. `logs/paper_data/` 규약대로 미러링 (그쪽 `README.md` 참조): artifact마다 디렉터리
   하나, `results.jsonl` + `report.md` + `SOURCE.txt`, `trace_prediction` /
   `trace_target`은 제거하고 `metrics._stripped_for_paper_data`에 기록.
4. 감사 스크립트를 확장해서 새 셀도 재계산되게 하기. `logs/paper_data/audit_table1.py`가
   템플릿입니다 — `main.tex`에서 표를 파싱해 셀마다 출처와 대조합니다.

---

## 5. 숫자를 읽을 때 알아야 할 함정 하나

경사 변형은 정의를 바꾸지 않지만 두 항의 **기준을 옮깁니다.** 생존률을 읽을 때 중요합니다.

Isaac Lab의 `root_height_below_minimum`은 절대 world `z`를 임계값과 비교하고, 자기
docstring에 *"평지에서만 지원됨"*이라고 적혀 있습니다. 이 지형에서는 지면이 대략
±1.2 m 움직입니다 (기울기 0.4인 피라미드가 플랫폼에서 가장자리까지 `0.4 × (8-2)/2` m
올라갑니다). 실측했습니다:

```
지면 z 편차            1.430 m  (-0.645 .. +0.785)
상대 z                 +0.783   전 로봇 동일
평지 규칙이면 종료      32대 중 2대   <- 똑바로 서 있는데
지형 상대 규칙이면 종료  0대
```

상류의 해법은 base 접촉으로 종료 조건을 바꾸는 것인데, 이 스터디는 그걸 택하지
않았습니다. 논문이 생존을 "base가 태스크별 높이 아래로 떨어짐"으로 정의하고, 경사
블록이 평지 블록 옆에서 읽히려면 둘이 같은 사건을 재야 하기 때문입니다. 그래서 정의는
유지하고 기준만 지역화했습니다 — `ppo_walk/mdp/terrain_mdp.py`가 레이캐스트 지면 높이를
빼줍니다. `base_height_l2` 보상도 같은 처리를 받는데, 이유는 다릅니다: 그대로 두면 로봇이
서 있는 지면의 고도에 대해 벌점을 물어서 `1.2²`에 가중치 `-1.0`이 되고, 추종 보상 전체를
덮어써서 평평한 곳의 고도만 찾아다니는 정책이 학습됩니다.

높이 스캐너는 **그 두 항에만** 쓰입니다. 관측 그룹에는 절대 들어가지 않습니다.
의도적입니다 — `height_scan` 관측을 추가하면 `schema.py`가 고정하고 `test_jose.py`가
단언하는 495차원 레이아웃이 깨지고, 그러면 추정기 주입 인덱스가, 즉 논문 전체가 다루는
그 인터페이스가 달라집니다. 계단이나 random height field 대신 경사면을 고른 이유이기도
합니다. 경사면은 국소적으로 평평해서 관측을 안 바꾸고도 blind 정책이 걸어갑니다.
