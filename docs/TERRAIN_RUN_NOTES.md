# 지형 스터디 실행 노트 — 원격 머신에서 돌리며 발견한 것

`docs/REMOTE_TERRAIN_STUDY.md`대로 slope / friction 두 변형을 실행하면서 나온 코드 변경과
이슈를 정리합니다. 실행 머신은 RTX 4090, 기반 커밋은 `2f5b45f`입니다.

---

## 1. 코드 변경 (커밋 `154ffdf`)

**지형 task id가 자식 프로세스에 등록되지 않아 3·4단계가 전혀 돌지 않았습니다.**

30개 job(변형당 method_comparison 12 + SET 3)이 전부 시작 1초 만에 죽었습니다:

```
gymnasium.error.NameNotFound: Environment
`Isaac-G1-PPO-Walk-Friction-Estimator-JOSE` doesn't exist.
Did you mean: `Isaac-G1-PPO-Walk-Estimator-JOSE`?
```

네 개의 지형 id를 등록하는 것은 `ppo_walk/__init__.py`의 `from . import terrain_tasks`
한 줄인데, 자식 스크립트들은 그 패키지를 지나가지 않습니다 — `evaluate_teacher.py`,
두 배포 트레이너, `train_set_baseline.py`는 `jose.estimator.*`, `jose.schema`,
`jose.teacher_setup`만 import합니다. `import jose`로는 `jose.ppo_walk`가 딸려오지
않으므로, 정작 `--task`를 해석하는 자리에 id가 없습니다.

`train_ppo_walk.py`와 `eval_ppo_walk.py`가 멀쩡했던 것은 **우연입니다.** 둘 다
`jose.ppo_walk.utils.cli_args`를 import하고, 그 과정에서 패키지가 딸려옵니다. teacher
학습과 게이트만 정상이었던 이유가 이것입니다.

**수정:** 두 러너(`run_method_comparison.py`, `run_set_baseline.py`)가 자식 명령을
부트스트랩으로 감쌉니다.

```python
_TASK_REGISTRY_BOOTSTRAP = (
    "import runpy, sys; import jose.ppo_walk; "
    "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
)
```

`runpy.run_path`는 `train_imu_distillation.py`가 이미 쓰는 관용구입니다.

**왜 러너인가.** 자식들이 공유하는 모듈(`__init__.py`, `evaluate_teacher.py`,
`estimator/adapters.py`, `schema.py`)은 전부 fingerprint 튜플에 있습니다. 거기에 import
한 줄을 넣으면 **이미 기록된 모든 스터디의 digest가 바뀌고 재개가 거부됩니다.** 두
러너는 어떤 튜플에도 없어서 기록된 digest가 하나도 움직이지 않습니다.
`jose.ppo_walk`는 gymnasium만 끌어오고 entry-point 문자열로 등록하므로 `AppLauncher`
이전에 import해도 안전합니다 (`ppo_walk/__init__.py:14` 주석이 그 조건을 명시).

**검증:** 스터디 첫 행의 `metrics` 키가 0개 → 47개.

---

## 2. 아직 고치지 않은 것 — 판단이 필요합니다

### 2.1 실패가 성공으로 기록됩니다 (가장 중요)

Isaac Sim은 파이썬 예외가 나도 **exit 0으로 종료**합니다. 그래서
`run_method_comparison.py:377`의

```python
"status": "ok" if returncode == 0 else "failed"
```

이 죽은 job 30개를 전부 `ok`로 기록했습니다. 유일한 증상은 `metrics: {}`였습니다.

```json
{ "status": "ok", "returncode": 0, "metrics": {} }
```

7시간짜리 스터디가 2분 만에 "성공"으로 끝난 것을 이상하게 여기지 않았다면 빈 결과가
그대로 번들에 실렸을 것입니다. 게다가 `run_method_comparison.py:189`가 재개할 때
`status == "ok"`인 행을 완료로 보므로, **재실행해도 계속 건너뛰었을 것입니다.**

1번 버그를 고쳐도 이 성질은 남습니다. 다른 이유로 자식이 죽으면 또 거짓 `ok`가 나옵니다.
`metrics`가 비어 있으면 `failed`로 기록하는 방어를 넣을지 검토가 필요합니다. 러너
파일이라 digest 영향은 없습니다.

### 2.2 `ppo_walk/terrain_tasks.py`의 docstring이 사실과 다릅니다

> `ablation_catalog.TASKS` names it in the task tuples ..., and **the runners import
> it before `gym.make`**.

러너는 import하지 않습니다. 저장소에서 이 모듈을 import하는 곳은
`ppo_walk/__init__.py`, `docs/check_env.py`, `test_jose.py` 셋뿐입니다.

### 2.3 프리플라이트가 실사용 경로를 재현하지 않습니다

`docs/check_env.py:125`가 `import jose.ppo_walk.terrain_tasks`를 **직접** 실행해서
확인합니다. 그래서 `terrain task ids register  4 ids`가 ok로 통과하고, 결함은 3단계까지
숨어 있었습니다. 자식 프로세스가 하는 방식(`import jose`만 한 상태에서 `gym.spec`)으로
검사해야 이 계열을 미리 잡습니다.

---

## 3. teacher 게이트가 두 변형 모두 5/6으로 실패했습니다

**진행하기로 결정하고 3·4단계를 돌렸습니다.** 게이트 출력은
`logs/jose_g1/terrain_<variant>/teacher_eval.json`에 그대로 남아 번들에 실립니다.

```
FAIL zero command does not lift feet repeatedly
       slope     1.506 lifts/s
       friction  2.199 lifts/s        (기준 < 0.5)
```

나머지 5개는 통과합니다. 정지 명령에서 로봇이 **제자리 행진**을 합니다 — 위치는 안
밀리고(drift 0.024 / 0.002 m/s), 속도도 0이고(측정 vx 0.023 / 0.001), 생존률 100%이며,
air-time 보상도 0입니다(보상 해킹이 아님). "멈추라"는 명령에 발만 안 멈춥니다.

`EVAL_COMMANDS` 주석이 이 체크의 의도를 명시합니다 — *"the only row that exposes a
policy that marches in place or creeps when told to stand still."* 즉 저자가 정확히
이 행동을 잡으려고 만든 체크입니다.

**원인으로 보이는 것:** `walk_env_cfg.py`의 명령 설정입니다.

```python
rel_standing_envs=0.02,                       # 정지 명령 노출 2%
ranges=Ranges(lin_vel_x=(0.0, 1.0), ...)
```

정지 명령 노출이 2%뿐이고, 제자리걸음을 막는 보상이 없습니다 — `feet_air_time`은 정지
명령에서 0으로 꺼지고, 남은 것은 `energy`(-0.008), `action_rate`(-0.157) 같은 일반
평활화 페널티뿐이라 걸음을 멈추게 할 만큼 크지 않습니다. 보상이 제자리걸음을 시키는 게
아니라 말릴 것이 없는 상태입니다.

**이 설정은 `walk_env_cfg.py`, 즉 평지 기본값에 있습니다.** slope / friction cfg는
scene·rewards·terminations·curriculum만 바꾸고 명령 설정은 건드리지 않습니다. 그래서
**평지 teacher도 같은 성질일 가능성이 높습니다.** 메인의 평지 `teacher_eval.json`에서
`foot_lifts_per_s`를 확인해 주세요:

- **0.5 미만** → 지형 변형에서 생긴 회귀. 원인을 더 파야 합니다.
- **0.5 이상** → 이 레시피에서 게이트가 통과한 적이 없고 지형 스터디가 물려받은 것.
  그러면 판단이 "게이트 기준을 어떻게 볼 것인가"의 문제가 됩니다.

임계값 0.5는 `TASK_SUCCESS_MAX_NORM_ERROR` 주석의 원칙대로 *"실측에 맞춰 튜닝하지 않고
명령 공간의 구조에서 정한"* 값입니다. 무결성을 위한 선택이지만, 통과하는 teacher로
검증된 적이 없을 수도 있다는 뜻이기도 합니다.

---

## 4. `run_terrain_study.sh`로는 3·4단계에 도달할 수 없습니다

2단계 게이트가 `set -euo pipefail` + `SystemExit`이라 실패 시 하드 스톱입니다(의도된
동작). 게이트를 넘어가기로 한 뒤에는 스크립트가 실행했을 명령 두 개를 직접 돌렸습니다:

```bash
run_method_comparison.py --case locomotion_<variant> <ckpt> --seeds 42 43 44
run_set_baseline.py      --case locomotion_<variant> <ckpt> --seeds 42 43 44
```

1단계 teacher 재사용 로직은 정상 동작합니다 — `iteration >= ITERATIONS-1`을 요구하고,
스모크 런의 `model_4.pt`를 실제로 걸러냈습니다.

---

## 5. 결과를 읽을 때 알아야 할 것

### 5.1 생존률이 두 개이고 서로 다른 얘기를 합니다

같은 실행에서 나오는데 결론이 반대로 읽힙니다. **리포트에는 둘 다, 출처와 단위를 붙여
싣습니다.**

| 지표 | 단위 | 출처 |
|---|---|---|
| `grid_survival_rate` | 0~1 비율 | `estimator/locomotion.py:492`. `EVAL_COMMANDS` 15개 **고정 명령 격자**에서 측정 창 끝까지 살아있던 환경 비율의 평균. 창이 짧고 명령이 고정 |
| `death_rate` | 0~100 % | `estimator/pipeline.py:340`. 환경 **자체의 무작위 명령 재샘플링**(10초마다) 아래 완주 에피소드 중 종료로 끝난 비율. 에피소드 기반(200 에피소드) |

slope 중간 결과(시드 2개)에서:

```
grid_survival_rate   Teacher 1.0000  IMU 1.0000  Joint-Only 0.9958  JOSE 0.9997
death_rate %         Teacher 0.00    IMU 0.00    Joint-Only 4.67    JOSE 3.00
```

격자 쪽은 네 방법을 구분하지 못하고, `death_rate`는 갈립니다. 하나만 실으면 "생존률로는
차이가 없다"와 "생존률에서 갈린다"가 둘 다 참인데 반대로 읽힙니다. 단위가 비율 대
퍼센트라 같은 열에 섞으면 100배 오차도 납니다.

`estimator/locomotion.py:490-491` 주석도 이 충돌을 명시합니다 — grid 지표에 접두사를
붙인 이유가 "에피소드 기반 평가 자신의 `success_rate`/`death_rate`와 겹치지 않게"이고,
둘은 나란히 보고되도록 설계돼 있습니다.

### 5.2 추정기 RMSE는 JOSE에만 있습니다

`IMU-BasedDistillation`과 `Joint-OnlyDistillation`은 추정기를 만들지 않고 정책을 직접
증류합니다. `rmse` / `r2` / `target_rmse` 열이 비어 있는 것이 정상이며, 추정기 품질
비교는 **SET baseline(4단계)이 있어야 성립합니다.**

### 5.3 MPJPE-G는 root 위치 적분 오차에 지배됩니다

추정 타깃 9개(`base_lin_vel` ×3, `base_ang_vel` ×3, `projected_gravity` ×3)에 절대
위치가 없어서, 전역 위치는 속도를 `mpjpe_horizon: 100`(2초) 적분해 얻습니다. 그 결과
모든 방법에서 `mpjpe_g ≈ root_position_error`입니다.

`PrivilegedTeacher` 행이 5021.7 mm로 학생들보다 나쁘게 나오는데(`mpjpe_l`은 1.4 mm로
정상), privileged 상태가 곧 정답인 teacher가 그럴 이유가 없습니다. **MPJPE를 표에 쓰실
계획이면 teacher 열의 계산 경로를 확인해 주세요.** 명령 RMSE / 추정기 RMSE / 생존률만
쓰신다면 무관합니다.

---

## 6. 소요 시간이 문서 추정의 약 1.7배입니다

| 단계 | 문서(4070 실측 / 4090 추정) | 실제(4090) |
|---|---|---|
| teacher 5000 iter | 2h33m / 1h20m~1h40m | **slope 3h05m, friction 2h07m** |
| method_comparison | 5h15m / ~3h | **~9h** (시드당 ~3h) |

시드당 job 시간은 PrivilegedTeacher 3분, IMU 60분, Joint-Only 51분, JOSE 48분입니다.

원인으로 짚이는 것은 이 머신의 상태입니다. 매 실행 로그에 찍힙니다:

```
[Warning] CPU performance profile is set to powersave.
[Warning] PCIe link generation current (1) and maximum (4) for device 0 don't match.
```

PCIe가 gen4 가능한데 gen1로 링크돼 있습니다. 둘 다 root 권한이 필요해 손대지
않았습니다. 4090 추정치를 문서에 확정하시기 전에 이 점을 감안해 주세요.

---

## 7. 전달 경로 관련

`logs/`가 gitignore이므로 **`logs/jose_g1/run_terrain_study.sh`와
`collect_terrain_results.sh`는 git으로 오지 않습니다.** 문서 §1 "무엇을 받아야 하나"에
`git clone` + `usd/` + `motions/`만 적혀 있는데, 이 두 스크립트도 목록에 넣어야 합니다.
처음 실행 시 이것 때문에 막혔습니다.
