# State-aware Recovery Bridge 구현 및 검증

작성일: 2026-09-03

판단: **State-aware Recovery baseline qualification GO**

## 1. 해결하려는 문제

기존 `Frozen BC + CBF`는 CBF 개입이 끝난 뒤 과거 controller event로 되감기거나, CBF가 만든 BC 분포 밖의 상태에서 바로 BC를 재개했다. 그 결과 안전성은 높지만 task progress가 파괴되어 timeout이 반복됐다.

이번 변경은 BC와 CBF를 재학습하거나 파라미터를 바꾸지 않고 다음 구조를 구현한다.

`Frozen BC -> CBF -> State-aware Recovery Bridge -> Frozen BC`

고정한 평가 계약은 다음과 같다.

- BC checkpoint: `v3_chan/policies/bc_pick_place_v2_release_settle.pt`
- 정책 입력에서 사람 관측 mask 유지
- CBF: safe gap 5 cm, activation gap 13 cm, gamma 8, prediction 0.15 s, buffer 8 cm
- strict task schema: `physical_event_driven_pick_place_v5`
- recovery schema: `state_aware_pick_place_recovery_bridge_v3`
- 성공: 실제 policy OPEN, grasp loss, 목표 위치·속도 조건, 12-step settle이 모두 확인된 경우만 인정
- horizon: 1,200 steps
- seed: 11
- development: clean19
- held-out: clean17; dev gate 통과 후 설정 변경 없이 1회 개봉

## 2. 구현 내용

### Cube를 들고 있을 때

1. 현재 cube와 EE 상태에서 안전 높이까지 lift
2. 현재 place target 위로 transport
3. live cube-to-EE offset을 반영해 place endpoint로 descend
4. CBF clear, cube 속도, joint 속도, place XY/Z 조건을 여러 step 확인
5. event 6 endpoint에서 frozen BC로 handoff
6. release와 settle은 frozen BC와 strict task semantics가 수행

### Cube를 놓쳤을 때

1. 기록된 과거 위치가 아니라 현재 cube pose로 pre-grasp anchor 생성
2. recovery controller로 이동
3. post-action 물리 상태를 다시 확인
4. event 1에서 BC로 handoff하여 re-grasp

### 안전·성공 판정 강화

- Recovery action도 마지막 CBF를 반드시 통과한다.
- CBF가 policy OPEN을 차단하면 release로 기록하지 않는다.
- 최종 CBF 이후 실제 `robot.apply_action`에 OPEN이 전달된 경우에만 release latch를 세운다.
- 목표 위에서 우연히 cube를 놓친 경우는 OPEN 명령이 없으면 성공이 아니다.
- 일시적인 grasp 신호 손실은 다중-step 확인 전까지 release/regrasp로 해석하지 않는다.
- branch snapshot은 strict/recovery FSM, pending handoff, gripper state까지 저장하고, config가 다르면 물리 상태를 바꾸기 전에 fail-closed한다.
- exact-prefix consumer는 strict/recovery branch state가 다르면 동일 prefix로 인정하지 않는다.

핵심 파일:

- `v3_chan/rl/state_aware_recovery.py`
- `v3_chan/rl/strict_task_semantics.py`
- `v3_chan/rl/pick_place_env.py`
- `v3_chan/evaluate_rollout_policy.py`
- `v3_chan/run_physical_safety_benchmark.py`
- `v3_chan/validate_release_bc_no_human.py`
- `v3_chan/run_bc_cbf_clean36_eval_v2.sh`

## 3. 검증 설계

동일 checkpoint, encounter, scene layout, seed, strict-v5 성공 정의로 2x2를 비교했다.

| Re-entry | Policy |
|---|---|
| 기존 re-entry | BC only |
| 기존 re-entry | BC + CBF |
| State-aware Recovery | BC only |
| State-aware Recovery | BC + CBF |

Treatment와 control의 설정 차이는 recovery enable과 이를 반영하는 관련 bool 3개뿐이다. Episode의 seed, encounter, layout, cube/target identity는 모두 exact match다. BC-only 결과도 두 re-entry 조건에서 완전히 동일하므로 비개입 경로에는 영향이 없었다.

## 4. 최종 결과

### clean36 합산

| Re-entry | Policy | Task success | 사람 충돌 | Near-miss | CBF 개입 episode 성공 | Recovery timeout |
|---|---|---:|---:|---:|---:|---:|
| 기존 | BC only | 35/36 (97.2%) | 12/36, 813 steps | 18/36, 652 steps | 해당 없음 | 0 |
| 기존 | BC + CBF | 12/36 (33.3%) | 0/36 | 0/36 | 1/25 (4.0%) | 해당 없음 |
| State-aware | BC only | 35/36 (97.2%) | 12/36, 813 steps | 18/36, 652 steps | 해당 없음 | 0 |
| **State-aware** | **BC + CBF** | **34/36 (94.4%)** | **0/36** | **0/36** | **23/25 (92.0%)** | **0** |

State-aware Recovery는 동일 strict-v5 대조군 대비 전체 성공률을 `12/36 -> 34/36`으로, 실제 CBF 개입 episode 성공률을 `1/25 -> 23/25`로 회복했다. 사람 충돌과 near-miss는 모두 0을 유지했다.

### Split별 결과

| 조건 | clean19 dev | clean17 held-out |
|---|---:|---:|
| BC only | 19/19 | 16/17 |
| 기존 BC + CBF | 6/19; 개입 1/14 | 6/17; 개입 0/11 |
| **BC + CBF + Recovery** | **17/19; 개입 12/14** | **17/17; 개입 11/11** |

### 행동 비용

| 지표 | BC only | 기존 BC + CBF | BC + CBF + Recovery |
|---|---:|---:|---:|
| 평균 steps | 715.9 | 1,036.8 | 775.8 |
| 평균 EE path | 1.244 m | 1.406 m | 1.435 m |
| 평균 RMS EE jerk | 170.5 | 159.8 | 198.1 |
| 평균 minimum gap | 3.749 cm | 7.373 cm | 7.090 cm |
| 전체 minimum gap | -3.500 cm | 4.028 cm | 3.491 cm |

Recovery가 task를 크게 회복했지만 path와 jerk는 증가했다. 이는 다음 human-aware/minimal-intervention 단계에서 개선할 대상이다.

## 5. 실패 2건과 해석 범위

Recovery 조건의 실패는 clean19 episode 8, 16 두 건이며 모두 `max_episode_steps=1200`이었다. 내부 recovery timeout, 충돌, near-miss는 없었다. 따라서 공식 결과에서는 deadlock 성공으로 재분류하지 않고 그대로 34/36으로 유지한다.

결과 해석에는 다음 제한이 있다.

- 5 cm는 CBF 설정값이지 관측된 hard separation guarantee가 아니다. Recovery 조건의 전체 minimum gap은 3.491 cm였다.
- 여기서 충돌은 측정 가능한 사람-로봇 충돌이다. 이 평가의 static/self collision signal은 `unavailable_not_inferred`다.
- 단일 seed, 36 encounter의 baseline qualification이며 논문 최종 통계는 아니다.
- 자동 `physical_feasibility.json`은 사전 고정 threshold 파일을 제공하지 않아 의도적으로 `NOT_EVALUATED/NO_GO`다. 위 GO는 사전 dev gate와 held-out 결과에 대한 engineering/research baseline 판단이지 자동 gate를 통과했다는 뜻이 아니다.

## 6. 무결성과 코드 검증

- Recovery ON/OFF no-human gate: 각각 1/1 PASS
- 두 gate 모두 policy OPEN step 673, strict success step 693
- clean36 2x2의 성공 rollout 116/116에서 실제 policy OPEN이 strict success보다 먼저 발생
- 성공 rollout의 release latch 위반: 0
- exact paired evaluation: dev 19/19, held-out 17/17
- 최종 관련 회귀: **210 passed**
- Python compile: PASS
- launcher `bash -n`: PASS
- `git diff --check`: PASS
- source/checkpoint/manifests SHA: snapshot과 exact match
- 종료 후 잔여 Isaac/evaluator/benchmark 프로세스: 0

Source snapshot:

`v3_chan/eval_results/state_aware_recovery_v3_strict_v5/source_manifest.json`

Result snapshot:

`v3_chan/eval_results/state_aware_recovery_v3_strict_v5/artifact_manifest.json`

## 7. 최종 판단

> **State-aware Recovery physical baseline qualification: GO**

현재 결과는 frozen BC와 CBF를 유지하면서 기존 re-entry가 만든 task failure를 구조적으로 복구했다. 다음 단계는 PPO 재학습이 아니라 이 baseline을 동결하고 intervention magnitude, duration, retreat, BC deviation, jerk, recovery time과 explicit accept/reject feedback을 수집하는 것이다.

## 8. 재현 명령

State-aware Recovery:

```bash
cd /home/railabchan/isaac_vr_project
STATE_AWARE_RECOVERY=1 \
TAG_STEM=bc_pick_place_v2_release_settle_state_aware_recovery_v3_strict_v5 \
DEV_TAG=bc_pick_place_v2_release_settle_state_aware_recovery_v3_strict_v5_clean19_dev \
HELDOUT_TAG=bc_pick_place_v2_release_settle_state_aware_recovery_v3_strict_v5_clean17_heldout \
FORCE=1 \
bash v3_chan/run_bc_cbf_clean36_eval_v2.sh all
```

기존 re-entry 대조군:

```bash
cd /home/railabchan/isaac_vr_project
STATE_AWARE_RECOVERY=0 \
TAG_STEM=bc_pick_place_v2_release_settle_existing_reentry_strict_v5 \
DEV_TAG=bc_pick_place_v2_release_settle_existing_reentry_strict_v5_clean19_dev \
HELDOUT_TAG=bc_pick_place_v2_release_settle_existing_reentry_strict_v5_clean17_heldout \
FORCE=1 \
bash v3_chan/run_bc_cbf_clean36_eval_v2.sh all
```

`FORCE=1`을 빼면 이미 존재하는 valid artifact를 재사용한다.
