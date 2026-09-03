# A/C Selective-Smoothing VR Explicit-Feedback 수집환경

이 문서는 Frozen BC와 동일한 physical-safety/Recovery 계약을 사용하는
`A_reactive`와 `C_smooth`를 비교하기 위한 전용 VR 수집환경의 운영 문서다.
schema는 `bc_cbf_recovery_ac_explicit_feedback_v1`이다. 이 수집기는 사람의
feedback으로 현재 또는 이후 trial의 condition을 바꾸지 않으며 critic 학습,
BC fine-tuning, CBF parameter adaptation, ErrP, 햅틱을 실행하지 않는다.

기존 3-cube 수집환경은 보존한다. 이 환경에서 한 trial은 BC-feasible
single-pick 1회, 의도된 hand crossing 1회, A 또는 C response 1회, HMD 안의
mandatory evaluation 1회다.

## 고정한 결정

| 항목 | 고정값 |
|---|---|
| Frozen BC | `v3_chan/policies/bc_pick_place_v2_release_settle.pt` |
| checkpoint SHA-256 | `d0fe9f9dc48a6049b77c8e1eab6de0905197207bb57ff858e0d7d332172b5230` |
| A | `A_reactive`, `joint_nominal`, `lambda_s=0` |
| C | `C_smooth`, `smooth_intervention`, `lambda_s=4.0` |
| 공통 control | 동일 CBF constraints/solver/fail-closed, strict task v5, Fixed State-aware Recovery v3 |
| 기본 설계 | practice 4 + evaluated 16 (`minimal_pilot`) |
| anchor 설계 | practice 4 + evaluated 20 (`pilot_with_anchors`, anchor-repeat 4) |
| 햅틱 | 항상 OFF; runner가 `BHAPTICS_ENABLED=0`, `HRI_HAPTIC_CONDITION=off`로 강제 |
| condition 변경 | trial `RESET`에서만 사전 배정값 적용; feedback 기반 변경 금지 |
| UI | condition을 숨긴 중앙 `XRSceneView` text panel |

16개 core trial은 task phase 4 × severity 2 × condition 2다. 방향과 속도는
별도 factorial dimension이 아니며 각각 8:8로 균형화한다. participant/session
가명과 seed로 schedule을 사전에 결정하고, phase·severity 안의 A/C 수를
동일하게 유지하며 같은 condition의 연속 길이를 최대 2로 제한한다.

Safety-response detector는 다음 phase를 구분한다.

```text
PRE_RESPONSE -> CBF_ACTIVE -> SMOOTH_TAIL -> RECOVERY_ACTIVE
             -> BC_RESUMED -> STABLE_TASK_RESUMPTION
```

onset은 active constraint와 `intervention_norm >= 0.05 rad/s`가 3 control
frame 연속일 때 confirm한다. stable resumption은 CBF/tail/Recovery가 모두
inactive이고 BC가 재개되었으며 `intervention_norm <= 0.01 rad/s`인 상태가
0.5초 유지될 때다. 설문은 이 stable 판정 뒤에만 표시한다.

Trial lifecycle은 다음 17개 상태를 한 방향으로만 통과한다.

```text
RESET -> LOAD_BC_FEASIBLE_SCENARIO -> START_FROZEN_BC
-> WAIT_FOR_TARGET_TASK_PHASE -> SHOW_HAND_CROSSING_CUE
-> EXECUTE_SINGLE_CROSSING -> RUN_ASSIGNED_RESPONSE_A_OR_C
-> TRACK_SAFETY_RESPONSE_EPISODE -> DETECT_STABLE_TASK_RESUMPTION
-> SHOW_RETURN_HAND_TO_NEUTRAL_CUE -> PAUSE_NOMINAL_TASK_PROGRESSION
-> SHOW_MANDATORY_FEEDBACK_UI -> SAVE_FEEDBACK
-> RESUME_TASK_TO_COMPLETION_OR_TERMINAL -> SAVE_TRIAL
-> VALIDATE_TRIAL -> NEXT_TRIAL_OR_END_SESSION
```

설문 중에는 nominal task progression만 safe hold로 멈추며 physics, CBF,
emergency abort는 계속 켜 둔다. 응답 후 task success/failure까지 진행한다.

## VR controller 입력

실시간 marker는 crossing하지 않는 반대쪽 controller를 사용한다.

| 상황 | safety concern | behavior anomaly |
|---|---|---|
| 왼손 crossing | 오른손 A | 오른손 B |
| 오른손 crossing | 왼손 X | 왼손 Y |

설문 navigation은 `X=이전`, `Y=다음`, `A=선택`, `B=확인`, `X+Y=뒤로`다.
`A+B`를 2초 유지하면 emergency abort다. 실제 OpenXR input/gesture 이름은
startup에서 cached controller device로 discovery하므로, 첫 HMD smoke에서
Quest/ALVR runtime이 이 logical mapping을 제공하는지 반드시 확인한다.
각 trial control step은 actuation 전과 feedback 처리 전에 양 controller와
X/Y/A/B의 known-state를 별도로 검사한다. pose tracking이 유효하더라도 button
channel이 끊기면 verified safe hold 후 `controller_input_unavailable...`로
abort하며 `.partial`에 이유를 남긴다. 이 장비 오류를 participant의
timeout/no-response와 합치지 않는다.

실시간 marker 두 종류는 독립 event다. 버튼을 누르지 않은 것은
`acceptable_as_is`가 아니다. Q1의 `uncertain`, timeout, no-response도
negative label로 바꾸지 않는다. 저장되는 `response_disposition`은 확정 응답
`answered`, 완료된 uncertain 응답 `uncertain_abstain`, timeout/no-response
`missing_abstain`으로 구분한다. Q1이
`needs_modification`이면 정해진 reason을 하나 이상 골라야 저장할 수 있다.
`query_answers`는 각 prompt cycle을 `answer_status`로 구분한다. 완료된 설문은
Q1–Q6 6개(필요할 때 reason까지 7개)가 모두 `confirmed`여야 한다. timeout이나
no-response는 이미 확정한 prefix 뒤에 현재 보이던 prompt를 `unconfirmed`
1행으로 남기고, 값은 JSON `null`, confirmed clock과 latency는 명시적 sentinel로
저장한다. 따라서 미확정 선택을 label로 승격하지 않으면서 prompt/first-input
audit은 보존된다.

## 기록과 validator

기본 출력은
`v3_chan/ac_selective_smoothing_feedback_data/<session>_<time>.hdf5`다. schedule,
trial/encounter/query/realtime-marker 식별자와 세 clock(simulation, monotonic,
Unix), raw hand tracking, robot/object/task state, BC raw 5D action, nominal·filtered·
applied command, A/C objective와 lambda, CBF correction memory/tail, Recovery,
gap/TTC/collision, response phase를 step별로 기록한다. trial summary에는 actual
crossing, phase별 duration/jerk, intervention integral/TV, task outcome, Q1–Q6,
reason과 입력 timestamp를 기록한다. `decision_context`에는 pre-solve 정보만
허용하고 future outcome은 금지한다.

완료 파일은 구조·provenance fail-closed validator를 통과해야만 유효하다.
구조적으로 유효하지만 corridor 이탈·onset 미검출 같은 protocol 사유가 있는
trial도 최종 `.hdf5` raw data로 보존하고 `analysis_exclude=true`, session
validator report의 `study_eligible=false`로 재수집 대상임을 표시한다. 실행 중단 또는 구조 검증
실패 파일만 `.partial`로 보존한다.

```bash
/home/railab/isaac-sim-4.5.0/python.sh \
  v3_chan/validate_ac_selective_smoothing_feedback.py \
  v3_chan/ac_selective_smoothing_feedback_data/<session>.hdf5 --json
```

## 실행 전 정적 확인

repository root에서 실행한다.

```bash
cd /home/railab/Desktop/Isaac_HRC/SeRT-Training-Platform-ac-selective-smoothing-v1

python v3_chan/runtime_contracts/ac_selective_smoothing_v1/validate_runtime_contract.py
python -m unittest -v \
  tests.test_v3_ac_selective_smoothing_feedback \
  tests.test_v3_ac_feedback_video

python v3_chan/collect_ac_selective_smoothing_feedback.py \
  --participant-id TEST_P01 --session-id dryrun_01 --dry-run

python v3_chan/collect_ac_selective_smoothing_feedback.py \
  --participant-id TEST_P01 --session-id dryrun_anchor_01 \
  --mode pilot_with_anchors --dry-run
```

Isaac Python 위치가 다르면 runner 호출 전에 절대경로를 지정한다.

```bash
ISAACSIM_PYTHON=/absolute/path/to/isaac-sim/python.sh \
  bash v3_chan/run_ac_selective_smoothing_feedback.sh --help
```

## HMD practice smoke

Participant 1명의 무작위 practice 1 trial:

```bash
bash v3_chan/run_ac_selective_smoothing_feedback.sh \
  --participant-id TEST_P01 --session-id hmd_smoke_random_01 \
  --practice-only --max-trials 1
```

A practice 1 trial:

```bash
bash v3_chan/run_ac_selective_smoothing_feedback.sh \
  --participant-id TEST_P01 --session-id hmd_smoke_A_01 \
  --practice-only --force-condition A_reactive --max-trials 1
```

C practice 1 trial:

```bash
bash v3_chan/run_ac_selective_smoothing_feedback.sh \
  --participant-id TEST_P01 --session-id hmd_smoke_C_01 \
  --practice-only --force-condition C_smooth --max-trials 1
```

`--force-condition`과 `--max-trials`는 practice-only 장비 점검에만 허용된다.
pilot schedule을 강제로 바꾸는 데 사용할 수 없다.

## Full feasibility pilot

현재 handoff contract는 의도적으로 `production_collection_ready=false`다.
A/C 각각의 HMD practice와 생성 로그 validator가 통과하고, 아래 수동 항목을
확인한 뒤 runtime contract를 새 version으로 명시적으로 승인해야 한다. 그
전에는 다음 명령이 fail-closed로 종료되는 것이 정상이다.

Minimal pilot session:

```bash
HRI_LIVE_HMD_QUALIFIED=1 \
HRI_AC_SOURCE_LINEAGE_RECONCILED=1 \
  bash v3_chan/run_ac_selective_smoothing_feedback.sh \
  --participant-id PILOT_P01 --session-id pilot_p01_s01 \
  --mode minimal_pilot
```

Anchor-repeat pilot session:

```bash
HRI_LIVE_HMD_QUALIFIED=1 \
HRI_AC_SOURCE_LINEAGE_RECONCILED=1 \
  bash v3_chan/run_ac_selective_smoothing_feedback.sh \
  --participant-id PILOT_P01 --session-id pilot_p01_anchor_s01 \
  --mode pilot_with_anchors
```

Pilot/production은 추가로 clean Git HEAD와 검증 가능한 code provenance를
요구한다. `HRI_LIVE_HMD_QUALIFIED=1`은 체크리스트를 실제로 통과한 operator만
설정한다.

## Production 전 HMD 수동 체크리스트

- [ ] room calibration, controller handedness, `XR_SWAP_HANDS`가 실제 손과 일치한다.
- [ ] startup discovery에서 양쪽 controller와 X/Y/A/B의 press/click gesture가 보인다.
- [ ] 버튼을 누른 채 연결하거나 tracking이 끊겨도 phantom marker가 생기지 않는다.
- [ ] trial/query 중 한 controller 또는 face-button channel을 끊으면 즉시 verified hold 후 `.partial`로 종료되고 timeout label로 저장되지 않는다.
- [ ] 중앙 panel의 한글과 모든 선택지가 HMD 안에서 읽히고 A/C identity가 노출되지 않는다.
- [ ] 왼손/오른손 crossing에서 반대 controller의 safety/anomaly marker가 각각 기록된다.
- [ ] no-marker가 acceptable로, uncertain/timeout이 negative로 변환되지 않는다.
- [ ] stable resumption 뒤 neutral-hand cue와 mandatory Q1–Q6이 정확히 한 번 나타난다.
- [ ] `needs_modification`은 reason 없이 확정되지 않고, back과 input timestamp가 기록된다.
- [ ] 설문 중 arm은 safe hold이고 physics/CBF는 ON이며 이후 task가 terminal까지 재개된다.
- [ ] A+B 2초 emergency abort가 동작하고 incomplete output은 `.partial`로만 남는다.
- [ ] 의도된 crossing이 trial당 정확히 1회이며 distal corridor 이탈은 `off_protocol`이다.
- [ ] A 로그는 `joint_nominal/0`, C 로그는 `smooth_intervention/4.0`이고 공통 CBF/Recovery 값은 같다.
- [ ] 실제 햅틱 출력과 pulse가 없고 session metadata의 `haptics_enabled=false`와 일치한다.
- [ ] A practice와 C practice의 최종 HDF5가 각각 fail-closed validator를 통과한다.
- [ ] spectator video를 켰다면 camera pose/FPS가 고정되고 frame index와 simulation time이 맞는다.
- [ ] held-out qualification manifest와 현재 runtime handoff의 source-SHA lineage 차이를 해소했다.
- [ ] 승인된 runtime contract가 `production_collection_ready=true`이고 수집 checkout이 clean하다.

정적 unit test와 dry-run은 실제 controller binding, HMD panel 가시성, 물리적
crossing corridor, emergency stop을 대신 검증하지 않는다. 이 항목들은 A/C
각각의 in-HMD practice 결과와 validator report로 승인한다.
