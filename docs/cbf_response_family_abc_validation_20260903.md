# CBF Safety-Response Family A/B/C 구현·검증 보고서

작성일: 2026-09-03  
대상 시스템: **Frozen BC + Fixed CBF + Fixed State-aware Recovery**

## 요약 및 최종 판정

최종 판정은 **`RESPONSE_VARIATION_NO_GO`** 이다.

- A는 기존 clean36 결과를 그대로 재현했다: task success `34/36`, human-hand collision `0/36`, intervention episode success `23/25`, no-intervention success `11/11`.
- C는 held-out에서 A 대비 CBF-active EE jerk를 `43.7%`, joint jerk를 `35.5%`, correction total variation을 `27.3%` 줄였고 task success `17/17`과 human-hand collision `0/17`을 유지했다.
- 그러나 B는 intended task-preservation family로 기능하지 않았다. Held-out exact response pairs에서 A 대비 paired path가 `47.0%`, progress-deficit AUC가 `96.0%`, recovery duration이 `9.6%`, CBF EE jerk가 `49.1%` 증가했다. 또한 task success `16/17`, object drop `1`, recovery internal timeout `1`이 발생했다.
- 따라서 smoothness 축 C는 성립했지만 task-preservation 축 B는 held-out에서 성립하지 않았다. 현재 A/B/C 세트를 participant에게 제시하지 않는다.

이 판정은 `SAFETY_AUDIT_NO_GO`가 아니다. A/B/C 모두 평가된 held-out에서 human collision `0`, logged `<2 cm` episode `0`, self-collision `0`이었고 3.491 cm discrepancy도 step-level evidence로 설명됐다. 문제는 **B의 response design/generalization**이다.

## 실험 계약과 데이터 분리

- Development: clean19, B/C 소규모 sweep과 representative selection에만 사용.
- Frozen selection: `B_task_consistent_eps_0p20`, `C_smooth_intervention_lambda_4p00`.
- Held-out: clean17, selection artifact SHA를 검증한 뒤 열었으며 결과를 보고 재선택하지 않았다.
- BC checkpoint SHA-256: `d0fe9f9dc48a6049b77c8e1eab6de0905197207bb57ff858e0d7d332172b5230`.
- Development run manifest SHA-256: `ee06af821ec69e280e86ca549245cfa1fb6cf111263ea2d2da6437bca9f993dc`.
- Frozen selection SHA-256: `a521ffa4c65cb125846d26550c70725672b9b0527472dff93fc79a548e8990e9`.
- Held-out manifest가 위 frozen-selection SHA를 직접 pin한다.
- A/B/C의 실제 첫 applied response 직전 state, response step, prefix digest가 일치한 held-out response encounter는 `11/11`이다. intervention이 없는 6개도 initial state, layout, replay, seed 계약은 같지만 response-branch exactness의 대상은 아니다.
- BC-only는 동일 source configuration의 paired counterfactual이다. CBF response가 없으므로 A/B/C와 같은 response-branch identity를 주장하지 않는다.

공유 모듈에는 objective switch와 진단 logging을 추가했지만, A의 `joint_nominal` solve 경로와 safety/recovery 설정은 튜닝하지 않았다. 기존 frozen A artifacts와 새 A의 22개 core field를 `27,930` step-row에서 비교한 결과 mismatch는 `0`이었다.

관련 계약:

- [Development run manifest](../v3_chan/eval_results/cbf_response_family_v1/dev/run_manifest.json)
- [Frozen selection](../v3_chan/eval_results/cbf_response_family_v1/dev/frozen_selection.json)
- [Held-out run manifest](../v3_chan/eval_results/cbf_response_family_v1/heldout/run_manifest.json)
- [Pairing verification](../v3_chan/eval_results/cbf_response_family_v1/heldout_tradeoff_analysis/pairing_verification.csv)

## A. Current baseline audit

### A.1 CBF 최적화 문제

최적화 변수는 Panda arm의 joint velocity `qdot in R^7`이다. RMPFlow가 velocity target을 주면 그대로 `qdot_nominal`로 사용하고, position target을 주면 현재 joint position `q`에 대해 다음처럼 변환한다.

```text
qdot_nominal = (q_target - q) / dt
```

A의 objective는 task-space term이 없는 **joint-space nominal-command preservation**이다.

```text
min_qdot  1/2 ||qdot - qdot_nominal||^2
```

각 유효 hand와 활성 distal-link pair에 대한 constraint는 다음과 같다.

```text
n^T J_point(q) qdot >= n^T v_hand - gamma h

h = d_surface - d_effective
d_effective = d_safe + min(b_max, T_pred * max(0, v_closing))
```

고정 설정:

| 항목 | 값 |
|---|---:|
| configured safe-gap parameter `d_safe` | 0.05 m |
| activation gap | 0.13 m |
| `gamma` | 8.0 /s |
| prediction horizon `T_pred` | 0.15 s |
| maximum prediction buffer `b_max` | 0.08 m |
| joint speed bound | asset bound와 2.0 rad/s 중 작은 값 |
| control/CBF period | 1/60 s |
| protected geometry | `panda_link6`, `panda_link7`, flange-equivalent `panda_link8/panda_hand`, left/right finger |

`n`은 hand center에서 선택된 robot surface point 방향, `J_point`는 해당 surface point의 translational Jacobian이다. 왼손/오른손 각각 최대 하나의 closest active constraint를 만든다.

### A.2 Slack과 solver

- Applied safety problem에는 slack penalty가 없다.
- 원 문제 infeasibility가 의심될 때 공통 non-negative slack을 계산하지만, 이는 **진단 evidence**일 뿐 relaxed command를 적용하기 위한 학습/목적함수 항이 아니다.
- 기본 `stop_on_infeasible=True`이므로 원 feasible set을 인증하지 못하면 zero arm command로 fail closed한다.
- A의 주 solver는 box와 half-space에 대한 Dykstra Euclidean projection이다.
- 필요할 때 SciPy HiGHS로 feasibility를 확인하고 SLSQP로 closest feasible velocity를 구한다.
- Held-out A/B/C에서 solver infeasibility와 objective-solver fallback은 모두 0이었다.
- 별도로 A/B/C 각각 2 episode에서 총 2 step의 `fallback_stop_invalid_active_hand`가 있었다. 이는 solver 실패가 아니라 hand-velocity 첫 sample이 유효하지 않을 때 적용한 보수적 zero-command measurement fail-closed이며 결과에서 숨기지 않는다.

### A.3 전체 command provenance

| Stage | 표현과 단위 | 차원 | 순서/변환 |
|---|---|---:|---|
| Frozen BC | normalized `[dx, dy, dz, dyaw, gripper]` | 5 | 이 checkpoint의 `action_v1_controller_target_delta`: 현재 EE 기준 최대 0.75 m controller-target offset, yaw 0.15 rad/step, 이후 workspace clip |
| Fixed Recovery | Cartesian recovery target + gripper state | 3 + state | active일 때 BC target 대신 사용; A/B/C 공통 |
| RMPFlow | articulation position/velocity target | arm 7 | Cartesian target을 nominal joint command로 변환 |
| CBF A/B/C | `qdot_nominal -> qdot_safe` | arm 7, rad/s | 동일 safety half-spaces와 speed box 안에서 objective만 변경 |
| Position reconstruction | `q_safe = q + qdot_safe*dt` | arm 7, rad | joint-position limit clip; velocity target도 함께 기록 |
| Gripper merge | event/policy open-close action | fingers | gripper-only tick이면 arm target은 의도적으로 commit하지 않음 |
| Isaac physics | articulation drive/physics state | robot | command 적용 후 1 physics step; 실제 `q`, `qdot`을 post-step 기록 |

코드상 CBF 뒤에는 acceleration limiter, low-pass filter, slew-rate limiter, interpolation 또는 external action smoother가 없다. 존재하는 것은 CBF 내부 joint-speed box, joint-position clip, 그리고 physics articulation drive의 tracking dynamics이다. 따라서 C의 intervention-smoothness objective는 기존 post-CBF smoother와 중복되지 않는다.

## B. A/B/C implementation

### B.1 공통 불변 요소

세 family 모두 다음을 공유한다.

- 동일 Frozen BC checkpoint와 observation/action normalization.
- 동일 hand replay, initial robot/cube/target state, physics seed/configuration.
- 동일 protected geometry와 CBF half-space constraints.
- 동일 `safe gap=0.05`, `activation=0.13`, `gamma=8`, `prediction=0.15`, `max buffer=0.08`, `max joint speed=2.0`.
- 동일 strict task semantics와 **Fixed State-aware Recovery v3**.
- 동일 fail-closed policy.

Runtime control-path 변경은 다음 4개 파일에 제한했다. 분석·렌더링·forensic 도구는 아래 별도 산출물 코드로 추가했다.

- `v3_chan/physical_safety_controllers.py`: objective mode A/B/C와 solver diagnostics.
- `v3_chan/rl/pick_place_env.py`: exact applied-command provenance와 C correction-memory commit semantics.
- `v3_chan/evaluate_rollout_policy.py`: step fingerprint, geometry/contact, command/solver/task diagnostics logging.
- `v3_chan/run_cbf_response_family_experiment.py`: split/SHA fail-closed experiment runner.

### B.2 A — Reactive baseline

```text
min 1/2 ||qdot - qdot_nominal||^2
subject to unchanged CBF constraints and joint-speed bounds
```

Objective mode는 `joint_nominal`이다. 기존 A의 mathematical control path를 그대로 사용했다.

### B.3 B — Task-consistent

BC가 실제로 제어하는 arm semantics만 사용해 world XYZ와 world yaw를 보존한다. Gripper는 별도 actuator이므로 objective에서 제외하며 임의의 roll/pitch 보존 항을 추가하지 않았다.

```text
J_task = [J_xyz; l_yaw * J_world_yaw]
v_task_nominal = J_task qdot_nominal

min 1/2 * w_task * ||J_task qdot - v_task_nominal||^2
  + 1/2 * epsilon * ||qdot - qdot_nominal||^2
```

- `w_task=1.0`
- yaw length scale `l_yaw=0.10 m/rad`
- dev sweep `epsilon in {0.01, 0.05, 0.20}`
- selected `epsilon=0.20`
- A의 feasible projection을 seed로 사용한 SLSQP quadratic solve; 인증 실패 시 A projection으로 fail-safe fallback

전체 objective의 공통 양의 배율은 argmin을 바꾸지 않으므로 `w_task`를 1로 고정하고 실제 trade-off를 정하는 비율 `epsilon / w_task`만 sweep했다.

현재 Isaac articulation에서 RMPFlow의 configured gripper target frame에 가장 가까운 available Jacobian인 `panda_hand`를 사용했다. 이 frame 근사와 instantaneous 4-D velocity objective가 실제 multi-phase task progress를 충분히 대표하지 못했을 가능성은 B held-out failure의 중요한 설계 한계다.

### B.4 C — Smooth intervention

`r_t = qdot_safe,t - qdot_nominal,t`로 두고 전체 nominal motion이 아니라 CBF correction만 smooth하게 만든다.

```text
min 1/2 ||qdot - qdot_nominal||^2
  + lambda_s/2 ||(qdot - qdot_nominal) - r_(t-1)||^2
```

이는 같은 feasible set에서 다음 reference를 Euclidean projection하는 convex equivalent다.

```text
qdot_reference = qdot_nominal + lambda_s/(1 + lambda_s) * r_(t-1)
```

- dev sweep `lambda_s in {0.25, 1.0, 4.0}`
- selected `lambda_s=4.0`
- external low-pass filter 없음
- active constraint가 끝난 뒤에는 correction이 objective 내부에서 감쇠하는 smooth tail을 허용
- correction memory는 실제 arm command가 commit된 tick에만 갱신하고 reset/intentional human absence에서 지움

## C. Qualification

### C.1 Development sweep와 freeze

아래 수치는 response가 실제 발생하고 exact-paired인 dev encounter 14개의 평균이며, success는 전체 clean19 기준이다.

| Condition | Success | ΔPath m | Progress deficit AUC s | Recovery s | CBF EE jerk RMS | Joint jerk RMS | Correction TV |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 17/19 | 0.2680 | 0.9810 | 3.694 | 394.7 | 686.6 | 10.35 |
| B eps=.01 | 16/19 | 0.2237 | 0.9287 | 3.285 | 207.1 | 885.6 | 8.71 |
| B eps=.05 | 16/19 | 0.2130 | 1.0215 | 3.445 | 201.9 | 518.8 | 4.89 |
| **B eps=.20** | **17/19** | **0.2252** | **0.9685** | **3.273** | 291.3 | 619.5 | 6.57 |
| C lambda=.25 | 17/19 | 0.2585 | 0.9600 | 3.525 | 330.6 | 643.7 | 8.72 |
| C lambda=1 | 18/19 | 0.2499 | 0.8965 | 3.523 | 241.9 | 513.3 | 5.71 |
| **C lambda=4** | **18/19** | 0.2608 | 0.9650 | 3.576 | **176.3** | **373.7** | **3.33** |

B eps=.20은 A와 동일한 `17/19` success/failure identity를 유지하면서 path와 recovery를 줄인 dev task-region 대표로 선택했다. 더 작은 epsilon은 `16/19`로 떨어졌다. C lambda=4는 qualified 후보 중 가장 강한 smoothness-region 대표로 선택했다. 이 선택을 SHA-bound artifact로 freeze한 뒤에만 held-out을 열었다.

### C.2 Held-out qualification

Near-miss는 **human collision이 아니면서 `0 < signed surface gap <= 0.02 m`**인 상태다. 별도의 logged `<2 cm` metric은 signed post-step gap `<0.02 m`인 모든 step을 세므로 collision/penetration도 포함한다. `below 5 cm`는 constraint-violation 명칭이 아니라 `logged_surface_gap_below_configured_margin`이다.

| Condition | Strict success | Grasp | Release | Human collision | Min logged gap | Logged `<2 cm` ep. | Near-miss ep. | Below 5 cm ep. | Mean episode min TTC | Static contact ep./steps | Self collision | Drop | Global / recovery timeout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BC-only reference | 16/17 | 17/17 | 17/17 | 7/17 | -3.500 cm | 9 | 9 | 10 | 0.222 s | 17/17 / 4331 | 0 | 0 | 0 / 0 |
| **A** | **17/17** | 17/17 | 17/17 | **0/17** | 3.491 cm | **0** | **0** | 7 | 0.397 s | 17/17 / 4277 | **0** | 0 | 0 / 0 |
| **B eps=.20** | **16/17** | 17/17 | 17/17 | **0/17** | 3.430 cm | **0** | **0** | 7 | 0.373 s | 17/17 / 4679 | **0** | **1** | 0 / **1** |
| **C lambda=4** | **17/17** | 17/17 | 17/17 | **0/17** | 3.674 cm | **0** | **0** | 6 | 0.382 s | 17/17 / 4274 | **0** | 0 | 0 / 0 |

Static contact의 absolute incidence가 17/17인 이유는 모든 조건에 존재하는 `gripper_finger || table` task contact 때문이다. 새 semantic static/self contact class나 새 collision episode는 없었다. 다만 B는 A보다 static-contact duration이 `402` step 길었으므로 광범위한 “collision-free”라고 표현하지 않는다. 정확한 표현은 **human-hand collision 0/17, self-collision 0/17, task-existing finger/table contact present**이다.

B 실패는 held-out episode 1, seed 12, encounter `e4dec154890c9930`이다.

- terminal reason: `state_aware_recovery_timeout`
- 1,148 steps
- grasp/loss 반복 7회와 object drop
- effective recovery 600 steps, CBF-blocked 99 steps
- final cube-goal distance 26.7 cm

전체 qualification table과 episode-level 근거:

- [Qualification summary](../v3_chan/eval_results/cbf_response_family_v1/heldout_tradeoff_analysis/qualification_summary.csv)
- [Per-episode paired trade-offs](../v3_chan/eval_results/cbf_response_family_v1/heldout_tradeoff_analysis/per_episode_paired_tradeoffs.csv)

## D. Held-out trade-off table

다음 평균은 A/B/C가 동일한 실제 첫 applied response state/prefix에서 갈라진 exact-verified response encounter `n=11`만 사용한다. Δ 값의 기준은 동일 source configuration의 BC-only counterfactual이다.

| Metric | A | B eps=.20 | C lambda=4 |
|---|---:|---:|---:|
| paired ΔPath | 0.2837 m | 0.4170 m | 0.2847 m |
| paired ΔCompletionTime | 1.173 s | 1.550 s | 1.127 s |
| Integrated Intervention | 0.2913 rad | 0.4788 rad | 0.3127 rad |
| Progress-deficit AUC | 0.7434 s | 1.4575 s | 0.8814 s |
| Recovery duration | 4.402 s | 4.826 s | 4.023 s |
| Retreat distance | 0.0507 m | 0.0378 m | 0.0457 m |
| CBF EE jerk RMS | 224.3 m/s^3 | 334.4 m/s^3 | **126.2 m/s^3** |
| CBF joint jerk RMS | 466.2 rad/s^3 | 2870.9 rad/s^3 | **300.5 rad/s^3** |
| Correction total variation | 3.926 rad/s | 16.519 rad/s | **2.853 rad/s** |

B는 retreat distance 하나만 작고 task-preservation 지표 전반과 motion quality가 나빠졌다. C는 A 대비 path `+0.35%`, integrated intervention `+7.36%`라는 작은 비용으로 completion delay `-3.9%`, recovery `-8.6%`, EE jerk `-43.7%`, joint jerk `-35.5%`, correction TV `-27.3%`를 보였다.

Task progress는 물리적 success를 대체하지 않는 controller-clock proxy다.

```text
progress = min(1, (clip(controller_event, 0, 8) + clip(controller_t, 0, 1)) / 8)
progress-deficit AUC = integral max(0, progress_BC - progress_response) dt
```

paused clock이 physical retreat처럼 해석되지 않도록 cube-goal regression도 별도 보존했다. Retreat는 response onset마다 grasp 전이면 EE-to-cube error, grasp 후면 cube-to-goal error가 onset 대비 증가한 최대값이다.

### D.1 Windowed jerk

아래는 동일 exact response pairs `n=11`의 condition별 평균이며 각 셀은 `RMS / peak`이다. `BC_RESUMED`는 recovery handoff 후 최대 0.5 s이다.

| Condition | Window | EE jerk m/s^3 | Joint jerk rad/s^3 |
|---|---|---:|---:|
| A | PRE_INTERVENTION | 18.2 / 56.0 | 159.0 / 447.1 |
| A | CBF_ACTIVE | 224.3 / 942.2 | 466.2 / 2079.2 |
| A | RECOVERY_ACTIVE | 42.9 / 379.0 | 223.8 / 1644.7 |
| A | BC_RESUMED | 500.7 / 1816.6 | 924.8 / 4448.6 |
| A | WHOLE_TASK | 180.8 / 2282.7 | 386.2 / 4647.0 |
| B | PRE_INTERVENTION | 6.0 / 19.8 | 70.3 / 236.3 |
| B | CBF_ACTIVE | 334.4 / 1313.6 | 2870.9 / 11295.0 |
| B | RECOVERY_ACTIVE | 283.9 / 3685.8 | 2286.0 / 24604.0 |
| B | BC_RESUMED | 483.2 / 1812.1 | 895.1 / 4447.5 |
| B | WHOLE_TASK | 348.3 / 5483.3 | 2017.7 / 27281.4 |
| C | PRE_INTERVENTION | 6.3 / 18.6 | 83.2 / 243.8 |
| C | CBF_ACTIVE | **126.2 / 986.0** | **300.5 / 2554.4** |
| C | RECOVERY_ACTIVE | **28.6 / 242.6** | **139.5 / 1074.3** |
| C | BC_RESUMED | **442.7 / 1633.3** | **819.8 / 4016.2** |
| C | WHOLE_TASK | **173.3 / 2283.0** | **357.0 / 4708.6** |

PRE window 길이는 intervention onset에 따라 달라지므로 condition 간 PRE 평균 자체를 treatment effect로 해석하지 않는다. 전체 원자료는 [windowed_jerk.csv](../v3_chan/eval_results/cbf_response_family_v1/heldout_tradeoff_analysis/windowed_jerk.csv)에 있다.

## E. Pareto analysis

생성된 held-out scatter:

1. [ΔPath vs CBF-window jerk](../v3_chan/eval_results/cbf_response_family_v1/heldout_tradeoff_analysis/pareto_path_vs_jerk.png)
2. [Integrated Intervention vs Recovery Duration](../v3_chan/eval_results/cbf_response_family_v1/heldout_tradeoff_analysis/pareto_intervention_vs_recovery.png)
3. [Retreat Distance vs ΔCompletionTime](../v3_chan/eval_results/cbf_response_family_v1/heldout_tradeoff_analysis/pareto_retreat_vs_completion.png)
4. [ΔProgress vs CBF-window jerk](../v3_chan/eval_results/cbf_response_family_v1/heldout_tradeoff_analysis/pareto_progress_vs_jerk.png)

각 point의 condition, task phase, encounter ID, non-dominated flag는 [per-episode table](../v3_chan/eval_results/cbf_response_family_v1/heldout_tradeoff_analysis/per_episode_paired_tradeoffs.csv)에 저장했다.

Dev에서는 B eps=.20과 C lambda=4가 서로 다른 task/smoothness region을 대표하는 것으로 보여 freeze했다. 그러나 held-out에서 C의 smoothness separation만 재현됐고 B는 여러 axis에서 dominated/high-cost response가 됐다. Held-out을 보고 B epsilon을 다시 선택하지 않는다. 새 B 설계는 새 development cycle과 새로운 untouched evaluation split이 필요하다.

## F. 동일 encounter visualization

동일 world axes, camera, human trajectory, time alignment로 A/B/C를 나란히 표시하고 EE/hands/cube trace, CBF-active 구간, recovery 구간을 overlay했다.

- 가장 큰 차이: [MP4](../v3_chan/eval_results/cbf_response_family_v1/heldout_visualizations/largest_difference/cbf_response_comparison.mp4), [render manifest](../v3_chan/eval_results/cbf_response_family_v1/heldout_visualizations/largest_difference/render_manifest.json)
  - episode 1, seed 12, encounter `e4dec154890c9930`
  - maximum pairwise EE separation 44.98 cm, mean 8.90 cm
  - A/C success, B recovery timeout/drop
- 거의 같은 차이: [MP4](../v3_chan/eval_results/cbf_response_family_v1/heldout_visualizations/almost_same/cbf_response_comparison.mp4), [render manifest](../v3_chan/eval_results/cbf_response_family_v1/heldout_visualizations/almost_same/render_manifest.json)
  - episode 14, seed 25, encounter `446476958c725aa2`
  - maximum pairwise EE separation 2.83 cm, mean 4.89 mm
  - A/B/C 모두 success

이 파일은 logged world-coordinate를 재생한 **3-D engineering trace**이며 Isaac RGB participant stimulus나 human-study evidence가 아니다. 개발 단계에서 trajectory 차이와 failure mode를 확인하기 위한 시각화다.

## G. 3.491 cm forensic audit

### G.1 Exact occurrence

- A held-out episode 1, seed 12, encounter `e4dec154890c9930`
- step 561, simulation time 9.3833338 s
- right hand vs `panda_link6/geometry/panda_link6`
- human query geometry: radius 0.035 m의 non-physical PhysX overlap sphere; 별도 physical hand-collider path는 없음
- CBF pre-step internal/raw surface gap: 0.0357666 m
- logger pre-step gap: 0.0357666 m
- one physics step 뒤 logged post-step gap: **0.0349121 m**

최초 configured-margin crossing은 step 557이다.

- previous post-gap: 0.0533447 m
- crossing post-gap: 0.0452881 m
- 16.667 ms 동안 gap change: -8.0566 mm
- CBF filtered hand speed: 0.1349 m/s
- logged backward-difference hand speed: 0.6689 m/s

최저-gap step에서도 filtered speed는 0.4192 m/s, logged backward-difference speed는 0.8211 m/s였다.

### G.2 배제·확인한 원인

| Category | 판정과 evidence |
|---|---|
| `DISTANCE_DEFINITION_MISMATCH` | 배제. CBF internal raw gap과 logger pre-gap 최대 차이 `2.78e-17 m` |
| `SOLVER_SLACK` | 배제. audited window 최대 slack `0.0 m/s` |
| `POST_CBF_COMMAND_MODIFICATION` | 배제. CBF output과 applied arm qdot 최대 norm difference `0.0 rad/s` |
| `GEOMETRY_APPROXIMATION` | audited pair에서는 배제. 같은 PhysX overlap query/collider가 CBF와 pre-step logger에 사용됨 |
| `ACTUATION_OR_TRACKING_LAG` | crossing/minimum의 주원인으로 배제. actual derivative residual은 각각 `+0.00263`, `+0.00721 m/s`; command tracking difference 최대 `0.0737 rad/s` |
| `DISCRETE_TIME_OVERSHOOT` | **주원인**. sampled derivative constraint를 만족해도 다음 60 Hz sample 전 hand가 빠르게 접근해 post-step gap이 margin을 넘음 |
| `OTHER:DYNAMIC_HAND_VELOCITY_ESTIMATION_LAG` | **공동 주원인**. 빠르게 가속하는 replay hand를 filtered velocity estimate가 과소평가 |

즉 5 cm는 continuous/discrete invariance가 실증된 guarantee가 아니라 **CBF가 설정된 safety-gap parameter**다. 허용 표현은 다음과 같다.

> The CBF was configured with a 5 cm safety-gap parameter. Across the held-out A episodes, no human-hand collision or logged gap below 2 cm was observed; the minimum logged post-step surface gap was 3.491 cm.

현재 empirical 결과를 보고하기 전에 알고리즘을 반드시 바꿀 필요는 없지만, 향후 5 cm invariance를 주장하려면 conservative acceleration/velocity bound와 discrete one-step-forward barrier를 추가하고 재검증해야 한다.

근거:

- [Forensic report](../v3_chan/eval_results/cbf_response_families_v1/gap_forensic_v1/minimum_gap_forensic_report.json)
- [Minimum step ±20 control-step dump](../v3_chan/eval_results/cbf_response_families_v1/gap_forensic_v1/minimum_gap_window.csv)

## H. Final decision

# `RESPONSE_VARIATION_NO_GO`

| Human-pilot gate | 결과 |
|---|---|
| A/B/C empirical human-safety qualification | PASS; 단 measurement fail-closed 2 step/condition 명시 |
| Baseline task competence 유지 | A/C PASS, B FAIL (`16/17`, drop 1, recovery timeout 1) |
| B task-preservation axis 개선 | **FAIL**; path/progress/recovery/jerk가 A보다 악화 |
| C smoothness axis 개선 | PASS; safety 약화 없이 CBF jerk/TV 감소 |
| 2개 이상 useful Pareto axes | FAIL; separation은 있으나 task 축 대표 B가 유효 candidate가 아님 |
| 동일 encounter에서 trajectory 차이 | PASS; 큰 차이와 거의 같은 사례 모두 존재 |
| 3.491 cm discrepancy 설명 | PASS |
| static/self logger | PASS; 기존 finger/table task contact를 별도 공개 |

다음 단계에서 필요한 것은 human feedback collector가 아니라 B objective의 재설계다. 우선 `panda_hand` instantaneous velocity preservation이 실제 RMPFlow target frame과 phase-level progress를 보존하는지 dev에서 검증하고, B가 A 대비 task axis를 개선할 때만 새로운 held-out split으로 재평가한다. 현재 held-out에 맞춰 epsilon을 다시 고르거나 recovery/safety constraint를 condition별로 바꾸면 안 된다.

## 구현 및 재현 명령

주요 구현:

- [CBF objectives](../v3_chan/physical_safety_controllers.py)
- [Environment command provenance](../v3_chan/rl/pick_place_env.py)
- [Evaluator logging](../v3_chan/evaluate_rollout_policy.py)
- [SHA-bound experiment runner](../v3_chan/run_cbf_response_family_experiment.py)
- [Trade-off analyzer](../v3_chan/analyze_cbf_response_tradeoffs.py)
- [Comparison renderer](../v3_chan/render_cbf_response_comparison.py)
- [Gap forensic analyzer](../v3_chan/analyze_cbf_gap_forensics.py)

```bash
# Development sweep
/home/railabchan/isaac-sim-4.5.0/python.sh \
  v3_chan/run_cbf_response_family_experiment.py \
  --stage dev \
  --output-root v3_chan/eval_results/cbf_response_family_v1

# Held-out: development selection을 먼저 수동 검토·freeze한 뒤에만 실행
/home/railabchan/isaac-sim-4.5.0/python.sh \
  v3_chan/run_cbf_response_family_experiment.py \
  --stage heldout \
  --selection-json v3_chan/eval_results/cbf_response_family_v1/dev/frozen_selection.json \
  --output-root v3_chan/eval_results/cbf_response_family_v1

# Held-out trade-off/Pareto regeneration
python v3_chan/analyze_cbf_response_tradeoffs.py \
  --result-root v3_chan/eval_results/cbf_response_family_v1/heldout \
  --output-directory v3_chan/eval_results/cbf_response_family_v1/heldout_tradeoff_analysis \
  --annotate-points

# 3.491 cm forensic regeneration
python v3_chan/analyze_cbf_gap_forensics.py \
  --steps-csv v3_chan/eval_results/cbf_response_families_v1/forensic_a_heldout_ep01/seed_11_steps.csv \
  --run-json v3_chan/eval_results/cbf_response_families_v1/forensic_a_heldout_ep01/seed_11.json \
  --output-directory v3_chan/eval_results/cbf_response_families_v1/gap_forensic_v1 \
  --window-steps 20 \
  --configured-margin-m 0.05
```

최종 검증 상태:

- 관련 control/environment/evaluator/analyzer/renderer regression suite: `177 passed`.
- 변경된 Python modules `py_compile`: PASS.
- source diff whitespace 검사: PASS.
- 현재 runtime source SHA는 held-out run manifest의 pinned SHA와 모두 일치한다.
