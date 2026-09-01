# BC + CBF public-server handoff v1

This directory freezes the verified pure-BC plus fixed-CBF runtime that will be
used as the base for the public-server explicit-feedback collector.

The handoff is **runtime-ready but not production-collection-ready**.  The
existing evaluator can exercise BC+CBF in live VR, but it does not yet provide
the explicit-feedback schema, complete nominal/filtered/applied joint-action
vectors, live-tracking dropout abort, or protocol-specific collection
validator.  Those locks are recorded in `runtime_config.json` and must be
cleared before collecting study data.

## Frozen policy and controller

- Checkpoint: `v3_chan/policies/bc_pick_place_v1_100eps.pt`
- Checkpoint SHA-256:
  `fc4801dfdd7ae4e3edf8760277a1d87fd540c12bf27aeb0a06a3eefdbd5ba8c3`
- Pure direct BC: 84-D robot/task observation to 5-D task action
- Human observations masked from BC
- Pseudo-ErrP disabled
- Strict event-driven single-pick semantics v3 with required release and 5 cm
  XY placement tolerance
- CBF: 5 cm safe gap, 13 cm activation gap, gamma 8/s, 0.15 s prediction
  horizon, 8 cm maximum prediction buffer, and 2 rad/s joint-speed cap

The CBF protects tracked hands against the configured distal Panda links.  It
does not establish a whole-body or real-robot safety claim.

## Verify after checkout

From the repository root:

```bash
python v3_chan/runtime_contracts/bc_cbf_explicit_feedback_v1/validate_runtime_contract.py
```

In the Isaac Python environment, also validate checkpoint metadata:

```bash
python v3_chan/runtime_contracts/bc_cbf_explicit_feedback_v1/validate_runtime_contract.py \
  --check-checkpoint-metadata
```

Run the clean-tree unit contract:

```bash
pytest -q -o addopts='' \
  tests/test_v3_strict_task_semantics.py \
  tests/test_v3_run_physical_safety_benchmark.py \
  tests/test_v3_physical_safety_controllers.py \
  tests/test_v3_robot_environment_safety.py
```

Expected result: `69 passed`.

## No-human runtime smoke

Use a new output path and do not reuse study outputs:

```bash
./launch_isaac.sh v3_chan/evaluate_rollout_policy.py \
  --checkpoint v3_chan/policies/bc_pick_place_v1_100eps.pt \
  --episodes 1 \
  --max-steps 4500 \
  --task-reward-version reward_v4_post_release_stability_hri_errp \
  --mask-human-obs-for-policy \
  --no-pseudo-errp \
  --fixed-orientation \
  --gripper-mode event \
  --require-release-for-success \
  --strict-task-semantics \
  --strict-place-xy-tolerance-m 0.05 \
  --physical-safety-controller cbf \
  --cbf-safe-gap-m 0.05 \
  --cbf-activation-gap-m 0.13 \
  --cbf-gamma-per-s 8.0 \
  --cbf-prediction-horizon-s 0.15 \
  --cbf-max-prediction-buffer-m 0.08 \
  --cbf-max-joint-speed-rad-s 2.0 \
  --output-json /tmp/bc_cbf_no_human_smoke.json \
  --output-csv /tmp/bc_cbf_no_human_smoke.csv \
  --output-step-csv /tmp/bc_cbf_no_human_smoke_steps.csv
```

With no tracked hand, the CBF must remain inactive with no fallback or solver
failure and finite diagnostics.  This smoke is not a human-safety evaluation.

## Live-VR runtime smoke

After XR tracking calibration, add `--live-vr`.  A deliberate controlled hand
crossing should produce valid tracked-hand geometry, constraints, and finite
CBF intervention diagnostics without fallback.  Do not require task success in
this controller-wiring smoke.

Do not start production collection from `evaluate_rollout_policy.py` directly.
Build and validate the dedicated explicit-feedback collector described by the
collection locks in `runtime_config.json` first.
