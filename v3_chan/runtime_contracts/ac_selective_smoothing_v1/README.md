# A/C selective-smoothing public-server handoff

This handoff freezes the runtime needed to build the VR explicit-feedback
collector. It is runtime-ready, but it is deliberately not marked as
production-collection-ready until the dedicated collector and its validator
have passed A and C practice trials.

## Frozen response conditions

- `A_reactive`: `objective_mode=joint_nominal`, effective `lambda_s=0`.
- `C_smooth`: `objective_mode=smooth_intervention`, `lambda_s=4.0`.

Both conditions use the same frozen BC checkpoint, CBF safety constraints,
protected geometry, solver/fail-closed behavior, strict task semantics, and
State-aware Recovery v3. Participant-facing collection must hide the
condition name and smoothing weight. Feedback must not change the current or
future trial assignment during data collection.

The current binary study selects between A and C; it does not continuously
regress a smoothing weight. Continuous `lambda_s` prediction is a future
extension.

## Verify immediately after checkout

```bash
python v3_chan/runtime_contracts/ac_selective_smoothing_v1/validate_runtime_contract.py
```

With the Isaac Python environment available:

```bash
./launch_isaac.sh \
  v3_chan/runtime_contracts/ac_selective_smoothing_v1/validate_runtime_contract.py \
  --check-checkpoint-metadata
```

The validator should report `valid: true`, `runtime_handoff_ready: true`, and
`production_collection_ready: false`.

## Collection boundary

Reuse the existing A/C implementation; do not rewrite or tune it in the
collector. The collector may assign only the two conditions above. It must
keep haptics off, BC frozen, pseudo-ErrP off, Recovery v3 frozen, and all CBF
safety parameters identical between A and C.

The 5 cm value is a configured safety-gap parameter, not an empirically
maintained 5 cm guarantee. The qualification report records a minimum logged
post-step gap below 5 cm and explains the discrete-time/hand-velocity lag.

See `docs/cbf_response_family_abc_validation_20260903.md` for the full A/C
qualification and claim boundaries.
