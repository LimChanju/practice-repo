from __future__ import annotations

import numpy as np
import pytest

from v3_chan.rl.decision_aligned_action_fusion import (
    ActionFusionDecisionSnapshot,
    DecisionAlignedActionFusionCache,
)
from v3_chan.rl.reentry import (
    ReentryEvidence,
    build_risk_complement_reentry_controller,
    select_shared_projection_hold_candidate,
)
from v3_chan.rl.task_recovery_adaptation import (
    PostReentryActorCreditTracker,
    TaskActionExecutionEvidence,
    TaskActorCreditConfig,
)


def _controller():
    return build_risk_complement_reentry_controller(
        enter_risk_threshold=0.4,
        exit_risk_ratio=0.75,
        readiness_threshold=0.75,
        minimum_hold_decisions=2,
        release_confirmation_decisions=1,
    )


def _evidence(risk: float) -> ReentryEvidence:
    return ReentryEvidence(
        risk_score=risk,
        readiness_score=1.0 - risk,
        physical_clearance=True,
        task_action_supported=True,
    )


def _tracker() -> PostReentryActorCreditTracker:
    return PostReentryActorCreditTracker(
        TaskActorCreditConfig(
            mode="executed_only_post_reentry",
            post_reentry_physics_steps=2,
            post_reentry_actor_weight=3.0,
        )
    )


def test_hard_release_executes_backup_then_credits_next_task_step():
    controller = _controller()
    assert controller.update(_evidence(0.8), decision_tick=True).intervene
    release = controller.update(_evidence(0.1), decision_tick=True)
    assert release.intervene
    assert release.release_after_step

    task = np.zeros(5, dtype=np.float32)
    backup = np.ones(5, dtype=np.float32)
    tracker = _tracker()
    release_credit = tracker.update(
        TaskActionExecutionEvidence(
            sampled_task_action=task,
            executed_action=backup,
            replacement_lambda=1.0,
            intervention_active=True,
            blend_active=False,
            release_edge=release.release_after_step,
            decision_tick=True,
            fresh_task_action=True,
        )
    )
    next_credit = tracker.update(
        TaskActionExecutionEvidence(
            sampled_task_action=task,
            executed_action=task,
            replacement_lambda=0.0,
            intervention_active=False,
            blend_active=False,
            release_edge=False,
            decision_tick=True,
            fresh_task_action=True,
        )
    )
    assert not release_credit.actor_valid
    assert release_credit.reason == "release_edge"
    assert next_credit.actor_valid
    assert next_credit.actor_weight == pytest.approx(3.0)


def test_projection_release_stays_positive_and_forces_fresh_next_tick():
    controller = _controller()
    controller.update(_evidence(0.8), decision_tick=True)
    release = controller.update(_evidence(0.1), decision_tick=True)
    assert release.release_after_step

    task = np.zeros(5, dtype=np.float32)
    backup = np.ones(5, dtype=np.float32)
    protected = select_shared_projection_hold_candidate(
        task,
        backup,
        lambda_grid=(0.0, 0.5, 1.0),
        candidate_risks=(0.1, 0.2, 0.3),
        candidate_allowed=(True, True, True),
        previous_lambda=0.5,
    )
    assert protected.lambda_value == pytest.approx(0.5)
    cache = DecisionAlignedActionFusionCache(action_repeat_steps=6)
    runtime = cache.resolve(
        ActionFusionDecisionSnapshot(
            task_action=task,
            limited_backup_action=backup[:4],
            full_backup_action=backup,
            executed_action=protected.action,
            executed_lambda=protected.lambda_value,
            minimum_allowed_lambda=0.0,
            task_risk=0.1,
            executed_risk=protected.risk,
            fallback_to_full_backup=protected.fallback,
            transition="shared_readiness_release_protected",
            candidate_lambdas=(0.0, 0.5, 1.0),
            candidate_risks=(0.1, 0.2, 0.3),
            candidate_allowed=(True, True, True),
            raw_intervention_request=False,
            risk_probability=0.1,
            action_delta_l2=float(np.linalg.norm(protected.action - task)),
        )
    )
    assert runtime.snapshot.executed_lambda > 0.0
    assert not runtime.release_edge
    cache.force_release_recompute_next_tick()
    assert cache.current_lambda == pytest.approx(0.0)
    assert cache.decision_tick
