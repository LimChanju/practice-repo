from types import SimpleNamespace

import numpy as np

from v3_chan.rl.pick_place_env import IsaacPickPlaceEnv


def _environment(event: int, progress: float) -> IsaacPickPlaceEnv:
    env = object.__new__(IsaacPickPlaceEnv)
    env.phase_event = int(event)
    env.phase_t = float(progress)
    env.phase_hold_steps = 17
    env.config = SimpleNamespace(success_dist=0.06, release_dist=0.07)
    return env


def _observation(*, grasped: bool, target_dist: float) -> dict[str, np.ndarray]:
    return {
        "has_grasped_cube": np.array([float(grasped)], dtype=np.float32),
        "cube_to_place_target": np.array(
            [float(target_dist), 0.0, 0.0], dtype=np.float32
        ),
    }


def test_task_phase_is_unchanged_while_safety_policy_is_active():
    env = _environment(event=2, progress=0.75)
    result = env._control_task_phase(
        _observation(grasped=False, target_dist=0.3),
        advance=False,
        reset_for_reentry=False,
    )

    assert env.phase_event == 2
    assert env.phase_t == 0.75
    assert env.phase_hold_steps == 17
    assert result["paused"] is True
    assert result["reentry"] is False


def test_reentry_resumes_paused_event_when_grasp_is_intact():
    env = _environment(event=5, progress=0.8)
    obs = _observation(grasped=True, target_dist=0.2)
    result = env._control_task_phase(
        obs,
        advance=False,
        reset_for_reentry=True,
    )

    assert env.phase_event == 5
    assert env.phase_t == 0.8
    assert env.phase_hold_steps == 17
    assert result["reason"] == "resume_paused_event"
    assert np.isclose(obs["controller_t"][0], 0.8)
    assert int(np.argmax(obs["controller_event"])) == 5


def test_reentry_rewinds_to_pick_when_cube_was_not_grasped():
    env = _environment(event=6, progress=0.9)
    obs = _observation(grasped=False, target_dist=0.3)
    result = env._control_task_phase(
        obs,
        advance=False,
        reset_for_reentry=True,
    )

    assert env.phase_event == 0
    assert env.phase_t == 0.0
    assert result["reason"] == "rewind_missing_grasp"
    assert int(np.argmax(obs["controller_event"])) == 0


def test_reentry_repositions_before_release_when_safety_detour_is_large():
    env = _environment(event=6, progress=0.9)
    obs = _observation(grasped=True, target_dist=0.3)
    result = env._control_task_phase(
        obs,
        advance=False,
        reset_for_reentry=True,
    )

    assert env.phase_event == 5
    assert result["reason"] == "reposition_before_release"
