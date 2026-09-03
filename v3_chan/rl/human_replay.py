from __future__ import annotations

import os
import importlib.util
import copy
import sys
from dataclasses import asdict, dataclass
from typing import Any, Literal

import h5py
import numpy as np

try:
    from .encounter_manifest import (
        SEVERITY_ORDER,
        extract_episode_source_configuration,
        load_encounter_manifest,
        parse_severity_mix,
        resolve_scenario_source,
        resolve_source_restoration,
    )
except ImportError:
    # Keep direct file imports used by the lightweight unit tests working.
    _manifest_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "encounter_manifest.py",
    )
    _manifest_spec = importlib.util.spec_from_file_location(
        "_v3_chan_encounter_manifest",
        _manifest_path,
    )
    if _manifest_spec is None or _manifest_spec.loader is None:
        raise
    _manifest_module = importlib.util.module_from_spec(_manifest_spec)
    sys.modules[_manifest_spec.name] = _manifest_module
    _manifest_spec.loader.exec_module(_manifest_module)
    SEVERITY_ORDER = _manifest_module.SEVERITY_ORDER
    extract_episode_source_configuration = (
        _manifest_module.extract_episode_source_configuration
    )
    load_encounter_manifest = _manifest_module.load_encounter_manifest
    parse_severity_mix = _manifest_module.parse_severity_mix
    resolve_scenario_source = _manifest_module.resolve_scenario_source
    resolve_source_restoration = _manifest_module.resolve_source_restoration


HumanReplayMode = Literal["step", "loop"]
HumanReplayEpisodePolicy = Literal["cycle", "random"]
EncounterReplayTimebase = Literal["recorded", "step"]

ENCOUNTER_AUGMENTATION_VERSION = (
    "encounter_augmentation_v1_rigid_spatiotemporal_low_frequency"
)

_LATEST_PRE_SUCCESS_TRIGGER_EVENT = {
    "approach_cube": 1,
    "grasp_cube": 3,
    "move_to_target": 5,
    "release_cube": 6,
}


def _scenario_human_motion_start_step(scenario: dict[str, Any]) -> int:
    value = scenario.get("human_motion_start_step", scenario.get("start_step", 0))
    return int(value)


def _scenario_human_motion_end_step(
    scenario: dict[str, Any],
    episode_length: int,
) -> int:
    window = scenario.get("human_motion_window")
    window = window if isinstance(window, dict) else {}
    if window and bool(scenario.get("human_motion_collection_eligible", False)):
        if not bool(window.get("recovery_complete", False)):
            raise ValueError(
                "Collection-eligible v4 scenario has no confirmed human recovery"
            )
        if window.get("end_step_exclusive") is None:
            raise ValueError(
                "Collection-eligible v4 scenario has no human motion endpoint"
            )
    complete = bool(window.get("recovery_complete", False))
    annotated_end = window.get(
        "end_step_exclusive",
        scenario.get("human_motion_end_step"),
    )
    if complete and annotated_end is not None:
        end_step = int(annotated_end)
    else:
        end_step = int(scenario["end_step"])
    start_step = _scenario_human_motion_start_step(scenario)
    if not start_step < end_step <= int(episode_length):
        raise ValueError(
            "Invalid synchronized human motion window: "
            f"[{start_step}, {end_step}) for episode length {episode_length}"
        )
    return end_step


def _validate_recovery_scenario_contract(scenario: dict[str, Any]) -> None:
    window = scenario.get("human_motion_window")
    if not isinstance(window, dict):
        raise ValueError("v4 scenario is missing human_motion_window")
    top_eligible = bool(scenario.get("human_motion_collection_eligible", False))
    nested_eligible = bool(window.get("collection_eligible", False))
    complete = bool(window.get("recovery_complete", False))
    end_step = window.get("end_step_exclusive")
    top_end_step = scenario.get("human_motion_end_step")
    if top_eligible != nested_eligible:
        raise ValueError("v4 human-motion eligibility fields disagree")
    if top_eligible:
        if not complete or end_step is None or top_end_step is None:
            raise ValueError("eligible v4 scenario has an incomplete recovery window")
        if int(end_step) != int(top_end_step):
            raise ValueError("v4 human-motion endpoint fields disagree")
        start_step = _scenario_human_motion_start_step(scenario)
        if int(end_step) <= start_step:
            raise ValueError("v4 human-motion window is empty or reversed")
    elif complete or end_step is not None or top_end_step is not None:
        raise ValueError("ineligible v4 scenario claims a completed recovery window")


@dataclass(frozen=True)
class HumanReplayInfo:
    path: str
    episode_count: int
    mode: str
    episode_policy: str


class HumanTrajectoryReplay:
    """Replay recorded human head/hand trajectories as an Isaac human_state_fn.

    The preferred source is the recorder's `/episodes/<ep>/human` group. Older
    trajectory files can still be replayed from `/obs/human_*` fields.
    """

    def __init__(
        self,
        path: str,
        *,
        mode: HumanReplayMode = "step",
        episode_policy: HumanReplayEpisodePolicy = "cycle",
        seed: int = 0,
    ) -> None:
        self.path = os.path.abspath(path)
        self.mode = mode
        self.episode_policy = episode_policy
        self.rng = np.random.default_rng(seed)
        self._file = h5py.File(self.path, "r")
        if "episodes" not in self._file:
            raise KeyError(f"Human replay file has no 'episodes' group: {self.path}")
        self._episodes = self._file["episodes"]
        self._episode_names = tuple(sorted(self._episodes.keys()))
        if not self._episode_names:
            raise ValueError(f"Human replay file has no episodes: {self.path}")
        self._episode_name = self._episode_names[0]
        self._episode = self._load_episode(self._episode_name)
        self._cursor = 0
        self._last_state: dict[str, Any] = {}

    @property
    def info(self) -> HumanReplayInfo:
        return HumanReplayInfo(
            path=self.path,
            episode_count=len(self._episode_names),
            mode=self.mode,
            episode_policy=self.episode_policy,
        )

    @property
    def episode_name(self) -> str:
        return self._episode_name

    def reset(self, episode_index: int = 0, *, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.episode_policy == "random":
            idx = int(self.rng.integers(0, len(self._episode_names)))
        elif self.episode_policy == "cycle":
            idx = int(episode_index) % len(self._episode_names)
        else:
            raise ValueError(f"Unknown human replay episode policy: {self.episode_policy}")
        self._episode_name = self._episode_names[idx]
        self._episode = self._load_episode(self._episode_name)
        self._cursor = 0
        self._last_state = {}
        return self.peek()

    def peek(self) -> dict[str, Any]:
        return self._state_at(self._cursor)

    def __call__(self) -> dict[str, Any]:
        state = self._state_at(self._cursor)
        self._last_state = state
        self._cursor += 1
        return state

    def close(self) -> None:
        self._file.close()

    def capture_state(self) -> dict[str, Any]:
        return {
            "schema_version": "human_trajectory_replay_state_v1",
            "episode_name": self._episode_name,
            "cursor": int(self._cursor),
            "last_state": _copy_replay_value(self._last_state),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != "human_trajectory_replay_state_v1":
            raise ValueError("Unsupported HumanTrajectoryReplay state schema")
        episode_name = str(state.get("episode_name", ""))
        if episode_name not in self._episode_names:
            raise ValueError(f"Unknown replay episode in state: {episode_name}")
        if episode_name != self._episode_name:
            self._episode_name = episode_name
            self._episode = self._load_episode(episode_name)
        self._cursor = int(state["cursor"])
        self._last_state = _copy_replay_value(state.get("last_state", {}))
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])

    def _state_at(self, idx: int) -> dict[str, Any]:
        length = int(self._episode["length"])
        if length <= 0:
            return {}
        if self.mode == "loop":
            sample_idx = int(idx) % length
        elif self.mode == "step":
            sample_idx = min(max(int(idx), 0), length - 1)
        else:
            raise ValueError(f"Unknown human replay mode: {self.mode}")

        valid_mask = self._episode["valid_mask"][sample_idx]
        state: dict[str, Any] = {
            "human_left_hand_vel": self._episode["left_hand_vel"][sample_idx],
            "human_right_hand_vel": self._episode["right_hand_vel"][sample_idx],
            "human_valid_mask": valid_mask,
        }
        if valid_mask[0] > 0.5:
            state["human_head_pos"] = self._episode["head_pos"][sample_idx]
        if valid_mask[1] > 0.5:
            state["human_left_hand_pos"] = self._episode["left_hand_pos"][sample_idx]
        if valid_mask[2] > 0.5:
            state["human_right_hand_pos"] = self._episode["right_hand_pos"][sample_idx]

        if self._episode["human_robot_collision"] is not None:
            collision = bool(self._episode["human_robot_collision"][sample_idx] > 0.5)
            state["recorded_human_robot_collision"] = collision
        if self._episode["near_human"] is not None:
            near_human = bool(self._episode["near_human"][sample_idx] > 0.5)
            state["recorded_near_human"] = near_human
        if self._episode["min_hand_gripper_dist_m"] is not None:
            state["recorded_min_hand_gripper_dist_m"] = float(
                self._episode["min_hand_gripper_dist_m"][sample_idx]
            )
        if self._episode["gripper_camera_occluded"] is not None:
            state["gripper_camera_occluded"] = float(
                np.clip(self._episode["gripper_camera_occluded"][sample_idx], 0.0, 1.0)
            )
        return state

    def _load_episode(self, episode_name: str) -> dict[str, Any]:
        return _load_episode_group(self._episodes[episode_name])


@dataclass(frozen=True)
class HumanEncounterReplayInfo:
    path: str
    scenario_count: int
    episode_policy: str
    anchor_mode: str
    phase_match: bool
    event_match: bool
    severity_mix: dict[str, float]
    playback_timebase: str
    playback_speed: float
    augmentation: dict[str, Any]


@dataclass(frozen=True)
class EncounterAugmentationConfig:
    """Train-only, body-coherent perturbations for encounter templates."""

    enabled: bool = False
    identity_probability: float = 0.25
    translation_xy_max_m: float = 0.03
    translation_z_max_m: float = 0.01
    yaw_max_deg: float = 10.0
    playback_speed_min: float = 0.8
    playback_speed_max: float = 1.2
    smooth_offset_max_m: float = 0.005
    smooth_cycles_min: float = 0.5
    smooth_cycles_max: float = 1.5
    post_window_mode: Literal["inactive", "hold_last_pose"] = "inactive"

    def validated(self) -> "EncounterAugmentationConfig":
        if not 0.0 <= float(self.identity_probability) <= 1.0:
            raise ValueError("identity_probability must be in [0, 1]")
        for name in (
            "translation_xy_max_m",
            "translation_z_max_m",
            "yaw_max_deg",
            "smooth_offset_max_m",
            "smooth_cycles_min",
            "smooth_cycles_max",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        speed_min = float(self.playback_speed_min)
        speed_max = float(self.playback_speed_max)
        if (
            not np.isfinite(speed_min)
            or not np.isfinite(speed_max)
            or speed_min <= 0.0
            or speed_max < speed_min
        ):
            raise ValueError(
                "playback speed bounds must be finite, positive, and ordered"
            )
        if float(self.smooth_cycles_max) < float(self.smooth_cycles_min):
            raise ValueError("smooth cycle bounds must be ordered")
        if self.post_window_mode not in ("inactive", "hold_last_pose"):
            raise ValueError(f"Unknown post_window_mode: {self.post_window_mode}")
        return self

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": ENCOUNTER_AUGMENTATION_VERSION,
            **asdict(self),
            "scope": "training_replay_only",
            "held_out_policy": "identity_only",
            "label_contract": (
                "source labels are metadata only; current surface gap, contact, "
                "TTC, closing speed, and agency must be recomputed"
            ),
        }


class HumanEncounterReplay:
    """Place one recorded encounter template into each full task rollout.

    A scenario remains inactive until the task reaches its recorded phase/event.
    Its human trajectory is then replayed once. A fixed translation can align the
    recorded end-effector anchor to the current end-effector pose. Safety labels
    from the source remain metadata; the environment recomputes current geometry.
    """

    def __init__(
        self,
        manifest_path: str,
        *,
        episode_policy: HumanReplayEpisodePolicy = "random",
        severity_mix: str | dict[str, float] | None = None,
        anchor_mode: Literal["ee", "world"] = "ee",
        phase_match: bool = True,
        event_match: bool = True,
        playback_timebase: EncounterReplayTimebase = "recorded",
        playback_speed: float = 1.0,
        augmentation: EncounterAugmentationConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.path = os.path.abspath(os.path.expanduser(manifest_path))
        self.manifest = load_encounter_manifest(self.path)
        self.episode_policy = episode_policy
        self.severity_mix = parse_severity_mix(severity_mix)
        self.anchor_mode = anchor_mode
        self.phase_match = bool(phase_match)
        self.event_match = bool(event_match)
        if playback_timebase not in ("recorded", "step"):
            raise ValueError(
                f"Unknown encounter playback timebase: {playback_timebase}"
            )
        if not np.isfinite(playback_speed) or float(playback_speed) <= 0.0:
            raise ValueError("encounter playback_speed must be finite and positive")
        self.playback_timebase = playback_timebase
        self.playback_speed = float(playback_speed)
        self.augmentation = (
            EncounterAugmentationConfig()
            if augmentation is None
            else augmentation
        ).validated()
        self.rng = np.random.default_rng(seed)
        self._augmentation_rng = np.random.default_rng(int(seed) + 7_919)
        manifest_scenarios = tuple(self.manifest["scenarios"])
        if self.manifest.get("schema_version") == "hri_encounter_manifest_v4":
            for scenario in manifest_scenarios:
                _validate_recovery_scenario_contract(scenario)
            manifest_scenarios = tuple(
                scenario
                for scenario in manifest_scenarios
                if bool(scenario.get("human_motion_collection_eligible", False))
            )
            if not manifest_scenarios:
                raise ValueError(
                    "Recovery-aware encounter manifest contains no collection-eligible "
                    "human motion windows."
                )
        self._scenarios = manifest_scenarios
        self._by_severity = {
            severity: tuple(
                scenario
                for scenario in self._scenarios
                if scenario.get("target_severity") == severity
            )
            for severity in SEVERITY_ORDER
        }
        self._files: dict[str, h5py.File] = {}
        self._scenario: dict[str, Any] = {}
        self._episode: dict[str, Any] = {}
        self._cursor = 0
        self._started = False
        self._finished = False
        self._anchor_offset = np.zeros(3, dtype=np.float32)
        self._runtime_context: dict[str, Any] = {}
        self._runtime_start_time_s: float | None = None
        self._source_start_time_s: float | None = None
        self._last_source_time_s: float | None = None
        self._last_active_state: dict[str, Any] = {}
        self._last_active_source_step = -1
        self._augmentation_parameters = _identity_augmentation_parameters()
        self._augmentation_pivot = np.zeros(3, dtype=np.float32)
        self.reset(0, seed=seed)

    @property
    def info(self) -> HumanEncounterReplayInfo:
        return HumanEncounterReplayInfo(
            path=self.path,
            scenario_count=len(self._scenarios),
            episode_policy=self.episode_policy,
            anchor_mode=self.anchor_mode,
            phase_match=self.phase_match,
            event_match=self.event_match,
            severity_mix=dict(self.severity_mix),
            playback_timebase=self.playback_timebase,
            playback_speed=self.playback_speed,
            augmentation=self.augmentation.metadata(),
        )

    @property
    def episode_name(self) -> str:
        return str(self._scenario.get("id", ""))

    @property
    def current_scenario(self) -> dict[str, Any]:
        return dict(self._scenario)

    @property
    def current_augmentation(self) -> dict[str, Any]:
        return _jsonable_replay_value(self._augmentation_parameters)

    def source_restoration(
        self,
        *,
        screening_seed: int,
        allow_legacy_fallback: bool = False,
    ) -> dict[str, Any]:
        return resolve_source_restoration(
            self._scenario,
            screening_seed=screening_seed,
            allow_legacy_fallback=allow_legacy_fallback,
        )

    def reset(self, episode_index: int = 0, *, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._augmentation_rng = np.random.default_rng(int(seed) + 7_919)
        self._scenario = self._select_scenario(episode_index)
        source_path = resolve_scenario_source(self._scenario, self.path)
        h5_file = self._files.get(source_path)
        if h5_file is None:
            h5_file = h5py.File(source_path, "r")
            self._files[source_path] = h5_file
        episode_name = str(self._scenario["episode_name"])
        episode_group = h5_file["episodes"][episode_name]
        if not isinstance(self._scenario.get("source_configuration"), dict):
            self._scenario["source_configuration"] = (
                extract_episode_source_configuration(
                    episode_group,
                    active_cube_index=int(self._scenario.get("cube_index", -1)),
                    source_anchor_step=int(
                        self._scenario.get(
                            "source_anchor_step",
                            self._scenario.get("start_step", 0),
                        )
                    ),
                )
            )
        self._episode = _load_episode_group(episode_group)
        self._cursor = _scenario_human_motion_start_step(self._scenario)
        self._started = False
        self._finished = False
        self._anchor_offset = np.zeros(3, dtype=np.float32)
        self._runtime_context = {}
        self._runtime_start_time_s = None
        self._source_start_time_s = None
        self._last_source_time_s = None
        self._last_active_state = {}
        self._last_active_source_step = -1
        self._augmentation_parameters = self._sample_augmentation()
        self._augmentation_pivot = np.zeros(3, dtype=np.float32)
        return self.peek()

    def set_runtime_context(
        self,
        *,
        step: int,
        task_phase: str,
        controller_event: int,
        controller_t: int,
        ee_pos: np.ndarray | None,
        playback_time_s: float | None = None,
    ) -> None:
        self._runtime_context = {
            "step": int(step),
            "task_phase": str(task_phase),
            "controller_event": int(controller_event),
            "controller_t": int(controller_t),
            "ee_pos": (
                None
                if ee_pos is None
                else np.asarray(ee_pos, dtype=np.float32).reshape(-1)[:3]
            ),
            "playback_time_s": (
                None
                if playback_time_s is None
                else float(playback_time_s)
            ),
        }

    def peek(self) -> dict[str, Any]:
        if not self._started:
            return self._inactive_state()
        if self._use_recorded_timebase():
            return self._state_at_recorded_time()
        return self._state_at_step(self._cursor, advance=False)

    def __call__(self) -> dict[str, Any]:
        if self._finished:
            return self._inactive_state()
        if not self._started:
            if not self._phase_is_ready():
                return self._inactive_state()
            self._start_playback()
        if self._use_recorded_timebase():
            return self._state_at_recorded_time()
        return self._state_at_step(self._cursor, advance=True)

    def close(self) -> None:
        for h5_file in self._files.values():
            h5_file.close()
        self._files.clear()

    def capture_state(self) -> dict[str, Any]:
        return {
            "schema_version": "human_encounter_replay_state_v1",
            "scenario_id": str(self._scenario.get("id", "")),
            "cursor": int(self._cursor),
            "started": bool(self._started),
            "finished": bool(self._finished),
            "anchor_offset": self._anchor_offset.copy(),
            "runtime_context": _copy_replay_value(self._runtime_context),
            "runtime_start_time_s": self._runtime_start_time_s,
            "source_start_time_s": self._source_start_time_s,
            "last_source_time_s": self._last_source_time_s,
            "last_active_state": _copy_replay_value(self._last_active_state),
            "last_active_source_step": int(self._last_active_source_step),
            "augmentation_parameters": _copy_replay_value(
                self._augmentation_parameters
            ),
            "augmentation_pivot": self._augmentation_pivot.copy(),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "augmentation_rng_state": copy.deepcopy(
                self._augmentation_rng.bit_generator.state
            ),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != "human_encounter_replay_state_v1":
            raise ValueError("Unsupported HumanEncounterReplay state schema")
        scenario_id = str(state.get("scenario_id", ""))
        current_id = str(self._scenario.get("id", ""))
        if scenario_id != current_id:
            raise ValueError(
                "Encounter state belongs to a different scenario: "
                f"{scenario_id!r} != {current_id!r}"
            )
        self._cursor = int(state["cursor"])
        self._started = bool(state["started"])
        self._finished = bool(state["finished"])
        self._anchor_offset = np.asarray(
            state["anchor_offset"], dtype=np.float32
        ).reshape(3).copy()
        self._runtime_context = _copy_replay_value(state.get("runtime_context", {}))
        self._runtime_start_time_s = _optional_float(
            state.get("runtime_start_time_s")
        )
        self._source_start_time_s = _optional_float(
            state.get("source_start_time_s")
        )
        self._last_source_time_s = _optional_float(state.get("last_source_time_s"))
        self._last_active_state = _copy_replay_value(
            state.get("last_active_state", {})
        )
        self._last_active_source_step = int(
            state.get("last_active_source_step", -1)
        )
        self._augmentation_parameters = _copy_replay_value(
            state.get(
                "augmentation_parameters",
                _identity_augmentation_parameters(),
            )
        )
        self._augmentation_pivot = np.asarray(
            state.get("augmentation_pivot", np.zeros(3)), dtype=np.float32
        ).reshape(3).copy()
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        if "augmentation_rng_state" in state:
            self._augmentation_rng.bit_generator.state = copy.deepcopy(
                state["augmentation_rng_state"]
            )

    def _select_scenario(self, episode_index: int) -> dict[str, Any]:
        if self.episode_policy == "cycle":
            return dict(self._scenarios[int(episode_index) % len(self._scenarios)])
        if self.episode_policy != "random":
            raise ValueError(
                f"Unknown encounter episode policy: {self.episode_policy}"
            )
        available = [
            severity
            for severity in SEVERITY_ORDER
            if self._by_severity[severity] and self.severity_mix[severity] > 0.0
        ]
        if not available:
            available = [
                severity
                for severity in SEVERITY_ORDER
                if self._by_severity[severity]
            ]
        weights = np.asarray(
            [self.severity_mix[severity] for severity in available],
            dtype=np.float64,
        )
        if float(weights.sum()) <= 0.0:
            weights = np.ones_like(weights)
        weights /= weights.sum()
        severity = str(self.rng.choice(available, p=weights))
        scenarios = self._by_severity[severity]
        return dict(scenarios[int(self.rng.integers(0, len(scenarios)))])

    def _phase_is_ready(self) -> bool:
        if not self.phase_match:
            return True
        current_phase = str(self._runtime_context.get("task_phase", ""))
        expected_phase = str(
            self._scenario.get(
                "trigger_task_phase",
                self._scenario.get("task_phase", ""),
            )
        )
        expected_event = int(
            self._scenario.get(
                "trigger_controller_event",
                self._scenario.get("controller_event", -1),
            )
        )
        current_event = int(self._runtime_context.get("controller_event", -1))
        if self.event_match and expected_event >= 0 and current_event >= 0:
            runtime_trigger_event = _runtime_trigger_event(
                expected_phase,
                expected_event,
            )
            return current_event == runtime_trigger_event
        return current_phase == expected_phase

    def _start_playback(self) -> None:
        self._started = True
        runtime_time = self._runtime_context.get("playback_time_s")
        if runtime_time is not None and np.isfinite(float(runtime_time)):
            self._runtime_start_time_s = float(runtime_time)
        source_times = self._episode.get("sample_time_s")
        if source_times is not None and len(source_times) > self._cursor:
            self._source_start_time_s = float(source_times[self._cursor])
            self._last_source_time_s = self._source_start_time_s
        if self.anchor_mode == "world":
            self._set_augmentation_pivot()
            return
        if self.anchor_mode != "ee":
            raise ValueError(f"Unknown encounter anchor mode: {self.anchor_mode}")
        source_anchor = self._scenario.get("source_anchor_ee_pos")
        current_anchor = self._runtime_context.get("ee_pos")
        if source_anchor is None or current_anchor is None:
            self._set_augmentation_pivot()
            return
        source = np.asarray(source_anchor, dtype=np.float32).reshape(-1)
        current = np.asarray(current_anchor, dtype=np.float32).reshape(-1)
        if (
            source.size >= 3
            and current.size >= 3
            and np.all(np.isfinite(source[:3]))
            and np.all(np.isfinite(current[:3]))
        ):
            self._anchor_offset = current[:3] - source[:3]
        self._set_augmentation_pivot()

    def _use_recorded_timebase(self) -> bool:
        if self.playback_timebase != "recorded":
            return False
        runtime_time = self._runtime_context.get("playback_time_s")
        source_times = self._episode.get("sample_time_s")
        return bool(
            runtime_time is not None
            and np.isfinite(float(runtime_time))
            and self._runtime_start_time_s is not None
            and self._source_start_time_s is not None
            and source_times is not None
            and len(source_times) == int(self._episode["length"])
        )

    def _state_at_recorded_time(self) -> dict[str, Any]:
        end_step = _scenario_human_motion_end_step(
            self._scenario,
            int(self._episode["length"]),
        )
        if self._cursor >= end_step or end_step <= 0:
            self._finished = True
            return self._inactive_state()
        runtime_time = float(self._runtime_context["playback_time_s"])
        elapsed_s = max(0.0, runtime_time - float(self._runtime_start_time_s))
        source_time_s = float(self._source_start_time_s) + (
            elapsed_s * self._effective_playback_speed()
        )
        source_times = np.asarray(self._episode["sample_time_s"], dtype=np.float64)
        end_time_s = float(source_times[end_step - 1])
        if source_time_s > end_time_s + 1e-9:
            self._finished = True
            return self._inactive_state()
        source_time_s = min(source_time_s, end_time_s)
        state, source_idx = _human_state_at_time(
            self._episode,
            source_time_s,
            start_step=int(self._scenario["start_step"]),
            end_step=end_step,
        )
        self._cursor = int(source_idx)
        self._last_source_time_s = float(source_time_s)
        state = self._apply_anchor(state)
        state = self._apply_augmentation(
            state,
            source_step=source_idx,
            source_time_s=source_time_s,
        )
        state.update(self._metadata_state(active=True, source_step=source_idx))
        state["encounter_source_time_s"] = float(source_time_s)
        self._remember_active_state(state, source_step=source_idx)
        return state

    def _state_at_step(self, source_idx: int, *, advance: bool) -> dict[str, Any]:
        end_step = _scenario_human_motion_end_step(
            self._scenario,
            int(self._episode["length"]),
        )
        if source_idx >= end_step:
            self._finished = True
            return self._inactive_state()
        state = _human_state_at(self._episode, source_idx)
        state = self._apply_anchor(state)
        state = self._apply_augmentation(
            state,
            source_step=source_idx,
            source_time_s=None,
        )
        state.update(self._metadata_state(active=True, source_step=source_idx))
        self._remember_active_state(state, source_step=source_idx)
        if advance:
            self._cursor += 1
            if self._cursor >= end_step:
                self._finished = True
        return state

    def _apply_anchor(self, state: dict[str, Any]) -> dict[str, Any]:
        for key in (
            "human_head_pos",
            "human_left_hand_pos",
            "human_right_hand_pos",
        ):
            if key in state:
                state[key] = (
                    np.asarray(state[key], dtype=np.float32) + self._anchor_offset
                )
        return state

    def _sample_augmentation(self) -> dict[str, Any]:
        config = self.augmentation
        if not bool(config.enabled) or (
            float(self._augmentation_rng.random())
            < float(config.identity_probability)
        ):
            return _identity_augmentation_parameters()

        translation = np.asarray(
            [
                self._augmentation_rng.uniform(
                    -config.translation_xy_max_m,
                    config.translation_xy_max_m,
                ),
                self._augmentation_rng.uniform(
                    -config.translation_xy_max_m,
                    config.translation_xy_max_m,
                ),
                self._augmentation_rng.uniform(
                    -config.translation_z_max_m,
                    config.translation_z_max_m,
                ),
            ],
            dtype=np.float32,
        )
        yaw_deg = float(
            self._augmentation_rng.uniform(-config.yaw_max_deg, config.yaw_max_deg)
        )
        direction = self._augmentation_rng.normal(size=3)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-12:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            direction = direction / direction_norm
        smooth_amplitude = float(
            self._augmentation_rng.uniform(0.0, config.smooth_offset_max_m)
        )
        smooth_cycles = float(
            self._augmentation_rng.uniform(
                config.smooth_cycles_min,
                config.smooth_cycles_max,
            )
        )
        return {
            "schema_version": ENCOUNTER_AUGMENTATION_VERSION,
            "applied": True,
            "translation_m": translation,
            "yaw_deg": yaw_deg,
            "playback_speed_factor": float(
                self._augmentation_rng.uniform(
                    config.playback_speed_min,
                    config.playback_speed_max,
                )
            ),
            "smooth_direction": np.asarray(direction, dtype=np.float32),
            "smooth_amplitude_m": smooth_amplitude,
            "smooth_cycles": smooth_cycles,
        }

    def _set_augmentation_pivot(self) -> None:
        runtime_anchor = self._runtime_context.get("ee_pos")
        if runtime_anchor is not None:
            value = np.asarray(runtime_anchor, dtype=np.float32).reshape(-1)
            if value.size >= 3 and np.all(np.isfinite(value[:3])):
                self._augmentation_pivot = value[:3].copy()
                return
        source_anchor = self._scenario.get("source_anchor_ee_pos")
        if source_anchor is not None:
            value = np.asarray(source_anchor, dtype=np.float32).reshape(-1)
            if value.size >= 3 and np.all(np.isfinite(value[:3])):
                self._augmentation_pivot = value[:3] + self._anchor_offset

    def _effective_playback_speed(self) -> float:
        return float(self.playback_speed) * float(
            self._augmentation_parameters.get("playback_speed_factor", 1.0)
        )

    def _augmentation_progress(
        self,
        *,
        source_step: int,
        source_time_s: float | None,
    ) -> tuple[float, float]:
        start_step = _scenario_human_motion_start_step(self._scenario)
        end_step = _scenario_human_motion_end_step(
            self._scenario,
            int(self._episode["length"]),
        )
        source_times = self._episode.get("sample_time_s")
        if source_times is not None and end_step > start_step:
            times = np.asarray(source_times, dtype=np.float64)
            start_time = float(times[start_step])
            end_time = float(times[end_step - 1])
            duration = max(end_time - start_time, 1e-6)
            value = (
                float(times[min(max(source_step, start_step), end_step - 1)])
                if source_time_s is None
                else float(source_time_s)
            )
            return float(np.clip((value - start_time) / duration, 0.0, 1.0)), duration
        duration_steps = max(end_step - start_step - 1, 1)
        return (
            float(np.clip((source_step - start_step) / duration_steps, 0.0, 1.0)),
            float(duration_steps),
        )

    def _apply_augmentation(
        self,
        state: dict[str, Any],
        *,
        source_step: int,
        source_time_s: float | None,
    ) -> dict[str, Any]:
        params = self._augmentation_parameters
        if not bool(params.get("applied", False)):
            return state

        yaw_rad = np.deg2rad(float(params["yaw_deg"]))
        cosine = float(np.cos(yaw_rad))
        sine = float(np.sin(yaw_rad))
        rotation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        progress, duration = self._augmentation_progress(
            source_step=source_step,
            source_time_s=source_time_s,
        )
        cycles = float(params["smooth_cycles"])
        amplitude = float(params["smooth_amplitude_m"])
        direction = np.asarray(params["smooth_direction"], dtype=np.float32)
        smooth = direction * amplitude * np.sin(2.0 * np.pi * cycles * progress)
        translation = np.asarray(params["translation_m"], dtype=np.float32)
        for key in (
            "human_head_pos",
            "human_left_hand_pos",
            "human_right_hand_pos",
        ):
            if key not in state:
                continue
            position = np.asarray(state[key], dtype=np.float32).reshape(3)
            state[key] = (
                self._augmentation_pivot
                + rotation @ (position - self._augmentation_pivot)
                + translation
                + smooth
            ).astype(np.float32)

        smooth_velocity = (
            direction
            * amplitude
            * (2.0 * np.pi * cycles / max(duration, 1e-6))
            * np.cos(2.0 * np.pi * cycles * progress)
            * self._effective_playback_speed()
        )
        for key in ("human_left_hand_vel", "human_right_hand_vel"):
            if key in state:
                state[key] = (
                    rotation @ np.asarray(state[key], dtype=np.float32).reshape(3)
                    * self._effective_playback_speed()
                    + smooth_velocity
                ).astype(np.float32)
        return state

    def _inactive_state(self) -> dict[str, Any]:
        hold_last_pose = bool(
            self._finished
            and self.augmentation.post_window_mode == "hold_last_pose"
            and self._last_active_state
        )
        if not hold_last_pose:
            state = self._metadata_state(active=False, source_step=-1)
            state["encounter_post_window_hold"] = 0.0
            return state

        state = _copy_replay_value(self._last_active_state)
        for key in ("human_left_hand_vel", "human_right_hand_vel"):
            if key in state:
                state[key] = np.zeros(3, dtype=np.float32)
        state.update(
            self._metadata_state(
                active=False,
                source_step=int(self._last_active_source_step),
            )
        )
        state["encounter_post_window_hold"] = 1.0
        return state

    def _remember_active_state(
        self,
        state: dict[str, Any],
        *,
        source_step: int,
    ) -> None:
        self._last_active_state = _copy_replay_value(state)
        self._last_active_source_step = int(source_step)

    def _metadata_state(self, *, active: bool, source_step: int) -> dict[str, Any]:
        core_start = int(self._scenario.get("core_start_step", -1))
        core_end = int(self._scenario.get("core_end_step", -1))
        risk_core_active = bool(
            active and core_start >= 0 and core_start <= source_step < core_end
        )
        recovery_active = bool(active and core_end >= 0 and source_step >= core_end)
        human_window = self._scenario.get("human_motion_window")
        human_window = human_window if isinstance(human_window, dict) else {}
        return {
            "encounter_id": str(self._scenario.get("id", "")),
            "encounter_target_severity": str(
                self._scenario.get("target_severity", "")
            ),
            "encounter_target_phase": str(
                self._scenario.get("task_phase", "")
            ),
            "encounter_target_event": int(
                self._scenario.get("controller_event", -1)
            ),
            "encounter_source_session": str(
                self._scenario.get("session_id", "")
            ),
            "encounter_source_episode": str(
                self._scenario.get("source_episode", "")
            ),
            "encounter_source_step": int(source_step),
            "encounter_active": float(active),
            "encounter_started": float(self._started),
            "encounter_finished": float(self._finished),
            "encounter_risk_core_active": float(risk_core_active),
            "encounter_human_recovery_active": float(recovery_active),
            "encounter_human_recovery_complete": float(
                bool(human_window.get("recovery_complete", False))
            ),
            "encounter_human_recovery_status": str(
                human_window.get("recovery_status", "legacy_not_annotated")
            ),
            "encounter_anchor_offset_m": self._anchor_offset.copy(),
            "encounter_playback_timebase": self.playback_timebase,
            "encounter_playback_speed": float(self.playback_speed),
            "encounter_effective_playback_speed": self._effective_playback_speed(),
            "encounter_augmentation_version": ENCOUNTER_AUGMENTATION_VERSION,
            "encounter_augmentation_applied": float(
                bool(self._augmentation_parameters.get("applied", False))
            ),
            "encounter_augmentation_translation_m": np.asarray(
                self._augmentation_parameters.get("translation_m", np.zeros(3)),
                dtype=np.float32,
            ),
            "encounter_augmentation_yaw_deg": float(
                self._augmentation_parameters.get("yaw_deg", 0.0)
            ),
            "encounter_post_window_hold": 0.0,
            "encounter_source_time_s": (
                -1.0
                if self._last_source_time_s is None
                else float(self._last_source_time_s)
            ),
        }


def _identity_augmentation_parameters() -> dict[str, Any]:
    return {
        "schema_version": ENCOUNTER_AUGMENTATION_VERSION,
        "applied": False,
        "translation_m": np.zeros(3, dtype=np.float32),
        "yaw_deg": 0.0,
        "playback_speed_factor": 1.0,
        "smooth_direction": np.zeros(3, dtype=np.float32),
        "smooth_amplitude_m": 0.0,
        "smooth_cycles": 0.0,
    }


def _load_episode_group(group: h5py.Group) -> dict[str, Any]:
    sim_time = _dataset_or_none(group, "sim_time", dtype=np.float64)
    pose_monotonic_ns = _dataset_or_none(
        group,
        "pose_monotonic_time_ns",
        dtype=np.int64,
    )
    if pose_monotonic_ns is None:
        pose_monotonic_ns = _dataset_or_none(
            group,
            "monotonic_time_ns",
            dtype=np.int64,
        )
    velocity_time = (
        np.asarray(pose_monotonic_ns, dtype=np.float64) * 1e-9
        if pose_monotonic_ns is not None
        else sim_time
    )
    if "human" in group:
        human = group["human"]
        head_pos = _dataset_or_zeros(human, "head_pos", (3,))
        left_hand_pos = _dataset_or_zeros(human, "left_hand_pos", (3,))
        right_hand_pos = _dataset_or_zeros(human, "right_hand_pos", (3,))
        left_hand_vel = _preferred_recorded_velocity(human, "left")
        if left_hand_vel is None:
            left_hand_vel = _finite_difference(left_hand_pos, velocity_time)
        right_hand_vel = _preferred_recorded_velocity(human, "right")
        if right_hand_vel is None:
            right_hand_vel = _finite_difference(right_hand_pos, velocity_time)
        valid_mask = _dataset_or_derived_valid_mask(
            human,
            head_pos,
            left_hand_pos,
            right_hand_pos,
        )
    else:
        obs = group["obs"]
        head_pos = _dataset_or_zeros(obs, "human_head_pos", (3,))
        left_hand_pos = _dataset_or_zeros(obs, "human_left_hand_pos", (3,))
        right_hand_pos = _dataset_or_zeros(obs, "human_right_hand_pos", (3,))
        left_hand_vel = _finite_difference(left_hand_pos, velocity_time)
        right_hand_vel = _finite_difference(right_hand_pos, velocity_time)
        valid_mask = _derived_valid_mask(
            head_pos,
            left_hand_pos,
            right_hand_pos,
        )

    length = int(head_pos.shape[0])
    return {
        "length": length,
        "sample_time_s": _preferred_sample_time_s(
            pose_monotonic_ns,
            sim_time,
            length,
        ),
        "head_pos": head_pos,
        "left_hand_pos": left_hand_pos,
        "right_hand_pos": right_hand_pos,
        "left_hand_vel": _align_length(left_hand_vel, length, (3,)),
        "right_hand_vel": _align_length(right_hand_vel, length, (3,)),
        "valid_mask": _align_length(valid_mask, length, (3,)),
        "human_robot_collision": _scalar_dataset_or_none(
            group,
            (
                "safety/contact_active",
                "safety/human_robot_collision",
                "obs/human_robot_collision",
            ),
        ),
        "near_human": _scalar_dataset_or_none(
            group,
            ("safety/near_human", "obs/near_human"),
        ),
        "min_hand_gripper_dist_m": _scalar_dataset_or_none(
            group,
            (
                "safety/min_hand_end_effector_surface_gap_m",
                "safety/end_effector_surface_gap_m",
                "obs/min_hand_end_effector_surface_gap",
                "safety/min_hand_gripper_surface_gap_m",
                "safety/min_hand_gripper_dist_m",
                "obs/min_hand_gripper_dist",
            ),
        ),
        "gripper_camera_occluded": _human_scalar_or_none(
            group,
            "gripper_camera_occluded",
        ),
    }


def _copy_replay_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _copy_replay_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_replay_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_replay_value(item) for item in value)
    return copy.deepcopy(value)


def _jsonable_replay_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable_replay_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_replay_value(item) for item in value]
    return copy.deepcopy(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("Replay state time must be finite or None")
    return result


def _human_state_at(episode: dict[str, Any], sample_idx: int) -> dict[str, Any]:
    valid_mask = episode["valid_mask"][sample_idx]
    state: dict[str, Any] = {
        "human_left_hand_vel": episode["left_hand_vel"][sample_idx],
        "human_right_hand_vel": episode["right_hand_vel"][sample_idx],
        "human_valid_mask": valid_mask,
    }
    if valid_mask[0] > 0.5:
        state["human_head_pos"] = episode["head_pos"][sample_idx]
    if valid_mask[1] > 0.5:
        state["human_left_hand_pos"] = episode["left_hand_pos"][sample_idx]
    if valid_mask[2] > 0.5:
        state["human_right_hand_pos"] = episode["right_hand_pos"][sample_idx]
    if episode["human_robot_collision"] is not None:
        state["recorded_human_robot_collision"] = bool(
            episode["human_robot_collision"][sample_idx] > 0.5
        )
    if episode["near_human"] is not None:
        state["recorded_near_human"] = bool(
            episode["near_human"][sample_idx] > 0.5
        )
    if episode["min_hand_gripper_dist_m"] is not None:
        state["recorded_min_hand_gripper_dist_m"] = float(
            episode["min_hand_gripper_dist_m"][sample_idx]
        )
    if episode["gripper_camera_occluded"] is not None:
        state["gripper_camera_occluded"] = float(
            np.clip(episode["gripper_camera_occluded"][sample_idx], 0.0, 1.0)
        )
    return state


def _human_state_at_time(
    episode: dict[str, Any],
    source_time_s: float,
    *,
    start_step: int,
    end_step: int,
) -> tuple[dict[str, Any], int]:
    times = np.asarray(episode["sample_time_s"], dtype=np.float64).reshape(-1)
    start = max(0, int(start_step))
    end = min(int(end_step), int(episode["length"]), times.size)
    if end <= start:
        return _human_state_at(episode, start), start

    clipped_time = float(np.clip(source_time_s, times[start], times[end - 1]))
    upper = int(np.searchsorted(times[start:end], clipped_time, side="right")) + start
    upper = min(max(upper, start + 1), end - 1)
    lower = max(start, upper - 1)
    t0 = float(times[lower])
    t1 = float(times[upper])
    alpha = 0.0 if t1 <= t0 else float(np.clip((clipped_time - t0) / (t1 - t0), 0.0, 1.0))
    nearest = lower if alpha < 0.5 else upper

    valid0 = np.asarray(episode["valid_mask"][lower], dtype=np.float32)
    valid1 = np.asarray(episode["valid_mask"][upper], dtype=np.float32)
    valid_mask = np.minimum(valid0, valid1)
    state: dict[str, Any] = {
        "human_left_hand_vel": _lerp(
            episode["left_hand_vel"][lower], episode["left_hand_vel"][upper], alpha
        ),
        "human_right_hand_vel": _lerp(
            episode["right_hand_vel"][lower], episode["right_hand_vel"][upper], alpha
        ),
        "human_valid_mask": valid_mask,
    }
    for valid_index, state_name, episode_name in (
        (0, "human_head_pos", "head_pos"),
        (1, "human_left_hand_pos", "left_hand_pos"),
        (2, "human_right_hand_pos", "right_hand_pos"),
    ):
        if valid_mask[valid_index] > 0.5:
            state[state_name] = _lerp(
                episode[episode_name][lower],
                episode[episode_name][upper],
                alpha,
            )
    if episode["human_robot_collision"] is not None:
        state["recorded_human_robot_collision"] = bool(
            episode["human_robot_collision"][nearest] > 0.5
        )
    if episode["near_human"] is not None:
        state["recorded_near_human"] = bool(
            episode["near_human"][nearest] > 0.5
        )
    if episode["min_hand_gripper_dist_m"] is not None:
        state["recorded_min_hand_gripper_dist_m"] = float(
            (1.0 - alpha) * episode["min_hand_gripper_dist_m"][lower]
            + alpha * episode["min_hand_gripper_dist_m"][upper]
        )
    if episode["gripper_camera_occluded"] is not None:
        state["gripper_camera_occluded"] = float(
            np.clip(episode["gripper_camera_occluded"][nearest], 0.0, 1.0)
        )
    return state, lower


def _preferred_sample_time_s(
    pose_monotonic_ns: np.ndarray | None,
    sim_time: np.ndarray | None,
    length: int,
) -> np.ndarray:
    if pose_monotonic_ns is not None and len(pose_monotonic_ns) == length:
        values = np.asarray(pose_monotonic_ns, dtype=np.float64).reshape(-1)
        if _valid_time_series(values):
            values = (values - values[0]) * 1e-9
            if _valid_time_series(values):
                return values
    if sim_time is not None and len(sim_time) == length:
        values = np.asarray(sim_time, dtype=np.float64).reshape(-1)
        if _valid_time_series(values):
            return values - values[0]
    return np.arange(length, dtype=np.float64) / 60.0


def _runtime_trigger_event(expected_phase: str, expected_event: int) -> int:
    latest_event = _LATEST_PRE_SUCCESS_TRIGGER_EVENT.get(str(expected_phase))
    if latest_event is None:
        return int(expected_event)
    return min(int(expected_event), int(latest_event))


def _valid_time_series(values: np.ndarray) -> bool:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return bool(
        values.size > 0
        and np.all(np.isfinite(values))
        and np.all(np.diff(values) >= 0.0)
        and (values.size == 1 or values[-1] > values[0])
    )


def _lerp(start, end, alpha: float) -> np.ndarray:
    start_arr = np.asarray(start, dtype=np.float32)
    end_arr = np.asarray(end, dtype=np.float32)
    return ((1.0 - alpha) * start_arr + alpha * end_arr).astype(np.float32)


def _dataset_or_zeros(group, name: str, item_shape: tuple[int, ...]) -> np.ndarray:
    if name in group:
        arr = np.asarray(group[name], dtype=np.float32)
        return arr.reshape((arr.shape[0],) + item_shape)
    length = _infer_group_length(group)
    return np.zeros((length,) + item_shape, dtype=np.float32)


def _dataset_or_none(
    group,
    name: str,
    *,
    dtype=np.float32,
) -> np.ndarray | None:
    if name not in group:
        return None
    return np.asarray(group[name], dtype=dtype).reshape(-1)


def _dataset_or_derived_valid_mask(
    human_group,
    head_pos: np.ndarray,
    left_hand_pos: np.ndarray,
    right_hand_pos: np.ndarray,
) -> np.ndarray:
    if "valid_mask" in human_group:
        arr = np.asarray(human_group["valid_mask"], dtype=np.float32)
        return arr.reshape((arr.shape[0], 3))
    return _derived_valid_mask(head_pos, left_hand_pos, right_hand_pos)


def _derived_valid_mask(
    head_pos: np.ndarray,
    left_hand_pos: np.ndarray,
    right_hand_pos: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            _valid_position_series(head_pos),
            _valid_position_series(left_hand_pos),
            _valid_position_series(right_hand_pos),
        ],
        axis=1,
    ).astype(np.float32)


def _valid_position_series(values: np.ndarray) -> np.ndarray:
    finite = np.all(np.isfinite(values), axis=1)
    nonzero = np.linalg.norm(values, axis=1) > 1e-6
    return np.logical_and(finite, nonzero).astype(np.float32)


def _preferred_recorded_velocity(human, hand: str) -> np.ndarray | None:
    for name in (
        f"{hand}_hand_vel_filtered_mps",
        f"{hand}_hand_vel",
        f"{hand}_hand_vel_raw_mps",
    ):
        if name in human:
            return np.asarray(human[name], dtype=np.float32)
    return None


def _finite_difference(values: np.ndarray, timestamps: np.ndarray | None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    vel = np.zeros_like(values)
    if values.shape[0] <= 1:
        return vel
    if timestamps is None or len(timestamps) != values.shape[0]:
        dt = np.ones((values.shape[0] - 1, 1), dtype=np.float32)
    else:
        dt = np.diff(timestamps).reshape(-1, 1).astype(np.float32)
        dt = np.maximum(dt, 1e-6)
    vel[1:] = (values[1:] - values[:-1]) / dt
    return vel


def _obs_scalar_or_none(group, name: str) -> np.ndarray | None:
    if "obs" not in group or name not in group["obs"]:
        return None
    return np.asarray(group["obs"][name], dtype=np.float32).reshape(-1)


def _scalar_dataset_or_none(group, paths: tuple[str, ...]) -> np.ndarray | None:
    for path in paths:
        if path in group:
            return np.asarray(group[path], dtype=np.float32).reshape(-1)
    return None


def _human_scalar_or_none(group, name: str) -> np.ndarray | None:
    if "human" not in group or name not in group["human"]:
        return None
    return np.asarray(group["human"][name], dtype=np.float32).reshape(-1)


def _align_length(arr: np.ndarray, length: int, item_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape == (length,) + item_shape:
        return arr
    result = np.zeros((length,) + item_shape, dtype=np.float32)
    count = min(length, arr.shape[0])
    if count > 0:
        result[:count] = arr[:count].reshape((count,) + item_shape)
    return result


def _infer_group_length(group) -> int:
    for value in group.values():
        if hasattr(value, "shape") and value.shape:
            return int(value.shape[0])
    return 0
