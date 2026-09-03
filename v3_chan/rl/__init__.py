"""RL utilities for SeRT trajectory collection and policy training."""

from .actions import (
    ACTION_DIM,
    ACTION_NAMES,
    ACTION_VERSION,
    CONTROLLER_TARGET_ACTION_VERSION,
    CONTROLLER_TARGET_MAX_DELTA_M,
    MAX_EE_DELTA_M,
    MAX_YAW_DELTA_RAD,
    TaskSpaceAction,
    clip_action,
    controller_target_action_from_target,
    controller_target_from_action,
    denormalize_action,
    expert_joint_action_vector,
    task_action_from_transition,
    zero_action,
)
from .observations import (
    AUXILIARY_OBSERVATION_FIELDS,
    BACKUP_LIMITER_ACTION_DIM,
    BACKUP_LIMITER_HISTORY_DIM,
    BACKUP_DYNAMIC_HRI_OBS_DIM,
    BACKUP_DYNAMIC_HRI_OBS_FIELD_NAMES,
    BACKUP_OMITTED_DYNAMIC_FIELD_NAMES,
    BACKUP_OBS_DIM,
    BACKUP_OBSERVATION_VERSION,
    DYNAMIC_HRI_OBS_DIM,
    DYNAMIC_HRI_OBS_FIELD_NAMES,
    DYNAMIC_HRI_OBSERVATION_FIELDS,
    DYNAMIC_HRI_OBSERVATION_VERSION,
    HRI_OBS_DIM,
    HRI_OBS_FIELD_NAMES,
    HRI_OBSERVATION_VERSION,
    LEGACY_BACKUP_OBS_DIM,
    LEGACY_BACKUP_OBSERVATION_VERSION,
    OBSERVATION_DIM,
    OBSERVATION_FIELDS,
    OBSERVATION_VERSION,
    RECORDED_OBSERVATION_FIELDS,
    TASK_PHASES,
    CONTROLLER_EVENT_COUNT,
    build_observation,
    apply_dynamic_hri_observation,
    controller_event_onehot,
    empty_observation,
    flatten_hri_observation,
    flatten_backup_dynamic_observation,
    flatten_backup_observation,
    flatten_backup_policy_observation,
    flatten_dynamic_hri_observation,
    flatten_observation,
    observation_slices,
    restore_dynamic_hri_observations_from_backup,
    validate_auxiliary_observation,
    validate_observation,
)
from .pseudo_errp import (
    DEFAULT_PSEUDO_ERRP_SOURCES,
    PSEUDO_ERRP_SOURCE_CODES,
    PseudoErrPResult,
    extract_pseudo_errp_aux_flags,
    parse_pseudo_errp_sources,
    pseudo_errp_from_observation,
)
from .errp_feedback import (
    ERRP_FEEDBACK_VERSION,
    CleanPseudoErrPSource,
    CreditAssignment,
    CreditConfig,
    ErrPEvent,
    ErrPEventDetector,
    ErrPFeedbackSample,
    EventDetectorConfig,
    EventLockedFeedbackBridge,
    PseudoErrPConfig,
    PseudoErrPSource,
    RealizedApproachAgency,
    credit_assignments,
    physical_risk_score,
    realized_approach_agency,
)
from .offline_errp import (
    OFFLINE_ERRP_REPLAY_VERSION,
    OfflineErrPReplaySource,
)
try:
    from .encounter_manifest import (
        MANIFEST_VERSION,
        SEVERITY_ORDER,
        SOURCE_CONFIGURATION_VERSION,
        EncounterBuildConfig,
        build_encounter_manifest,
        extract_episode_source_configuration,
        load_encounter_manifest,
        parse_severity_mix,
        resolve_source_restoration,
    )
    from .human_replay import (
        ENCOUNTER_AUGMENTATION_VERSION,
        EncounterAugmentationConfig,
        HumanEncounterReplay,
        HumanEncounterReplayInfo,
        HumanReplayInfo,
        HumanTrajectoryReplay,
    )
except ModuleNotFoundError as exc:
    if exc.name != "h5py":
        raise

    class HumanReplayInfo:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "h5py is required for HumanTrajectoryReplay. Install it in the active "
                "Python environment before using --human-replay-data."
            ) from exc

    class HumanTrajectoryReplay:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "h5py is required for HumanTrajectoryReplay. Install it in the active "
                "Python environment before using --human-replay-data."
            ) from exc

    class HumanEncounterReplay:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "h5py is required for HumanEncounterReplay. Install it in the active "
                "Python environment before using --encounter-manifest."
            ) from exc
from .rewards import (
    DEFAULT_ISAAC_FRANKA_DENSE_REWARD_WEIGHTS,
    DEFAULT_MINIMAL_REWARD_WEIGHTS,
    DEFAULT_REWARD_WEIGHTS,
    ISAAC_FRANKA_DENSE_REWARD_VERSION,
    LEGACY_REWARD_VERSION,
    MINIMAL_EVENT_POTENTIAL_REWARD_VERSION,
    REWARD_VERSION,
    SUCCESS_ONLY_REWARD_VERSION,
    IsaacFrankaDenseRewardWeights,
    MinimalEventPotentialRewardWeights,
    RewardResult,
    RewardWeights,
    compute_isaac_franka_dense_reward,
    compute_reward,
    compute_minimal_event_potential_reward,
    is_success,
    compute_strict_success_only_reward,
    isaac_franka_dense_reward_weights_dict,
    minimal_reward_weights_dict,
    reward_component_names,
    reward_weights_dict,
)
from .safemotions_style import (
    BACKUP_ACTION_DIM,
    BACKUP_ACTION_VERSION,
    BACKUP_XYZYAW_ACTION_DIM,
    BACKUP_XYZYAW_ACTION_VERSION,
    RISK_ESTIMATOR_VERSION,
    SAFEMOTIONS_REFERENCE_COMMIT,
    SAFEMOTIONS_PHASE_EXECUTION_CONTRACT,
    ActionReplacementResult,
    BackupActionLimitConfig,
    BackupActionLimiter,
    BackupEpisodeBoundary,
    BackupRewardConfig,
    BackupRewardResult,
    BackupStartDistribution,
    DecisionTimebase,
    FutureRiskConfig,
    FutureRiskResult,
    InterventionHandoffConfig,
    InterventionHandoffController,
    InterventionHandoffDecision,
    HandoffActionBlender,
    HandoffBlendDecision,
    RiskCandidateAction,
    SafeMotionsBackupRewardConfig,
    StateActionRiskEstimator,
    TaskManifoldState,
    backup_start_retry_source,
    backup_episode_boundary,
    backup_episode_should_start,
    classify_future_risk,
    compute_backup_reward,
    compute_safemotions_backup_reward,
    horizon_steps_from_seconds,
    oracle_intervention_from_gate,
    normalized_joint_limit_risk,
    replace_unsafe_task_action,
    risk_candidate_actions,
    risk_estimator_checkpoint,
    sample_backup_start_source,
    sample_task_manifold_candidates,
    scheduled_backup_start_source,
    state_action_features,
    task_reward_without_human_terms,
    time_steps_from_seconds,
)
from .artifacts import file_sha256
from .running_statistics import RunningMeanStd
try:
    from .pick_place_env import (
        IsaacPickPlaceEnv,
        PickPlaceBranchState,
        PickPlaceEnvConfig,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {
        "isaacsim",
        "omni",
        "panda_robot",
        "scene_setup",
    }:
        raise

    class PickPlaceEnvConfig:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "IsaacPickPlaceEnv requires Isaac Sim runtime modules. Create a "
                "SimulationApp first and run through launch_isaac.sh."
            ) from exc

    class IsaacPickPlaceEnv:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "IsaacPickPlaceEnv requires Isaac Sim runtime modules. Create a "
                "SimulationApp first and run through launch_isaac.sh."
            ) from exc

    class PickPlaceBranchState:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "PickPlaceBranchState requires the Isaac pick-place environment."
            ) from exc

try:
    from .trajectory_recorder import (
        EXPERT_JOINT_ACTION_DIM,
        TRAJECTORY_SCHEMA_VERSION,
        TrajectoryRecorder,
    )
except ModuleNotFoundError as exc:
    if exc.name != "h5py":
        raise
    EXPERT_JOINT_ACTION_DIM = 9
    TRAJECTORY_SCHEMA_VERSION = "trajectory_v0_transitions"

    class TrajectoryRecorder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "h5py is required for TrajectoryRecorder. Install it in the Isaac "
                "Python environment, or run collect_expert_trajectories.py with "
                "--install-missing-deps."
            ) from exc

__all__ = [
    "ACTION_DIM",
    "ACTION_NAMES",
    "ACTION_VERSION",
    "ActionReplacementResult",
    "AUXILIARY_OBSERVATION_FIELDS",
    "BACKUP_ACTION_DIM",
    "BACKUP_ACTION_VERSION",
    "BACKUP_LIMITER_ACTION_DIM",
    "BACKUP_LIMITER_HISTORY_DIM",
    "BACKUP_DYNAMIC_HRI_OBS_DIM",
    "BACKUP_DYNAMIC_HRI_OBS_FIELD_NAMES",
    "BACKUP_OMITTED_DYNAMIC_FIELD_NAMES",
    "BACKUP_OBS_DIM",
    "BACKUP_OBSERVATION_VERSION",
    "BACKUP_XYZYAW_ACTION_DIM",
    "BACKUP_XYZYAW_ACTION_VERSION",
    "BackupActionLimitConfig",
    "BackupActionLimiter",
    "BackupEpisodeBoundary",
    "BackupRewardConfig",
    "BackupRewardResult",
    "BackupStartDistribution",
    "CONTROLLER_EVENT_COUNT",
    "CONTROLLER_TARGET_ACTION_VERSION",
    "CONTROLLER_TARGET_MAX_DELTA_M",
    "DEFAULT_ISAAC_FRANKA_DENSE_REWARD_WEIGHTS",
    "DEFAULT_REWARD_WEIGHTS",
    "DEFAULT_MINIMAL_REWARD_WEIGHTS",
    "DEFAULT_PSEUDO_ERRP_SOURCES",
    "DYNAMIC_HRI_OBS_DIM",
    "DYNAMIC_HRI_OBS_FIELD_NAMES",
    "DYNAMIC_HRI_OBSERVATION_FIELDS",
    "DYNAMIC_HRI_OBSERVATION_VERSION",
    "DecisionTimebase",
    "EXPERT_JOINT_ACTION_DIM",
    "ERRP_FEEDBACK_VERSION",
    "CleanPseudoErrPSource",
    "CreditAssignment",
    "CreditConfig",
    "ErrPEvent",
    "ErrPEventDetector",
    "ErrPFeedbackSample",
    "EventDetectorConfig",
    "EventLockedFeedbackBridge",
    "FutureRiskConfig",
    "FutureRiskResult",
    "InterventionHandoffConfig",
    "InterventionHandoffController",
    "InterventionHandoffDecision",
    "HandoffActionBlender",
    "HandoffBlendDecision",
    "HRI_OBS_DIM",
    "HRI_OBS_FIELD_NAMES",
    "HRI_OBSERVATION_VERSION",
    "ISAAC_FRANKA_DENSE_REWARD_VERSION",
    "LEGACY_BACKUP_OBS_DIM",
    "LEGACY_BACKUP_OBSERVATION_VERSION",
    "HumanEncounterReplay",
    "HumanEncounterReplayInfo",
    "EncounterAugmentationConfig",
    "ENCOUNTER_AUGMENTATION_VERSION",
    "HumanReplayInfo",
    "HumanTrajectoryReplay",
    "IsaacFrankaDenseRewardWeights",
    "MAX_EE_DELTA_M",
    "MAX_YAW_DELTA_RAD",
    "MINIMAL_EVENT_POTENTIAL_REWARD_VERSION",
    "OBSERVATION_DIM",
    "OBSERVATION_FIELDS",
    "OBSERVATION_VERSION",
    "OFFLINE_ERRP_REPLAY_VERSION",
    "OfflineErrPReplaySource",
    "MinimalEventPotentialRewardWeights",
    "RECORDED_OBSERVATION_FIELDS",
    "SOURCE_CONFIGURATION_VERSION",
    "IsaacPickPlaceEnv",
    "LEGACY_REWARD_VERSION",
    "PickPlaceEnvConfig",
    "PickPlaceBranchState",
    "PSEUDO_ERRP_SOURCE_CODES",
    "PseudoErrPConfig",
    "PseudoErrPSource",
    "RealizedApproachAgency",
    "PseudoErrPResult",
    "REWARD_VERSION",
    "RISK_ESTIMATOR_VERSION",
    "RiskCandidateAction",
    "RewardResult",
    "RewardWeights",
    "StateActionRiskEstimator",
    "SAFEMOTIONS_REFERENCE_COMMIT",
    "SAFEMOTIONS_PHASE_EXECUTION_CONTRACT",
    "SafeMotionsBackupRewardConfig",
    "RunningMeanStd",
    "TASK_PHASES",
    "TRAJECTORY_SCHEMA_VERSION",
    "TaskSpaceAction",
    "TaskManifoldState",
    "TrajectoryRecorder",
    "build_observation",
    "apply_dynamic_hri_observation",
    "clip_action",
    "controller_event_onehot",
    "controller_target_action_from_target",
    "controller_target_from_action",
    "compute_reward",
    "compute_isaac_franka_dense_reward",
    "compute_minimal_event_potential_reward",
    "compute_backup_reward",
    "compute_safemotions_backup_reward",
    "horizon_steps_from_seconds",
    "credit_assignments",
    "classify_future_risk",
    "backup_episode_boundary",
    "backup_episode_should_start",
    "backup_start_retry_source",
    "denormalize_action",
    "empty_observation",
    "expert_joint_action_vector",
    "extract_episode_source_configuration",
    "extract_pseudo_errp_aux_flags",
    "flatten_hri_observation",
    "flatten_backup_dynamic_observation",
    "flatten_backup_observation",
    "flatten_backup_policy_observation",
    "flatten_dynamic_hri_observation",
    "flatten_observation",
    "restore_dynamic_hri_observations_from_backup",
    "file_sha256",
    "is_success",
    "isaac_franka_dense_reward_weights_dict",
    "minimal_reward_weights_dict",
    "oracle_intervention_from_gate",
    "normalized_joint_limit_risk",
    "observation_slices",
    "parse_pseudo_errp_sources",
    "pseudo_errp_from_observation",
    "physical_risk_score",
    "realized_approach_agency",
    "replace_unsafe_task_action",
    "reward_component_names",
    "reward_weights_dict",
    "risk_estimator_checkpoint",
    "risk_candidate_actions",
    "sample_backup_start_source",
    "sample_task_manifold_candidates",
    "scheduled_backup_start_source",
    "resolve_source_restoration",
    "task_action_from_transition",
    "state_action_features",
    "task_reward_without_human_terms",
    "time_steps_from_seconds",
    "validate_auxiliary_observation",
    "validate_observation",
    "zero_action",
]
