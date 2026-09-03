"""Frozen identifiers and shared protocol exception for A/C collection."""

from __future__ import annotations

COLLECTION_SCHEMA_VERSION = "bc_cbf_recovery_ac_explicit_feedback_v1"
ENCOUNTER_SCHEMA_VERSION = "ac_safety_response_episode_v1"
EXPLICIT_FEEDBACK_SCHEMA_VERSION = "ac_selective_smoothing_questionnaire_v1"
ACTION_TRACE_SCHEMA_VERSION = "ac_bc_rmpflow_cbf_applied_action_trace_v1"
CLOCK_SCHEMA_VERSION = "server_monotonic_and_unix_ns_v1"
ROW_SEMANTICS = "transition_obs_t_action_t_obs_t_plus_1_v1"

RUNTIME_CONTRACT_SCHEMA_VERSION = "ac_selective_smoothing_runtime_handoff_v1"
RUNTIME_CONTRACT_SHA256 = (
    "aee2507ef62782839bab4dc0973f3ef2933d5b0f3ea83df5cd1c9c2d808e7a64"
)
RUNTIME_HANDOFF_COMMIT = "1d7d07cbdc51bef9e96605e137ec24eb4dff391a"
RUNTIME_BASE_COMMIT = "3682056fd8888d2881290f1e1f65f3cce10270d9"
POLICY_SHA256 = "d0fe9f9dc48a6049b77c8e1eab6de0905197207bb57ff858e0d7d332172b5230"
POLICY_SIZE_BYTES = 421_408
POLICY_RELATIVE_PATH = "v3_chan/policies/bc_pick_place_v2_release_settle.pt"

QUESTION_ID = "reuse_robot_response_needs_modification_v1"
QUESTION_TEXT_KO = "같은 상황에서 방금 로봇 반응을 다시 사용한다면 수정이 필요합니까?"
QUESTION_TEXT_EN = "Would this robot response need modification before reuse in the same situation?"
RESPONSE_CODES = (
    "needs_modification",
    "acceptable_as_is",
    "uncertain",
)
RESPONSE_LABELS_KO = {
    "needs_modification": "수정이 필요함",
    "acceptable_as_is": "그대로 사용 가능함",
    "uncertain": "잘 모르겠음",
}


class ProtocolViolation(RuntimeError):
    """A collection invariant was violated and collection must fail closed."""
