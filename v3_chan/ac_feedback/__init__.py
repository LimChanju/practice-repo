"""Fail-closed BC+CBF explicit-feedback collection components.

The package is intentionally Isaac-independent at import time.  The collection
entry point creates ``SimulationApp`` first and only then imports the runtime
objects that depend on Isaac Sim.
"""

from .schema import (
    COLLECTION_SCHEMA_VERSION,
    ENCOUNTER_SCHEMA_VERSION,
    EXPLICIT_FEEDBACK_SCHEMA_VERSION,
    QUESTION_ID,
    RESPONSE_CODES,
)
from .action_trace import (
    ActionStepTrace,
    ActionTraceRecorder,
    CanonicalJointAction,
    canonicalize_joint_action,
)

__all__ = [
    "COLLECTION_SCHEMA_VERSION",
    "ENCOUNTER_SCHEMA_VERSION",
    "EXPLICIT_FEEDBACK_SCHEMA_VERSION",
    "QUESTION_ID",
    "RESPONSE_CODES",
    "ActionStepTrace",
    "ActionTraceRecorder",
    "CanonicalJointAction",
    "canonicalize_joint_action",
]
