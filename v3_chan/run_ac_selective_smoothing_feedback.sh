#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_isaac_python="${ISAACSIM_PYTHON:-/home/railab/isaac-sim-4.5.0/python.sh}"

if [[ ! -x "$_isaac_python" ]]; then
    printf '[ACFeedback] Isaac Python is not executable: %s\n' "$_isaac_python" >&2
    printf '[ACFeedback] Set ISAACSIM_PYTHON to the absolute path of python.sh.\n' >&2
    exit 2
fi

# Haptics are outside this study and are unconditionally disabled for practice,
# pilot, and production collection.  Caller-provided values cannot override it.
export BHAPTICS_ENABLED=0
export HRI_HAPTIC_CONDITION=off
export HRI_HAPTICS_ENABLED=0
export HRI_HAPTIC_FEEDBACK=0
export PYTHONUNBUFFERED=1

exec "$_isaac_python" \
    "$_script_dir/collect_ac_selective_smoothing_feedback.py" \
    "$@"
