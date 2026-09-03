#!/usr/bin/env python3
"""Fail-closed validator for the frozen A/C VR feedback study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


try:
    from v3_chan.ac_feedback.config import DEFAULT_CONFIG
    from v3_chan.ac_feedback.validator import (
        validate_collection,
        validate_static_contract,
    )
except ModuleNotFoundError:  # direct execution from inside v3_chan/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from v3_chan.ac_feedback.config import DEFAULT_CONFIG
    from v3_chan.ac_feedback.validator import (
        validate_collection,
        validate_static_contract,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed validation of the frozen A/C config, deterministic "
            "counterbalanced schedule, and optionally one finalized HDF5 collection."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="finalized .h5/.hdf5 collection; omit for a static config/schedule audit",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"frozen pilot config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--participant-id",
        default=None,
        help="expected pseudonymous participant ID; static default is STATIC-P00",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="expected session ID; static default is STATIC-S00",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="base schedule seed; static default is 11",
    )
    parser.add_argument(
        "--mode",
        choices=("minimal_pilot", "pilot_with_anchors"),
        default=None,
        help="schedule mode override (otherwise read from config/artifact)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one machine-readable JSON report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.path is None:
        report = validate_static_contract(
            args.config,
            participant_id=args.participant_id or "STATIC-P00",
            session_id=args.session_id or "STATIC-S00",
            seed=11 if args.seed is None else args.seed,
            mode=args.mode,
            raise_on_error=False,
        )
    else:
        report = validate_collection(
            args.path,
            config_path=args.config,
            participant_id=args.participant_id,
            session_id=args.session_id,
            seed=args.seed,
            mode=args.mode,
            allow_partial_path=False,
            raise_on_error=False,
        )
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
    elif report.valid:
        if report.collection_checked:
            eligibility = "study-eligible" if report.study_eligible else "valid raw/practice artifact"
            print(
                f"VALID {eligibility}: {report.path} "
                f"({report.trial_count} trials, {report.query_count} queries, "
                f"{report.encounter_count} encounters, "
                f"{report.realtime_marker_count} markers)"
            )
        else:
            print(
                "VALID frozen A/C config and schedule: "
                f"config_sha256={report.config_sha256} "
                f"schedule_seed={report.schedule_seed} "
                f"group={report.counterbalancing_group}"
            )
    else:
        target = report.path or report.config_path
        print(f"INVALID fail-closed: {target}", file=sys.stderr)
        for issue in report.issues:
            print(f"- [{issue.code}] {issue.path}: {issue.message}", file=sys.stderr)
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
