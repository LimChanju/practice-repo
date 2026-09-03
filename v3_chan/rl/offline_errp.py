from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .errp_feedback import ErrPEvent


OFFLINE_ERRP_REPLAY_VERSION = "offline_errp_replay_source_v2"


class OfflineErrPReplaySource:
    """Replay public EEG decoder outputs at synthetic simulator events.

    Label-conditioned sampling is a simulation assumption: a threat event draws
    from public ErrP-labeled epochs and a safe probe draws from non-ErrP epochs.
    The sampled EEG was not recorded from the current simulator participant.
    """

    source_name = "offline_eegnet_edl_replay"

    def __init__(
        self,
        bundle_path: str,
        *,
        subjects: Sequence[str] | None = None,
        scenarios: Sequence[str] | None = None,
        sampling_mode: str = "label_conditioned",
    ) -> None:
        if sampling_mode not in (
            "label_conditioned",
            "unconditioned",
            "shuffled_label",
        ):
            raise ValueError(f"Unknown offline EEG sampling mode: {sampling_mode}")
        self.path = os.path.abspath(os.path.expanduser(bundle_path))
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        self.sampling_mode = sampling_mode
        with np.load(self.path, allow_pickle=False) as archive:
            self.labels = np.asarray(archive["y"], dtype=np.uint8).reshape(-1)
            self.probabilities = np.asarray(
                archive["p_errp"], dtype=np.float32
            ).reshape(-1)
            self.uncertainties = np.asarray(
                archive["uncertainty"], dtype=np.float32
            ).reshape(-1)
            self.predictions = np.asarray(
                archive["prediction"], dtype=np.uint8
            ).reshape(-1)
            self.subjects = np.asarray(archive["subject"]).astype(str)
            self.scenarios = (
                np.asarray(archive["scenario"]).astype(str)
                if "scenario" in archive.files
                else np.repeat(
                    np.asarray(["batzianoulis_static_obstacle"], dtype=str),
                    len(self.labels),
                )
            )
            self.source_files = np.asarray(archive["source_file"]).astype(str)
            source_index_key = (
                "source_epoch_index"
                if "source_epoch_index" in archive.files
                else "source_row_index"
            )
            if source_index_key not in archive.files:
                raise ValueError(
                    "Offline ErrP bundle needs source_epoch_index or source_row_index"
                )
            self.source_epoch_indices = np.asarray(
                archive[source_index_key], dtype=np.int64
            ).reshape(-1)
            self.source_trial_ids = (
                np.asarray(archive["source_trial_id"]).astype(str)
                if "source_trial_id" in archive.files
                else np.repeat(np.asarray([""], dtype=str), len(self.labels))
            )
            self.oof_folds = (
                np.asarray(archive["oof_fold"], dtype=np.int16).reshape(-1)
                if "oof_fold" in archive.files
                else np.full(len(self.labels), -1, dtype=np.int16)
            )
        sizes = {
            len(self.labels),
            len(self.probabilities),
            len(self.uncertainties),
            len(self.predictions),
            len(self.subjects),
            len(self.scenarios),
            len(self.source_files),
            len(self.source_epoch_indices),
            len(self.source_trial_ids),
            len(self.oof_folds),
        }
        if len(sizes) != 1 or next(iter(sizes)) == 0:
            raise ValueError("Offline ErrP bundle arrays have inconsistent lengths")
        if not np.all(np.isin(self.labels, (0, 1))):
            raise ValueError("Offline ErrP labels must be binary")
        if not (
            np.all(np.isfinite(self.probabilities))
            and np.all((self.probabilities >= 0.0) & (self.probabilities <= 1.0))
            and np.all(np.isfinite(self.uncertainties))
            and np.all((self.uncertainties >= 0.0) & (self.uncertainties <= 1.0))
        ):
            raise ValueError("Offline ErrP decoder outputs must be finite in [0, 1]")
        selected = np.ones(len(self.labels), dtype=bool)
        if subjects:
            selected &= np.isin(self.subjects, np.asarray(tuple(subjects)))
        if scenarios:
            selected &= np.isin(self.scenarios, np.asarray(tuple(scenarios)))
        self.indices = np.flatnonzero(selected)
        if self.indices.size == 0:
            raise ValueError("Offline ErrP filters selected no epochs")
        self.indices_by_label = {
            label: self.indices[self.labels[self.indices] == label]
            for label in (0, 1)
        }
        if sampling_mode == "label_conditioned" and any(
            values.size == 0 for values in self.indices_by_label.values()
        ):
            raise ValueError("Label-conditioned replay requires both EEG classes")
        self._manifest = self._read_manifest()
        self._validate_decoder_provenance()
        self.sample_count = 0
        self.sample_count_by_label = {0: 0, 1: 0}

    def sample(
        self,
        event: ErrPEvent,
        *,
        rng: np.random.Generator,
    ) -> tuple[float, float, int, int, str]:
        if self.sampling_mode == "label_conditioned":
            candidates = self.indices_by_label[int(event.expected_label)]
        elif self.sampling_mode == "shuffled_label":
            # Preserve a balanced class-conditioned sampling channel while
            # deliberately breaking the simulator-event/EEG-label relation.
            shuffled_label = int(rng.integers(0, 2))
            candidates = self.indices_by_label[shuffled_label]
        else:
            candidates = self.indices
        index = int(candidates[int(rng.integers(0, candidates.size))])
        label = int(self.labels[index])
        self.sample_count += 1
        self.sample_count_by_label[label] += 1
        epoch_id = str(self.source_trial_ids[index]).strip()
        if not epoch_id:
            epoch_id = (
                f"{Path(self.source_files[index]).stem}:"
                f"{int(self.source_epoch_indices[index]):05d}"
            )
        return (
            float(self.probabilities[index]),
            float(self.uncertainties[index]),
            label,
            int(self.predictions[index]),
            epoch_id,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": OFFLINE_ERRP_REPLAY_VERSION,
            "bundle": self.path,
            "sampling_mode": self.sampling_mode,
            "selected_epoch_count": int(self.indices.size),
            "selected_class_counts": {
                str(label): int(indices.size)
                for label, indices in self.indices_by_label.items()
            },
            "selected_subjects": sorted(np.unique(self.subjects[self.indices]).tolist()),
            "selected_scenarios": sorted(
                np.unique(self.scenarios[self.indices]).tolist()
            ),
            "selected_oof_folds": sorted(
                int(value)
                for value in np.unique(self.oof_folds[self.indices])
                if int(value) >= 0
            ),
            "sample_count": int(self.sample_count),
            "sample_count_by_label": {
                str(key): int(value)
                for key, value in self.sample_count_by_label.items()
            },
            "claim_scope": (
                "Event-triggered replay of public EEG decoder outputs; not "
                "participant-synchronized online EEG."
            ),
            "bundle_manifest": self._manifest,
        }

    def _read_manifest(self) -> dict[str, Any]:
        path = os.path.splitext(self.path)[0] + ".json"
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        supported = {
            "offline_errp_replay_bundle_v1",
            "batzianoulis_errp_subject_oof_replay_v1",
        }
        if payload.get("schema_version") not in supported:
            raise ValueError(
                f"Unsupported offline ErrP bundle manifest: "
                f"{payload.get('schema_version')}"
            )
        return payload

    def _validate_decoder_provenance(self) -> None:
        schema = str(self._manifest.get("schema_version", ""))
        if schema != "batzianoulis_errp_subject_oof_replay_v1":
            return
        if np.any(self.oof_folds <= 0):
            raise ValueError(
                "Batzianoulis replay requires a positive held-out fold for every epoch"
            )
        split = self._manifest.get("split_contract", {})
        if not bool(split.get("each_epoch_tested_once", False)):
            raise ValueError(
                "Batzianoulis replay manifest does not guarantee OOF predictions"
            )
