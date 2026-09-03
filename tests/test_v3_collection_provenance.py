from pathlib import Path

import pytest

from v3_chan.collection_provenance import (
    resolve_code_version,
    source_tree_sha256,
    validate_production_metadata,
)


def _source_tree(tmp_path: Path) -> Path:
    (tmp_path / "v3_chan").mkdir()
    (tmp_path / "v3_chan" / "main.py").write_text("print('v1')\n", encoding="ascii")
    (tmp_path / "v3_chan" / "trajectories").mkdir()
    (tmp_path / "v3_chan" / "trajectories" / "data.md").write_text(
        "ignored", encoding="ascii"
    )
    return tmp_path


def test_source_hash_is_stable_and_ignores_output_directories(tmp_path):
    root = _source_tree(tmp_path)
    first = source_tree_sha256(root)
    (root / "v3_chan" / "trajectories" / "data.md").write_text(
        "changed output", encoding="ascii"
    )
    assert source_tree_sha256(root) == first


def test_source_hash_ignores_ac_feedback_video_sidecars(tmp_path):
    root = _source_tree(tmp_path)
    first = source_tree_sha256(root)
    frames = (
        root
        / "v3_chan"
        / "ac_selective_smoothing_feedback_data"
        / "session_spectator_frames"
    )
    frames.mkdir(parents=True)
    (frames / "encode_mp4.sh").write_text(
        "#!/usr/bin/env bash\necho generated\n", encoding="ascii"
    )
    assert source_tree_sha256(root) == first


def test_source_hash_changes_with_runnable_source(tmp_path):
    root = _source_tree(tmp_path)
    first = source_tree_sha256(root)
    (root / "v3_chan" / "main.py").write_text("print('v2')\n", encoding="ascii")
    assert source_tree_sha256(root) != first


def test_resolve_code_version_uses_explicit_or_source_hash(tmp_path):
    root = _source_tree(tmp_path)
    explicit, source, digest = resolve_code_version(root, "release-42")
    fallback, fallback_source, fallback_digest = resolve_code_version(root, "unknown")
    assert explicit == "release-42"
    assert source == "HRI_CODE_VERSION"
    assert len(digest) == 64
    assert fallback == f"source-sha256:{fallback_digest}"
    assert fallback_source == "source_tree_sha256"


def test_production_validation_rejects_placeholders():
    with pytest.raises(RuntimeError, match="PARTICIPANT"):
        validate_production_metadata(
            production_mode=True,
            participant_id="unspecified",
            code_version="source-sha256:abc",
        )
    with pytest.raises(RuntimeError, match="CODE_VERSION"):
        validate_production_metadata(
            production_mode=True,
            participant_id="P01",
            code_version="unknown",
        )
    validate_production_metadata(
        production_mode=True,
        participant_id="P01",
        code_version="source-sha256:abc",
    )
