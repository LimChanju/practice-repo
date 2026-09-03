import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "v3_chan" / "hri_video_recorder.py"


def _load_recorder_module(monkeypatch):
    pxr = types.ModuleType("pxr")
    for name in ("Gf", "Sdf", "UsdGeom", "UsdLux"):
        setattr(pxr, name, types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    spec = importlib.util.spec_from_file_location("hri_video_recorder_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_capture_refreshes_render_without_advancing_physics(monkeypatch, tmp_path):
    module = _load_recorder_module(monkeypatch)
    recorder = module.HRIOverviewVideoRecorder(
        enabled=True,
        record_dir=str(tmp_path),
        interval_steps=1,
    )

    calls = []
    orchestrator = types.SimpleNamespace(step=lambda **kwargs: calls.append(kwargs))
    recorder._rep = types.SimpleNamespace(orchestrator=orchestrator)
    recorder._writer = object()

    recorder.capture()

    assert calls == [{"delta_time": 0.0, "pause_timeline": False}]
    assert recorder._capture_count == 1
