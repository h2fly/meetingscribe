"""
Tests for the first-use model download notice: which models a transcription
or caption session would still have to download, and the sentence that tells
the user about it. Nothing here touches the network or the real caches —
MODELSCOPE_CACHE / HF_HUB_CACHE / CONFIG_DIR are pointed into tmp_path.
Run: pytest tests/
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import meetingscribe as ms


@pytest.fixture(autouse=True)
def empty_caches(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path / "ms"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    monkeypatch.setattr(ms, "CONFIG_DIR", tmp_path / "cfg")
    return tmp_path


def _put_funasr(tmp_path, alias, weight="model.pt"):
    mid = ms._FUNASR_MODEL_INFO[alias][0]
    d = tmp_path / "ms" / "hub" / "models" / mid
    d.mkdir(parents=True)
    (d / "configuration.json").write_text("{}")
    if weight:
        (d / weight).write_bytes(b"x")


def _labels(missing):
    return [name for name, _ in missing]


def test_transcription_lists_all_three_on_a_fresh_machine():
    missing = ms._models_missing_for_transcription({})
    assert _labels(missing) == ["paraformer-large", "FSMN-VAD", "CT-Punc"]
    assert sum(mb for _, mb in missing) > 2000


def test_transcription_nothing_when_cached(empty_caches):
    for alias in ("paraformer-zh", "fsmn-vad", "ct-punc"):
        _put_funasr(empty_caches, alias)
    assert ms._models_missing_for_transcription({}) == []


def test_config_without_weights_still_counts_as_missing(empty_caches):
    # An interrupted download leaves configuration.json but no weights.
    _put_funasr(empty_caches, "ct-punc", weight=None)
    assert "CT-Punc" in _labels(ms._models_missing_for_transcription({}))


def test_diarization_adds_campplus():
    cfg = {"stt": {"funasr": {"spk_model": "cam++"}}}
    assert "cam++" in _labels(ms._models_missing_for_transcription(cfg))


def test_cloud_transcription_needs_nothing():
    assert ms._models_missing_for_transcription({"transcribe_provider": "openai"}) == []


def test_local_model_dir_is_never_announced(tmp_path):
    d = tmp_path / "my-model"
    d.mkdir()
    assert ms._funasr_model_missing(str(d)) is None


def test_captions_en_with_hymt(empty_caches):
    lc = {"language": "en", "translate_to": "zh", "mt_provider": "hymt",
          "speaker_id": {"enabled": False}}
    missing = ms._models_missing_for_captions({}, lc)
    assert _labels(missing) == ["Nemotron Speech EN 0.6B", "HY-MT 1.5 1.8B"]
    # English sessions never refine, so the FunASR stack is not announced.
    assert "paraformer-large" not in _labels(missing)


def test_captions_translation_off_skips_mt():
    lc = {"language": "zh", "translate_to": "off", "mt_provider": "hymt",
          "refine": False, "speaker_id": {"enabled": False}}
    assert _labels(ms._models_missing_for_captions({}, lc)) == [
        "Streaming Paraformer zh-en"]


def test_captions_zh_refine_and_speaker_share_funasr_entries():
    lc = {"language": "zh", "translate_to": "off", "refine": True}
    labels = _labels(ms._models_missing_for_captions({}, lc))
    assert labels == ["Streaming Paraformer zh-en", "paraformer-large",
                      "FSMN-VAD", "CT-Punc", "cam++"]


def test_captions_nothing_once_downloaded(empty_caches):
    lc = {"language": "en", "translate_to": "zh", "mt_provider": "hymt",
          "speaker_id": {"enabled": False}}
    models = empty_caches / "cfg" / "models"
    stem = "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"
    (models / stem).mkdir(parents=True)
    (models / stem / "tokens.txt").write_text("a 0")
    (models / "HY-MT1.5-1.8B-Q4_K_M.gguf").write_bytes(b"x")
    assert ms._models_missing_for_captions({}, lc) == []


def test_marian_checks_hf_cache(empty_caches):
    lc = {"language": "en", "translate_to": "zh", "mt_provider": "marian",
          "asr_model_dir": "/somewhere", "speaker_id": {"enabled": False}}
    assert len(ms._models_missing_for_captions({}, lc)) == 2
    for repo in ("Helsinki-NLP/opus-mt-zh-en", "Helsinki-NLP/opus-mt-en-zh"):
        snap = (empty_caches / "hf" / ("models--" + repo.replace("/", "--"))
                / "snapshots" / "abc")
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_bytes(b"x")
    assert ms._models_missing_for_captions({}, lc) == []


def test_explicit_gguf_path_is_not_announced():
    lc = {"language": "en", "translate_to": "zh", "mt_provider": "hymt",
          "asr_model_dir": "/x", "hymt": {"model_path": "/nope.gguf"},
          "speaker_id": {"enabled": False}}
    assert ms._models_missing_for_captions({}, lc) == []


def test_notice_wording_and_total():
    missing = [("paraformer-large", 950), ("CT-Punc", 1100), ("FSMN-VAD", 4)]
    zh = ms._model_download_notice(missing, "transcribe")
    assert zh.startswith("首次转写需要先下载模型，共约 2.0 GB")
    assert "CT-Punc（约 1.1 GB）" in zh and "FSMN-VAD（约 4 MB）" in zh
    en = ms._model_download_notice(missing, "captions", "en")
    assert en.startswith("First use of live captions") and "2.0 GB" in en
    assert ms._model_download_notice([], "transcribe") == ""


def test_download_hook_reports_progress(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ms, "_model_download_hook",
                        lambda *a: calls.append(a))

    class FakeResp:
        headers = {"Content-Length": str(4 * 1024 * 1024)}

        def __init__(self):
            self.left = 4

        def read(self, n):
            if not self.left:
                return b""
            self.left -= 1
            return b"\0" * (1024 * 1024)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: FakeResp())
    ms._caption_download("https://example.invalid/m.gguf", tmp_path / "m.gguf.part")
    assert calls and calls[-1] == ("m.gguf", 4, 4)
