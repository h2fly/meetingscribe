"""Tests for speaker diarization (stt.funasr.spk_model → cam++).

Pure-Python: FunASR is faked via ``sys.modules`` / monkeypatched loaders, so
no model download happens. Covers the line formatter, the model-cache key
(which must keep sharing the non-diarized instance with live captions), and
the gate that forces serial transcription when diarization is on.
"""

import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import meetingscribe as ms


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_wav(path: Path, duration_secs: float, sample_rate: int = 16000,
              channels: int = 2) -> Path:
    frames = int(duration_secs * sample_rate)
    shape = (frames, channels) if channels > 1 else (frames,)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(np.zeros(shape, dtype=np.int16).tobytes())
    return path


def _sent(text, start, spk):
    return {"text": text, "start": start, "spk": spk}


class _FakeModel:
    """Stands in for funasr.AutoModel: records every generate() call."""

    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self._payload


# ── _spk_lines ────────────────────────────────────────────────────────────────

class TestSpkLines:
    def test_labels_and_timestamps(self):
        items = [{"sentence_info": [
            _sent("排期能不能提前？", 300, 1),
            _sent("可以。", 4100, 0),
        ]}]
        assert ms._spk_lines(items) == [
            "[000.3s] [说话人1] 排期能不能提前？",
            "[004.1s] [说话人2] 可以。",
        ]

    def test_consecutive_same_speaker_merged(self):
        items = [{"sentence_info": [
            _sent("可以，", 4100, 0),
            _sent("但要看兼容测试。", 5200, 0),
            _sent("我补充一下。", 9800, 3),
        ]}]
        lines = ms._spk_lines(items)
        assert lines == [
            "[004.1s] [说话人1] 可以，但要看兼容测试。",
            "[009.8s] [说话人2] 我补充一下。",
        ]

    def test_labels_numbered_by_first_appearance(self):
        # cam++ cluster ids are arbitrary; display numbering must follow the
        # transcript so 说话人1 is whoever spoke first.
        items = [{"sentence_info": [_sent("甲", 0, 7), _sent("乙", 1000, 2),
                                    _sent("丙", 2000, 7)]}]
        assert ms._spk_lines(items) == [
            "[000.0s] [说话人1] 甲",
            "[001.0s] [说话人2] 乙",
            "[002.0s] [说话人1] 丙",
        ]

    def test_string_speaker_ids(self):
        items = [{"sentence_info": [{"text": "甲", "start": 0, "spk": "spk_a"},
                                    {"text": "乙", "start": 500, "spk": "spk_b"}]}]
        assert ms._spk_lines(items) == [
            "[000.0s] [说话人1] 甲",
            "[000.5s] [说话人2] 乙",
        ]

    def test_offset_applied(self):
        items = [{"sentence_info": [_sent("甲", 500, 0)]}]
        assert ms._spk_lines(items, offset_s=60.0) == ["[060.5s] [说话人1] 甲"]

    def test_falls_back_to_timestamp_field(self):
        items = [{"sentence_info": [
            {"text": "甲", "spk": 0, "timestamp": [[1500, 1800]]}]}]
        assert ms._spk_lines(items) == ["[001.5s] [说话人1] 甲"]

    def test_missing_start_keeps_label_without_stamp(self):
        items = [{"sentence_info": [{"text": "甲", "spk": 0}]}]
        assert ms._spk_lines(items) == ["[说话人1] 甲"]

    def test_malformed_start_does_not_raise(self):
        items = [{"sentence_info": [{"text": "甲", "spk": 0, "start": "abc"}]}]
        assert ms._spk_lines(items) == ["[说话人1] 甲"]

    def test_no_sentence_info_returns_empty(self):
        assert ms._spk_lines([{"text": "整段文本"}]) == []
        assert ms._spk_lines([]) == []
        assert ms._spk_lines(None) == []

    def test_skips_blank_and_non_dict_entries(self):
        items = [{"sentence_info": [_sent("", 0, 0), "junk",
                                    _sent("  ", 100, 0), _sent("甲", 200, 0)]}]
        assert ms._spk_lines(items) == ["[000.2s] [说话人1] 甲"]


# ── model cache key ───────────────────────────────────────────────────────────

class TestCacheKey:
    def test_no_spk_keeps_three_tuple_for_sharing(self):
        # Live captions' refine pass keys on the 3-tuple; a 4-tuple here would
        # load a second ~1 GB paraformer-large instead of reusing the cached one.
        assert ms._funasr_cache_key("a", "v", "p") == ("a", "v", "p")
        assert ms._funasr_cache_key("a", "v", "p", "") == ("a", "v", "p")

    def test_spk_gets_its_own_entry(self):
        assert ms._funasr_cache_key("a", "v", "p", "cam++") == (
            "a", "v", "p", "cam++")


class TestAutoModelSpkKwarg:
    def _fake_funasr(self, monkeypatch, sink):
        mod = types.ModuleType("funasr")

        def _auto_model(**kwargs):
            sink.append(kwargs)
            return object()

        mod.AutoModel = _auto_model
        monkeypatch.setitem(sys.modules, "funasr", mod)

    def test_spk_model_forwarded(self, monkeypatch):
        seen: list = []
        self._fake_funasr(monkeypatch, seen)
        ms._load_funasr_automodel("a", "v", "p", "cam++")
        assert seen[-1]["spk_model"] == "cam++"

    def test_no_spk_model_key_when_disabled(self, monkeypatch):
        seen: list = []
        self._fake_funasr(monkeypatch, seen)
        ms._load_funasr_automodel("a", "v", "p", "")
        assert "spk_model" not in seen[-1]


# ── _transcribe_funasr integration ────────────────────────────────────────────

class TestTranscribeWithDiarization:
    def _run(self, monkeypatch, tmp_path, payload, pcfg, duration=900.0):
        wav = _make_wav(tmp_path / "rec.wav", duration)
        model = _FakeModel(payload)
        monkeypatch.setattr(ms, "_funasr_model_cache", {})
        monkeypatch.setattr(
            ms, "_load_funasr_automodel",
            lambda *a, **k: model)
        text = ms._transcribe_funasr(wav, pcfg)
        return text, model

    def test_diarization_forces_single_pass(self, monkeypatch, tmp_path):
        # 15 min at chunk_secs=300 would normally be 3 parallel chunks; with
        # diarization it must be one call, otherwise speaker ids from
        # different chunks would be conflated by the notes LLM.
        payload = [{"sentence_info": [_sent("甲说", 0, 0), _sent("乙说", 5000, 1)]}]
        text, model = self._run(
            monkeypatch, tmp_path, payload,
            {"chunk_secs": 300, "spk_model": "cam++", "workers": 3})
        assert len(model.calls) == 1
        assert text.splitlines() == [
            "[000.0s] [说话人1] 甲说",
            "[005.0s] [说话人2] 乙说",
        ]

    def test_without_diarization_still_chunks(self, monkeypatch, tmp_path):
        payload = [{"text": "普通文本", "timestamp": [[0, 500]]}]
        _, model = self._run(
            monkeypatch, tmp_path, payload,
            {"chunk_secs": 300, "spk_model": "", "workers": 3})
        assert len(model.calls) == 3  # 15 min / 5 min chunks

    def test_hotword_still_forwarded_with_diarization(self, monkeypatch, tmp_path):
        payload = [{"sentence_info": [_sent("甲说", 0, 0)]}]
        _, model = self._run(
            monkeypatch, tmp_path, payload,
            {"chunk_secs": 300, "spk_model": "cam++", "hotword": "GKE Kong"})
        assert model.calls[0]["hotword"] == "GKE Kong"

    def test_missing_sentence_info_falls_back_to_plain_lines(
            self, monkeypatch, tmp_path):
        # A FunASR version that ignores spk_model must not cost us the
        # transcript — unlabelled lines beat an empty file.
        payload = [{"text": "普通文本", "timestamp": [[0, 500]]}]
        logged: list = []
        monkeypatch.setattr(ms, "_log", lambda cat, msg: logged.append((cat, msg)))
        text, _ = self._run(
            monkeypatch, tmp_path, payload,
            {"chunk_secs": 300, "spk_model": "cam++"})
        assert text == "[000.0s] 普通文本"
        assert any(cat == "WARN" and "sentence_info" in msg for cat, msg in logged)

    def test_short_recording_unaffected(self, monkeypatch, tmp_path):
        payload = [{"sentence_info": [_sent("甲说", 0, 0)]}]
        text, model = self._run(
            monkeypatch, tmp_path, payload,
            {"chunk_secs": 300, "spk_model": "cam++"}, duration=60.0)
        assert len(model.calls) == 1
        assert text == "[000.0s] [说话人1] 甲说"


# ── prompt contract ───────────────────────────────────────────────────────────

def test_polish_prompt_tells_llm_to_keep_speaker_labels():
    """The polish step strips timestamps; it must NOT strip speaker labels,
    or diarization would be lost before the notes prompt ever sees it."""
    prompt = ms._resolve_prompt(ms.load_config(), "polish")
    assert "[说话人N]" in prompt
    assert "不要把不同 N 合并成一个人" in prompt
