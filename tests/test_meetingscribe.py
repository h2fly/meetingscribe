"""
Tests for meetingscribe.py.
Run: pytest tests/
"""

import io
import json
import sys
import subprocess
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import urllib.error

sys.path.insert(0, str(Path(__file__).parent.parent))
import meetingscribe as ms


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_wav(
    path: Path,
    duration_secs: float = 1.0,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    frames = int(duration_secs * sample_rate)
    shape = (frames, channels) if channels > 1 else (frames,)
    data = np.zeros(shape, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())
    return path


def _url_mock(body: bytes) -> MagicMock:
    """Simulate urllib.request.urlopen used as a context manager."""
    m = MagicMock()
    m.__enter__.return_value = m
    m.read.return_value = body
    return m


def _http_error(code: int, msg: str = "Error") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(None, code, msg, {}, io.BytesIO(msg.encode()))


def _whisper_mock(texts=("hello",)) -> MagicMock:
    model = MagicMock()
    info = MagicMock(language="zh", language_probability=0.99)
    segs = [MagicMock(text=t, start=float(i)) for i, t in enumerate(texts)]
    model.transcribe.return_value = (iter(segs), info)
    return model


def _gemini_body(text: str) -> bytes:
    return json.dumps(
        {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    ).encode()


# ── _strip_jsonc_comments ─────────────────────────────────────────────────────

class TestStripJsoncComments:
    def test_plain_json_unchanged(self):
        src = '{"key": 1}'
        assert ms._strip_jsonc_comments(src) == src

    def test_inline_comment_removed(self):
        result = ms._strip_jsonc_comments('{"k": 1} // comment')
        assert "//" not in result
        assert '"k"' in result

    def test_url_inside_string_preserved(self):
        src = '{"url": "https://api.openai.com/v1"}'
        assert "https://api.openai.com/v1" in ms._strip_jsonc_comments(src)

    def test_standalone_comment_line(self):
        src = "// header\n{\"a\": 1}"
        parsed = json.loads(ms._strip_jsonc_comments(src))
        assert parsed == {"a": 1}

    def test_multiple_inline_comments(self):
        src = '{\n  "a": 1, // first\n  "b": 2  // second\n}'
        parsed = json.loads(ms._strip_jsonc_comments(src))
        assert parsed == {"a": 1, "b": 2}


# ── _deep_merge ───────────────────────────────────────────────────────────────

class TestDeepMerge:
    def test_flat_override(self):
        assert ms._deep_merge({"a": 1, "b": 2}, {"b": 99}) == {"a": 1, "b": 99}

    def test_new_key_added(self):
        assert ms._deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_nested_partial_override_preserves_other_keys(self):
        base = {"stt": {"whisper": {"model": "base", "workers": 2}}}
        over = {"stt": {"whisper": {"model": "large-v3"}}}
        r = ms._deep_merge(base, over)
        assert r["stt"]["whisper"]["model"] == "large-v3"
        assert r["stt"]["whisper"]["workers"] == 2

    def test_does_not_mutate_base(self):
        base = {"a": {"x": 1}}
        ms._deep_merge(base, {"a": {"x": 99}})
        assert base["a"]["x"] == 1

    def test_non_dict_value_replaces_nested_dict(self):
        r = ms._deep_merge({"a": {"x": 1}}, {"a": "string"})
        assert r["a"] == "string"


# ── load_config ───────────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_returns_defaults_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ms, "CONFIG_FILE", tmp_path / "missing.jsonc")
        monkeypatch.setattr(ms, "CONFIG_DIR", tmp_path)
        cfg = ms.load_config()
        assert cfg["transcribe_provider"] == "funasr"
        assert cfg["mode"] == "meeting"

    def test_merges_on_disk_values(self, tmp_path, monkeypatch):
        (tmp_path / "cfg.jsonc").write_text('{"mode": "interview"}')
        monkeypatch.setattr(ms, "CONFIG_FILE", tmp_path / "cfg.jsonc")
        monkeypatch.setattr(ms, "CONFIG_DIR", tmp_path)
        cfg = ms.load_config()
        assert cfg["mode"] == "interview"
        assert cfg["transcribe_provider"] == "funasr"  # default preserved

    def test_deep_merges_nested_stt_config(self, tmp_path, monkeypatch):
        (tmp_path / "cfg.jsonc").write_text(
            '{"stt": {"whisper": {"model": "large-v3"}}}'
        )
        monkeypatch.setattr(ms, "CONFIG_FILE", tmp_path / "cfg.jsonc")
        monkeypatch.setattr(ms, "CONFIG_DIR", tmp_path)
        cfg = ms.load_config()
        assert cfg["stt"]["whisper"]["model"] == "large-v3"
        assert cfg["stt"]["whisper"]["workers"] == 2  # default preserved

    def test_no_file_deepcopy_does_not_mutate_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ms, "CONFIG_FILE", tmp_path / "missing.jsonc")
        monkeypatch.setattr(ms, "CONFIG_DIR", tmp_path)
        cfg = ms.load_config()
        cfg["stt"]["whisper"]["model"] = "MUTATED"
        assert ms.DEFAULT_CONFIG["stt"]["whisper"]["model"] == "base"

    def test_jsonc_comments_stripped_before_parse(self, tmp_path, monkeypatch):
        (tmp_path / "cfg.jsonc").write_text('// comment\n{"mode": "interview"}')
        monkeypatch.setattr(ms, "CONFIG_FILE", tmp_path / "cfg.jsonc")
        monkeypatch.setattr(ms, "CONFIG_DIR", tmp_path)
        assert ms.load_config()["mode"] == "interview"


# ── DualStreamRecorder.save ───────────────────────────────────────────────────

class TestDualStreamRecorderSave:
    def test_both_streams_produces_stereo_wav(self, tmp_path):
        rec = ms.DualStreamRecorder("sys", "mic", 48000)
        rec._sys_frames = [np.zeros((4800, 2), np.float32)]
        rec._mic_frames = [np.zeros((4800, 1), np.float32)]
        path = tmp_path / "out.wav"
        assert rec.save(path) is True
        with wave.open(str(path)) as wf:
            assert wf.getnchannels() == 2
            assert wf.getnframes() == 4800

    def test_sys_only_still_produces_stereo(self, tmp_path):
        rec = ms.DualStreamRecorder("sys", "mic", 48000)
        rec._sys_frames = [np.zeros((4800, 2), np.float32)]
        path = tmp_path / "out.wav"
        rec.save(path)
        with wave.open(str(path)) as wf:
            assert wf.getnchannels() == 2

    def test_mic_only_still_produces_stereo(self, tmp_path):
        rec = ms.DualStreamRecorder("sys", "mic", 48000)
        rec._mic_frames = [np.zeros((4800, 1), np.float32)]
        path = tmp_path / "out.wav"
        rec.save(path)
        with wave.open(str(path)) as wf:
            assert wf.getnchannels() == 2

    def test_no_frames_returns_false(self, tmp_path):
        rec = ms.DualStreamRecorder("sys", "mic", 48000)
        assert rec.save(tmp_path / "out.wav") is False

    def test_audio_clipped_to_int16_range(self, tmp_path):
        rec = ms.DualStreamRecorder("sys", "mic", 48000)
        rec._sys_frames = [np.full((100, 2), 2.0, np.float32)]
        rec._mic_frames = [np.full((100, 1), 2.0, np.float32)]
        path = tmp_path / "out.wav"
        rec.save(path)
        with wave.open(str(path)) as wf:
            samples = np.frombuffer(wf.readframes(100), dtype=np.int16)
        assert samples.max() == 32767

    def test_unequal_stream_lengths_truncated_to_shorter(self, tmp_path):
        rec = ms.DualStreamRecorder("sys", "mic", 48000)
        rec._sys_frames = [np.zeros((4800, 2), np.float32)]
        rec._mic_frames = [np.zeros((3000, 1), np.float32)]
        path = tmp_path / "out.wav"
        rec.save(path)
        with wave.open(str(path)) as wf:
            assert wf.getnframes() == 3000


# ── DualStreamRecorder.stop ───────────────────────────────────────────────────

class TestDualStreamRecorderStop:
    def test_streams_set_to_none_after_stop(self):
        rec = ms.DualStreamRecorder("s", "m", 48000)
        rec._sys_stream = MagicMock()
        rec._mic_stream = MagicMock()
        rec.stop()
        assert rec._sys_stream is None
        assert rec._mic_stream is None

    def test_stop_calls_stop_and_close_on_each_stream(self):
        rec = ms.DualStreamRecorder("s", "m", 48000)
        sys_s = MagicMock()
        mic_s = MagicMock()
        rec._sys_stream = sys_s
        rec._mic_stream = mic_s
        rec.stop()
        sys_s.stop.assert_called_once()
        sys_s.close.assert_called_once()
        mic_s.stop.assert_called_once()
        mic_s.close.assert_called_once()

    def test_stop_is_idempotent(self):
        rec = ms.DualStreamRecorder("s", "m", 48000)
        mock_s = MagicMock()
        rec._sys_stream = mock_s
        rec.stop()
        rec.stop()  # second call: stream is None, should not raise
        mock_s.stop.assert_called_once()

    def test_stop_exception_does_not_propagate(self):
        rec = ms.DualStreamRecorder("s", "m", 48000)
        mock_s = MagicMock()
        mock_s.stop.side_effect = RuntimeError("already stopped")
        rec._sys_stream = mock_s
        rec.stop()  # must not raise

    def test_close_called_even_when_stop_raises(self):
        rec = ms.DualStreamRecorder("s", "m", 48000)
        mock_s = MagicMock()
        mock_s.stop.side_effect = RuntimeError("oops")
        rec._sys_stream = mock_s
        rec.stop()
        mock_s.close.assert_called_once()

    def test_start_partial_failure_cleans_up_sys_stream(self):
        rec = ms.DualStreamRecorder("sys", "mic", 48000)
        mock_sys = MagicMock()
        mock_mic = MagicMock()
        mock_mic.start.side_effect = RuntimeError("device not found")
        with patch("meetingscribe.sd.InputStream", side_effect=[mock_sys, mock_mic]):
            with pytest.raises(RuntimeError, match="device not found"):
                rec.start()
        mock_sys.stop.assert_called_once()
        mock_sys.close.assert_called_once()


# ── _llm_claude_cli ───────────────────────────────────────────────────────────

class TestLlmClaudeCli:
    def test_success_strips_whitespace(self):
        result = MagicMock(returncode=0, stdout="  result\n  ")
        with patch("meetingscribe.subprocess.run", return_value=result):
            assert ms._llm_claude_cli("p", "lbl", 60) == "result"

    def test_command_not_found_exits(self):
        with patch("meetingscribe.subprocess.run", side_effect=FileNotFoundError()):
            with pytest.raises(SystemExit):
                ms._llm_claude_cli("p", "lbl", 60)

    def test_timeout_exits(self):
        with patch(
            "meetingscribe.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["claude"], 60),
        ):
            with pytest.raises(SystemExit):
                ms._llm_claude_cli("p", "lbl", 60)

    def test_nonzero_returncode_exits(self):
        result = MagicMock(returncode=1, stderr="error")
        with patch("meetingscribe.subprocess.run", return_value=result):
            with pytest.raises(SystemExit):
                ms._llm_claude_cli("p", "lbl", 60)


# ── _llm_openai ───────────────────────────────────────────────────────────────

class TestLlmOpenai:
    _pcfg = {"api_key": "k", "model": "gpt-4o", "base_url": "https://api.openai.com/v1"}

    def test_success_returns_content(self):
        body = json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()
        with patch("urllib.request.urlopen", return_value=_url_mock(body)):
            assert ms._llm_openai("p", self._pcfg, "lbl", 60) == "hi"

    def test_empty_choices_list_exits(self):
        body = json.dumps({"choices": []}).encode()
        with patch("urllib.request.urlopen", return_value=_url_mock(body)):
            with pytest.raises(SystemExit):
                ms._llm_openai("p", self._pcfg, "lbl", 60)

    def test_invalid_json_exits(self):
        with patch("urllib.request.urlopen", return_value=_url_mock(b"not json")):
            with pytest.raises(SystemExit):
                ms._llm_openai("p", self._pcfg, "lbl", 60)

    def test_http_error_exits(self):
        with patch("urllib.request.urlopen", side_effect=_http_error(401)):
            with pytest.raises(SystemExit):
                ms._llm_openai("p", self._pcfg, "lbl", 60)

    def test_url_timeout_exits(self):
        err = urllib.error.URLError(TimeoutError("timed out"))
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(SystemExit):
                ms._llm_openai("p", self._pcfg, "lbl", 60)

    def test_url_network_error_exits(self):
        err = urllib.error.URLError("connection refused")
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(SystemExit):
                ms._llm_openai("p", self._pcfg, "lbl", 60)


# ── _llm_gemini ───────────────────────────────────────────────────────────────

class TestLlmGemini:
    _pcfg = {"api_key": "k", "model": "gemini-1.5-pro"}

    def test_success_returns_text(self):
        with patch("urllib.request.urlopen", return_value=_url_mock(_gemini_body("ok"))):
            assert ms._llm_gemini("p", self._pcfg, "lbl", 60) == "ok"

    def test_empty_candidates_exits(self):
        body = json.dumps({"candidates": []}).encode()
        with patch("urllib.request.urlopen", return_value=_url_mock(body)):
            with pytest.raises(SystemExit):
                ms._llm_gemini("p", self._pcfg, "lbl", 60)

    def test_invalid_json_exits(self):
        with patch("urllib.request.urlopen", return_value=_url_mock(b"!!")):
            with pytest.raises(SystemExit):
                ms._llm_gemini("p", self._pcfg, "lbl", 60)

    def test_http_error_exits(self):
        with patch("urllib.request.urlopen", side_effect=_http_error(403)):
            with pytest.raises(SystemExit):
                ms._llm_gemini("p", self._pcfg, "lbl", 60)

    def test_url_error_exits(self):
        err = urllib.error.URLError("network error")
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(SystemExit):
                ms._llm_gemini("p", self._pcfg, "lbl", 60)


# ── _transcribe_openai ────────────────────────────────────────────────────────

class TestTranscribeOpenai:
    _pcfg = {
        "api_key": "k",
        "model": "whisper-1",
        "base_url": "https://api.openai.com/v1",
    }

    def test_success_returns_text(self, tmp_path):
        wav = _make_wav(tmp_path / "a.wav")
        with patch("urllib.request.urlopen", return_value=_url_mock(b"hello world")):
            assert ms._transcribe_openai(wav, self._pcfg) == "hello world"

    def test_http_error_exits(self, tmp_path):
        wav = _make_wav(tmp_path / "a.wav")
        with patch("urllib.request.urlopen", side_effect=_http_error(401)):
            with pytest.raises(SystemExit):
                ms._transcribe_openai(wav, self._pcfg)

    def test_network_error_exits(self, tmp_path):
        wav = _make_wav(tmp_path / "a.wav")
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("refused"),
        ):
            with pytest.raises(SystemExit):
                ms._transcribe_openai(wav, self._pcfg)


# ── _transcribe_gemini ────────────────────────────────────────────────────────

class TestTranscribeGemini:
    _pcfg = {"api_key": "k", "model": "gemini-2.0-flash"}

    def test_success_returns_transcript(self, tmp_path):
        wav = _make_wav(tmp_path / "a.wav")
        with patch(
            "urllib.request.urlopen", return_value=_url_mock(_gemini_body("transcript"))
        ):
            assert ms._transcribe_gemini(wav, self._pcfg) == "transcript"

    def test_empty_candidates_exits(self, tmp_path):
        wav = _make_wav(tmp_path / "a.wav")
        body = json.dumps({"candidates": []}).encode()
        with patch("urllib.request.urlopen", return_value=_url_mock(body)):
            with pytest.raises(SystemExit):
                ms._transcribe_gemini(wav, self._pcfg)

    def test_http_error_exits(self, tmp_path):
        wav = _make_wav(tmp_path / "a.wav")
        with patch("urllib.request.urlopen", side_effect=_http_error(400)):
            with pytest.raises(SystemExit):
                ms._transcribe_gemini(wav, self._pcfg)


# ── _transcribe_whisper ───────────────────────────────────────────────────────

class TestTranscribeWhisper:
    _pcfg_serial = {"model": "base", "chunk_secs": 300, "workers": 2, "cpu_threads": 1}
    _pcfg_parallel = {"model": "base", "chunk_secs": 5, "workers": 2, "cpu_threads": 1}

    @pytest.fixture(autouse=True)
    def _fw(self):
        # WhisperModel is imported inside _transcribe_whisper, so mock via sys.modules
        mock_fw = MagicMock()
        with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
            yield mock_fw

    def test_serial_path_includes_timestamps(self, tmp_path, _fw):
        wav = _make_wav(tmp_path / "a.wav", duration_secs=5)
        _fw.WhisperModel.return_value = _whisper_mock(["hello"])
        result = ms._transcribe_whisper(wav, self._pcfg_serial)
        assert "[000.0s]" in result
        assert "hello" in result

    def test_serial_path_empty_segments_returns_empty_string(self, tmp_path, _fw):
        wav = _make_wav(tmp_path / "a.wav", duration_secs=5)
        _fw.WhisperModel.return_value = _whisper_mock([])
        assert ms._transcribe_whisper(wav, self._pcfg_serial) == ""

    def test_chunk_secs_zero_forces_serial_even_for_long_file(self, tmp_path, _fw):
        wav = _make_wav(tmp_path / "a.wav", duration_secs=600, sample_rate=8000)
        pcfg = {**self._pcfg_serial, "chunk_secs": 0}
        _fw.WhisperModel.return_value = _whisper_mock(["full"])
        result = ms._transcribe_whisper(wav, pcfg)
        assert "full" in result

    def test_parallel_path_creates_one_model_per_chunk(self, tmp_path, _fw):
        wav = _make_wav(tmp_path / "a.wav", duration_secs=10, sample_rate=8000)
        call_count = [0]

        def make_model(*a, **kw):
            call_count[0] += 1
            return _whisper_mock([f"seg{call_count[0]}"])

        _fw.WhisperModel.side_effect = make_model
        result = ms._transcribe_whisper(wav, self._pcfg_parallel)

        assert call_count[0] == 2
        assert "seg1" in result and "seg2" in result

    def test_parallel_path_results_are_in_time_order(self, tmp_path, _fw):
        wav = _make_wav(tmp_path / "a.wav", duration_secs=10, sample_rate=8000)
        call_count = [0]

        def make_model(*a, **kw):
            call_count[0] += 1
            n = call_count[0]
            return _whisper_mock([f"chunk{n}"])

        _fw.WhisperModel.side_effect = make_model
        result = ms._transcribe_whisper(wav, self._pcfg_parallel)

        lines = result.splitlines()
        assert len(lines) == 2
        ts0 = float(lines[0].split("s]")[0].lstrip("["))
        ts1 = float(lines[1].split("s]")[0].lstrip("["))
        assert ts0 < ts1

    def test_on_progress_called_once_per_chunk(self, tmp_path, _fw):
        wav = _make_wav(tmp_path / "a.wav", duration_secs=10, sample_rate=8000)
        calls = []
        _fw.WhisperModel.return_value = _whisper_mock(["t"])
        ms._transcribe_whisper(wav, self._pcfg_parallel, on_progress=calls.append)
        assert len(calls) == 2
        assert all(5 <= v <= 40 for v in calls)


# ── polish_transcript ────────────────────────────────────────────────────────

class TestPolishTranscript:
    def test_empty_input_returns_empty_string(self):
        assert ms.polish_transcript("", "claude", ms.DEFAULT_CONFIG) == ""

    def test_single_chunk_returns_llm_output(self):
        with patch("meetingscribe._llm_run", return_value="polished"):
            result = ms.polish_transcript("raw", "claude", ms.DEFAULT_CONFIG)
        assert result == "polished"

    def test_multiple_chunks_joined_with_double_newline(self):
        cfg = {**ms.DEFAULT_CONFIG, "polish_chunk_size": 5}
        transcript = "\n".join(["abc"] * 10)
        with patch("meetingscribe._llm_run", return_value="P"):
            result = ms.polish_transcript(transcript, "claude", cfg)
        parts = result.split("\n\n")
        assert len(parts) > 1
        assert all(p == "P" for p in parts)


# ── save_minutes ──────────────────────────────────────────────────────────────

class TestSaveMinutes:
    def test_writes_md_next_to_wav(self, tmp_path):
        wav = tmp_path / "20260101_120000.wav"
        out = ms.save_minutes("# Notes", wav)
        assert out == tmp_path / "20260101_120000.md"
        assert out.read_text(encoding="utf-8") == "# Notes"

    def test_overwrites_existing_md_file(self, tmp_path):
        wav = tmp_path / "test.wav"
        ms.save_minutes("first", wav)
        ms.save_minutes("second", wav)
        assert (tmp_path / "test.md").read_text(encoding="utf-8") == "second"


# ── get_device_volume ─────────────────────────────────────────────────────────

class TestGetDeviceVolume:
    def test_returns_none_on_non_darwin(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        assert ms.get_device_volume("any device") is None

    def test_returns_none_for_nonexistent_device(self):
        # Passes through CoreAudio on darwin but finds no match; returns None elsewhere too
        assert ms.get_device_volume("__nonexistent_device_xyz__") is None

    def test_return_type_is_float_or_none(self):
        import sys as _sys
        if _sys.platform != "darwin":
            pytest.skip("CoreAudio only on macOS")
        result = ms.get_device_volume(ms._get_current_output_device() or "")
        assert result is None or isinstance(result, float)

    def test_value_in_range_when_device_found(self):
        import sys as _sys
        if _sys.platform != "darwin":
            pytest.skip("CoreAudio only on macOS")
        dev = ms._get_current_output_device()
        if dev is None:
            pytest.skip("no default output device")
        v = ms.get_device_volume(dev)
        if v is not None:
            assert 0.0 <= v <= 1.0


# ── set_device_volume ─────────────────────────────────────────────────────────

class TestSetDeviceVolume:
    def test_noop_on_non_darwin(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        ms.set_device_volume("any device", 0.5)  # must not raise

    def test_noop_for_nonexistent_device(self):
        ms.set_device_volume("__nonexistent_device_xyz__", 0.5)  # must not raise

    def test_volume_clamped_no_exception_on_out_of_range(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        ms.set_device_volume("dev", -1.0)   # clamped to 0.0
        ms.set_device_volume("dev", 2.0)    # clamped to 1.0

    def test_roundtrip_restores_original_volume(self):
        import sys as _sys
        if _sys.platform != "darwin":
            pytest.skip("CoreAudio only on macOS")
        dev = ms._get_current_output_device()
        if dev is None:
            pytest.skip("no default output device")
        original = ms.get_device_volume(dev)
        if original is None:
            pytest.skip("device does not expose volume property")
        ms.set_device_volume(dev, original)  # set to same value — must not raise
        restored = ms.get_device_volume(dev)
        assert restored is not None
        assert abs(restored - original) < 0.02  # within 2% tolerance
