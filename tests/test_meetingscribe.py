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
        base = {"stt": {"funasr": {"model": "paraformer-zh", "workers": 2}}}
        over = {"stt": {"funasr": {"model": "paraformer-en"}}}
        r = ms._deep_merge(base, over)
        assert r["stt"]["funasr"]["model"] == "paraformer-en"
        assert r["stt"]["funasr"]["workers"] == 2

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
            '{"stt": {"funasr": {"model": "paraformer-en"}}}'
        )
        monkeypatch.setattr(ms, "CONFIG_FILE", tmp_path / "cfg.jsonc")
        monkeypatch.setattr(ms, "CONFIG_DIR", tmp_path)
        cfg = ms.load_config()
        assert cfg["stt"]["funasr"]["model"] == "paraformer-en"
        assert cfg["stt"]["funasr"]["workers"] == 4  # default preserved

    def test_no_file_deepcopy_does_not_mutate_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ms, "CONFIG_FILE", tmp_path / "missing.jsonc")
        monkeypatch.setattr(ms, "CONFIG_DIR", tmp_path)
        cfg = ms.load_config()
        cfg["stt"]["funasr"]["model"] = "MUTATED"
        assert ms.DEFAULT_CONFIG["stt"]["funasr"]["model"] == "paraformer-zh"

    def test_jsonc_comments_stripped_before_parse(self, tmp_path, monkeypatch):
        (tmp_path / "cfg.jsonc").write_text('// comment\n{"mode": "interview"}')
        monkeypatch.setattr(ms, "CONFIG_FILE", tmp_path / "cfg.jsonc")
        monkeypatch.setattr(ms, "CONFIG_DIR", tmp_path)
        assert ms.load_config()["mode"] == "interview"


# ── MultiStreamRecorder.save ──────────────────────────────────────────────────

class TestMultiStreamRecorderSave:
    def test_both_streams_produces_stereo_wav(self, tmp_path):
        rec = ms.MultiStreamRecorder(["sys", "mic"], 48000, role_labels=["system", "mic"])
        rec._frames = {
            "sys": [np.zeros((4800, 1), np.float32)],
            "mic": [np.zeros((4800, 1), np.float32)],
        }
        path = tmp_path / "out.wav"
        assert rec.save(path) is True
        with wave.open(str(path)) as wf:
            assert wf.getnchannels() == 2
            assert wf.getnframes() == 4800
        assert not (tmp_path / "out.warnings.txt").exists()

    def test_sys_only_still_produces_stereo(self, tmp_path):
        rec = ms.MultiStreamRecorder(["sys", "mic"], 48000, role_labels=["system", "mic"])
        rec._frames = {"sys": [np.zeros((4800, 1), np.float32)], "mic": []}
        path = tmp_path / "out.wav"
        rec.save(path)
        with wave.open(str(path)) as wf:
            assert wf.getnchannels() == 2

    def test_mic_only_still_produces_stereo(self, tmp_path):
        rec = ms.MultiStreamRecorder(["sys", "mic"], 48000, role_labels=["system", "mic"])
        rec._frames = {"sys": [], "mic": [np.zeros((4800, 1), np.float32)]}
        path = tmp_path / "out.wav"
        rec.save(path)
        with wave.open(str(path)) as wf:
            assert wf.getnchannels() == 2

    def test_no_frames_returns_false(self, tmp_path):
        rec = ms.MultiStreamRecorder(["sys", "mic"], 48000, role_labels=["system", "mic"])
        rec._frames = {"sys": [], "mic": []}
        assert rec.save(tmp_path / "out.wav") is False

    def test_audio_clipped_to_int16_range(self, tmp_path):
        rec = ms.MultiStreamRecorder(["sys", "mic"], 48000, role_labels=["system", "mic"])
        rec._frames = {
            "sys": [np.full((100, 1), 2.0, np.float32)],
            "mic": [np.full((100, 1), 2.0, np.float32)],
        }
        path = tmp_path / "out.wav"
        rec.save(path)
        with wave.open(str(path)) as wf:
            samples = np.frombuffer(wf.readframes(100), dtype=np.int16)
        assert samples.max() == 32767

    def test_unequal_stream_lengths_truncated_to_shorter(self, tmp_path):
        rec = ms.MultiStreamRecorder(["sys", "mic"], 48000, role_labels=["system", "mic"])
        rec._frames = {
            "sys": [np.zeros((4800, 1), np.float32)],
            "mic": [np.zeros((3000, 1), np.float32)],
        }
        path = tmp_path / "out.wav"
        rec.save(path)
        with wave.open(str(path)) as wf:
            assert wf.getnframes() == 3000


# ── MultiStreamRecorder.stop ──────────────────────────────────────────────────

class TestMultiStreamRecorderStop:
    def test_streams_cleared_after_stop(self):
        rec = ms.MultiStreamRecorder(["sys", "mic"], 48000)
        rec._streams = {"sys": MagicMock(), "mic": MagicMock()}
        rec.recording = True
        rec.stop()
        assert rec._streams == {}
        assert rec.recording is False

    def test_stop_calls_stop_and_close_on_each_stream(self):
        rec = ms.MultiStreamRecorder(["sys", "mic"], 48000)
        sys_s, mic_s = MagicMock(), MagicMock()
        rec._streams = {"sys": sys_s, "mic": mic_s}
        rec.stop()
        sys_s.stop.assert_called_once()
        sys_s.close.assert_called_once()
        mic_s.stop.assert_called_once()
        mic_s.close.assert_called_once()

    def test_stop_is_idempotent(self):
        rec = ms.MultiStreamRecorder(["sys"], 48000)
        mock_s = MagicMock()
        rec._streams = {"sys": mock_s}
        rec.stop()
        rec.stop()  # second call: streams are gone, must not raise
        mock_s.stop.assert_called_once()

    def test_stop_exception_does_not_propagate(self):
        rec = ms.MultiStreamRecorder(["sys"], 48000)
        mock_s = MagicMock()
        mock_s.stop.side_effect = RuntimeError("already stopped")
        rec._streams = {"sys": mock_s}
        rec.stop()  # must not raise

    def test_close_called_even_when_stop_raises(self):
        rec = ms.MultiStreamRecorder(["sys"], 48000)
        mock_s = MagicMock()
        mock_s.stop.side_effect = RuntimeError("oops")
        rec._streams = {"sys": mock_s}
        rec.stop()
        mock_s.close.assert_called_once()


# ── MultiStreamRecorder stop race regression guard ────────────────────────────

class TestMultiStreamRecorderStopRace:
    """Regression tests for the bug where `_monitor_iteration`'s `_try_open`
    ran concurrently with `stop()`'s `_streams.clear()`, wiping captured
    frames via `_frames[device] = []` and emitting a false
    `system-audio-not-opened` warning. Fixed by (a) `stop()` joining the
    monitor thread before clearing streams and (b) `_try_open`'s defensive
    re-check of `self.recording` inside the lock."""

    def test_try_open_returns_false_when_not_recording(self):
        """Defensive guard inside _try_open: if `self.recording` flipped to
        False after the stream object was created but before the install,
        the stream must be discarded — NOT installed and NOT used to reset
        `self._frames[device]`."""
        rec = ms.MultiStreamRecorder(["BlackHole 2ch"], 48000)
        # Simulate captured audio already in _frames from a previous stream.
        rec._frames = {"BlackHole 2ch": [np.ones((480, 1), np.float32) for _ in range(5)]}
        rec.recording = False  # simulate "stop() already ran"

        mock_stream = MagicMock()
        with patch.object(ms.sd, "InputStream", return_value=mock_stream), \
             patch.object(ms, "_portaudio_lock"):
            # _portaudio_lock is a real RLock — we don't really need to patch
            # it but keeping the test isolated is fine.
            result = rec._try_open("BlackHole 2ch")

        assert result is False
        # Stream was created but NEVER installed into _streams.
        assert "BlackHole 2ch" not in rec._streams
        # _frames was NOT reset — the 5 pre-existing frames are still there.
        assert len(rec._frames["BlackHole 2ch"]) == 5
        # The orphan stream was stopped and closed.
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    def test_try_open_installs_when_recording_true(self):
        """The defensive guard MUST NOT break the happy path: when
        recording is True, _try_open installs the stream and resets the
        device's frame list normally."""
        rec = ms.MultiStreamRecorder(["mic"], 48000)
        rec.recording = True
        rec._frames = {"mic": [np.ones((480, 1), np.float32)]}

        mock_stream = MagicMock()
        with patch.object(ms.sd, "InputStream", return_value=mock_stream):
            result = rec._try_open("mic")

        assert result is True
        assert rec._streams["mic"] is mock_stream
        # Happy path resets _frames as before.
        assert rec._frames["mic"] == []

    def test_stop_joins_monitor_thread_before_clearing_streams(self):
        """stop() must join the monitor thread before clearing _streams.
        Verify by giving the recorder a real Thread that records the order
        of when it sees `self.recording` flip to False vs. when
        `self._streams` is empty."""
        rec = ms.MultiStreamRecorder(["mic"], 48000)
        # Pre-populate as if recording is in progress.
        rec.recording = True
        rec._streams = {"mic": MagicMock()}
        rec._frames = {"mic": [np.ones((480, 1), np.float32) for _ in range(3)]}

        observations: list[tuple[str, bool, bool]] = []

        def fake_monitor_target():
            # Loop until recording flips. Record what we observe at each tick.
            import time as _t
            for _ in range(20):  # cap at 1 s
                observations.append((
                    "tick",
                    rec.recording,
                    bool(rec._streams),
                ))
                if not rec.recording:
                    return
                _t.sleep(0.05)

        thread = ms.threading.Thread(target=fake_monitor_target, daemon=True)
        thread.start()
        rec._monitor_thread = thread

        # Let the monitor run a few ticks first.
        import time as _t
        _t.sleep(0.15)
        rec.stop()

        # Monitor thread must be done by the time stop() returns.
        assert not thread.is_alive(), "stop() did not join the monitor thread"
        assert rec._monitor_thread is None
        # No tick observed an empty _streams while recording was still True
        # (the race scenario we're guarding against).
        for tag, was_recording, had_streams in observations:
            if was_recording:
                # If recording is True, streams must not yet be cleared.
                assert had_streams, (
                    "race detected: monitor saw _streams cleared while "
                    "recording was still True"
                )

    def test_captured_frames_survive_concurrent_stop_and_try_open(self):
        """End-to-end regression of the symptom in the user's log:
        BlackHole's captured frames must survive even if a concurrent
        _try_open call races with stop()."""
        rec = ms.MultiStreamRecorder(["BlackHole 2ch", "External Microphone"], 48000)
        # Simulate 9 seconds of captured audio at 48 kHz / 100 ms blocks → 90 blocks.
        rec.recording = True
        rec._streams = {"BlackHole 2ch": MagicMock(), "External Microphone": MagicMock()}
        captured_blocks = 90
        rec._frames = {
            "BlackHole 2ch": [np.full((4800, 1), 0.5, np.float32) for _ in range(captured_blocks)],
            "External Microphone": [np.full((4800, 1), 0.3, np.float32) for _ in range(captured_blocks)],
        }

        # First flip recording to False (as stop() does), THEN attempt a
        # _try_open that would previously have reset _frames.
        rec.recording = False
        mock_new_stream = MagicMock()
        with patch.object(ms.sd, "InputStream", return_value=mock_new_stream):
            result = rec._try_open("BlackHole 2ch")

        assert result is False
        # Captured frames are PRESERVED.
        assert len(rec._frames["BlackHole 2ch"]) == captured_blocks
        # Each captured block still has the original signal (0.5).
        for block in rec._frames["BlackHole 2ch"]:
            assert np.allclose(block, 0.5)

    def test_user_log_scenario_no_false_warning_no_silent_blackhole(self):
        """Replays the user's exact 23:06:16-23:06:26 scenario in-memory and
        verifies save() does NOT emit `system-audio-not-opened` and the
        BlackHole channel has its full audio. Without the fix, this test
        would fail."""
        rec = ms.MultiStreamRecorder(
            ["BlackHole 2ch", "External Microphone"],
            48000,
            role_labels=["system", "mic"],
        )
        # Simulate the recording captured 9 seconds of audio (the user's
        # actual duration). BlackHole has detectable music; mic has voice.
        rec.recording = True
        captured_blocks = 90
        music_signal = np.full((4800, 1), 0.0973, np.float32)  # RMS from user's log
        rec._frames = {
            "BlackHole 2ch": [music_signal.copy() for _ in range(captured_blocks)],
            "External Microphone": [np.full((4800, 1), 0.002, np.float32) for _ in range(captured_blocks)],
        }
        rec._streams = {"BlackHole 2ch": MagicMock(), "External Microphone": MagicMock()}

        # Simulate the race: the monitor's _try_open fires for BlackHole
        # right when stop() is mid-flight. The new guard must reject it.
        rec.recording = False  # simulate stop()'s recording=False
        mock_orphan = MagicMock()
        with patch.object(ms.sd, "InputStream", return_value=mock_orphan):
            rec._try_open("BlackHole 2ch")

        # BlackHole's captured music is intact (this is the entire bug).
        assert len(rec._frames["BlackHole 2ch"]) == captured_blocks
        for block in rec._frames["BlackHole 2ch"]:
            rms = float(np.sqrt(np.mean(block ** 2)))
            assert rms > 0.05, f"BlackHole block RMS dropped to {rms} (frames were wiped)"

        # save() does NOT emit "system-audio-not-opened" because frames exist.
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "out.wav"
            rec.save(out_path)
            assert out_path.exists()
            assert "system-audio-not-opened" not in rec.warnings, (
                f"false warning emitted; rec.warnings={rec.warnings}"
            )


# ── MultiStreamRecorder mid-recording mic/system swap ─────────────────────────

class TestMultiStreamRecorderMicSwap:
    """Tests for the mid-recording role swap: when the resolver picks a
    different mic (e.g. USB headset plugged in), the recorder closes the old
    mic stream, opens the new one, and prepends the old frames into the new
    stream's frame list so audio continuity is preserved."""

    def _devices(self, names):
        return _qd([{"name": n, "in": 1, "out": 0} for n in names])

    def _plan(self, mic, sys_src="BlackHole 2ch"):
        return ms.AudioPlan(
            mic_name=mic,
            sys_source_name=sys_src,
            restore_output_name="MacBook Air Speakers",
        )

    def test_mic_swap_closes_old_opens_new(self):
        # Recording started with built-in mic; USB mic appears in the live
        # device list and resolver promotes it.
        rec = ms.MultiStreamRecorder(
            ["BlackHole 2ch", "MacBook Air Microphone"],
            48000,
            role_labels=["system", "mic"],
        )
        # Simulate streams already open from rec.start()
        builtin_mic_stream = MagicMock()
        sys_stream = MagicMock()
        rec._streams = {
            "BlackHole 2ch": sys_stream,
            "MacBook Air Microphone": builtin_mic_stream,
        }
        rec._frames = {
            "BlackHole 2ch": [np.zeros((480, 1), np.float32)],
            "MacBook Air Microphone": [np.zeros((480, 1), np.float32) for _ in range(3)],
        }
        rec.recording = True

        devices = self._devices(["BlackHole 2ch", "MacBook Air Microphone", "USB Audio CODEC"])
        new_plan = self._plan(mic="USB Audio CODEC")

        opened_devices: list[str] = []

        def fake_try_open(dev):
            opened_devices.append(dev)
            mock_stream = MagicMock()
            rec._streams[dev] = mock_stream
            rec._frames[dev] = []
            return True

        with patch.object(ms.sd, "query_devices", return_value=devices), \
             patch.object(ms, "resolve_audio_devices", return_value=new_plan), \
             patch.object(rec, "_try_open", side_effect=fake_try_open):
            rec._monitor_iteration()

        # The built-in mic stream was closed and removed.
        assert "MacBook Air Microphone" not in rec._streams
        builtin_mic_stream.stop.assert_called_once()
        builtin_mic_stream.close.assert_called_once()
        # The USB mic stream was opened.
        assert "USB Audio CODEC" in opened_devices
        assert "USB Audio CODEC" in rec._streams
        # The system-audio stream is untouched.
        sys_stream.stop.assert_not_called()
        sys_stream.close.assert_not_called()
        # `wanted` reflects the new plan and role labels stay aligned.
        assert rec.wanted == ["BlackHole 2ch", "USB Audio CODEC"]
        assert rec.role_labels == ["system", "mic"]

    def test_mic_swap_carries_old_frames_into_new_device(self):
        """Audio continuity: pre-swap mic frames must be prepended to the new
        mic device's frame list so they're included when save() iterates
        wanted devices."""
        rec = ms.MultiStreamRecorder(
            ["BlackHole 2ch", "MacBook Air Microphone"],
            48000,
            role_labels=["system", "mic"],
        )
        pre_swap_frames = [np.full((480, 1), 0.1, np.float32) for _ in range(5)]
        rec._streams = {
            "BlackHole 2ch": MagicMock(),
            "MacBook Air Microphone": MagicMock(),
        }
        rec._frames = {
            "BlackHole 2ch": [np.zeros((480, 1), np.float32)],
            "MacBook Air Microphone": list(pre_swap_frames),  # copy
        }
        rec.recording = True

        devices = self._devices(["BlackHole 2ch", "MacBook Air Microphone", "USB Audio CODEC"])
        new_plan = self._plan(mic="USB Audio CODEC")

        def fake_try_open(dev):
            rec._streams[dev] = MagicMock()
            rec._frames[dev] = []  # mirrors the real _try_open behaviour
            return True

        with patch.object(ms.sd, "query_devices", return_value=devices), \
             patch.object(ms, "resolve_audio_devices", return_value=new_plan), \
             patch.object(rec, "_try_open", side_effect=fake_try_open):
            rec._monitor_iteration()

        # USB Audio CODEC's frame list begins with the 5 pre-swap frames.
        carried = rec._frames["USB Audio CODEC"]
        assert len(carried) == 5
        for original, carried_frame in zip(pre_swap_frames, carried):
            assert np.array_equal(original, carried_frame)
        # The old device's frame entry is gone (popped during the carry).
        assert "MacBook Air Microphone" not in rec._frames

    def test_system_source_unchanged_across_mic_swap(self):
        """The BlackHole 2ch stream must NOT be reopened by a mic swap."""
        rec = ms.MultiStreamRecorder(
            ["BlackHole 2ch", "MacBook Air Microphone"],
            48000,
            role_labels=["system", "mic"],
        )
        sys_stream_original = MagicMock()
        rec._streams = {
            "BlackHole 2ch": sys_stream_original,
            "MacBook Air Microphone": MagicMock(),
        }
        rec._frames = {"BlackHole 2ch": [], "MacBook Air Microphone": []}
        rec.recording = True

        devices = self._devices(["BlackHole 2ch", "MacBook Air Microphone", "USB Audio CODEC"])
        new_plan = self._plan(mic="USB Audio CODEC")

        def fake_try_open(dev):
            rec._streams[dev] = MagicMock()
            rec._frames[dev] = []
            return True

        with patch.object(ms.sd, "query_devices", return_value=devices), \
             patch.object(ms, "resolve_audio_devices", return_value=new_plan), \
             patch.object(rec, "_try_open", side_effect=fake_try_open):
            rec._monitor_iteration()

        # Same object as before — never replaced.
        assert rec._streams["BlackHole 2ch"] is sys_stream_original
        sys_stream_original.stop.assert_not_called()
        sys_stream_original.close.assert_not_called()

    def test_no_swap_when_plan_unchanged(self):
        """Resolver returns the same plan → no streams open/close, no
        `wanted` mutation, no carry."""
        rec = ms.MultiStreamRecorder(
            ["BlackHole 2ch", "MacBook Air Microphone"],
            48000,
            role_labels=["system", "mic"],
        )
        sys_stream = MagicMock()
        mic_stream = MagicMock()
        rec._streams = {"BlackHole 2ch": sys_stream, "MacBook Air Microphone": mic_stream}
        rec._frames = {"BlackHole 2ch": [], "MacBook Air Microphone": []}
        rec.recording = True

        devices = self._devices(["BlackHole 2ch", "MacBook Air Microphone"])
        same_plan = self._plan(mic="MacBook Air Microphone")

        with patch.object(ms.sd, "query_devices", return_value=devices), \
             patch.object(ms, "resolve_audio_devices", return_value=same_plan), \
             patch.object(rec, "_try_open") as try_open:
            rec._monitor_iteration()

        try_open.assert_not_called()
        sys_stream.stop.assert_not_called()
        mic_stream.stop.assert_not_called()
        assert rec.wanted == ["BlackHole 2ch", "MacBook Air Microphone"]


# ── resolve_audio_devices + restore policy ────────────────────────────────────

def _qd(items):
    """Helper: build a sd.query_devices()-style list from short dicts."""
    out = []
    for it in items:
        out.append({
            "name": it["name"],
            "max_input_channels": it.get("in", 0),
            "max_output_channels": it.get("out", 0),
        })
    return out


class TestResolveAudioDevices:
    def test_prefers_external_mic_over_builtin(self):
        devices = _qd([
            {"name": "MacBook Air Microphone", "in": 1, "out": 0},
            {"name": "USB Audio CODEC", "in": 1, "out": 2},
            {"name": "MacBook Air Speakers", "in": 0, "out": 2},
        ])
        transport = {
            "MacBook Air Microphone": "bltn",
            "USB Audio CODEC": "usb ",
            "MacBook Air Speakers": "bltn",
        }
        with patch.object(ms.sd, "query_devices", return_value=devices), \
             patch.object(ms, "_coreaudio_device_info", return_value=transport), \
             patch.object(ms, "_get_system_output_device", return_value="MacBook Air Speakers"), \
             patch.object(ms, "_get_current_output_device", return_value="MacBook Air Speakers"):
            plan = ms.resolve_audio_devices()
        assert plan.mic_name == "USB Audio CODEC"

    def test_restore_target_excludes_aggregate(self):
        # User's current default is the Multi-Output Device (aggregate). Restore
        # must walk to the first physical device by transport priority.
        devices = _qd([
            {"name": "BlackHole 2ch", "in": 2, "out": 2},
            {"name": "MacBook Air Microphone", "in": 1, "out": 0},
            {"name": "MacBook Air Speakers", "in": 0, "out": 2},
            {"name": "External Headphones", "in": 0, "out": 2},
            {"name": "多输出设备", "in": 0, "out": 2},
        ])
        transport = {
            "BlackHole 2ch": "virt",
            "MacBook Air Microphone": "bltn",
            "MacBook Air Speakers": "bltn",
            "External Headphones": "usb ",
            "多输出设备": "aggt",
        }
        with patch.object(ms.sd, "query_devices", return_value=devices), \
             patch.object(ms, "_coreaudio_device_info", return_value=transport), \
             patch.object(ms, "_get_system_output_device", return_value="多输出设备"), \
             patch.object(ms, "_get_current_output_device", return_value="多输出设备"):
            plan = ms.resolve_audio_devices()
        # Restore target must be the external headphones (priority 0), not the
        # aggregate (rejected) and not the built-in speakers (priority 1).
        assert plan.restore_output_name == "External Headphones"
        assert plan.is_external_output is True
        assert plan.multi_output_name == "多输出设备"
        assert plan.sys_source_name == "BlackHole 2ch"

    def test_only_builtin_devices(self):
        devices = _qd([
            {"name": "BlackHole 2ch", "in": 2, "out": 2},
            {"name": "MacBook Air Microphone", "in": 1, "out": 0},
            {"name": "MacBook Air Speakers", "in": 0, "out": 2},
        ])
        transport = {
            "BlackHole 2ch": "virt",
            "MacBook Air Microphone": "bltn",
            "MacBook Air Speakers": "bltn",
        }
        with patch.object(ms.sd, "query_devices", return_value=devices), \
             patch.object(ms, "_coreaudio_device_info", return_value=transport), \
             patch.object(ms, "_get_system_output_device", return_value="MacBook Air Speakers"), \
             patch.object(ms, "_get_current_output_device", return_value="MacBook Air Speakers"):
            plan = ms.resolve_audio_devices()
        assert plan.mic_name == "MacBook Air Microphone"
        assert plan.restore_output_name == "MacBook Air Speakers"
        assert plan.is_external_output is False
        assert "no-multi-output-device" in plan.warnings


class TestRestoreOutputIfNeeded:
    def setup_method(self):
        # The default reason "post-recording" is gated on _recording_did_switch.
        # These tests exercise the underlying dOut/sOut restore mechanics, so
        # we simulate "a recording-start switch happened" by setting the flag.
        # The new TestRestoreOutputGate class covers the gate itself separately.
        ms._recording_did_switch.set()

    def teardown_method(self):
        ms._recording_did_switch.clear()

    def test_skips_both_when_both_already_physical(self):
        # dOut and sOut both pointing at the same physical device → complete no-op.
        plan = ms.AudioPlan(restore_output_name="MacBook Air Speakers")
        with patch.object(ms, "_coreaudio_device_info", return_value={"MacBook Air Speakers": "bltn"}), \
             patch.object(ms, "_get_current_output_device", return_value="MacBook Air Speakers"), \
             patch.object(ms, "_get_system_output_device", return_value="MacBook Air Speakers"), \
             patch.object(ms, "switch_output") as sw_d, \
             patch.object(ms, "switch_system_output") as sw_s:
            result = ms._restore_output_if_needed(plan)
        sw_d.assert_not_called()
        sw_s.assert_not_called()
        assert result == "MacBook Air Speakers"

    def test_writes_only_sout_when_dout_already_physical(self):
        # Common bug case: dOut is back on the speakers but sOut is still the
        # Multi-Output Device, so hardware volume keys don't work. Fix sOut only.
        plan = ms.AudioPlan(restore_output_name="MacBook Air Speakers")
        devices = _qd([
            {"name": "MacBook Air Speakers", "in": 0, "out": 2},
            {"name": "多输出设备", "in": 0, "out": 2},
        ])
        transport = {"MacBook Air Speakers": "bltn", "多输出设备": "aggt"}
        with patch.object(ms, "_coreaudio_device_info", return_value=transport), \
             patch.object(ms, "_get_current_output_device", return_value="MacBook Air Speakers"), \
             patch.object(ms, "_get_system_output_device", return_value="多输出设备"), \
             patch.object(ms.sd, "query_devices", return_value=devices), \
             patch.object(ms, "switch_output") as sw_d, \
             patch.object(ms, "switch_system_output") as sw_s:
            result = ms._restore_output_if_needed(plan)
        sw_d.assert_not_called()
        sw_s.assert_called_once_with("MacBook Air Speakers")
        assert result == "MacBook Air Speakers"

    def test_writes_both_when_both_non_physical(self):
        # Both selectors point at the Multi-Output Device → write the same physical
        # target to both, so media and volume keys land on the same device.
        plan = ms.AudioPlan(restore_output_name="External Headphones")
        devices = _qd([
            {"name": "MacBook Air Speakers", "in": 0, "out": 2},
            {"name": "External Headphones", "in": 0, "out": 2},
            {"name": "多输出设备", "in": 0, "out": 2},
        ])
        transport = {
            "MacBook Air Speakers": "bltn",
            "External Headphones": "usb ",
            "多输出设备": "aggt",
        }
        with patch.object(ms, "_coreaudio_device_info", return_value=transport), \
             patch.object(ms, "_get_current_output_device", return_value="多输出设备"), \
             patch.object(ms, "_get_system_output_device", return_value="多输出设备"), \
             patch.object(ms.sd, "query_devices", return_value=devices), \
             patch.object(ms, "switch_output") as sw_d, \
             patch.object(ms, "switch_system_output") as sw_s:
            result = ms._restore_output_if_needed(plan)
        sw_d.assert_called_once_with("External Headphones")
        sw_s.assert_called_once_with("External Headphones")
        assert result == "External Headphones"

    def test_falls_back_when_target_missing(self):
        # Resolver picked External Headphones but they've been unplugged before
        # the actual switch — must walk priority list and land on the built-in.
        # Verifies the fallback target is written to BOTH dOut and sOut.
        plan = ms.AudioPlan(restore_output_name="External Headphones")
        devices = _qd([
            {"name": "MacBook Air Speakers", "in": 0, "out": 2},
            {"name": "多输出设备", "in": 0, "out": 2},
        ])
        transport = {"MacBook Air Speakers": "bltn", "多输出设备": "aggt"}
        with patch.object(ms, "_coreaudio_device_info", return_value=transport), \
             patch.object(ms, "_get_current_output_device", return_value="多输出设备"), \
             patch.object(ms, "_get_system_output_device", return_value="多输出设备"), \
             patch.object(ms.sd, "query_devices", return_value=devices), \
             patch.object(ms, "switch_output") as sw_d, \
             patch.object(ms, "switch_system_output") as sw_s:
            result = ms._restore_output_if_needed(plan)
        sw_d.assert_called_once_with("MacBook Air Speakers")
        sw_s.assert_called_once_with("MacBook Air Speakers")
        assert result == "MacBook Air Speakers"


# ── Post-recording restore gate ───────────────────────────────────────────────

class TestRestoreOutputGate:
    """The post-recording gate on `_restore_output_if_needed`: when the start
    path didn't switch dOut to the Multi-Output Device, the stop path must be
    a no-op so music apps are not paused unnecessarily."""

    def setup_method(self):
        ms._recording_did_switch.clear()

    def teardown_method(self):
        ms._recording_did_switch.clear()

    def _plan(self):
        return ms.AudioPlan(restore_output_name="MacBook Air Speakers")

    def test_post_recording_with_flag_clear_returns_none_no_writes(self):
        with patch.object(ms, "_coreaudio_device_info") as ca, \
             patch.object(ms, "_get_current_output_device") as gdo, \
             patch.object(ms, "_get_system_output_device") as gso, \
             patch.object(ms, "switch_output") as sw_d, \
             patch.object(ms, "switch_system_output") as sw_s:
            result = ms._restore_output_if_needed(self._plan(), reason="post-recording")
        assert result is None
        # When gated out, no CoreAudio queries even run.
        ca.assert_not_called()
        gdo.assert_not_called()
        gso.assert_not_called()
        sw_d.assert_not_called()
        sw_s.assert_not_called()

    def test_post_recording_with_flag_set_runs_existing_evaluation(self):
        ms._recording_did_switch.set()
        # dOut and sOut both on the Multi-Output Device → both halves get written.
        plan = ms.AudioPlan(restore_output_name="MacBook Air Speakers")
        devices = _qd([
            {"name": "MacBook Air Speakers", "in": 0, "out": 2},
            {"name": "多输出设备", "in": 0, "out": 2},
        ])
        transport = {"MacBook Air Speakers": "bltn", "多输出设备": "aggt"}
        with patch.object(ms, "_coreaudio_device_info", return_value=transport), \
             patch.object(ms, "_get_current_output_device", return_value="多输出设备"), \
             patch.object(ms, "_get_system_output_device", return_value="多输出设备"), \
             patch.object(ms.sd, "query_devices", return_value=devices), \
             patch.object(ms, "switch_output") as sw_d, \
             patch.object(ms, "switch_system_output") as sw_s:
            result = ms._restore_output_if_needed(plan, reason="post-recording")
        sw_d.assert_called_once_with("MacBook Air Speakers")
        sw_s.assert_called_once_with("MacBook Air Speakers")
        assert result == "MacBook Air Speakers"

    def test_idle_event_bypasses_gate(self):
        # Flag is clear (no recording happened) but an idle-event restore must
        # still run when dOut needs alignment.
        plan = ms.AudioPlan(restore_output_name="External Headphones")
        devices = _qd([
            {"name": "External Headphones", "in": 0, "out": 2},
            {"name": "多输出设备", "in": 0, "out": 2},
        ])
        transport = {"External Headphones": "usb ", "多输出设备": "aggt"}
        with patch.object(ms, "_coreaudio_device_info", return_value=transport), \
             patch.object(ms, "_get_current_output_device", return_value="多输出设备"), \
             patch.object(ms, "_get_system_output_device", return_value="多输出设备"), \
             patch.object(ms.sd, "query_devices", return_value=devices), \
             patch.object(ms, "switch_output") as sw_d, \
             patch.object(ms, "switch_system_output") as sw_s:
            result = ms._restore_output_if_needed(plan, reason="idle-event")
        assert result == "External Headphones"
        sw_d.assert_called_once_with("External Headphones")
        sw_s.assert_called_once_with("External Headphones")

    def test_unknown_reason_logs_warn_and_gates_as_post_recording(self):
        log_buf = io.StringIO()
        with patch.object(ms, "_log_file_handle", log_buf):
            result = ms._restore_output_if_needed(self._plan(), reason="bogus")
        assert result is None
        log = log_buf.getvalue()
        assert "[WARN]" in log and "unknown reason='bogus'" in log

    def test_recorder_start_does_not_touch_did_switch_flag(self):
        """The flag is now cleared at the entry of the recording-start
        lifecycle (cmd_record / cmd_ui), NOT in MultiStreamRecorder.start().
        Calling rec.start() must leave the flag's existing value alone, so a
        start-path switch_output(...) that set the flag before constructing
        the recorder survives into the stop-time gate check."""
        ms._recording_did_switch.set()  # simulate start-path having performed a switch
        rec = ms.MultiStreamRecorder(["dummy"], 48000)
        with patch.object(rec, "_try_open", return_value=False), \
             patch.object(ms.sd, "query_devices", return_value=_qd([{"name": "dummy", "in": 1, "out": 0}])), \
             patch.object(ms.threading, "Thread") as thread_cls:
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False
            thread_cls.return_value = mock_thread
            rec.start()
        # The flag SURVIVES MultiStreamRecorder.start() — that's the whole fix.
        assert ms._recording_did_switch.is_set()
        rec.stop()


# ── MultiStreamRecorder warnings + sidecar ────────────────────────────────────

class TestMultiStreamRecorderWarnings:
    def test_emits_mic_warning_on_failure_and_writes_sidecar(self, tmp_path):
        rec = ms.MultiStreamRecorder(["sys", "mic"], 48000, role_labels=["system", "mic"])
        seen: list[str] = []
        rec.on_warning = seen.append
        # System captured frames; mic captured nothing → mic-not-opened
        rec._frames = {"sys": [np.zeros((4800, 1), np.float32)], "mic": []}
        path = tmp_path / "out.wav"
        assert rec.save(path) is True
        assert "mic-not-opened" in seen
        assert "mic-not-opened" in rec.warnings
        sidecar = tmp_path / "out.warnings.txt"
        assert sidecar.exists()
        assert "mic-not-opened" in sidecar.read_text()

    def test_emits_system_warning_on_mic_only_recording(self, tmp_path):
        rec = ms.MultiStreamRecorder(["sys", "mic"], 48000, role_labels=["system", "mic"])
        seen: list[str] = []
        rec.on_warning = seen.append
        rec._frames = {"sys": [], "mic": [np.zeros((4800, 1), np.float32)]}
        path = tmp_path / "out.wav"
        rec.save(path)
        assert "system-audio-not-opened" in seen

    def test_no_warning_no_sidecar_on_clean_recording(self, tmp_path):
        rec = ms.MultiStreamRecorder(["sys", "mic"], 48000, role_labels=["system", "mic"])
        rec._frames = {
            "sys": [np.zeros((4800, 1), np.float32)],
            "mic": [np.zeros((4800, 1), np.float32)],
        }
        path = tmp_path / "out.wav"
        rec.save(path)
        assert rec.warnings == []
        assert not (tmp_path / "out.warnings.txt").exists()

    def test_warning_deduplicated(self):
        rec = ms.MultiStreamRecorder(["mic"], 48000, role_labels=["mic"])
        seen: list[str] = []
        rec.on_warning = seen.append
        rec._emit_warning("mic-not-opened")
        rec._emit_warning("mic-not-opened")  # duplicate must be a no-op
        assert seen == ["mic-not-opened"]
        assert rec.warnings == ["mic-not-opened"]


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
    def test_writes_meeting_md_by_default(self, tmp_path):
        wav = tmp_path / "20260101_120000.wav"
        out = ms.save_minutes("# Notes", wav)
        # Default mode='meeting' → <stem>.meeting.md (interview mode produces
        # <stem>.interview.md). The mode-tagged suffix lets the same .wav
        # carry both a meeting summary AND an interview report side by side.
        assert out == tmp_path / "20260101_120000.meeting.md"
        assert out.read_text(encoding="utf-8") == "# Notes"

    def test_writes_interview_md_for_interview_mode(self, tmp_path):
        wav = tmp_path / "20260101_120000.wav"
        out = ms.save_minutes("# Report", wav, mode="interview")
        assert out == tmp_path / "20260101_120000.interview.md"
        assert out.read_text(encoding="utf-8") == "# Report"

    def test_overwrites_existing_md_file(self, tmp_path):
        wav = tmp_path / "test.wav"
        ms.save_minutes("first", wav)
        ms.save_minutes("second", wav)
        assert (tmp_path / "test.meeting.md").read_text(encoding="utf-8") == "second"

    def test_unknown_mode_falls_back_to_meeting_suffix(self, tmp_path):
        wav = tmp_path / "x.wav"
        out = ms.save_minutes("body", wav, mode="bogus")
        assert out.name == "x.meeting.md"


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


# ── _log helper ───────────────────────────────────────────────────────────────

import re as _re


def _capture_log(category, message, console=False):
    """Drive _log() against a StringIO and return what was written to the file
    and (if console=True) to stderr."""
    buf = io.StringIO()
    with patch.object(ms, "_log_file_handle", buf), \
         patch.object(ms, "_LOG_TO_CONSOLE", console), \
         patch.object(sys, "stderr", io.StringIO()) as stderr_buf:
        ms._log(category, message)
    return buf.getvalue(), stderr_buf.getvalue()


class TestLog:
    def test_writes_timestamped_category_line_to_file(self):
        file_text, stderr_text = _capture_log("DEVICE", "test message")
        assert _re.match(r"^\[\d\d:\d\d:\d\d\] \[DEVICE\] test message\n$", file_text)
        assert stderr_text == ""

    def test_no_op_when_handle_is_none(self, capsys):
        with patch.object(ms, "_log_file_handle", None), \
             patch.object(ms, "_LOG_TO_CONSOLE", False):
            ms._log("REC", "should be silent")
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""

    def test_console_mirror_when_LOG_TO_CONSOLE_set(self):
        file_text, stderr_text = _capture_log("AUDIO", "switched", console=True)
        assert "[AUDIO] switched" in file_text
        assert "[AUDIO] switched" in stderr_text

    def test_concurrent_writes_are_serialised(self):
        import threading as _t
        buf = io.StringIO()
        with patch.object(ms, "_log_file_handle", buf), \
             patch.object(ms, "_LOG_TO_CONSOLE", False):
            threads = [
                _t.Thread(target=lambda i=i: [ms._log("REC", f"line-{i}-{j}") for j in range(10)])
                for i in range(20)
            ]
            for t in threads: t.start()
            for t in threads: t.join()
        lines = [ln for ln in buf.getvalue().splitlines() if ln]
        assert len(lines) == 200
        for ln in lines:
            assert _re.match(r"^\[\d\d:\d\d:\d\d\] \[REC\] line-\d+-\d+$", ln)

    def test_log_does_not_raise_when_handle_write_fails(self):
        # Logging itself must never raise — caller paths can't be polluted by
        # broken handles.
        bad = MagicMock()
        bad.write.side_effect = OSError("disk full")
        with patch.object(ms, "_log_file_handle", bad), \
             patch.object(ms, "_LOG_TO_CONSOLE", False):
            ms._log("ERR", "should not raise")


# ── Resolver decision logging ─────────────────────────────────────────────────

class TestResolverLogsDecision:
    def test_resolver_emits_plan_line(self):
        devices = _qd([
            {"name": "BlackHole 2ch", "in": 2, "out": 2},
            {"name": "USB Mic", "in": 1, "out": 0},
            {"name": "MacBook Air Speakers", "in": 0, "out": 2},
            {"name": "多输出设备", "in": 0, "out": 2},
        ])
        transport = {
            "BlackHole 2ch": "virt",
            "USB Mic": "usb ",
            "MacBook Air Speakers": "bltn",
            "多输出设备": "aggt",
        }
        buf = io.StringIO()
        with patch.object(ms.sd, "query_devices", return_value=devices), \
             patch.object(ms, "_coreaudio_device_info", return_value=transport), \
             patch.object(ms, "_get_system_output_device", return_value="MacBook Air Speakers"), \
             patch.object(ms, "_get_current_output_device", return_value="MacBook Air Speakers"), \
             patch.object(ms, "_log_file_handle", buf):
            ms.resolve_audio_devices()
        log = buf.getvalue()
        assert "[DEVICE] plan:" in log
        assert "mic='USB Mic'" in log
        assert "restore='MacBook Air Speakers'" in log
        assert "multi='多输出设备'" in log


# ── Restore policy logging ────────────────────────────────────────────────────

class TestRestoreLogs:
    def setup_method(self):
        # Same rationale as TestRestoreOutputIfNeeded — these tests check the
        # internal log lines emitted by the per-property restore, so the
        # post-recording gate must be open.
        ms._recording_did_switch.set()

    def teardown_method(self):
        ms._recording_did_switch.clear()

    def _run(self, plan, dout, sout, transport, devices):
        buf = io.StringIO()
        with patch.object(ms, "_coreaudio_device_info", return_value=transport), \
             patch.object(ms, "_get_current_output_device", return_value=dout), \
             patch.object(ms, "_get_system_output_device", return_value=sout), \
             patch.object(ms.sd, "query_devices", return_value=devices), \
             patch.object(ms, "switch_output"), \
             patch.object(ms, "switch_system_output"), \
             patch.object(ms, "_log_file_handle", buf):
            ms._restore_output_if_needed(plan)
        return buf.getvalue()

    def test_logs_both_physical_no_op(self):
        plan = ms.AudioPlan(restore_output_name="MacBook Air Speakers")
        log = self._run(
            plan,
            dout="MacBook Air Speakers", sout="MacBook Air Speakers",
            transport={"MacBook Air Speakers": "bltn"},
            devices=_qd([{"name": "MacBook Air Speakers", "in": 0, "out": 2}]),
        )
        assert "[RESTORE]" in log
        assert "no-op" in log
        # No [AUDIO] dOut/sOut → ... line should appear in no-op path.
        assert "dOut →" not in log
        assert "sOut →" not in log

    def test_logs_per_property_writes(self):
        plan = ms.AudioPlan(restore_output_name="External Headphones")
        log = self._run(
            plan,
            dout="多输出设备", sout="多输出设备",
            transport={
                "External Headphones": "usb ",
                "MacBook Air Speakers": "bltn",
                "多输出设备": "aggt",
            },
            devices=_qd([
                {"name": "External Headphones", "in": 0, "out": 2},
                {"name": "MacBook Air Speakers", "in": 0, "out": 2},
                {"name": "多输出设备", "in": 0, "out": 2},
            ]),
        )
        assert "[AUDIO] dOut → 'External Headphones'" in log
        assert "[AUDIO] sOut → 'External Headphones'" in log


# ── Stream lifecycle logging ──────────────────────────────────────────────────

class TestStreamLifecycleLogs:
    def test_close_one_logs_with_reason(self):
        rec = ms.MultiStreamRecorder(["sys"], 48000, role_labels=["system"])
        mock_stream = MagicMock()
        rec._streams = {"sys": mock_stream}
        buf = io.StringIO()
        with patch.object(ms, "_log_file_handle", buf):
            rec._close_one("sys", reason="disappeared")
        assert "[STREAM] closed device='sys' reason=disappeared" in buf.getvalue()

    def test_save_logs_sidecar_creation(self, tmp_path):
        rec = ms.MultiStreamRecorder(["sys", "mic"], 48000, role_labels=["system", "mic"])
        rec._frames = {"sys": [np.zeros((4800, 1), np.float32)], "mic": []}
        buf = io.StringIO()
        with patch.object(ms, "_log_file_handle", buf):
            rec.save(tmp_path / "out.wav")
        log = buf.getvalue()
        assert "[REC] sidecar written" in log
        assert "[WARN] recorder: mic-not-opened" in log


# ── AudioDeviceMonitor ────────────────────────────────────────────────────────

class TestAudioDeviceMonitor:
    """Unit tests for the 1 Hz dedicated audio-device monitor thread.

    Tests use `tick_once()` (the public test seam) to drive the monitor body
    deterministically, bypassing the real 1 s sleep loop. The
    `_recording_active` module event is toggled directly to simulate idle vs.
    recording state.
    """

    @staticmethod
    def _plan(restore="Speakers", mic="Mic", sys_src="BlackHole"):
        return ms.AudioPlan(
            mic_name=mic,
            sys_source_name=sys_src,
            restore_output_name=restore,
        )

    def setup_method(self):
        # Ensure each test starts in the idle branch.
        ms._recording_active.clear()
        ms._hotplug_event.clear()

    def test_first_idle_tick_always_restores(self):
        plan = self._plan()
        mon = ms.AudioDeviceMonitor()
        with patch.object(ms, "resolve_audio_devices", return_value=plan) as resolver, \
             patch.object(ms, "_restore_output_if_needed", return_value="Speakers") as restore:
            mon.tick_once()
        # Idle-driven restore bypasses the post-recording did-switch gate
        # by passing reason="idle-event".
        restore.assert_called_once_with(plan, reason="idle-event")
        # Idle ticks MUST refresh the PortAudio cache so newly-attached devices
        # are picked up within 1 s.
        resolver.assert_called_once_with(query_fresh=True)

    def test_second_idle_tick_with_same_triple_skips_restore(self):
        plan = self._plan()
        mon = ms.AudioDeviceMonitor()
        with patch.object(ms, "resolve_audio_devices", return_value=plan), \
             patch.object(ms, "_restore_output_if_needed", return_value="Speakers") as restore:
            mon.tick_once()
            mon.tick_once()
        assert restore.call_count == 1

    def test_idle_tick_with_changed_output_target_re_restores(self):
        plans = [self._plan(restore="Speakers"), self._plan(restore="Headphones")]
        mon = ms.AudioDeviceMonitor()
        with patch.object(ms, "resolve_audio_devices", side_effect=plans), \
             patch.object(ms, "_restore_output_if_needed", return_value=None) as restore:
            mon.tick_once()
            mon.tick_once()
        assert restore.call_count == 2

    def test_idle_tick_with_changed_mic_re_restores(self):
        plans = [self._plan(mic="Built-in Mic"), self._plan(mic="USB Mic")]
        mon = ms.AudioDeviceMonitor()
        with patch.object(ms, "resolve_audio_devices", side_effect=plans), \
             patch.object(ms, "_restore_output_if_needed", return_value=None) as restore:
            mon.tick_once()
            mon.tick_once()
        assert restore.call_count == 2

    def test_idle_tick_with_changed_sys_source_re_restores(self):
        plans = [self._plan(sys_src="BlackHole 2ch"), self._plan(sys_src="多输出设备")]
        mon = ms.AudioDeviceMonitor()
        with patch.object(ms, "resolve_audio_devices", side_effect=plans), \
             patch.object(ms, "_restore_output_if_needed", return_value=None) as restore:
            mon.tick_once()
            mon.tick_once()
        assert restore.call_count == 2

    def test_recording_branch_fires_on_recording_plan_change_callback(self):
        """When the recording-branch detects a plan change, the optional
        on_recording_plan_change callback fires with the new plan. The GUI
        uses this to rebind the volume slider to the user's current physical
        output after a mid-recording hotplug."""
        mon = ms.AudioDeviceMonitor()
        received: list = []
        mon.on_recording_plan_change = lambda plan: received.append(plan)
        ms._recording_active.set()
        try:
            plan_a = self._plan(restore="MacBook Air Speakers")
            plan_b = self._plan(restore="External Headphones")
            with patch.object(ms, "resolve_audio_devices",
                              side_effect=[plan_a, plan_a, plan_b]), \
                 patch.object(ms, "_reconcile_recording_mutes"):
                mon.tick_once()
                mon.tick_once()  # same plan: callback should NOT fire again
                mon.tick_once()  # new plan: callback fires
        finally:
            ms._recording_active.clear()
            mon._prev_mute_triple = None
        assert len(received) == 2, (
            f"expected callback to fire once per distinct plan, got {len(received)}"
        )
        assert received[0].restore_output_name == "MacBook Air Speakers"
        assert received[1].restore_output_name == "External Headphones"

    def test_recording_branch_callback_exception_is_logged_not_raised(self):
        """A misbehaving callback MUST NOT crash the monitor thread."""
        mon = ms.AudioDeviceMonitor()
        mon.on_recording_plan_change = lambda plan: (_ for _ in ()).throw(
            RuntimeError("boom"))
        ms._recording_active.set()
        try:
            with patch.object(ms, "resolve_audio_devices",
                              return_value=self._plan()), \
                 patch.object(ms, "_reconcile_recording_mutes"):
                # Must not raise
                mon.tick_once()
        finally:
            ms._recording_active.clear()
            mon._prev_mute_triple = None

    def test_recording_branch_reconciles_mutes_but_no_dout_writes(self):
        """During recording the monitor MUST NOT switch dOut or call the
        post-recording restore — those would pause music apps. It MAY call
        _reconcile_recording_mutes to keep Multi-Output sub-device mutes in
        sync with hotplug events (e.g. headphones plugged in mid-recording).
        And it does call resolve_audio_devices(query_fresh=False) to detect
        plan changes (query_fresh=False is critical: streams are open so
        terminating PortAudio would break the recording)."""
        mon = ms.AudioDeviceMonitor()
        ms._recording_active.set()
        try:
            fake_plan = self._plan()
            with patch.object(ms, "resolve_audio_devices",
                              return_value=fake_plan) as resolver, \
                 patch.object(ms, "_restore_output_if_needed") as restore, \
                 patch.object(ms, "_reconcile_recording_mutes") as reconcile, \
                 patch.object(ms, "switch_output") as sw_d, \
                 patch.object(ms, "switch_system_output") as sw_s, \
                 patch.object(ms, "_refresh_portaudio") as refresh:
                mon.tick_once()
                # Second tick with identical plan: reconcile must NOT be
                # invoked again (mute_triple memoization).
                mon.tick_once()
            # The recorder's _monitor is the authoritative input-stream
            # reconciler, so we must not switch dOut or trigger a restore.
            restore.assert_not_called()
            sw_d.assert_not_called()
            sw_s.assert_not_called()
            # The resolver IS called, but only with query_fresh=False so we
            # never terminate PortAudio while streams are open.
            for call in resolver.call_args_list:
                assert call.kwargs.get("query_fresh") is False, (
                    f"recording-branch must call resolve_audio_devices with "
                    f"query_fresh=False; got {call}"
                )
            refresh.assert_not_called()
            # Reconcile fires exactly once (the first tick); the second tick's
            # mute_triple matches the cached value so it's a no-op.
            assert reconcile.call_count == 1
        finally:
            ms._recording_active.clear()
            # Reset the monitor's memo so subsequent tests start clean.
            mon._prev_mute_triple = None

    def test_recording_branch_never_invokes_portaudio_refresh(self):
        """Regression guard for the BlackHole-goes-silent bug: spy on
        _refresh_portaudio and assert the monitor never calls it during a
        recording, no matter how many ticks fire."""
        mon = ms.AudioDeviceMonitor()
        ms._recording_active.set()
        try:
            with patch.object(ms, "_refresh_portaudio") as refresh_spy:
                mon.tick_once()
                mon.tick_once()
                mon.tick_once()
            refresh_spy.assert_not_called()
        finally:
            ms._recording_active.clear()

    def test_idle_event_triggers_extra_tick(self):
        """After the initial start() tick, setting _hotplug_event must wake the
        blocked wait and trigger one additional tick."""
        plan = self._plan()
        mon = ms.AudioDeviceMonitor()
        with patch.object(ms, "resolve_audio_devices", return_value=plan) as resolver, \
             patch.object(ms, "_restore_output_if_needed", return_value=None):
            mon.start()
            # Give the initial-tick thread a moment, then poke an event.
            import time as _t
            _t.sleep(0.05)
            initial_calls = resolver.call_count
            ms._hotplug_event.set()
            _t.sleep(0.05)
            mon.stop(timeout=1.5)
        assert resolver.call_count >= initial_calls + 1, (
            f"event did not trigger an extra tick: {initial_calls} → {resolver.call_count}"
        )

    def test_stop_unblocks_event_wait(self):
        """stop() must release the blocking _hotplug_event.wait() and the
        thread must exit within the join timeout, even with no real event."""
        plan = self._plan()
        mon = ms.AudioDeviceMonitor()
        with patch.object(ms, "resolve_audio_devices", return_value=plan), \
             patch.object(ms, "_restore_output_if_needed", return_value=None):
            mon.start()
            import time as _t
            _t.sleep(0.05)  # let the initial tick finish, then block on wait
            mon.stop(timeout=1.5)
        assert not mon.is_alive()

    def test_stop_joins_within_timeout(self):
        plan = self._plan()
        mon = ms.AudioDeviceMonitor()
        with patch.object(ms, "resolve_audio_devices", return_value=plan), \
             patch.object(ms, "_restore_output_if_needed", return_value=None):
            mon.start()
            assert mon.is_alive()
            mon.stop(timeout=1.5)
        assert not mon.is_alive()

    def test_tick_exception_is_caught_and_logged(self):
        mon = ms.AudioDeviceMonitor()
        log_buf = io.StringIO()
        with patch.object(ms, "resolve_audio_devices", side_effect=RuntimeError("boom")), \
             patch.object(ms, "_log_file_handle", log_buf):
            # tick_once() does NOT catch — exceptions propagate to the run()
            # loop where the try/except lives. Drive the real run() via
            # start()/stop() so we exercise the loop body, not the test seam.
            mon.start()
            mon.stop(timeout=1.5)
        log = log_buf.getvalue()
        assert "[ERR] monitor tick: RuntimeError: boom" in log


# ── Silent-except regression guard ────────────────────────────────────────────

class TestNoSilentExcepts:
    def test_zero_unlogged_pass_blocks_in_audio_paths(self):
        """Regression test: ensure no new `except: pass` blocks slip in where
        we expect logging. The 5 deliberately-silent sites are listed by their
        unique surrounding context so the test fails loudly if any are added."""
        src = Path(ms.__file__).read_text(encoding="utf-8")
        # Count `except*:\n*pass` patterns.
        pat = _re.compile(r"^\s+except[^#\n]*:\s*\n\s+pass\s*$", _re.MULTILINE)
        matches = pat.findall(src)
        # Expected silent-pass sites (must stay silent — see design doc D6):
        #   • _log file-write recursion guard
        #   • _log console-mirror recursion guard
        #   • stdlib-logging → file Handler.emit (avoid logging-from-logging recursion)
        #   • _setup_log_file._restore: removeHandler (idempotent cleanup)
        #   • _setup_log_file._restore: fh.close (already-closed handle is fine)
        #   • CoreAudio listener callback (must not raise into HAL thread)
        #   • _atexit_restore_mutes outer (atexit never raises)
        #   • _atexit_restore_mutes inner-log fallback (logging itself might fail)
        #   • module-level _recover_persisted_mutes inner-log fallback (never
        #     raise from module import)
        #   • config _set value-cast fallback loop
        #   • _poll _q.Empty sentinel
        #   • cmd_ui _start_recording dOut-failure mute-rollback (best-effort;
        #     we're already in an error path, don't shadow the original failure)
        # If you add a new silent pass, document it here AND in the design doc.
        assert len(matches) <= 12, (
            f"Found {len(matches)} silent except: pass blocks — expected ≤12. "
            f"Either log the failure or document it as deliberate."
        )


# ── Per-device mute lifecycle for the recording session ─────────────────────


class _MuteFixtureMixin:
    """Setup/teardown for tests that touch the mute lifecycle state.

    Each test starts with a clean _active_mutes dict and a redirected
    _MUTE_STATE_FILE pointing into a tmp dir, so production state is never
    touched and parallel runs don't interfere.
    """

    def setup_method(self):
        self._saved_state_file = ms._MUTE_STATE_FILE
        self._saved_active_mutes = dict(ms._active_mutes)
        ms._active_mutes.clear()

    def teardown_method(self):
        ms._active_mutes.clear()
        ms._active_mutes.update(self._saved_active_mutes)
        ms._MUTE_STATE_FILE = self._saved_state_file


class TestMuteBindings:
    """_ca_get_device_mute / _ca_set_device_mute."""

    def test_get_returns_none_on_non_darwin(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        assert ms._ca_get_device_mute("any device") is None

    def test_set_returns_false_on_non_darwin(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        assert ms._ca_set_device_mute("any device", True) is False

    def test_get_returns_none_for_nonexistent_device(self):
        # On darwin or otherwise: device-not-found yields None / False
        assert ms._ca_get_device_mute("__nonexistent_device_xyz__") is None

    def test_set_returns_false_for_nonexistent_device(self):
        assert ms._ca_set_device_mute("__nonexistent_device_xyz__", True) is False

    def test_roundtrip_preserves_state(self):
        if sys.platform != "darwin":
            pytest.skip("CoreAudio only on macOS")
        dev = ms._get_current_output_device()
        if dev is None:
            pytest.skip("no default output device")
        original = ms._ca_get_device_mute(dev)
        if original is None:
            pytest.skip("device does not expose mute property")
        # Set to same value — should be a successful no-op
        assert ms._ca_set_device_mute(dev, original) is True
        # State preserved
        assert ms._ca_get_device_mute(dev) is original


class TestReconcileRecordingMutes(_MuteFixtureMixin):
    """_reconcile_recording_mutes asymmetric tracking and idempotency."""

    def _plan(self, multi="多输出设备", restore="External Headphones"):
        return ms.AudioPlan(multi_output_name=multi, restore_output_name=restore)

    def test_no_multi_output_is_noop(self):
        plan = self._plan(multi=None)
        with patch.object(ms, "_ca_set_device_mute") as set_m, \
             patch.object(ms, "_ca_get_device_mute") as get_m:
            ms._reconcile_recording_mutes(plan)
        set_m.assert_not_called()
        get_m.assert_not_called()
        assert ms._active_mutes == {}

    def test_no_physical_subs_is_noop(self):
        plan = self._plan()
        with patch.object(ms, "_get_multi_output_physical_subs", return_value=[]), \
             patch.object(ms, "_ca_set_device_mute") as set_m:
            ms._reconcile_recording_mutes(plan)
        set_m.assert_not_called()
        assert ms._active_mutes == {}

    def test_no_restore_target_is_noop(self):
        plan = self._plan(restore=None)
        with patch.object(ms, "_get_multi_output_physical_subs",
                          return_value=["MacBook Air Speakers"]), \
             patch.object(ms, "_ca_set_device_mute") as set_m:
            ms._reconcile_recording_mutes(plan)
        set_m.assert_not_called()
        assert ms._active_mutes == {}

    def test_active_sub_already_unmuted_no_change_no_track(self):
        plan = self._plan(restore="MacBook Air Speakers")
        with patch.object(ms, "_get_multi_output_physical_subs",
                          return_value=["MacBook Air Speakers"]), \
             patch.object(ms, "_ca_get_device_mute", return_value=False), \
             patch.object(ms, "_ca_set_device_mute") as set_m, \
             patch.object(ms, "_persist_mutes"):
            ms._reconcile_recording_mutes(plan)
        set_m.assert_not_called()
        assert ms._active_mutes == {}

    def test_inactive_sub_unmuted_gets_muted_and_tracked(self):
        plan = self._plan(restore="External Headphones")
        with patch.object(ms, "_get_multi_output_physical_subs",
                          return_value=["MacBook Air Speakers", "External Headphones"]), \
             patch.object(ms, "_ca_get_device_mute") as get_m, \
             patch.object(ms, "_ca_set_device_mute", return_value=True) as set_m, \
             patch.object(ms, "_persist_mutes"):
            get_m.side_effect = lambda name: False  # all start unmuted
            ms._reconcile_recording_mutes(plan)
        # Speakers (inactive) gets muted; original False is tracked
        set_m.assert_any_call("MacBook Air Speakers", True)
        assert ms._active_mutes == {"MacBook Air Speakers": False}

    def test_active_sub_premuted_gets_unmuted_but_not_tracked(self):
        """Asymmetric rule: pre-existing mute on the active listening target is
        cleared (so the user hears audio) but NOT tracked for restore — we
        don't want to re-impose a stale silence at stop."""
        plan = self._plan(restore="External Headphones")
        with patch.object(ms, "_get_multi_output_physical_subs",
                          return_value=["External Headphones"]), \
             patch.object(ms, "_ca_get_device_mute") as get_m, \
             patch.object(ms, "_ca_set_device_mute", return_value=True) as set_m, \
             patch.object(ms, "_persist_mutes"):
            get_m.side_effect = lambda name: True  # active sub is pre-muted
            ms._reconcile_recording_mutes(plan)
        # We DID unmute the device …
        set_m.assert_called_once_with("External Headphones", False)
        # … but did NOT track it
        assert ms._active_mutes == {}

    def test_virtual_subs_already_filtered_by_helper(self):
        """_get_multi_output_physical_subs is responsible for excluding BlackHole
        and other virtual sub-devices; reconcile trusts its output."""
        plan = self._plan(restore="External Headphones")
        # Helper would never return BlackHole; verify reconcile doesn't add
        # spurious mute calls on whatever physical subs it gets.
        with patch.object(ms, "_get_multi_output_physical_subs",
                          return_value=["MacBook Air Speakers"]), \
             patch.object(ms, "_ca_get_device_mute", return_value=False), \
             patch.object(ms, "_ca_set_device_mute", return_value=True) as set_m, \
             patch.object(ms, "_persist_mutes"):
            ms._reconcile_recording_mutes(plan)
        set_calls = [c for c in set_m.call_args_list if "BlackHole" in str(c)]
        assert set_calls == []

    def test_hotplug_transition_pops_tracked_entry_on_unmute(self):
        """Round 1: speakers (inactive) muted, tracked. Round 2 (hotplug →
        speakers become active): speakers unmuted; the asymmetric pop removes
        the tracking entry so restore_all has nothing to do for this device."""
        # Single physical sub so we can isolate the pop without a second
        # inactive sub being muted in round 2 and re-populating the dict.
        plan_phones = self._plan(restore="External Headphones")
        plan_speakers = self._plan(restore="MacBook Air Speakers")
        physical = ["MacBook Air Speakers"]
        mute_state = {"MacBook Air Speakers": False}

        with patch.object(ms, "_get_multi_output_physical_subs", return_value=physical), \
             patch.object(ms, "_ca_get_device_mute",
                          side_effect=lambda n: mute_state.get(n)), \
             patch.object(ms, "_ca_set_device_mute",
                          side_effect=lambda n, v: mute_state.__setitem__(n, v) or True), \
             patch.object(ms, "_persist_mutes"):
            # Round 1: speakers inactive → muted + tracked
            ms._reconcile_recording_mutes(plan_phones)
            assert ms._active_mutes == {"MacBook Air Speakers": False}
            assert mute_state["MacBook Air Speakers"] is True
            # Round 2: speakers becomes active → unmuted; entry popped
            ms._reconcile_recording_mutes(plan_speakers)
            assert ms._active_mutes == {}
            assert mute_state["MacBook Air Speakers"] is False

    def test_unsupported_mute_property_is_skipped(self):
        plan = self._plan(restore="External Headphones")
        with patch.object(ms, "_get_multi_output_physical_subs",
                          return_value=["MacBook Air Speakers"]), \
             patch.object(ms, "_ca_get_device_mute", return_value=None), \
             patch.object(ms, "_ca_set_device_mute") as set_m, \
             patch.object(ms, "_persist_mutes"):
            ms._reconcile_recording_mutes(plan)
        set_m.assert_not_called()
        assert ms._active_mutes == {}


class TestRestoreAllRecordingMutes(_MuteFixtureMixin):
    def test_empty_is_noop(self):
        with patch.object(ms, "_ca_set_device_mute") as set_m:
            ms._restore_all_recording_mutes()
        set_m.assert_not_called()

    def test_restores_each_to_recorded_original(self):
        ms._active_mutes["A"] = False
        ms._active_mutes["B"] = True
        with patch.object(ms, "_ca_set_device_mute", return_value=True) as set_m, \
             patch.object(ms, "_persist_mutes"):
            ms._restore_all_recording_mutes()
        set_m.assert_any_call("A", False)
        set_m.assert_any_call("B", True)
        assert ms._active_mutes == {}

    def test_failed_restore_keeps_entry(self):
        """If the CoreAudio set call returns False, leave the entry in
        _active_mutes so a subsequent retry (or atexit) can have another go."""
        ms._active_mutes["A"] = False
        with patch.object(ms, "_ca_set_device_mute", return_value=False), \
             patch.object(ms, "_persist_mutes"):
            ms._restore_all_recording_mutes()
        assert "A" in ms._active_mutes


class TestPersistAndRecoverMutes(_MuteFixtureMixin):
    def test_persist_writes_atomically_and_deletes_when_empty(self, tmp_path):
        f = tmp_path / ".active_mutes.json"
        ms._MUTE_STATE_FILE = f
        ms._active_mutes["A"] = False
        ms._persist_mutes()
        assert f.exists()
        data = json.loads(f.read_text())
        assert data["pid"] == ms.os.getpid()
        assert data["muted"] == {"A": False}
        assert data["schema_version"] == 1

        # Clearing the dict and re-persisting deletes the file
        ms._active_mutes.clear()
        ms._persist_mutes()
        assert not f.exists()

    def test_recover_with_dead_pid_restores_and_deletes(self, tmp_path):
        f = tmp_path / ".active_mutes.json"
        ms._MUTE_STATE_FILE = f
        f.write_text(json.dumps({"schema_version": 1, "pid": 99999,
                                 "muted": {"A": False, "B": True}}))
        with patch.object(ms.os, "kill", side_effect=ProcessLookupError), \
             patch.object(ms, "_ca_set_device_mute", return_value=True) as set_m:
            ms._recover_persisted_mutes()
        set_m.assert_any_call("A", False)
        set_m.assert_any_call("B", True)
        assert not f.exists()

    def test_recover_with_live_pid_refuses_and_keeps_file(self, tmp_path):
        f = tmp_path / ".active_mutes.json"
        ms._MUTE_STATE_FILE = f
        f.write_text(json.dumps({"schema_version": 1, "pid": 1000,
                                 "muted": {"A": False}}))
        with patch.object(ms.os, "kill", return_value=None), \
             patch.object(ms, "_ca_set_device_mute") as set_m:
            ms._recover_persisted_mutes()
        set_m.assert_not_called()
        assert f.exists()

    def test_recover_with_self_pid_restores(self, tmp_path):
        """Self-pid means this process left the file behind earlier in its
        lifetime (e.g. an unhandled exception bypassed atexit). Recover."""
        f = tmp_path / ".active_mutes.json"
        ms._MUTE_STATE_FILE = f
        f.write_text(json.dumps({"schema_version": 1, "pid": ms.os.getpid(),
                                 "muted": {"A": False}}))
        with patch.object(ms, "_ca_set_device_mute", return_value=True) as set_m:
            ms._recover_persisted_mutes()
        set_m.assert_called_once_with("A", False)
        assert not f.exists()

    def test_recover_with_corrupted_file_deletes(self, tmp_path):
        f = tmp_path / ".active_mutes.json"
        ms._MUTE_STATE_FILE = f
        f.write_text("{not valid json")
        with patch.object(ms, "_ca_set_device_mute") as set_m:
            ms._recover_persisted_mutes()
        set_m.assert_not_called()
        assert not f.exists()

    def test_recover_with_no_file_is_noop(self, tmp_path):
        ms._MUTE_STATE_FILE = tmp_path / "nonexistent.json"
        with patch.object(ms, "_ca_set_device_mute") as set_m:
            ms._recover_persisted_mutes()
        set_m.assert_not_called()


class TestMuteAtexit(_MuteFixtureMixin):
    def test_atexit_swallows_exceptions(self):
        """The atexit handler MUST NOT propagate exceptions or block process
        exit, even if the underlying restore raises."""
        with patch.object(ms, "_restore_all_recording_mutes",
                          side_effect=RuntimeError("boom")):
            # Must not raise
            ms._atexit_restore_mutes()


class TestGetMultiOutputPhysicalSubs:
    def test_returns_empty_for_none(self):
        assert ms._get_multi_output_physical_subs(None) == []
        assert ms._get_multi_output_physical_subs("") == []

    def test_returns_empty_on_non_darwin(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        assert ms._get_multi_output_physical_subs("多输出设备") == []

    def test_filters_virtual_and_aggregate_subs(self):
        """The helper SHALL skip 'virt' (BlackHole) / 'aggt' / 'grup' subs and
        return only physical device names."""
        dump = [
            {"id": 53, "name": "多输出设备", "uid": "agg-uid", "class_id": "aagg",
             "class_status": 0, "transport": "grup", "transport_status": 0,
             "sub_uids": ["bh-uid", "spk-uid", "hp-uid"]},
            {"id": 92, "name": "BlackHole 2ch", "uid": "bh-uid", "class_id": "adev",
             "class_status": 0, "transport": "virt", "transport_status": 0, "sub_uids": []},
            {"id": 80, "name": "MacBook Air Speakers", "uid": "spk-uid", "class_id": "adev",
             "class_status": 0, "transport": "bltn", "transport_status": 0, "sub_uids": []},
            {"id": 172, "name": "External Headphones", "uid": "hp-uid", "class_id": "adev",
             "class_status": 0, "transport": "bltn", "transport_status": 0, "sub_uids": []},
        ]
        with patch.object(ms, "_coreaudio_device_raw_dump", return_value=dump):
            result = ms._get_multi_output_physical_subs("多输出设备")
        assert result == ["MacBook Air Speakers", "External Headphones"]

    def test_returns_empty_when_multi_output_not_in_dump(self):
        with patch.object(ms, "_coreaudio_device_raw_dump", return_value=[]):
            assert ms._get_multi_output_physical_subs("多输出设备") == []

    def test_returns_empty_when_multi_output_has_no_subs(self):
        dump = [
            {"id": 53, "name": "多输出设备", "uid": "agg-uid", "class_id": "aagg",
             "class_status": 0, "transport": "grup", "transport_status": 0, "sub_uids": []},
        ]
        with patch.object(ms, "_coreaudio_device_raw_dump", return_value=dump):
            assert ms._get_multi_output_physical_subs("多输出设备") == []


# ── save_config JSONC comment preservation ─────────────────────────────────


class TestSaveConfigPreservesComments:
    """`save_config` is invoked by `config --set k=v`. Before this fix it
    used `json.dump` and silently stripped all comments. The in-place editor
    walks the existing JSONC text, finds the lines whose values diverge from
    `cfg`, and rewrites only those values — preserving every comment,
    blank line and key ordering exactly. When the change is too structural
    to patch line-locally (new key, malformed file, missing file), the
    function falls back to plain `json.dump`."""

    SAMPLE_JSONC = (
        "{\n"
        "  // ── 并发控制 ────────────────────────────\n"
        "  // LLM 超时（秒）\n"
        "  \"llm_timeout\": 600,\n"
        "  // 并发 worker 数\n"
        "  \"polish_max_workers\": 0,\n"
        "\n"
        "  // ── 运行模式 ────────────────────────────\n"
        "  \"mode\": \"meeting\",\n"
        "\n"
        "  // STT 配置\n"
        "  \"stt\": {\n"
        "    \"funasr\": {\n"
        "      \"workers\": 0,        // 0 = auto\n"
        "      \"chunk_secs\": 300,\n"
        "      \"model\": \"paraformer-zh\"\n"
        "    }\n"
        "  }\n"
        "}\n"
    )

    def _patch_paths(self, tmp_path):
        cfg_dir = tmp_path
        cfg_file = cfg_dir / "config.jsonc"
        cfg_file.write_text(self.SAMPLE_JSONC, encoding="utf-8")
        return patch.multiple(ms, CONFIG_DIR=cfg_dir, CONFIG_FILE=cfg_file)

    def _parsed_cfg(self):
        """Parse SAMPLE_JSONC directly — bypass load_config's DEFAULT_CONFIG
        merge so we can drive `save_config` with a dict that contains only
        the keys we wrote into the test file. (The merge would otherwise
        inject every DEFAULT_CONFIG default and make them look like new
        keys to the in-place patcher.)"""
        return json.loads(ms._strip_jsonc_comments(self.SAMPLE_JSONC))

    def test_top_level_value_change_preserves_all_comments(self, tmp_path):
        with self._patch_paths(tmp_path):
            cfg = self._parsed_cfg()
            cfg["polish_max_workers"] = 5
            ms.save_config(cfg)
            text = ms.CONFIG_FILE.read_text(encoding="utf-8")
        # Every comment line from the original survives byte-for-byte.
        original_comment_lines = [l for l in self.SAMPLE_JSONC.splitlines() if "//" in l]
        new_comment_lines = [l for l in text.splitlines() if "//" in l]
        assert new_comment_lines == original_comment_lines
        assert '"polish_max_workers": 5' in text
        # Untouched values stay verbatim
        assert '"llm_timeout": 600' in text

    def test_nested_value_change_preserves_inline_comment(self, tmp_path):
        with self._patch_paths(tmp_path):
            cfg = self._parsed_cfg()
            cfg["stt"]["funasr"]["workers"] = 7
            ms.save_config(cfg)
            text = ms.CONFIG_FILE.read_text(encoding="utf-8")
        # Inline `// 0 = auto` comment on the workers line must survive.
        assert '"workers": 7,        // 0 = auto' in text

    def test_string_value_change_round_trips(self, tmp_path):
        with self._patch_paths(tmp_path):
            cfg = self._parsed_cfg()
            cfg["mode"] = "interview"
            ms.save_config(cfg)
            text = ms.CONFIG_FILE.read_text(encoding="utf-8")
        assert '"mode": "interview"' in text
        assert "// ── 运行模式 ────" in text

    def test_no_change_is_noop_and_keeps_file_byte_identical(self, tmp_path):
        with self._patch_paths(tmp_path):
            cfg = self._parsed_cfg()
            ms.save_config(cfg)
            text = ms.CONFIG_FILE.read_text(encoding="utf-8")
        assert text == self.SAMPLE_JSONC

    def test_new_top_level_key_splices_in_preserving_comments(self, tmp_path):
        """A NEW top-level key gets spliced in before the root closing
        brace — existing comments are preserved. Regression test for
        the schema-bump case (e.g. introducing a ``prompts`` section
        into a user config that predated it)."""
        with self._patch_paths(tmp_path):
            cfg = self._parsed_cfg()
            cfg["brand_new_setting"] = "hello"
            cfg["another_new_key"] = {"nested": True}
            ms.save_config(cfg)
            text = ms.CONFIG_FILE.read_text(encoding="utf-8")
        reloaded = json.loads(ms._strip_jsonc_comments(text))
        # Round-trip preserves data
        assert reloaded["brand_new_setting"] == "hello"
        assert reloaded["another_new_key"] == {"nested": True}
        assert reloaded["polish_max_workers"] == 0
        # Existing comments are preserved (no json.dump fallback)
        assert "//" in text

    def test_new_nested_key_still_falls_back_to_plain_json(self, tmp_path):
        """Nested missing-key inserts can't be done line-locally without
        knowing which scope's closing brace to splice into — those still
        fall back to json.dump (the comment-losing rewrite is preferable
        to a corrupted file)."""
        with self._patch_paths(tmp_path):
            cfg = self._parsed_cfg()
            # Inject a missing NESTED key (top-level "stt" exists but
            # "stt.brand_new_subkey" does not).
            cfg.setdefault("stt", {})["brand_new_subkey"] = "x"
            ms.save_config(cfg)
            text = ms.CONFIG_FILE.read_text(encoding="utf-8")
        reloaded = json.loads(text)
        assert reloaded["stt"]["brand_new_subkey"] == "x"
        # Fallback path discards comments — confirms we did NOT try to
        # splice line-locally for nested inserts.
        assert "//" not in text

    def test_missing_file_falls_back_to_plain_json(self, tmp_path):
        cfg_dir = tmp_path
        cfg_file = cfg_dir / "config.jsonc"
        assert not cfg_file.exists()
        with patch.multiple(ms, CONFIG_DIR=cfg_dir, CONFIG_FILE=cfg_file):
            ms.save_config({"polish_max_workers": 3})
            text = cfg_file.read_text(encoding="utf-8")
        # Plain JSON, no comments
        assert json.loads(text) == {"polish_max_workers": 3}

    def test_corrupt_existing_file_falls_back_to_plain_json(self, tmp_path):
        cfg_dir = tmp_path
        cfg_file = cfg_dir / "config.jsonc"
        cfg_file.write_text("{ not valid json", encoding="utf-8")
        with patch.multiple(ms, CONFIG_DIR=cfg_dir, CONFIG_FILE=cfg_file):
            ms.save_config({"polish_max_workers": 4})
            text = cfg_file.read_text(encoding="utf-8")
        assert json.loads(text) == {"polish_max_workers": 4}


# ── Meeting filename parsing + rename ───────────────────────────────────────


class TestSplitMeetingStem:
    """`_split_meeting_stem` extracts the timestamp prefix (always required)
    and an optional custom-name segment from a recording stem."""

    def test_timestamp_only(self):
        assert ms._split_meeting_stem("20260518_174926") == ("20260518_174926", None)

    def test_with_custom_name(self):
        assert ms._split_meeting_stem("20260518_174926.客户访谈") == (
            "20260518_174926", "客户访谈"
        )

    def test_with_custom_name_containing_dots(self):
        # Once we've matched a timestamp prefix, every dot afterwards is part
        # of the custom name — we capture greedily so a multi-dot custom
        # label like "weekly.standup" survives intact.
        assert ms._split_meeting_stem("20260518_174926.weekly.standup") == (
            "20260518_174926", "weekly.standup"
        )

    def test_unmatched_stem_returns_none(self):
        assert ms._split_meeting_stem("randomname") == (None, None)
        assert ms._split_meeting_stem("not_a_timestamp.foo") == (None, None)
        assert ms._split_meeting_stem("") == (None, None)


class TestSanitizeMeetingCustomName:
    def test_strips_filesystem_reserved_chars(self):
        # /, \, :, *, ?, ", <, >, |, plus C0 control chars are mapped to _.
        out = ms._sanitize_meeting_custom_name('a/b\\c:d*e?f"g<h>i|j')
        assert out == "a_b_c_d_e_f_g_h_i_j"

    def test_strips_leading_trailing_dots_and_whitespace(self):
        assert ms._sanitize_meeting_custom_name("  .foo.  ") == "foo"

    def test_empty_input_returns_empty(self):
        assert ms._sanitize_meeting_custom_name("") == ""
        assert ms._sanitize_meeting_custom_name(None) == ""  # type: ignore[arg-type]


class TestRenameMeetingFiles:
    def _seed(self, dir_: Path, stem: str, suffixes: list[str]) -> Path:
        wav = dir_ / f"{stem}.wav"
        wav.write_bytes(b"")
        for s in suffixes:
            (dir_ / f"{stem}{s}").write_text(f"{s} content", encoding="utf-8")
        return wav

    def test_renames_wav_and_all_companions(self, tmp_path):
        wav = self._seed(tmp_path, "20260518_174926",
                         [".raw.txt", ".polish.txt", ".meeting.md"])
        new_wav = ms._rename_meeting_files(wav, "客户访谈")
        assert new_wav == tmp_path / "20260518_174926.客户访谈.wav"
        assert new_wav.exists()
        assert (tmp_path / "20260518_174926.客户访谈.raw.txt").exists()
        assert (tmp_path / "20260518_174926.客户访谈.polish.txt").exists()
        assert (tmp_path / "20260518_174926.客户访谈.meeting.md").exists()
        # Old files are gone
        assert not wav.exists()
        assert not (tmp_path / "20260518_174926.raw.txt").exists()

    def test_can_strip_custom_name_back_to_bare_timestamp(self, tmp_path):
        wav = self._seed(tmp_path, "20260518_174926.客户访谈",
                         [".raw.txt", ".meeting.md"])
        new_wav = ms._rename_meeting_files(wav, "")
        assert new_wav == tmp_path / "20260518_174926.wav"
        assert (tmp_path / "20260518_174926.raw.txt").exists()
        assert (tmp_path / "20260518_174926.meeting.md").exists()
        assert not wav.exists()

    def test_collision_aborts_without_renaming(self, tmp_path):
        wav = self._seed(tmp_path, "20260518_174926", [".raw.txt"])
        # Pre-existing target with the new stem
        (tmp_path / "20260518_174926.客户访谈.wav").write_bytes(b"taken")
        result = ms._rename_meeting_files(wav, "客户访谈")
        assert result is None
        # Original files untouched
        assert wav.exists()
        assert (tmp_path / "20260518_174926.raw.txt").exists()

    def test_no_change_is_noop_and_returns_path(self, tmp_path):
        wav = self._seed(tmp_path, "20260518_174926.客户访谈", [".raw.txt"])
        result = ms._rename_meeting_files(wav, "客户访谈")
        assert result == wav  # no-op short-circuit
        assert wav.exists()
        assert (tmp_path / "20260518_174926.客户访谈.raw.txt").exists()

    def test_bad_stem_refuses_rename(self, tmp_path):
        wav = tmp_path / "not_a_timestamp.wav"
        wav.write_bytes(b"")
        assert ms._rename_meeting_files(wav, "anything") is None
        assert wav.exists()

    def test_missing_file_returns_none(self, tmp_path):
        assert ms._rename_meeting_files(tmp_path / "nope.wav", "x") is None

    def test_sanitises_filesystem_hostile_chars(self, tmp_path):
        wav = self._seed(tmp_path, "20260518_174926", [".raw.txt"])
        new_wav = ms._rename_meeting_files(wav, "a/b:c*d")
        # Reserved chars → _
        assert new_wav == tmp_path / "20260518_174926.a_b_c_d.wav"
        assert new_wav.exists()


# ── _resolve_prompt ──────────────────────────────────────────────────────────


class TestResolvePrompt:
    """Pipeline prompts are externalised in ``cfg["prompts"]`` with a
    fallback to ``DEFAULT_CONFIG["prompts"]``. ``polish`` lives at the top
    level (mode-agnostic); ``notes_zh`` / ``notes_en`` live under each
    mode (``meeting`` / ``interview``). Each value may be a string used
    verbatim, or a list of strings joined with ``\\n``."""

    def test_default_polish_top_level(self):
        # Empty user cfg + mode=None → top-level default polish.
        result = ms._resolve_prompt({}, "polish")
        assert "校对助手" in result
        assert "{transcript}" in result

    def test_default_notes_mode_scoped(self):
        # mode-scoped keys require the mode arg.
        meeting_zh = ms._resolve_prompt({}, "notes_zh", mode="meeting")
        assert "会议纪要助手" in meeting_zh
        interview_zh = ms._resolve_prompt({}, "notes_zh", mode="interview")
        assert "面试评估助手" in interview_zh

    def test_string_override_polish(self):
        cfg = {"prompts": {"polish": "custom polish {transcript}"}}
        assert ms._resolve_prompt(cfg, "polish") == "custom polish {transcript}"

    def test_list_override_joined_with_newline(self):
        cfg = {"prompts": {"polish": ["line A", "", "line C"]}}
        assert ms._resolve_prompt(cfg, "polish") == "line A\n\nline C"

    def test_partial_override_keeps_other_defaults(self):
        # Override only polish — notes for both modes still default.
        cfg = {"prompts": {"polish": "X"}}
        assert ms._resolve_prompt(cfg, "polish") == "X"
        assert "会议纪要助手" in ms._resolve_prompt(cfg, "notes_zh", mode="meeting")
        assert "面试评估助手" in ms._resolve_prompt(cfg, "notes_zh", mode="interview")

    def test_partial_override_in_mode_block(self):
        cfg = {"prompts": {"meeting": {"notes_zh": "Y"}}}
        assert ms._resolve_prompt(cfg, "notes_zh", mode="meeting") == "Y"
        # interview side unaffected.
        assert "面试评估助手" in ms._resolve_prompt(cfg, "notes_zh", mode="interview")

    def test_invalid_type_raises_typeerror(self):
        cfg = {"prompts": {"polish": 42}}
        with pytest.raises(TypeError, match="must be str or list"):
            ms._resolve_prompt(cfg, "polish")

    def test_unknown_key_raises_keyerror(self):
        # Programming bug, not a user one — surface it loudly.
        with pytest.raises(KeyError):
            ms._resolve_prompt({}, "no_such_key")
        with pytest.raises(KeyError):
            ms._resolve_prompt({}, "no_such_key", mode="meeting")

    def test_transcript_token_preserved(self):
        # _resolve_prompt does NOT substitute — callers do via .replace().
        cfg: dict = {}
        assert "{transcript}" in ms._resolve_prompt(cfg, "polish")
        for mode in ("meeting", "interview"):
            for key in ("notes_zh", "notes_en"):
                assert "{transcript}" in ms._resolve_prompt(cfg, key, mode=mode), \
                    f"{mode}.{key} missing {{transcript}} token"

    def test_non_dict_prompts_falls_back_to_default(self):
        # A malformed user cfg with prompts as a non-dict should still
        # return the default, not crash.
        cfg = {"prompts": "oops not a dict"}
        assert "{transcript}" in ms._resolve_prompt(cfg, "polish")
        assert "{transcript}" in ms._resolve_prompt(cfg, "notes_zh", mode="meeting")

    def test_polish_speaker_label_instruction_present(self):
        # The generalised speaker-labelling rule lives in the unified
        # polish prompt now; both interviewer/candidate and meeting users
        # see the same instruction.
        polish = ms._resolve_prompt({}, "polish")
        assert "区分不同发言者" in polish
