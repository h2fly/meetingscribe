"""Tests for the `captions <wav>` replay subcommand.

The point of the command is that caption changes become comparable: the same
file, fed straight into the engine, twice. Playing a file through speakers and
letting the microphone re-record it is not equivalent — it adds a speaker →
room → mic round trip, and a role-separated recording puts its speech on a
channel the capture device may not carry.
"""

import argparse
import queue
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import meetingscribe as ms


def _write_wav(path: Path, channels: "list[np.ndarray]", sr: int = 16000) -> Path:
    data = (np.column_stack(channels) if len(channels) > 1 else channels[0])
    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(len(channels))
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


class TestWavChannelsByRole:
    def test_stereo_splits_into_system_and_mic(self, tmp_path):
        sysc = np.full(1600, 0.2, dtype=np.float32)
        micc = np.full(1600, -0.4, dtype=np.float32)
        p = _write_wav(tmp_path / "two.wav", [sysc, micc])
        tracks, sr = ms._wav_channels_by_role(p)
        assert sr == 16000
        assert set(tracks) == {"system", "mic"}
        # Channel 0 is system audio, channel 1 the mic — the `wanted` order
        # MultiStreamRecorder.save() writes.
        assert np.allclose(tracks["system"], 0.2, atol=1e-3)
        assert np.allclose(tracks["mic"], -0.4, atol=1e-3)

    def test_mono_is_treated_as_microphone(self, tmp_path):
        p = _write_wav(tmp_path / "one.wav", [np.full(800, 0.1, dtype=np.float32)])
        tracks, sr = ms._wav_channels_by_role(p)
        assert set(tracks) == {"mic"} and len(tracks["mic"]) == 800

    def test_lengths_match_the_file(self, tmp_path):
        p = _write_wav(tmp_path / "len.wav",
                       [np.zeros(4000, dtype=np.float32)] * 2)
        tracks, _ = ms._wav_channels_by_role(p)
        assert all(len(v) == 4000 for v in tracks.values())


class _FakeEngine:
    """Records what the replay feeds it and emits a scripted event script."""

    instances: list = []

    def __init__(self, cfg, on_event):
        self.cfg = cfg
        self.on_event = on_event
        self.fed: list = []
        self.started = self.stopped = False
        self._mt_queue = queue.Queue()
        self._refine_queue = queue.Queue()
        self._spk_queue = queue.Queue()
        self._review_enabled = bool(
            ((cfg.get("live_captions") or {}).get("review") or {}).get("enabled"))
        self.backlog = 0.0
        _FakeEngine.instances.append(self)

    def backlog_secs(self):
        return self.backlog

    def start(self):
        self.started = True

    def feed(self, role, samples, sr):
        self.fed.append((role, int(samples.size), sr))

    def stop(self):
        self.stopped = True
        self.on_event({"type": "final", "id": 1, "text": "第一句原文", "role": "mic"})
        self.on_event({"type": "speaker", "id": 1, "speaker": 2, "side": "mic"})
        self.on_event({"type": "refined", "id": 1, "text": "第一句修正后的原文"})
        self.on_event({"type": "translation", "id": 1, "text": "First line."})

    def _emit(self, **ev):
        self.on_event(ev)

    def _take_review_batch(self):
        return []


@pytest.fixture
def fake_engine(monkeypatch):
    _FakeEngine.instances.clear()
    monkeypatch.setattr(ms, "LiveCaptionEngine", _FakeEngine)
    monkeypatch.setattr(ms, "_setup_log_file", lambda: (None, lambda: None))
    return _FakeEngine


def _args(file, **kw):
    base = dict(file=str(file), start=0, seconds=0, fast=True,
                review=False, trace=False, wait=0, quiet=False)
    base.update(kw)
    return argparse.Namespace(**base)


class TestReplay:
    def test_feeds_both_roles_as_separate_sources(self, tmp_path, fake_engine, capsys):
        p = _write_wav(tmp_path / "r.wav",
                       [np.full(16000, 0.1, dtype=np.float32),
                        np.full(16000, 0.2, dtype=np.float32)])
        ms._cmd_captions_body(_args(p), {})
        eng = fake_engine.instances[-1]
        assert eng.started and eng.stopped
        roles = {r for r, _, _ in eng.fed}
        assert roles == {"system", "mic"}      # not pre-mixed into one stream
        # 1 s of audio at the engine's own pacer interval
        assert sum(n for _, n, _ in eng.fed) == 32000

    def test_review_is_off_unless_asked(self, tmp_path, fake_engine):
        p = _write_wav(tmp_path / "r.wav", [np.zeros(1600, dtype=np.float32)] * 2)
        ms._cmd_captions_body(_args(p), {"live_captions": {"review": {"enabled": True}}})
        assert fake_engine.instances[-1]._review_enabled is False

    def test_caller_config_is_not_mutated(self, tmp_path, fake_engine):
        p = _write_wav(tmp_path / "r.wav", [np.zeros(1600, dtype=np.float32)] * 2)
        cfg = {"live_captions": {"review": {"enabled": True}}}
        ms._cmd_captions_body(_args(p), cfg)
        assert cfg["live_captions"]["review"]["enabled"] is True

    def test_start_and_seconds_slice_the_audio(self, tmp_path, fake_engine):
        p = _write_wav(tmp_path / "r.wav",
                       [np.zeros(16000 * 4, dtype=np.float32)] * 2)
        ms._cmd_captions_body(_args(p, start=1, seconds=2), {})
        eng = fake_engine.instances[-1]
        per_role = sum(n for r, n, _ in eng.fed if r == "mic")
        assert per_role == 32000          # exactly the requested 2 s

    def test_streams_lines_while_running(self, tmp_path, fake_engine, capsys):
        """The live stream deliberately shows the provisional text and marks
        the revision — a 34-minute file must not sit silent until the end."""
        p = _write_wav(tmp_path / "r.wav", [np.zeros(1600, dtype=np.float32)] * 2)
        ms._cmd_captions_body(_args(p), {})
        stream = capsys.readouterr().out.split("── 字幕 (")[0]
        assert "[  1] 第一句原文" in stream
        assert "↻ 第一句修正后的原文" in stream
        assert "→ First line." in stream
        assert "⇢ 说话人2" in stream

    def test_report_shows_settled_state_only(self, tmp_path, fake_engine, capsys):
        p = _write_wav(tmp_path / "r.wav", [np.zeros(1600, dtype=np.float32)] * 2)
        ms._cmd_captions_body(_args(p), {})
        report = capsys.readouterr().out.split("── 字幕 (")[1]
        assert "第一句修正后的原文" in report      # the settled text
        assert "第一句原文\n" not in report        # not the superseded one
        assert "[说话人2]" in report
        assert "First line." in report
        assert "被修正的行  : 1" in report
        assert "识别出说话人: 1" in report

    def test_quiet_suppresses_the_stream_but_keeps_the_report(
            self, tmp_path, fake_engine, capsys):
        p = _write_wav(tmp_path / "r.wav", [np.zeros(1600, dtype=np.float32)] * 2)
        ms._cmd_captions_body(_args(p, quiet=True), {})
        out = capsys.readouterr().out
        stream = out.split("── 字幕 (")[0]
        assert "[  1]" not in stream
        assert "第一句修正后的原文" in out.split("── 字幕 (")[1]

    def test_missing_file_exits(self, tmp_path, fake_engine):
        with pytest.raises(SystemExit):
            ms._cmd_captions_body(_args(tmp_path / "nope.wav"), {})

    def test_trace_lists_events_but_not_partials(self, tmp_path, fake_engine, capsys):
        p = _write_wav(tmp_path / "r.wav", [np.zeros(1600, dtype=np.float32)] * 2)

        class _Chatty(_FakeEngine):
            def stop(self):
                self.on_event({"type": "partial", "text": "进行中"})
                super().stop()

        fake_engine.instances.clear()
        import meetingscribe as m
        m.LiveCaptionEngine = _Chatty
        try:
            ms._cmd_captions_body(_args(p, trace=True), {})
        finally:
            m.LiveCaptionEngine = _FakeEngine
        out = capsys.readouterr().out
        assert "事件轨迹" in out
        assert "refined" in out
        assert "进行中" not in out          # partials are too noisy to diff


class TestSpeedFidelity:
    def test_default_is_realtime_paced(self, tmp_path, fake_engine, monkeypatch):
        """Feeding faster hands the recognizer bigger chunks, and is_endpoint()
        is checked once per accept() — measured on one 104 s passage, the fast
        path produced 3 run-on segments where realtime produced 6. So realtime
        is the default and --fast warns."""
        slept = []
        monkeypatch.setattr(ms.time, "sleep", lambda s: slept.append(s))
        p = _write_wav(tmp_path / "r.wav", [np.zeros(16000, dtype=np.float32)] * 2)
        ms._cmd_captions_body(_args(p, fast=False), {})
        assert slept and all(s == ms._CAPTION_PACER_INTERVAL for s in slept)

    def test_fast_mode_warns_about_segmentation(self, tmp_path, fake_engine, capsys):
        p = _write_wav(tmp_path / "r.wav", [np.zeros(1600, dtype=np.float32)] * 2)
        ms._cmd_captions_body(_args(p, fast=True), {})
        assert "断句会明显变少变长" in capsys.readouterr().out


class TestPacing:
    def test_returns_immediately_when_backlog_is_small(self):
        eng = _FakeEngine({}, lambda ev: None)
        eng.backlog = 1.0
        ms._captions_pace(eng)          # must not spin

    def test_waits_while_the_asr_is_behind(self, monkeypatch):
        """Unpaced, the replay pushes audio faster than the recognizer eats
        it and feed() silently drops past _CAPTION_RING_SECONDS — the file
        would only be partly transcribed."""
        eng = _FakeEngine({}, lambda ev: None)
        eng.backlog = ms._CAPTION_RING_SECONDS      # over the half-full mark
        slept = []
        monkeypatch.setattr(ms.time, "sleep", lambda s: slept.append(s))
        ms._captions_pace(eng)
        assert len(slept) == 400        # backed off to its ceiling, no hang


class TestDrain:
    def test_returns_once_queues_are_empty(self):
        eng = _FakeEngine({}, lambda ev: None)
        ms._captions_drain(eng, timeout=5)     # nothing queued → immediate

    def test_times_out_without_hanging(self, monkeypatch):
        logged = []
        monkeypatch.setattr(ms, "_log", lambda cat, msg: logged.append(msg))
        eng = _FakeEngine({}, lambda ev: None)
        eng._mt_queue.put(1)                   # never drains
        ms._captions_drain(eng, timeout=0.5)
        assert any("drain timed out" in m for m in logged)


class TestCliWiring:
    def test_subcommand_is_dispatched(self):
        src = Path(ms.__file__).read_text(encoding="utf-8")
        assert '"captions": cmd_captions' in src

    def test_argparse_accepts_the_flags(self):
        import subprocess
        r = subprocess.run(
            [sys.executable, str(Path(ms.__file__)), "captions", "--help"],
            capture_output=True, text=True, timeout=60)
        assert r.returncode == 0
        for flag in ("--start", "--seconds", "--fast", "--review",
                     "--trace", "--wait", "--quiet"):
            assert flag in r.stdout
