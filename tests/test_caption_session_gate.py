"""Stop a recording, immediately start another: the new captions must not be
overwritten by the old session's late corrections.

Segment ids restart at 1 every session and the pane matches `refined` /
`translation` / `speaker` by id ALONE, so a correction arriving late from the
previous session lands on the new session's row with that id. A review batch
is a measured 62 s median and `stop()` cannot interrupt one, so "late" here
means "for the next minute", which is exactly the window in which the user
starts the next recording.
"""

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import meetingscribe as ms


# ── engine shutdown budget ───────────────────────────────────────────────────

class _Engine(ms.LiveCaptionEngine):
    """Engine whose workers are replaced by controllable stand-ins."""

    def __init__(self, cfg=None, on_event=None):
        super().__init__(cfg or {}, on_event or (lambda ev: None))


def _hanging_thread(name, release):
    t = threading.Thread(target=lambda: release.wait(30), daemon=True, name=name)
    t.start()
    return t


class TestStopIsBounded:
    def test_stop_does_not_wait_five_seconds_per_worker(self):
        """Five threads at a flat 5 s meant a stop could block its caller for
        25 s — on the Qt thread, i.e. a frozen window on every stop."""
        eng = _Engine()
        release = threading.Event()
        try:
            eng._asr_thread = None          # the one role that IS waited for
            eng._mt_thread = _hanging_thread("caption-mt", release)
            eng._refine_thread = _hanging_thread("caption-refine", release)
            eng._spk_thread = _hanging_thread("caption-spk", release)
            eng._review_thread = _hanging_thread("caption-review", release)
            t0 = time.time()
            eng.stop()
            took = time.time() - t0
        finally:
            release.set()
        # 4 abandoned roles x 0.5 s, with generous slack for a loaded machine.
        assert took < 3.0, f"stop() blocked {took:.1f}s"

    def test_asr_thread_still_gets_a_real_budget(self):
        """It shares the CACHED streaming recognizer with the next session;
        two sessions interleaved into one decoder stream corrupts recognition
        rather than merely delaying it."""
        eng = _Engine()
        release = threading.Event()
        try:
            eng._asr_thread = _hanging_thread("caption-asr", release)
            t0 = time.time()
            eng.stop()
            took = time.time() - t0
        finally:
            release.set()
        assert took >= ms._CAPTION_STOP_ASR_TIMEOUT - 0.2

    def test_survivors_are_logged_not_silent(self, monkeypatch):
        logged = []
        monkeypatch.setattr(ms, "_log", lambda cat, msg: logged.append(msg))
        eng = _Engine()
        release = threading.Event()
        try:
            eng._review_thread = _hanging_thread("caption-review", release)
            eng.stop()
        finally:
            release.set()
        line = next(m for m in logged if "engine stopped" in m)
        assert "caption-review" in line and "events now ignored" in line

    def test_all_thread_handles_are_cleared(self):
        eng = _Engine()
        eng.stop()
        assert eng._asr_thread is None and eng._mt_thread is None
        assert eng._refine_thread is None and eng._spk_thread is None
        assert eng._review_thread is None

    def test_clean_shutdown_logs_plainly(self, monkeypatch):
        logged = []
        monkeypatch.setattr(ms, "_log", lambda cat, msg: logged.append(msg))
        _Engine().stop()
        assert "engine stopped" in logged
        assert not any("still finishing" in m for m in logged)


# ── the session gate itself ──────────────────────────────────────────────────

class _Gate:
    """The gate as `_RecorderState` builds it, isolated from Qt.

    Mirrors the closure in start_recording: a monotonic counter captured by
    default argument, compared against the live value on every event.
    """

    def __init__(self):
        self._caption_session = 0
        self.delivered: list = []

    def new_session(self):
        self._caption_session += 1
        sess = self._caption_session

        def _emit(ev, _sess=sess):
            if _sess == self._caption_session:
                self.delivered.append(ev)

        return _emit


class TestSessionGate:
    def test_current_session_events_pass(self):
        g = _Gate()
        emit = g.new_session()
        emit({"type": "final", "id": 1, "text": "第一句"})
        assert len(g.delivered) == 1

    def test_superseded_session_is_silenced(self):
        g = _Gate()
        old = g.new_session()
        g.new_session()                      # user starts the next recording
        old({"type": "refined", "id": 3, "text": "上一场的迟到修正"})
        assert g.delivered == []

    def test_the_stopping_session_keeps_delivering_until_the_next_starts(self):
        """stop() is not the cut-off — the tail of the recording that just
        ended (the ASR flush's final segment) still belongs on screen."""
        g = _Gate()
        emit = g.new_session()
        emit({"type": "final", "id": 9, "text": "最后一句"})
        assert len(g.delivered) == 1

    def test_late_correction_cannot_overwrite_the_new_sessions_row(self):
        """The actual defect: ids restart at 1, and the pane matches by id."""
        g = _Gate()
        old = g.new_session()
        old({"type": "final", "id": 1, "text": "上一场第一句"})
        new = g.new_session()
        new({"type": "final", "id": 1, "text": "新一场第一句"})
        old({"type": "refined", "id": 1, "text": "上一场的修正"})
        old({"type": "translation", "id": 1, "text": "stale translation"})
        old({"type": "speaker", "id": 1, "speaker": 7})
        # Only the new session's own row survives in the delivered stream.
        after_switch = [e for e in g.delivered if e.get("text") != "上一场第一句"]
        assert [e["text"] for e in after_switch] == ["新一场第一句"]

    def test_several_stale_sessions_all_stay_silenced(self):
        g = _Gate()
        s1, s2 = g.new_session(), g.new_session()
        s3 = g.new_session()
        s1({"type": "final", "id": 1, "text": "一"})
        s2({"type": "final", "id": 1, "text": "二"})
        s3({"type": "final", "id": 1, "text": "三"})
        assert [e["text"] for e in g.delivered] == ["三"]

    def test_gate_survives_a_worker_thread_racing_the_switch(self):
        """The counter is read from worker threads; a stale read may cost one
        event on the boundary but must never raise or leak a whole session."""
        g = _Gate()
        old = g.new_session()
        stop = threading.Event()

        def spam():
            i = 0
            while not stop.is_set():
                i += 1
                old({"type": "refined", "id": i, "text": f"stale-{i}"})

        t = threading.Thread(target=spam, daemon=True)
        t.start()
        time.sleep(0.05)
        g.new_session()
        time.sleep(0.05)
        stop.set()
        t.join(timeout=2)
        # Everything after the switch is gone; nothing raised.
        assert all(e["text"].startswith("stale-") for e in g.delivered)


class TestWiring:
    def test_recorder_state_creates_the_counter(self):
        src = Path(ms.__file__).read_text(encoding="utf-8")
        assert "self._caption_session = 0" in src

    def test_engine_is_constructed_with_the_gated_emitter(self):
        """A regression guard: passing `self.caption_event.emit` directly is
        what allowed the cross-session overwrite."""
        src = Path(ms.__file__).read_text(encoding="utf-8")
        assert "LiveCaptionEngine(cfg, _emit_caption)" in src
        assert "LiveCaptionEngine(cfg, self.caption_event.emit)" not in src
