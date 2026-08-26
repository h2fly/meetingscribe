"""Tests for live-caption speaker identification (voice print + clustering).

The cam++ model itself is faked — these cover the clustering maths, the
side vote, and the engine's worker/queue wiring. Reference numbers for the
threshold come from cam++'s own sample voices, measured on this machine: same
speaker 0.694, different speakers -0.084 / 0.007.

Identity comes from the voice print ALONE. Channel role only says which side
the audio arrived on, because one microphone can carry a whole meeting room.
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import meetingscribe as ms


def vec(*values, dim=192):
    """A deterministic unit-ish embedding: first components set, rest zero."""
    v = np.zeros(dim, dtype=np.float32)
    for i, x in enumerate(values):
        v[i] = x
    return v


A = vec(1.0, 0.0, 0.0)          # speaker A
A2 = vec(0.95, 0.31, 0.0)       # A again, slightly different (cos ≈ 0.95)
B = vec(0.0, 1.0, 0.0)          # speaker B (orthogonal → cos 0)
C = vec(0.0, 0.0, 1.0)          # speaker C


class TestSpeakerClusterer:
    def test_first_embedding_creates_speaker(self):
        c = ms._SpeakerClusterer()
        assert c.assign(A) == 0

    def test_same_voice_joins_existing_cluster(self):
        c = ms._SpeakerClusterer(threshold=0.5)
        assert c.assign(A) == 0
        assert c.assign(A2) == 0

    def test_different_voice_starts_new_cluster(self):
        c = ms._SpeakerClusterer(threshold=0.5)
        c.assign(A)
        assert c.assign(B) == 1
        assert c.assign(C) == 2

    def test_alternating_speakers_stay_stable(self):
        c = ms._SpeakerClusterer(threshold=0.5)
        assert [c.assign(e) for e in (A, B, A2, B, A)] == [0, 1, 0, 1, 0]

    def test_threshold_controls_splitting(self):
        loose = ms._SpeakerClusterer(threshold=0.1)
        loose.assign(A)
        assert loose.assign(A2) == 0
        strict = ms._SpeakerClusterer(threshold=0.99)
        strict.assign(A)
        assert strict.assign(A2) == 1       # same person, split by a high bar

    def test_max_speakers_folds_into_nearest(self):
        c = ms._SpeakerClusterer(threshold=0.9, max_speakers=2)
        c.assign(A)
        c.assign(B)
        # A third distinct voice cannot create 说话人3; it joins the nearest.
        assert c.assign(C) in (0, 1)
        assert len(c._centroids) == 2

    def test_centroid_is_a_running_mean(self):
        c = ms._SpeakerClusterer(threshold=0.5)
        c.assign(A)
        c.assign(A2)
        assert c._counts[0] == 2
        assert np.isclose(np.linalg.norm(c._centroids[0]), 1.0, atol=1e-5)

    def test_new_cluster_centroid_is_the_embedding(self):
        c = ms._SpeakerClusterer()
        c.assign(vec(3.0, 4.0))            # unnormalized input
        assert np.isclose(np.linalg.norm(c._centroids[0]), 1.0, atol=1e-6)
        assert c._counts[0] == 1

    def test_zero_embedding_rejected(self):
        c = ms._SpeakerClusterer()
        assert c.assign(np.zeros(0, dtype=np.float32)) == -1


class TestSideVote:
    """Channel role is a SIDE, never an identity — one microphone can carry a
    whole meeting room, so "arrived on the mic" must not mean "is the user"."""

    def test_mic_majority_reports_mic_side(self):
        c = ms._SpeakerClusterer(threshold=0.5)
        for _ in range(3):
            c.assign(A, "mic")
        assert c.majority_side(0) == "mic"

    def test_system_majority_reports_system_side(self):
        c = ms._SpeakerClusterer(threshold=0.5)
        for _ in range(3):
            c.assign(B, "system")
        assert c.majority_side(0) == "system"

    def test_mixed_votes_follow_the_majority(self):
        # Channel bleed (listening on speakers) puts a few of the remote
        # speaker's segments on the mic channel; the majority must still win.
        c = ms._SpeakerClusterer(threshold=0.5)
        c.assign(B, "mic")
        for _ in range(4):
            c.assign(B, "system")
        assert c.majority_side(0) == "system"

    def test_no_role_information(self):
        c = ms._SpeakerClusterer()
        c.assign(A)
        assert c.majority_side(0) is None

    def test_out_of_range_index(self):
        assert ms._SpeakerClusterer().majority_side(5) is None


class TestDisplayNumbering:
    def test_numbered_in_discovery_order_across_channels(self):
        c = ms._SpeakerClusterer(threshold=0.5)
        assert c.display_number(c.assign(A, "mic")) == 1
        assert c.display_number(c.assign(B, "system")) == 2
        assert c.display_number(c.assign(C, "system")) == 3

    def test_several_people_on_one_microphone_get_separate_numbers(self):
        """A meeting room sharing one mic: everyone on the mic channel must
        still be told apart, which is why numbering is global and no cluster
        is special-cased as "me"."""
        c = ms._SpeakerClusterer(threshold=0.5)
        n1 = c.display_number(c.assign(A, "mic"))
        n2 = c.display_number(c.assign(B, "mic"))
        n3 = c.display_number(c.assign(C, "mic"))
        assert (n1, n2, n3) == (1, 2, 3)
        assert all(c.majority_side(i) == "mic" for i in range(3))

    def test_numbers_are_stable_across_repeats(self):
        c = ms._SpeakerClusterer(threshold=0.5)
        c.assign(A, "mic")
        first = c.display_number(c.assign(B, "system"))
        for _ in range(5):
            assert c.display_number(c.assign(B, "system")) == first

    def test_both_sides_multi_speaker(self):
        c = ms._SpeakerClusterer(threshold=0.5)
        seen = {}
        for emb, role in [(A, "mic"), (B, "mic"), (C, "system"),
                          (A, "mic"), (C, "system"), (B, "mic")]:
            idx = c.assign(emb, role)
            seen.setdefault(c.display_number(idx), c.majority_side(idx))
        assert sorted(seen) == [1, 2, 3]
        assert seen[1] == "mic" and seen[2] == "mic" and seen[3] == "system"


# ── engine wiring ────────────────────────────────────────────────────────────

class _FakeSpeakerId:
    """Returns a fixed embedding per call, from a scripted list."""

    def __init__(self, embeddings):
        self._embs = list(embeddings)
        self.calls = 0

    def embed(self, audio):
        self.calls += 1
        return self._embs[min(self.calls - 1, len(self._embs) - 1)]


def _engine(embeddings, cfg=None):
    eng = ms.LiveCaptionEngine(cfg or {}, lambda ev: None)
    eng._load_speaker_backend = lambda: _FakeSpeakerId(embeddings)
    return eng


class TestEngineSpeakerWorker:
    def _run(self, eng, sink, segments):
        eng.on_event = sink
        eng._spk_thread = threading.Thread(
            target=eng._speaker_worker, daemon=True)
        eng._spk_thread.start()
        for sid, role in segments:
            eng._spk_queue.put((sid, np.zeros(16000, dtype=np.float32), role))
        deadline = time.time() + 5
        while time.time() < deadline:
            if len([e for e in sink.events if e["type"] == "speaker"]) >= len(segments):
                break
            time.sleep(0.02)
        eng._spk_queue.put(None)
        eng._spk_thread.join(timeout=2)

    def test_emits_speaker_events(self):
        events = []
        sink = type("S", (), {"events": events, "__call__":
                              lambda self, ev: events.append(ev)})()
        eng = _engine([A, B, A])
        self._run(eng, sink, [(1, "mic"), (2, "system"), (3, "mic")])
        spk = [e for e in events if e["type"] == "speaker"]
        assert [e["id"] for e in spk] == [1, 2, 3]
        assert spk[0]["side"] == "mic" and spk[1]["side"] == "system"
        assert spk[0]["speaker"] == 1 and spk[1]["speaker"] == 2
        assert spk[2]["speaker"] == 1        # speaker 1 again, same voice

    def test_backend_failure_is_non_fatal(self):
        events = []
        sink = type("S", (), {"events": events, "__call__":
                              lambda self, ev: events.append(ev)})()
        eng = ms.LiveCaptionEngine({}, sink)

        def boom():
            raise RuntimeError("model missing")

        eng._load_speaker_backend = boom
        self._run(eng, sink, [(1, "mic")])
        assert not [e for e in events if e["type"] == "speaker"]

    def test_short_segments_are_not_queued(self):
        eng = ms.LiveCaptionEngine({}, lambda ev: None)
        eng._finalize_segment("太短", np.zeros(8000, dtype=np.float32))  # 0.5 s
        assert eng._spk_queue.qsize() == 0

    def test_long_enough_segment_is_queued(self):
        eng = ms.LiveCaptionEngine({}, lambda ev: None)
        eng._finalize_segment("够长的一段话", np.zeros(24000, dtype=np.float32))
        assert eng._spk_queue.qsize() == 1

    def test_backlog_cap_skips_segments(self):
        eng = ms.LiveCaptionEngine(
            {"live_captions": {"speaker_id": {"max_backlog": 2}}}, lambda ev: None)
        audio = np.zeros(24000, dtype=np.float32)
        for i in range(5):
            eng._finalize_segment(f"第{i}段话内容", audio)
        assert eng._spk_queue.qsize() == 2
        assert eng._spk_dropped == 3

    def test_disabled_by_config(self):
        eng = ms.LiveCaptionEngine({"live_captions": {"speaker_id": {"enabled": False}}}, lambda ev: None)
        eng._finalize_segment("够长的一段话", np.zeros(24000, dtype=np.float32))
        assert eng._spk_queue.qsize() == 0
        assert eng._spk_thread is None


# ── refine must not re-decode English with a Chinese model ───────────────────

def test_english_segment_skips_refine():
    """Measured on a real session: paraformer-zh "corrected" 53 of 56 English
    segments at 1.7–2.5 s each, all of them for the worse."""
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    assert eng._refine_enabled
    eng._finalize_segment("Thank you everyone for joining this call",
                          np.zeros(24000, dtype=np.float32))
    assert eng._refine_queue.qsize() == 0


def test_chinese_segment_still_refines():
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    eng._finalize_segment("我们把流量切到新集群上",
                          np.zeros(24000, dtype=np.float32))
    assert eng._refine_queue.qsize() == 1


def test_english_segment_still_gets_speaker_id():
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    eng._finalize_segment("Thank you everyone for joining this call",
                          np.zeros(24000, dtype=np.float32))
    assert eng._spk_queue.qsize() == 1


# ── role attribution normalises per source ───────────────────────────────────

def test_quiet_mic_still_wins_its_own_segment():
    """A mic at -30 dBFS against playback at -6 dBFS: comparing raw RMS made
    the loud channel win nearly every line (observed: almost everything
    tagged 对方). Scoring relative to each source's own peak fixes it."""
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    eng._running = True
    # Establish each source's normal level.
    for _ in range(4):
        eng.feed("system", np.full(4000, 0.5, dtype=np.float32), 16000)
        eng.feed("mic", np.full(4000, 0.03, dtype=np.float32), 16000)
        eng._drain_mixed()
    eng._take_segment_role()
    # Now the local user speaks (loud for the mic) while the call is quiet.
    eng.feed("mic", np.full(8000, 0.03, dtype=np.float32), 16000)
    eng.feed("system", np.full(8000, 0.02, dtype=np.float32), 16000)
    eng._drain_mixed()
    assert eng._take_segment_role() == "mic"


# ── noise segments must never reach the pane or the clusterer ────────────────

class TestNoiseSegments:
    """Measured on a real 34-minute meeting: 4 of the 8 voice-print clusters
    came from ONE junk segment each (`M`, `N`, `A`, `啊。`), and they used up
    max_speakers so later real speakers were folded into existing clusters."""

    JUNK = ["M", "N", "A", "n", "啊。", "嗯", "嗯嗯", "。", "  ", "M。", "，"]
    REAL = ["对。", "好的", "可以", "有对。", "那我们。", "OK", "一百二，我现在走。"]

    def test_junk_classified_as_noise(self):
        assert all(ms._caption_is_noise(t) for t in self.JUNK)

    def test_short_real_answers_kept(self):
        # 对 / 好 / 可以 are content in a Chinese meeting — dropping them
        # would lose actual decisions.
        assert not any(ms._caption_is_noise(t) for t in self.REAL)

    def test_noise_segment_emits_nothing(self):
        events = []
        eng = ms.LiveCaptionEngine({}, lambda ev: events.append(ev))
        eng._finalize_segment("M", np.zeros(24000, dtype=np.float32))
        assert events == []
        assert eng._spk_queue.qsize() == 0
        assert eng._refine_queue.qsize() == 0
        assert eng._mt_queue.qsize() == 0

    def test_noise_segment_does_not_consume_a_segment_id(self):
        eng = ms.LiveCaptionEngine({}, lambda ev: None)
        eng._finalize_segment("啊。", np.zeros(24000, dtype=np.float32))
        assert eng._seg_id == 0

    def test_real_segment_still_flows(self):
        events = []
        eng = ms.LiveCaptionEngine({}, lambda ev: events.append(ev))
        eng._finalize_segment("这个方案可以", np.zeros(24000, dtype=np.float32))
        assert [e["type"] for e in events] == ["final"]
        assert eng._spk_queue.qsize() == 1

    def test_junk_no_longer_creates_phantom_speakers(self):
        """Replays the real session's shape: 3 speakers plus junk lines. The
        junk must not turn into clusters 3/4/6/7 like it did."""
        eng = ms.LiveCaptionEngine({}, lambda ev: None)
        audio = np.zeros(24000, dtype=np.float32)
        for text in ["第一个人在说话", "M", "第二个人补充一句", "N",
                     "第三个人也说了", "啊。", "A"]:
            eng._finalize_segment(text, audio)
        assert eng._spk_queue.qsize() == 3      # only the real lines queued


# ── "heard" detection drives the side tint ───────────────────────────────────

def test_builtin_mic_level_counts_as_heard():
    """A real session logged side=None for all 208 segments: the old 0.01
    per-window bar is above a built-in mic's conversational level."""
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    eng._running = True
    for _ in range(3):
        eng.feed("mic", np.full(4000, 0.006, dtype=np.float32), 16000)
        eng.feed("system", np.full(4000, 0.30, dtype=np.float32), 16000)
        eng._drain_mixed()
    assert eng._role_seen == {"mic", "system"}
    assert eng._take_segment_role() in ("mic", "system")


def test_digital_silence_is_not_heard():
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    eng._running = True
    eng.feed("mic", np.full(4000, 0.20, dtype=np.float32), 16000)
    eng.feed("system", np.zeros(4000, dtype=np.float32), 16000)
    eng._drain_mixed()
    assert eng._role_seen == {"mic"}
    assert eng._take_segment_role() is None    # only one side heard → no tag
