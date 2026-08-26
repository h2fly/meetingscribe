"""
Tests for the live bilingual caption engine (LiveCaptionEngine + helpers).
Run: pytest tests/
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import meetingscribe as ms


# ── helpers ───────────────────────────────────────────────────────────────────

class _EventSink:
    def __init__(self):
        self.events = []
        self._lock = threading.Lock()

    def __call__(self, ev):
        with self._lock:
            self.events.append(ev)

    def types(self):
        with self._lock:
            return [e["type"] for e in self.events]

    def wait_for(self, ev_type, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                for e in self.events:
                    if e["type"] == ev_type:
                        return e
            time.sleep(0.05)
        return None


class _FakeASR:
    """Finalizes one segment as soon as ≥0.1 s of audio has arrived."""

    def __init__(self, engine):
        self.engine = engine
        self.total = 0
        self.done = False

    def accept(self, samples):
        self.total += samples.size
        if not self.done and self.total >= 1600:
            self.done = True
            self.engine._emit_partial("部分文本")
            self.engine._finalize_segment("你好世界")

    def flush(self):
        pass


class _FakeMT:
    def translate(self, text):
        return "EN:" + text


def _make_engine(sink):
    eng = ms.LiveCaptionEngine({}, sink)
    eng._load_asr_backend = lambda: _FakeASR(eng)
    eng._load_mt_backend = lambda: _FakeMT()
    return eng


# ── _caption_resample ─────────────────────────────────────────────────────────

def test_resample_identity_same_rate():
    x = np.random.rand(1600).astype(np.float32)
    out = ms._caption_resample(x, 16000, 16000)
    assert out is x or np.array_equal(out, x)


def test_resample_48k_to_16k_length():
    x = np.zeros(4800, dtype=np.float32)
    out = ms._caption_resample(x, 48000, 16000)
    assert len(out) == 1600
    assert out.dtype == np.float32


def test_resample_empty():
    out = ms._caption_resample(np.zeros(0, dtype=np.float32), 48000, 16000)
    assert out.size == 0


# ── _caption_is_english ───────────────────────────────────────────────────────

def test_language_sniff_chinese():
    assert not ms._caption_is_english("我们讨论一下架构方案。")


def test_language_sniff_english():
    assert ms._caption_is_english("Let's talk about the routing layer design.")


def test_language_sniff_mixed_mostly_chinese():
    assert not ms._caption_is_english("我们用 GKE 跑 workload，没问题的。")


# ── _pick_model_file ──────────────────────────────────────────────────────────

def test_pick_model_file_prefers_int8(tmp_path):
    (tmp_path / "encoder-epoch-99-avg-1.onnx").write_bytes(b"x")
    (tmp_path / "encoder-epoch-99-avg-1.int8.onnx").write_bytes(b"x")
    assert ms._pick_model_file(tmp_path, "encoder").endswith("int8.onnx")


def test_pick_model_file_fallback_fp32(tmp_path):
    (tmp_path / "decoder-epoch-99-avg-1.onnx").write_bytes(b"x")
    assert ms._pick_model_file(tmp_path, "decoder").endswith(".onnx")


def test_pick_model_file_missing(tmp_path):
    with pytest.raises(RuntimeError):
        ms._pick_model_file(tmp_path, "joiner")


# ── _ensure_sherpa_model ──────────────────────────────────────────────────────

def test_ensure_sherpa_model_skips_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "CONFIG_DIR", tmp_path)
    name = ms._SHERPA_ZIPFORMER_URL.rsplit("/", 1)[-1].replace(".tar.bz2", "")
    target = tmp_path / "models" / name
    target.mkdir(parents=True)
    (target / "tokens.txt").write_text("a 1\n")
    # No network mock needed: presence of tokens.txt short-circuits download.
    assert ms._ensure_sherpa_model() == target


# ── LiveCaptionEngine ─────────────────────────────────────────────────────────

def test_engine_full_event_flow():
    sink = _EventSink()
    eng = _make_engine(sink)
    eng.start()
    try:
        assert sink.wait_for("status") is not None
        # 0.2 s of 16 kHz audio → crosses the fake backend's 1600-sample gate
        eng.feed("mic", np.full(3200, 0.1, dtype=np.float32), 16000)
        final = sink.wait_for("final")
        assert final and final["text"] == "你好世界"
        trans = sink.wait_for("translation")
        assert trans and trans["text"] == "EN:你好世界"
        assert trans["id"] == final["id"]
    finally:
        eng.stop()
    types = sink.types()
    assert "partial" in types
    assert {"state": "loading"} == {
        "state": next(e["state"] for e in sink.events if e["type"] == "status")}
    assert sink.events[-1] == {"type": "status", "state": "stopped"}


def test_engine_asr_load_failure_emits_error():
    sink = _EventSink()
    eng = ms.LiveCaptionEngine({}, sink)

    def boom():
        raise RuntimeError("no backend installed")

    eng._load_asr_backend = boom
    eng._load_mt_backend = lambda: _FakeMT()
    eng.start()
    err = sink.wait_for("error")
    eng.stop()
    assert err and "no backend installed" in err["message"]
    assert eng._running is False


def test_feed_ignored_when_not_running():
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    eng.feed("mic", np.zeros(1600, dtype=np.float32), 16000)
    assert not eng._buffers


def test_feed_backlog_cap():
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    eng._running = True
    one_sec = np.zeros(16000, dtype=np.float32)
    for _ in range(40):  # 40 s pushed; cap is 30 s
        eng.feed("mic", one_sec, 16000)
    total = sum(len(c) / sr for c, sr in eng._buffers["mic"])
    assert total <= ms._CAPTION_RING_SECONDS + 1.0


def test_drain_mixed_sums_and_resamples():
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    eng._running = True
    eng.feed("system", np.full(4800, 0.25, dtype=np.float32), 48000)  # 0.1 s
    eng.feed("mic", np.full(1600, 0.25, dtype=np.float32), 16000)     # 0.1 s
    mixed = eng._drain_mixed()
    assert len(mixed) == 1600
    assert np.allclose(mixed[100:1500], 0.5, atol=1e-3)
    # buffers were drained
    assert eng._drain_mixed().size == 0


def test_stereo_feed_downmixes_all_channels():
    """Channel 0 alone silently discarded content: replaying a role-separated
    stereo recording through BlackHole put every word on channel 1, so the
    digital copy read as silence and only the microphone's acoustic
    re-recording reached the captions."""
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    eng._running = True
    left_only = np.column_stack([
        np.full(1600, 0.3, dtype=np.float32),
        np.zeros(1600, dtype=np.float32),
    ])
    eng.feed("mic", left_only, 16000)
    assert np.allclose(eng._drain_mixed(), 0.15, atol=1e-6)


def test_stereo_feed_hears_right_channel_only_content():
    """The case that broke playback testing: content exclusively on ch1."""
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    eng._running = True
    right_only = np.column_stack([
        np.zeros(1600, dtype=np.float32),
        np.full(1600, 0.4, dtype=np.float32),
    ])
    eng.feed("system", right_only, 16000)
    mixed = eng._drain_mixed()
    assert np.allclose(mixed, 0.2, atol=1e-6)   # was 0.0 before the fix


def test_mono_feed_unchanged():
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    eng._running = True
    eng.feed("mic", np.full(1600, 0.3, dtype=np.float32), 16000)
    assert np.allclose(eng._drain_mixed(), 0.3, atol=1e-6)


def test_engine_refine_flow():
    sink = _EventSink()
    eng = ms.LiveCaptionEngine({}, sink)

    class _AudioASR(_FakeASR):
        def accept(self, samples):
            self.total += samples.size
            if not self.done and self.total >= 1600:
                self.done = True
                self.engine._finalize_segment(
                    "原始文本", np.zeros(16000, dtype=np.float32))

    class _FakeRefine:
        def generate(self, input, batch_size_s):
            return [{"text": "校验后文本"}]

    eng._load_asr_backend = lambda: _AudioASR(eng)
    eng._load_mt_backend = lambda: _FakeMT()
    eng._load_refine_backend = lambda: _FakeRefine()
    assert eng._refine_enabled
    eng.start()
    try:
        eng.feed("mic", np.full(3200, 0.1, dtype=np.float32), 16000)
        refined = sink.wait_for("refined")
        assert refined and refined["text"] == "校验后文本"
        deadline = time.time() + 5
        got = None
        while time.time() < deadline and got is None:
            got = next((e for e in list(sink.events)
                        if e["type"] == "translation"
                        and e["text"] == "EN:校验后文本"), None)
            time.sleep(0.05)
        assert got and got["id"] == refined["id"]
    finally:
        eng.stop()


def test_engine_partial_translation():
    sink = _EventSink()
    eng = _make_engine(sink)  # accurate-mode fakes; refine disabled
    eng.start()
    try:
        eng._emit_partial("这是一个足够长的进行中句子")
        pt = sink.wait_for("partial_translation")
        assert pt and pt["text"] == "EN:这是一个足够长的进行中句子"
    finally:
        eng.stop()


def test_short_partial_not_translated():
    sink = _EventSink()
    eng = _make_engine(sink)
    eng.start()
    try:
        eng._emit_partial("短句")  # below the 6-char gate
        assert sink.wait_for("partial_translation", timeout=1.0) is None
    finally:
        eng.stop()


def test_tidy_zh_collapses_stutter_ngrams():
    assert ms._caption_tidy_zh("所以就像所以就像刚刚温森也提到") == "所以就像刚刚温森也提到"
    assert ms._caption_tidy_zh("但是但是每一个月") == "但是每一个月"
    assert ms._caption_tidy_zh("有的话有的话我们可以") == "有的话我们可以"


def test_tidy_zh_collapses_triple_chars():
    assert ms._caption_tidy_zh("是不是都都都") == "是不是都"


def test_tidy_zh_keeps_legit_reduplication():
    assert ms._caption_tidy_zh("谢谢大家") == "谢谢大家"
    assert ms._caption_tidy_zh("刚刚好") == "刚刚好"


def test_tidy_zh_drops_fillers():
    assert ms._caption_tidy_zh("主要还是嗯涉及到一些。") == "主要还是涉及到一些。"
    assert ms._caption_tidy_zh("是讲的是说我们哎呀哎。") == "是讲的是说我们。"
    assert ms._caption_tidy_zh("呃这个方案呃我觉得可行") == "这个方案我觉得可行"


def test_tidy_zh_keeps_sentence_particles():
    assert ms._caption_tidy_zh("它运行的时候呢") == "它运行的时候呢"
    assert ms._caption_tidy_zh("这样可以吧") == "这样可以吧"


def test_tidy_zh_cleans_orphan_punctuation():
    assert ms._caption_tidy_zh("嗯，标记也标记一下") == "标记也标记一下"


def test_tidy_zh_leaves_english_alone():
    s = "This is is a very very long English sentence."
    assert ms._caption_tidy_zh(s) == s


def test_final_text_is_tidied():
    sink = _EventSink()
    eng = _make_engine(sink)
    eng._finalize_segment("但是但是每一个月")
    ev = sink.wait_for("final", timeout=1.0)
    assert ev and ev["text"] == "但是每一个月"


def test_partial_throttle():
    events = []
    eng = ms.LiveCaptionEngine({"live_captions": {"partial_interval_ms": 60000}}, lambda ev: events.append(ev))
    eng._emit_partial("第一次出字内容")
    eng._emit_partial("第一次出字内容第二次追加")
    partials = [e for e in events if e["type"] == "partial"]
    assert len(partials) == 1  # second update throttled…
    assert eng._partial_pending == "第一次出字内容第二次追加"  # …but MT sees it


def test_translation_coalescing():
    sink = _EventSink()
    eng = _make_engine(sink)
    # Queue two texts under the same id BEFORE the worker starts: the
    # worker must translate only the latest and skip the stale duplicate.
    eng._queue_translation(1, "文本甲")
    eng._queue_translation(1, "文本乙")
    eng.start()
    try:
        trans = sink.wait_for("translation")
        assert trans and trans["text"] == "EN:文本乙"
        time.sleep(0.6)  # give the worker a chance to (wrongly) emit again
        all_trans = [e for e in sink.events if e["type"] == "translation"]
        assert len(all_trans) == 1
    finally:
        eng.stop()


def test_refine_passes_hotword():
    sink = _EventSink()
    eng = ms.LiveCaptionEngine({"stt": {"funasr": {"hotword": "sandbox API网关"}}}, sink)

    class _AudioASR(_FakeASR):
        def accept(self, samples):
            self.total += samples.size
            if not self.done and self.total >= 1600:
                self.done = True
                self.engine._finalize_segment(
                    "原始文本", np.zeros(16000, dtype=np.float32))

    class _KwRefine:
        kwargs = None
        def generate(self, **kw):
            _KwRefine.kwargs = kw
            return [{"text": "校验后文本"}]

    eng._load_asr_backend = lambda: _AudioASR(eng)
    eng._load_mt_backend = lambda: _FakeMT()
    eng._load_refine_backend = lambda: _KwRefine()
    eng.start()
    try:
        eng.feed("mic", np.full(3200, 0.1, dtype=np.float32), 16000)
        assert sink.wait_for("refined") is not None
    finally:
        eng.stop()
    assert _KwRefine.kwargs["hotword"] == "sandbox API网关"


def test_refine_enabled_by_default():
    assert ms.LiveCaptionEngine({}, lambda ev: None)._refine_enabled is True


def test_refine_config_off():
    eng = ms.LiveCaptionEngine(
        {"live_captions": {"refine": False}}, lambda ev: None)
    assert eng._refine_enabled is False


# ── MultiStreamRecorder audio tap ─────────────────────────────────────────────

def test_recorder_tap_receives_role_and_frames():
    rec = ms.MultiStreamRecorder(["DevA"], 16000, role_labels=["system"])
    rec.recording = True
    rec._frames["DevA"] = []
    calls = []
    rec.on_audio_chunk = lambda role, data, sr: calls.append((role, data, sr))
    cb = rec._make_cb("DevA")
    block = np.ones((160, 1), dtype=np.float32)
    cb(block, 160, None, None)
    assert len(rec._frames["DevA"]) == 1
    assert calls and calls[0][0] == "system" and calls[0][2] == 16000
    assert np.array_equal(calls[0][1], block)


def test_recorder_tap_exception_never_breaks_capture():
    rec = ms.MultiStreamRecorder(["DevA"], 16000, role_labels=["mic"])
    rec.recording = True
    rec._frames["DevA"] = []

    def bad_hook(role, data, sr):
        raise RuntimeError("boom")

    rec.on_audio_chunk = bad_hook
    cb = rec._make_cb("DevA")
    cb(np.ones((160, 1), dtype=np.float32), 160, None, None)
    assert len(rec._frames["DevA"]) == 1  # capture unaffected


def test_recorder_no_tap_by_default():
    rec = ms.MultiStreamRecorder(["DevA"], 16000)
    assert rec.on_audio_chunk is None


# ── config plumbing ───────────────────────────────────────────────────────────

def test_default_config_has_live_captions_block():
    lc = ms.DEFAULT_CONFIG["live_captions"]
    # One recognition path: the former mode/fast/accurate nesting is gone.
    assert "mode" not in lc and "fast" not in lc and "accurate" not in lc
    assert {"asr_model_dir", "mt_zh_en", "mt_en_zh", "refine"} <= set(lc)


def test_config_jsonc_live_captions_parses():
    raw = (Path(__file__).parent.parent / "config.jsonc").read_text(encoding="utf-8")
    import json
    parsed = json.loads(ms._strip_jsonc_comments(raw))
    lc = parsed.get("live_captions")
    assert lc is not None
    assert "mode" not in lc


# ── display-layer paragraph merging (④) ──────────────────────────────────────

def test_join_caption_texts_cjk_bare_latin_spaced():
    assert ms._join_caption_texts(["你好", "世界"]) == "你好世界"
    assert ms._join_caption_texts(["hello", "world"]) == "hello world"
    assert ms._join_caption_texts(["切到GKE", "然后看"]) == "切到GKE 然后看"
    assert ms._join_caption_texts(["", "  ", "你好"]) == "你好"


def test_group_caption_rows_merges_close_rows():
    rows = [
        {"src": "第一句", "t": 100.0},
        {"src": "第二句", "t": 101.0},   # 1s gap → merge
        {"src": "第三句", "t": 110.0},   # 9s gap → new group
    ]
    groups = ms._group_caption_rows(rows, gap_secs=2.5)
    assert [len(g) for g in groups] == [2, 1]


def test_group_caption_rows_respects_max_chars():
    rows = [{"src": "长" * 80, "t": 100.0 + i * 0.5} for i in range(3)]
    groups = ms._group_caption_rows(rows, gap_secs=2.5, max_chars=120)
    # first group reaches 160 chars after two rows → third starts anew
    assert [len(g) for g in groups] == [2, 1]


def test_group_caption_rows_gap_zero_disables_merging():
    rows = [{"src": "a", "t": 100.0}, {"src": "b", "t": 100.1}]
    assert [len(g) for g in ms._group_caption_rows(rows, gap_secs=0)] == [1, 1]


# ── MT worker correct+translate path (①) ─────────────────────────────────────

class _FakeFixMT:
    def __init__(self):
        self.calls = []

    def translate(self, text):
        return "EN:" + text

    def correct_and_translate(self, text, context):
        self.calls.append((text, list(context)))
        return "FIXED:" + text, "EN:" + text


def test_mt_worker_uses_correct_and_translate_for_finals():
    sink = _EventSink()
    eng = ms.LiveCaptionEngine({}, sink)
    backend = _FakeFixMT()
    eng._load_asr_backend = lambda: _FakeASR(eng)
    eng._load_mt_backend = lambda: backend
    eng.start()
    try:
        eng.feed("mic", np.random.rand(4800).astype(np.float32) * 0.5, 16000)
        refined = sink.wait_for("refined")
        assert refined and refined["text"].startswith("FIXED:")
        tr = sink.wait_for("translation")
        assert tr and tr["text"].startswith("EN:")
        # rolling context recorded the corrected line
        assert list(eng._mt_context) == ["FIXED:你好世界"]
    finally:
        eng.stop()


def test_mt_worker_plain_translate_backend_still_works():
    sink = _EventSink()
    eng = _make_engine(sink)
    eng.start()
    try:
        eng.feed("mic", np.random.rand(4800).astype(np.float32) * 0.5, 16000)
        tr = sink.wait_for("translation")
        assert tr and tr["text"] == "EN:你好世界"
        assert "refined" not in sink.types()
    finally:
        eng.stop()


# ── refine backlog + instrumentation ─────────────────────────────────────────

def test_refine_backlog_cap_skips_excess_segments(monkeypatch):
    """A re-decode costs ~1.2 s; when speech outruns it the queue must stop
    growing, otherwise corrections land after the line scrolled away."""
    logged: list = []
    monkeypatch.setattr(ms, "_log", lambda cat, msg: logged.append((cat, msg)))
    eng = ms.LiveCaptionEngine({"live_captions": {"refine_max_backlog": 2}}, lambda ev: None)
    assert eng._refine_enabled and eng._refine_max_backlog == 2
    audio = np.zeros(16000, dtype=np.float32)
    for i in range(5):
        eng._finalize_segment(f"第{i}句话", audio)
    # queue holds the cap, the rest are dropped and accounted for
    assert eng._refine_queue.qsize() == 2
    assert eng._refine_dropped == 3
    assert sum("refine skipped" in msg for _, msg in logged) == 3


def test_refine_backlog_default_from_config():
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    assert eng._refine_max_backlog == 3


def test_refine_queue_carries_pre_refine_text():
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    eng._finalize_segment("原始文本", np.zeros(16000, dtype=np.float32))
    sid, audio, before = eng._refine_queue.get_nowait()
    assert (sid, before) == (1, "原始文本") and audio.size == 16000


def test_refine_logs_duration_and_whether_text_changed(monkeypatch):
    logged: list = []
    monkeypatch.setattr(ms, "_log", lambda cat, msg: logged.append((cat, msg)))
    sink = _EventSink()
    eng = ms.LiveCaptionEngine({}, sink)

    class _AudioASR(_FakeASR):
        def accept(self, samples):
            self.total += samples.size
            if not self.done and self.total >= 1600:
                self.done = True
                self.engine._finalize_segment(
                    "原始文本", np.zeros(16000, dtype=np.float32))

    class _FakeRefine:
        def generate(self, input, batch_size_s):
            return [{"text": "校验后文本"}]

    eng._load_asr_backend = lambda: _AudioASR(eng)
    eng._load_mt_backend = lambda: _FakeMT()
    eng._load_refine_backend = lambda: _FakeRefine()
    eng.start()
    try:
        eng.feed("mic", np.full(3200, 0.1, dtype=np.float32), 16000)
        assert sink.wait_for("refined")
    finally:
        eng.stop()
    line = next((msg for cat, msg in logged
                 if cat == "CAPTION" and msg.startswith("refine seg=")), None)
    assert line and "took=" in line and "changed=True" in line


def test_mt_latency_logged(monkeypatch):
    logged: list = []
    monkeypatch.setattr(ms, "_log", lambda cat, msg: logged.append((cat, msg)))
    sink = _EventSink()
    eng = _make_engine(sink)
    eng.start()
    try:
        eng.feed("mic", np.random.rand(4800).astype(np.float32) * 0.5, 16000)
        assert sink.wait_for("translation")
    finally:
        eng.stop()
    line = next((msg for cat, msg in logged
                 if cat == "CAPTION" and msg.startswith("mt seg=")), None)
    assert line and "took=" in line and "corrected=False" in line


def test_partial_gate_uses_partial_cost_not_fix_cost():
    """The grammar-constrained fix call runs ~1.8 s; if the partial gate read
    that shared timer it would switch partial translation off for the rest of
    the session. It must read the partial path's own cost.

    No audio is fed on purpose: `_finalize_segment` clears `_partial_pending`,
    which would race with the assignment below.
    """
    sink = _EventSink()
    eng = _make_engine(sink)
    eng._mt_last_secs = 5.0       # poisoned by a slow finalized-line call
    eng._mt_partial_secs = 0.1    # the partial path itself is cheap
    eng.start()
    try:
        with eng._partial_lock:
            eng._partial_pending = "这是一句进行中的字幕"
        assert sink.wait_for("partial_translation", timeout=5.0)
    finally:
        eng.stop()


def test_slow_partial_path_still_disables_partials():
    sink = _EventSink()
    eng = _make_engine(sink)
    eng._mt_partial_secs = 2.0   # e.g. NLLB on CPU
    eng.start()
    try:
        with eng._partial_lock:
            eng._partial_pending = "这是一句进行中的字幕"
        time.sleep(1.0)
        assert "partial_translation" not in sink.types()
    finally:
        eng.stop()


# ── MT repetition loops (reported: 「雅 的 内容」×30 then "Ya"×100) ────────────

def test_collapse_mt_repeats_kills_cjk_loop():
    bad = "然后，好是" + "雅 的 内容 " * 12
    out = ms._collapse_mt_repeats(bad)
    assert out.count("雅 的 内容") == 1
    assert len(out) < 30


def test_collapse_mt_repeats_kills_latin_loop():
    assert ms._collapse_mt_repeats("Ya Ya Ya Ya Ya Ya Ya Ya") == "Ya"


def test_collapse_mt_repeats_kills_phrase_loop():
    out = ms._collapse_mt_repeats("the plan the plan the plan is done")
    assert out == "the plan is done"


def test_collapse_mt_repeats_keeps_legit_english_doubling():
    # "very very" is an intensifier, not a decoding loop: two occurrences
    # must survive (this is why char-level rules skip English).
    s = "It's a very very good plan."
    assert ms._collapse_mt_repeats(s) == s


def test_collapse_mt_repeats_keeps_legit_chinese_reduplication():
    assert ms._collapse_mt_repeats("谢谢，好好聊一下") == "谢谢，好好聊一下"


def test_collapse_mt_repeats_leaves_normal_text_alone():
    for s in ["我们把流量切到新集群上", "We switch traffic to the GKE cluster.", ""]:
        assert ms._collapse_mt_repeats(s) == s


def test_mt_backends_pass_repetition_guards(monkeypatch):
    """The real fix is decode-side: greedy decoding must be forbidden from
    repeating an n-gram at all."""
    seen = {}

    class _Tok:
        def __call__(self, *a, **k):
            return {}
        def decode(self, *a, **k):
            return "out"

    class _Mdl:
        def generate(self, **kwargs):
            seen.update(kwargs)
            return [[0]]
        def eval(self):
            return self

    mt = ms._MarianCaptionMT.__new__(ms._MarianCaptionMT)
    mt._ids = {"zh-en": "x", "en-zh": "y"}
    mt._models = {"zh-en": (_Tok(), _Mdl())}
    assert mt.translate("你好世界") == "out"
    assert seen["no_repeat_ngram_size"] == ms._MT_NO_REPEAT_NGRAM
    assert seen["repetition_penalty"] == ms._MT_REPETITION_PENALTY


def test_translation_events_are_collapsed():
    class _LoopMT:
        def translate(self, text):
            return "Ya Ya Ya Ya Ya Ya"

    sink = _EventSink()
    eng = ms.LiveCaptionEngine({}, sink)
    eng._load_asr_backend = lambda: _FakeASR(eng)
    eng._load_mt_backend = lambda: _LoopMT()
    eng.start()
    try:
        eng.feed("mic", np.random.rand(4800).astype(np.float32) * 0.5, 16000)
        tr = sink.wait_for("translation")
        assert tr and tr["text"] == "Ya"
    finally:
        eng.stop()


# ── ALL-CAPS English from the fast streaming model ───────────────────────────

def test_fix_case_sentence_cases_shouted_english():
    assert ms._caption_fix_case("AND THEN OKAY WERE CON YEAH") == \
        "And then okay were con yeah"


def test_fix_case_restores_hotword_spelling():
    out = ms._caption_fix_case("THE CHATOPS FLOW ON GKE IS FINE",
                               ["ChatOps", "GKE", "gVisor"])
    assert out == "The ChatOps flow on GKE is fine"


def test_fix_case_capitalizes_standalone_i():
    assert ms._caption_fix_case("I THINK I'LL JOIN") == "I think I'll join"


def test_fix_case_starts_each_sentence():
    assert ms._caption_fix_case("WE GO NOW. THEY JOIN LATER") == \
        "We go now. They join later"


def test_fix_case_leaves_mixed_case_alone():
    for s in ["Maybe james, i think bella join as well.",
              "这个 GKE 集群没问题", "GKE", "", "OK"]:
        assert ms._caption_fix_case(s, ["GKE"]) == s


def test_fix_case_leaves_chinese_with_acronym_alone():
    s = "我们把流量切到 GKE 集群上"
    assert ms._caption_fix_case(s, ["GKE"]) == s


# ── live-caption speaker roles ───────────────────────────────────────────────

def _feed_role(eng, source, amplitude, n=8000):
    eng._running = True   # feed() drops chunks when the engine isn't running
    eng.feed(source, np.full(n, amplitude, dtype=np.float32), 16000)
    eng._drain_mixed()    # energy is tallied while draining, as in _asr_worker


def test_segment_attributed_to_louder_source():
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    _feed_role(eng, "mic", 0.3)
    _feed_role(eng, "system", 0.05)
    assert eng._take_segment_role() == "mic"


def test_role_switches_with_energy():
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    _feed_role(eng, "mic", 0.3)
    _feed_role(eng, "system", 0.3)   # both heard → labelling is meaningful
    eng._take_segment_role()
    _feed_role(eng, "system", 0.4)
    assert eng._take_segment_role() == "system"


def test_no_role_when_only_one_source_heard():
    """Labelling every line 「我」 in a solo recording tells the user nothing."""
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    for _ in range(3):
        _feed_role(eng, "mic", 0.3)
    assert eng._take_segment_role() is None


def test_role_disabled_by_config():
    eng = ms.LiveCaptionEngine(
        {"live_captions": {"speaker_labels": False}}, lambda ev: None)
    _feed_role(eng, "mic", 0.3)
    _feed_role(eng, "system", 0.3)
    assert eng._take_segment_role() is None


def test_role_tally_resets_between_segments():
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    _feed_role(eng, "mic", 0.3)
    _feed_role(eng, "system", 0.3)
    eng._take_segment_role()
    assert eng._role_energy == {}
    assert eng._take_segment_role() is None  # no energy since the reset


def test_final_event_carries_role():
    sink = _EventSink()
    eng = ms.LiveCaptionEngine({}, sink)
    _feed_role(eng, "mic", 0.3)
    _feed_role(eng, "system", 0.02)
    eng._finalize_segment("你好世界")
    ev = sink.wait_for("final")
    assert ev and ev["role"] == "mic"


def test_quiet_source_does_not_count_as_heard():
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    _feed_role(eng, "mic", 0.3)
    _feed_role(eng, "system", 0.001)   # below the RMS floor: silent line-in
    assert eng._take_segment_role() is None


# ── grouping must not merge across speakers ──────────────────────────────────

def test_group_rows_split_on_role_change():
    rows = [
        {"src": "甲说", "dst": "", "t": 100.0, "role": "mic"},
        {"src": "乙说", "dst": "", "t": 100.5, "role": "system"},
    ]
    groups = ms._group_caption_rows(rows, gap_secs=2.5)
    assert len(groups) == 2


def test_group_rows_merge_same_role():
    rows = [
        {"src": "甲说", "dst": "", "t": 100.0, "role": "mic"},
        {"src": "还有", "dst": "", "t": 100.5, "role": "mic"},
    ]
    assert len(ms._group_caption_rows(rows, gap_secs=2.5)) == 1


def test_group_rows_without_roles_still_merge():
    rows = [{"src": "甲说", "dst": "", "t": 100.0},
            {"src": "还有", "dst": "", "t": 100.4}]
    assert len(ms._group_caption_rows(rows, gap_secs=2.5)) == 1


# ── scroll-back retention (3 h by time, row cap as memory backstop) ──────────

def _rows(n, start_t=1000.0, step=1.0):
    return [{"id": i, "src": f"line{i}", "dst": "", "t": start_t + i * step}
            for i in range(n)]


def test_prune_drops_rows_outside_time_window():
    rows = _rows(100, start_t=0.0, step=60.0)      # one row per minute
    dropped = ms._prune_caption_rows(rows, now=100 * 60,
                                     max_secs=30 * 60, max_rows=0)
    assert dropped == 70
    assert len(rows) == 30
    assert rows[0]["id"] == 70                     # oldest kept is 30 min back


def test_prune_is_batched_not_per_row():
    """Evicting one row per append would shift every paragraph in the pane and
    defeat _CaptionDocRenderer's tail patching — the pane would re-layout in
    full on every line. Eviction must wait until the slack is exceeded."""
    rows = _rows(100, start_t=0.0, step=1.0)       # one row per second
    # 1 s past the window: inside the 60 s slack, so nothing moves yet.
    assert ms._prune_caption_rows(rows, now=100.0, max_secs=99.0,
                                  max_rows=0) == 0
    assert len(rows) == 100
    # 61 s past: slack exceeded, the whole overflow goes at once.
    assert ms._prune_caption_rows(rows, now=160.0, max_secs=99.0,
                                  max_rows=0) == 61
    assert len(rows) == 39


def test_prune_row_cap_also_batched():
    rows = _rows(1150, start_t=0.0, step=0.1)
    # 150 over the cap but within the 200-row slack → untouched.
    assert ms._prune_caption_rows(rows, now=115.0, max_secs=0,
                                  max_rows=1000) == 0
    assert len(rows) == 1150
    rows.extend(_rows(100, start_t=200.0, step=0.1))
    assert ms._prune_caption_rows(rows, now=300.0, max_secs=0,
                                  max_rows=1000) == 250
    assert len(rows) == 1000


def test_prune_slack_is_configurable():
    rows = _rows(100, start_t=0.0, step=1.0)
    assert ms._prune_caption_rows(rows, now=100.0, max_secs=50.0,
                                  max_rows=0, slack_secs=0) == 50
    assert len(rows) == 50


def test_prune_keeps_everything_inside_window():
    rows = _rows(50, start_t=0.0, step=60.0)
    assert ms._prune_caption_rows(rows, now=50 * 60,
                                  max_secs=180 * 60, max_rows=0) == 0
    assert len(rows) == 50


def test_prune_row_cap_is_a_backstop():
    rows = _rows(500, start_t=0.0, step=0.1)       # 50 s of very fast speech
    dropped = ms._prune_caption_rows(rows, now=50.0, max_secs=180 * 60,
                                     max_rows=100, slack_rows=0)
    assert dropped == 400 and len(rows) == 100
    assert rows[-1]["id"] == 499                   # newest rows are the ones kept


def test_prune_time_window_disabled():
    rows = _rows(10, start_t=0.0, step=3600.0)     # spread over 10 hours
    assert ms._prune_caption_rows(rows, now=10 * 3600,
                                  max_secs=0, max_rows=0) == 0


def test_prune_three_hour_window_at_realistic_cadence():
    # 30 finalized lines/minute for 4 hours; a 3 h window keeps the last 3 h.
    rows = _rows(30 * 60 * 4, start_t=0.0, step=2.0)
    ms._prune_caption_rows(rows, now=4 * 3600, max_secs=3 * 3600,
                           max_rows=12000, slack_secs=0)
    assert len(rows) == 5400                        # 3 h × 30 lines/min
    assert rows[0]["t"] >= 3600


def test_prune_empty_is_safe():
    rows = []
    assert ms._prune_caption_rows(rows, now=0.0, max_secs=60, max_rows=10) == 0


# ── mt_provider can pin the MT model independently of the ASR mode ───────────

def _mt_provider_engine(provider):
    return ms.LiveCaptionEngine(
        {"live_captions": {"mt_provider": provider}}, lambda ev: None)


def test_mt_provider_defaults_to_marian(monkeypatch):
    """With one recognition path there is no mode to follow: opus-mt is the
    default, NLLB stays reachable by name."""
    monkeypatch.setattr(ms, "_caption_backend_cache", {})
    monkeypatch.setattr(ms, "_MarianCaptionMT", lambda lc: "marian-backend")
    monkeypatch.setattr(ms, "_NLLBCaptionMT", lambda lc: "nllb-backend")
    for provider in ("default", "", "marian", "opus", "opus-mt", " MARIAN "):
        assert _mt_provider_engine(provider)._load_mt_backend() == "marian-backend"


def test_mt_provider_nllb_selected_by_name(monkeypatch):
    monkeypatch.setattr(ms, "_caption_backend_cache", {})
    monkeypatch.setattr(ms, "_MarianCaptionMT", lambda lc: "marian-backend")
    monkeypatch.setattr(ms, "_NLLBCaptionMT", lambda lc: "nllb-backend")
    assert _mt_provider_engine("nllb")._load_mt_backend() == "nllb-backend"


def test_marian_and_nllb_cached_separately(monkeypatch):
    calls = []
    monkeypatch.setattr(ms, "_caption_backend_cache", {})
    monkeypatch.setattr(ms, "_MarianCaptionMT",
                        lambda lc: calls.append("m") or "marian-backend")
    monkeypatch.setattr(ms, "_NLLBCaptionMT",
                        lambda lc: calls.append("n") or "nllb-backend")
    _mt_provider_engine("marian")._load_mt_backend()
    _mt_provider_engine("nllb")._load_mt_backend()
    _mt_provider_engine("marian")._load_mt_backend()      # cached
    assert calls == ["m", "n"]


def test_nllb_reads_its_own_config_block(monkeypatch):
    """mt_model / mt_int8 moved out of the deleted "accurate" block."""
    seen = {}
    monkeypatch.setattr(ms, "_caption_backend_cache", {})
    monkeypatch.setattr(ms, "_NLLBCaptionMT",
                        lambda lc: seen.setdefault("nllb", lc.get("nllb")))
    ms.LiveCaptionEngine(
        {"live_captions": {"mt_provider": "nllb",
                           "nllb": {"mt_model": "x/y", "mt_int8": False}}},
        lambda ev: None)._load_mt_backend()
    assert seen["nllb"] == {"mt_model": "x/y", "mt_int8": False}


# ── glossary: fuzzy matching is for CJK-mixed lines only ─────────────────────

def test_glossary_no_fuzzy_noise_on_english():
    hw = ["Acme", "ChatOps", "GKE", "Portland"]
    assert ms._glossary_candidates(
        "Now we were told about this RFP around four days", hw) == []


def test_glossary_literal_hits_still_work_on_english():
    hw = ["ChatOps", "GKE"]
    out = ms._glossary_candidates(
        "We use ChatOps for approvals and GKE for workloads", hw)
    assert out == ["ChatOps", "GKE"]


def test_glossary_fuzzy_still_works_on_cjk_mixed():
    assert "ChatOps" in ms._glossary_candidates("拆特ops的审批流程", ["ChatOps"])


# ── the in-flight line gets the same stutter collapse as finalized ones ──────

def test_partial_is_tidied():
    """Measured on a real meeting, the partial line carried 4-19x the
    duplication of the offline transcript of the same audio, because the
    collapse only ran at finalize. 「大大大家大家」 was on screen verbatim."""
    sink = _EventSink()
    eng = ms.LiveCaptionEngine({}, sink)
    eng._partial_min_interval = 0.0
    eng._emit_partial("因为因为我现在现在大大大家大家嗯嗯也有在在在在上")
    ev = sink.wait_for("partial")
    assert ev
    assert "大大大家大家" not in ev["text"]
    assert "因为因为" not in ev["text"]
    assert "在在在在" not in ev["text"]
    assert "嗯" not in ev["text"]          # filler, dropped


def test_partial_translation_input_is_tidied():
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    eng._emit_partial("训练练练习记者已已经")
    with eng._partial_lock:
        pending = eng._partial_pending
    assert "练练练" not in pending


def test_partial_keeps_legitimate_reduplication():
    sink = _EventSink()
    eng = ms.LiveCaptionEngine({}, sink)
    eng._partial_min_interval = 0.0
    eng._emit_partial("谢谢，我们好好聊一下")
    ev = sink.wait_for("partial")
    assert ev and ev["text"] == "谢谢，我们好好聊一下"


def test_partial_english_untouched():
    sink = _EventSink()
    eng = ms.LiveCaptionEngine({}, sink)
    eng._partial_min_interval = 0.0
    eng._emit_partial("This is a very very long English sentence.")
    ev = sink.wait_for("partial")
    assert ev and ev["text"] == "This is a very very long English sentence."


# ── chunk-boundary dedup: streaming decoders commit twice at the join ────────

class _FakeStream:
    def accept_waveform(self, sr, samples):
        pass


class _FakeRecognizer:
    """Replays a scripted sequence of get_result() values."""

    def __init__(self, results):
        self._results = list(results)
        self._i = -1
        self.endpoint_at = set()
        self.resets = 0

    def create_stream(self):
        return _FakeStream()

    def is_ready(self, stream):
        return False

    def decode_stream(self, stream):
        pass

    def step(self):
        self._i += 1

    def get_result(self, stream):
        return self._results[min(self._i, len(self._results) - 1)]

    def is_endpoint(self, stream):
        return self._i in self.endpoint_at

    def reset(self, stream):
        self.resets += 1


def _sherpa_with(results):
    asr = ms._SherpaCaptionASR.__new__(ms._SherpaCaptionASR)
    asr._rec = _FakeRecognizer(results)
    asr._stream = asr._rec.create_stream()
    asr._hotwords = []
    asr._seg_audio = []
    asr._raw_hyp = ""
    asr._clean_hyp = ""
    asr._emit_partial = lambda t: None
    asr._emit_final = lambda t, a=None: None
    return asr


class TestAppendHypothesis:
    def test_strips_duplicate_at_the_join(self):
        # 「今天」 committed, then the next chunk commits it again.
        assert ms._append_hypothesis("因为今天", "今天天看") == "因为今天天看"

    def test_strips_longer_overlap(self):
        assert ms._append_hypothesis("一万七十三", "十三你要") == "一万七十三你要"

    def test_leaves_genuine_continuation(self):
        assert ms._append_hypothesis("我们把流量", "切到新集群") == "我们把流量切到新集群"

    def test_single_char_overlap_left_alone(self):
        # 2-char minimum on purpose: 「看看」 split across the boundary is
        # real Chinese, and losing it reads as an error.
        assert ms._append_hypothesis("你看", "看这个") == "你看看这个"

    def test_empty_delta(self):
        assert ms._append_hypothesis("已有文本", "") == "已有文本"

    def test_empty_clean(self):
        assert ms._append_hypothesis("", "第一段文本") == "第一段文本"


class TestSherpaHypothesisAdvance:
    def test_dedups_across_chunks(self):
        asr = _sherpa_with(["因为今天", "因为今天今天天看"])
        asr._rec.step(); asr._advance_hypothesis()
        asr._rec.step()
        assert asr._advance_hypothesis() == "因为今天天看"

    def test_tracks_raw_separately_from_clean(self):
        """The recognizer keeps its own append-only history; ours diverges."""
        asr = _sherpa_with(["因为今天", "因为今天今天天看"])
        asr._rec.step(); asr._advance_hypothesis()
        asr._rec.step(); asr._advance_hypothesis()
        assert asr._raw_hyp == "因为今天今天天看"      # untouched
        assert asr._clean_hyp == "因为今天天看"         # deduped

    def test_unchanged_result_is_idempotent(self):
        asr = _sherpa_with(["评估是两万", "评估是两万"])
        asr._rec.step(); first = asr._advance_hypothesis()
        asr._rec.step()
        assert asr._advance_hypothesis() == first

    def test_recognizer_rewrite_is_trusted_wholesale(self):
        # A build that rescores earlier text breaks the append-only
        # assumption; splicing two disagreeing histories would corrupt.
        asr = _sherpa_with(["原来的假设", "完全不同的新假设"])
        asr._rec.step(); asr._advance_hypothesis()
        asr._rec.step()
        assert asr._advance_hypothesis() == "完全不同的新假设"

    def test_endpoint_resets_hypothesis_state(self):
        asr = _sherpa_with(["第一段内容", ""])
        finals = []
        asr._emit_final = lambda t, a=None: finals.append(t)
        asr._rec.endpoint_at = {0}
        asr._rec.step()
        asr.accept(np.zeros(1600, dtype=np.float32))
        assert finals == ["第一段内容"]
        assert asr._raw_hyp == "" and asr._clean_hyp == ""

    def test_flush_resets_state(self):
        asr = _sherpa_with(["未断句的内容"])
        finals = []
        asr._emit_final = lambda t, a=None: finals.append(t)
        asr._rec.step()
        asr.flush()
        assert finals == ["未断句的内容"]
        assert asr._raw_hyp == "" and asr._clean_hyp == ""

    def test_realistic_growth_sequence(self):
        """The exact shape measured off real audio."""
        asr = _sherpa_with([
            "评估是两万还是",
            "评估是两万还是第一般还",
            "评估是两万还是第一般还可以在",
            "评估是两万还是第一般还可以在再再再去",
            "评估是两万还是第一般还可以在再再再去多贵一点因为今天",
            "评估是两万还是第一般还可以在再再再去多贵一点因为今天今天天看",
        ])
        out = ""
        for _ in range(6):
            asr._rec.step()
            out = asr._advance_hypothesis()
        assert "今天今天天" not in out      # the at-join duplicate is gone
        assert out.endswith("因为今天天看")


def test_ring_overflow_is_logged(monkeypatch):
    """Dropping audio used to be completely silent, which made "why did the
    captions miss that" unanswerable — and the replay tool hit it at once by
    feeding faster than the recognizer consumes."""
    logged = []
    monkeypatch.setattr(ms, "_log", lambda cat, msg: logged.append(msg))
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    eng._running = True
    one_sec = np.zeros(16000, dtype=np.float32)
    for _ in range(int(ms._CAPTION_RING_SECONDS) + 5):
        eng.feed("mic", one_sec, 16000)
    assert eng._dropped_secs > 0
    assert any("ring buffer overflow" in m for m in logged)


def test_no_overflow_log_when_keeping_up(monkeypatch):
    logged = []
    monkeypatch.setattr(ms, "_log", lambda cat, msg: logged.append(msg))
    eng = ms.LiveCaptionEngine({}, lambda ev: None)
    eng._running = True
    for _ in range(5):
        eng.feed("mic", np.zeros(16000, dtype=np.float32), 16000)
        eng._drain_mixed()
    assert eng._dropped_secs == 0
    assert not [m for m in logged if "overflow" in m]
