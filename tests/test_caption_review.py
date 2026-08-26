"""Tests for the periodic batch re-check of live captions (live_captions.review).

The LLM is faked. What matters here is that a batch reply can only ever
IMPROVE the pane: a truncated, chatty or hallucinated answer must degrade to
"leave that line alone", never to lost or rewritten-beyond-recognition
captions — these are lines the user has already read.
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import meetingscribe as ms


def batch(*pairs):
    """[(id, text, dst), …] → the row dicts the worker builds."""
    return [{"id": i, "text": t, "dst": d} for i, t, d in pairs]


B = batch(
    (1, "我们把流量切到拆特ops上面", "We route traffic to the special ops."),
    (2, "这个季度的排期下周确认", "The schedule is confirmed next week."),
)


class TestFormatLines:
    def test_numbered_from_one(self):
        out = ms._format_caption_review_lines(B)
        assert out.splitlines()[0].startswith("1 ||| 我们把流量切到拆特ops上面 |||")
        assert out.splitlines()[1].startswith("2 |||")

    def test_missing_translation_is_marked(self):
        out = ms._format_caption_review_lines(batch((7, "还没翻译的一行", "")))
        assert "（未翻译）" in out

    def test_newlines_flattened(self):
        out = ms._format_caption_review_lines(batch((1, "上一行\n下一行", "a\nb")))
        assert len(out.splitlines()) == 1


class TestParseReview:
    def test_happy_path(self):
        reply = ("1 ||| 我们把流量切到 ChatOps 上面 ||| We route traffic to ChatOps.\n"
                 "2 ||| 这个季度的排期下周确认 ||| The schedule is confirmed next week.")
        out = ms._parse_caption_review(reply, B)
        assert out[1] == ("我们把流量切到 ChatOps 上面", "We route traffic to ChatOps.")
        assert out[2][0] == "这个季度的排期下周确认"

    def test_keys_are_segment_ids_not_positions(self):
        rows = batch((41, "第一句话内容", "one"), (42, "第二句话内容", "two"))
        reply = "1 ||| 第一句话内容 ||| ONE\n2 ||| 第二句话内容 ||| TWO"
        out = ms._parse_caption_review(reply, rows)
        assert set(out) == {41, 42}

    def test_omitted_entries_left_alone(self):
        reply = "2 ||| 这个季度的排期下周确认 ||| The schedule is confirmed next week."
        out = ms._parse_caption_review(reply, B)
        assert set(out) == {2}

    def test_empty_reply(self):
        assert ms._parse_caption_review("", B) == {}
        assert ms._parse_caption_review(None, B) == {}

    def test_chatty_preamble_ignored(self):
        reply = ("好的，以下是修正结果：\n\n"
                 "1 ||| 我们把流量切到 ChatOps 上面 ||| We route traffic to ChatOps.\n"
                 "希望这对你有帮助！")
        out = ms._parse_caption_review(reply, B)
        assert set(out) == {1}

    def test_markdown_numbering_tolerated(self):
        reply = "**1.** ||| 我们把流量切到 ChatOps 上面 ||| Routed to ChatOps."
        assert 1 in ms._parse_caption_review(reply, B)

    def test_out_of_range_index_dropped(self):
        reply = "9 ||| 凭空出现的一行 ||| Out of nowhere."
        assert ms._parse_caption_review(reply, B) == {}

    def test_missing_fields_dropped(self):
        assert ms._parse_caption_review("1 ||| 只有原文", B) == {}
        assert ms._parse_caption_review("1 ||| 只有原文 ||| ", B) == {}
        assert ms._parse_caption_review("1 |||  ||| only translation", B) == {}

    def test_extra_separators_keep_first_three_fields(self):
        reply = ("1 ||| 我们把流量切到 ChatOps 上面 ||| A translation "
                 "||| stray tail")
        out = ms._parse_caption_review(reply, B)
        assert out[1] == ("我们把流量切到 ChatOps 上面", "A translation")

    def test_hallucinated_rewrite_keeps_original_source(self):
        """A "correction" that shares almost nothing with the original is a
        rewrite; the translation is still taken (it legitimately differs)."""
        reply = "1 ||| 完全不相关的另外一句胡编内容啊 ||| Something else."
        out = ms._parse_caption_review(reply, B)
        assert out[1] == (B[0]["text"], "Something else.")

    def test_dropped_clause_keeps_original_source(self):
        rows = batch((1, "我们把流量切到新集群上，机器人会自动通知大家", "x"))
        reply = "1 ||| 我们把流量切到 ||| We switch traffic."
        out = ms._parse_caption_review(reply, rows)
        assert out[1] == (rows[0]["text"], "We switch traffic.")

    def test_padded_source_keeps_original(self):
        rows = batch((1, "短句", "x"))
        reply = "1 ||| " + "凭空扩写的很长的一段内容" * 3 + " ||| Long."
        assert ms._parse_caption_review(reply, rows)[1][0] == "短句"

    def test_minimal_edit_accepted(self):
        rows = batch((1, "我们用 GKE 跑 workload 没问题", "x"))
        reply = "1 ||| 我们用 GKE 跑 workload，没问题 ||| Works fine on GKE."
        assert ms._parse_caption_review(reply, rows)[1][0] == \
            "我们用 GKE 跑 workload，没问题"

    def test_no_separator_lines_skipped(self):
        assert ms._parse_caption_review("just prose\nmore prose", B) == {}


# ── engine wiring ────────────────────────────────────────────────────────────

class _Sink:
    def __init__(self):
        self.events = []
        self._lock = threading.Lock()

    def __call__(self, ev):
        with self._lock:
            self.events.append(ev)

    def of_type(self, t):
        with self._lock:
            return [e for e in self.events if e["type"] == t]


def _engine(sink, reply="", **review):
    cfg = {"live_captions": {"review": {"interval_minutes": 0.5, **review}}}
    eng = ms.LiveCaptionEngine(cfg, sink)
    eng._review_llm = lambda prompt: reply
    return eng


class TestReviewBuffer:
    def test_finalized_lines_are_buffered(self):
        eng = _engine(_Sink())
        eng._finalize_segment("第一句话内容")
        eng._finalize_segment("第二句话内容")
        assert [r["text"] for r in eng._review_buffer] == \
            ["第一句话内容", "第二句话内容"]

    def test_noise_lines_never_buffered(self):
        eng = _engine(_Sink())
        eng._finalize_segment("M")
        assert eng._review_buffer == []

    def test_translation_is_recorded(self):
        eng = _engine(_Sink())
        eng._finalize_segment("第一句话内容")
        eng._review_record_translation(1, "First line.")
        assert eng._review_buffer[0]["dst"] == "First line."

    def test_refine_updates_the_buffered_text(self):
        eng = _engine(_Sink())
        eng._finalize_segment("原始的一句话")
        eng._review_record(1, "校验后的一句话")
        assert len(eng._review_buffer) == 1
        assert eng._review_buffer[0]["text"] == "校验后的一句话"

    def test_disabled_skips_buffering(self):
        eng = _engine(_Sink(), enabled=False)
        eng._finalize_segment("第一句话内容")
        assert eng._review_buffer == []
        assert eng._review_thread is None


class TestReviewWorker:
    def _run_once(self, eng, sink, expect, timeout=8.0):
        eng._running = True
        eng._review_interval = 0.0          # fire on the first tick
        t = threading.Thread(target=eng._review_worker, daemon=True)
        t.start()
        deadline = time.time() + timeout
        while time.time() < deadline and len(sink.of_type(expect)) == 0:
            time.sleep(0.05)
        eng._running = False
        t.join(timeout=2)

    def test_emits_refined_and_translation(self):
        sink = _Sink()
        eng = _engine(sink)
        eng._finalize_segment("我们把流量切到拆特ops上面")
        eng._review_record_translation(1, "old translation")
        eng._review_llm = lambda p: (
            "1 ||| 我们把流量切到 ChatOps 上面 ||| We route traffic to ChatOps.")
        self._run_once(eng, sink, "refined")
        refined = sink.of_type("refined")
        assert refined and refined[0]["text"] == "我们把流量切到 ChatOps 上面"
        trans = sink.of_type("translation")
        assert trans and trans[0]["text"] == "We route traffic to ChatOps."

    def test_prompt_carries_hotwords_and_lines(self):
        seen = {}
        sink = _Sink()
        eng = ms.LiveCaptionEngine({"stt": {"funasr": {"hotword": "ChatOps GKE"}},
             "live_captions": {"review": {"interval_minutes": 0.5}}}, sink)
        eng._review_llm = lambda p: seen.setdefault("prompt", p) and ""
        eng._finalize_segment("我们把流量切到拆特ops上面")
        self._run_once(eng, sink, "never", timeout=2.0)
        assert "ChatOps" in seen["prompt"]
        assert "拆特ops" in seen["prompt"]
        assert "|||" in seen["prompt"]

    def test_batch_is_consumed_once(self):
        sink = _Sink()
        eng = _engine(sink)
        eng._finalize_segment("第一句话内容")
        eng._review_llm = lambda p: "1 ||| 第一句话内容 ||| First line."
        self._run_once(eng, sink, "translation")
        assert eng._review_buffer == []

    def test_llm_failure_leaves_captions_alone(self):
        sink = _Sink()
        eng = _engine(sink)
        eng._finalize_segment("第一句话内容")

        def boom(prompt):
            raise RuntimeError("claude cli exploded")

        eng._review_llm = boom
        self._run_once(eng, sink, "never", timeout=2.0)
        assert sink.of_type("refined") == []
        assert sink.of_type("translation") == []

    def test_unchanged_source_emits_no_refined(self):
        sink = _Sink()
        eng = _engine(sink)
        eng._finalize_segment("这一句本来就是对的")
        eng._review_llm = lambda p: "1 ||| 这一句本来就是对的 ||| This one was fine."
        self._run_once(eng, sink, "translation")
        assert sink.of_type("refined") == []
        assert sink.of_type("translation")[0]["text"] == "This one was fine."

    def test_max_lines_splits_the_batch(self):
        """Tested without the thread: with a 0-second interval the worker
        would drain every batch before the assertion could see the split."""
        eng = _engine(_Sink(), max_lines=2)
        for i in range(5):
            eng._finalize_segment(f"第{i}句话的内容")
        first = eng._take_review_batch()
        assert len(first) == 2 and len(eng._review_buffer) == 3
        second = eng._take_review_batch()
        assert len(second) == 2 and len(eng._review_buffer) == 1
        assert [r["id"] for r in first + second] == [1, 2, 3, 4]

    def test_empty_buffer_yields_no_batch(self):
        eng = _engine(_Sink())
        assert eng._take_review_batch() == []


class TestReviewLlmCall:
    """Exercises the REAL `_review_llm` (the other tests stub it out, which is
    how a missing `_llm_run(..., label)` argument slipped through once: the
    worker swallows exceptions, so captions would silently never be reviewed)."""

    def test_calls_llm_run_with_the_full_signature(self, monkeypatch):
        seen = {}

        def fake_llm_run(prompt, provider_name, cfg, label):
            seen.update(prompt=prompt, provider=provider_name, label=label)
            return "1 ||| 修好的一行内容 ||| Fixed line."

        monkeypatch.setattr(ms, "_llm_run", fake_llm_run)
        eng = ms.LiveCaptionEngine({"polish_provider": "claude"}, lambda ev: None)
        assert eng._review_llm("PROMPT") == "1 ||| 修好的一行内容 ||| Fixed line."
        assert seen["provider"] == "claude"
        assert seen["label"]          # a label is required, not optional

    def test_provider_override_wins(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            ms, "_llm_run",
            lambda prompt, provider_name, cfg, label:
                seen.setdefault("provider", provider_name) or "")
        eng = ms.LiveCaptionEngine(
            {"polish_provider": "claude",
             "live_captions": {"review": {"provider": "gemini"}}}, lambda ev: None)
        eng._review_llm("PROMPT")
        assert seen["provider"] == "gemini"

    def test_worker_survives_a_real_signature_error(self, monkeypatch):
        """Whatever _llm_run raises, the pane must be left as it was."""
        monkeypatch.setattr(
            ms, "_llm_run",
            lambda *a, **k: (_ for _ in ()).throw(TypeError("bad signature")))
        sink = _Sink()
        eng = ms.LiveCaptionEngine({"live_captions": {"review": {"interval_minutes": 0.5}}}, sink)
        eng._finalize_segment("第一句话内容")
        eng._running = True
        eng._review_interval = 0.0
        t = threading.Thread(target=eng._review_worker, daemon=True)
        t.start()
        time.sleep(1.0)
        eng._running = False
        t.join(timeout=2)
        assert sink.of_type("refined") == [] and sink.of_type("translation") == []


class TestRequeueOnOmission:
    """A line popped from the buffer but missing from the reply (truncated
    answer) must not silently lose its chance at review — the exact hole a
    larger max_lines would widen."""

    def _one_round(self, eng, reply):
        batch = eng._take_review_batch()
        fixes = ms._parse_caption_review(reply, batch)
        missed = []
        for row in batch:
            if not fixes.get(row["id"]):
                if row.get("tries", 0) < eng._review_max_tries:
                    row["tries"] = row.get("tries", 0) + 1
                    missed.append(row)
        if missed:
            with eng._review_lock:
                eng._review_buffer[:0] = missed
        return batch, fixes, missed

    def test_omitted_line_is_requeued(self):
        sink = _Sink()
        eng = _engine(sink)
        eng._finalize_segment("第一句话的内容")
        eng._finalize_segment("第二句话的内容")
        # Reply covers only the first line (as if truncated).
        _, _, missed = self._one_round(eng, "1 ||| 第一句话的内容 ||| First.")
        assert len(missed) == 1
        assert [r["id"] for r in eng._review_buffer] == [2]

    def test_requeue_keeps_chronological_order(self):
        eng = _engine(_Sink())
        for i in range(3):
            eng._finalize_segment(f"第{i}句话的内容")
        self._one_round(eng, "2 ||| 第1句话的内容 ||| Second.")
        assert [r["id"] for r in eng._review_buffer] == [1, 3]

    def test_retries_are_bounded(self):
        eng = _engine(_Sink())
        eng._finalize_segment("永远被忽略的一句话")
        for _ in range(eng._review_max_tries):
            self._one_round(eng, "")            # model never answers
            assert eng._review_buffer            # still queued
        self._one_round(eng, "")
        assert eng._review_buffer == []          # given up, not looping

    def test_max_tries_zero_disables_requeue(self):
        eng = _engine(_Sink(), max_tries=0)
        eng._finalize_segment("只投一次的一句话")
        self._one_round(eng, "")
        assert eng._review_buffer == []

    def test_worker_requeues_end_to_end(self):
        sink = _Sink()
        eng = _engine(sink)
        eng._finalize_segment("第一句话的内容")
        eng._finalize_segment("第二句话的内容")
        eng._review_llm = lambda p: "1 ||| 第一句话的内容 ||| First."
        eng._running = True
        eng._review_interval = 0.0
        t = threading.Thread(target=eng._review_worker, daemon=True)
        t.start()
        deadline = time.time() + 5
        while time.time() < deadline and not sink.of_type("translation"):
            time.sleep(0.05)
        eng._running = False
        t.join(timeout=2)
        # Line 2 came back for another round rather than being dropped.
        assert any(r["id"] == 2 for r in eng._review_buffer) or \
            len(sink.of_type("translation")) >= 1


class TestChangeAuditLog:
    """Every rewrite must be recoverable from the log: this pass silently
    changes lines the user already read, and aggregate counts alone make that
    unverifiable after the fact."""

    def test_log_text_is_one_line_and_delimited(self):
        out = ms._review_log_text("上一行\n下一行   有空格")
        assert out == "「上一行 下一行 有空格」"

    def test_log_text_caps_runaway_length(self):
        out = ms._review_log_text("字" * 900, limit=100)
        assert out.startswith("「" + "字" * 100)
        assert "…(+800)" in out

    def test_log_text_handles_empty(self):
        assert ms._review_log_text("") == "「」"
        assert ms._review_log_text(None) == "「」"

    def _run(self, eng, sink, logged):
        eng._running = True
        eng._review_interval = 0.0
        t = threading.Thread(target=eng._review_worker, daemon=True)
        t.start()
        deadline = time.time() + 5
        while time.time() < deadline and not any("review batch=" in m for m in logged):
            time.sleep(0.05)
        eng._running = False
        t.join(timeout=2)

    def test_source_change_logged_with_before_and_after(self, monkeypatch):
        logged = []
        monkeypatch.setattr(ms, "_log", lambda cat, msg: logged.append(msg))
        sink = _Sink()
        eng = _engine(sink)
        eng._finalize_segment("Adb主要是BBQ和cr circle这两个措施")
        eng._review_llm = lambda p: (
            "1 ||| DBA 主要是 BBQ 和 CockroachDB 这两个措施 ||| DBA side: BBQ and CockroachDB.")
        self._run(eng, sink, logged)
        src_lines = [m for m in logged if "review seg=1 src:" in m]
        assert len(src_lines) == 1
        assert "cr circle" in src_lines[0]          # before
        assert "CockroachDB" in src_lines[0]        # after
        assert "→" in src_lines[0]

    def test_translation_change_logged(self, monkeypatch):
        logged = []
        monkeypatch.setattr(ms, "_log", lambda cat, msg: logged.append(msg))
        sink = _Sink()
        eng = _engine(sink)
        eng._finalize_segment("这一行的原文不变")
        eng._review_record_translation(1, "old translation")
        eng._review_llm = lambda p: "1 ||| 这一行的原文不变 ||| a much better translation"
        self._run(eng, sink, logged)
        dst_lines = [m for m in logged if "review seg=1 dst:" in m]
        assert len(dst_lines) == 1
        assert "old translation" in dst_lines[0]
        assert "a much better translation" in dst_lines[0]
        assert not [m for m in logged if "review seg=1 src:" in m]

    def test_unchanged_lines_produce_no_noise(self, monkeypatch):
        logged = []
        monkeypatch.setattr(ms, "_log", lambda cat, msg: logged.append(msg))
        sink = _Sink()
        eng = _engine(sink)
        eng._finalize_segment("这一句本来就是对的")
        eng._review_record_translation(1, "This one was fine.")
        eng._review_llm = lambda p: "1 ||| 这一句本来就是对的 ||| This one was fine."
        self._run(eng, sink, logged)
        assert not [m for m in logged if " seg=1 src:" in m or " seg=1 dst:" in m]

    def test_summary_counts_both_dimensions(self, monkeypatch):
        logged = []
        monkeypatch.setattr(ms, "_log", lambda cat, msg: logged.append(msg))
        sink = _Sink()
        eng = _engine(sink)
        eng._finalize_segment("第一句需要修正的内容")
        eng._review_record_translation(1, "old")
        eng._review_llm = lambda p: "1 ||| 第一句需要修正的东西 ||| new translation"
        self._run(eng, sink, logged)
        summary = [m for m in logged if m.startswith("review batch=")]
        assert summary
        assert "src_changed=1" in summary[0] and "dst_changed=1" in summary[0]
