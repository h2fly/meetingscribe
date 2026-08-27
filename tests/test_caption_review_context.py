"""Tests for the review pass's backlog cap (P2) and read-only context (P1).

Both exist because of measured behaviour on one real 51-minute session: the
pending queue had no total bound (only per-batch), and each batch was reviewed
with no knowledge of what earlier batches had already decided, so terminology
could oscillate between windows.
"""

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import meetingscribe as ms


def _engine(**review):
    cfg = {"live_captions": {"review": {"enabled": True, **review}}}
    return ms.LiveCaptionEngine(cfg, lambda ev: None)


# ── P2: backlog cap ──────────────────────────────────────────────────────────

class TestBufferCap:
    def test_cap_never_below_one_batch(self):
        """A cap under max_lines would throw away rows a single call could
        have handled."""
        eng = _engine(max_lines=120, max_buffer=10)
        assert eng._review_max_buffer == 120

    def test_oldest_rows_are_dropped(self):
        eng = _engine(max_lines=2, max_buffer=3)
        for sid in range(1, 6):
            eng._review_record(sid, f"第{sid}句")
        assert [r["id"] for r in eng._review_buffer] == [3, 4, 5]

    def test_drop_is_logged_with_the_id_range(self, monkeypatch):
        """Silent truncation would let the pass report 'batch=8 parsed=8'
        while having thrown rows away — that reads as full coverage."""
        logged = []
        monkeypatch.setattr(ms, "_log", lambda cat, msg: logged.append(msg))
        eng = _engine(max_lines=1, max_buffer=2)
        for sid in range(1, 5):
            eng._review_record(sid, f"第{sid}句")
        drops = [m for m in logged if "review buffer full" in m]
        assert drops
        assert "seg=1..1" in drops[0] and "dropped 1" in drops[0]

    def test_backlog_warning_fires_before_anything_is_lost(self, monkeypatch):
        logged = []
        monkeypatch.setattr(ms, "_log", lambda cat, msg: logged.append(msg))
        eng = _engine(max_lines=2, max_buffer=100)
        for sid in range(1, 5):
            eng._review_record(sid, f"第{sid}句")
        assert any("provider behind" in m for m in logged)
        assert not any("buffer full" in m for m in logged)

    def test_warning_is_not_repeated_every_row(self, monkeypatch):
        logged = []
        monkeypatch.setattr(ms, "_log", lambda cat, msg: logged.append(msg))
        eng = _engine(max_lines=2, max_buffer=100)
        for sid in range(1, 30):
            eng._review_record(sid, f"第{sid}句")
        assert sum("provider behind" in m for m in logged) == 1

    def test_no_cap_effect_under_normal_load(self):
        eng = _engine(max_lines=120, max_buffer=240)
        for sid in range(1, 31):          # a 5-minute window at measured rate
            eng._review_record(sid, f"第{sid}句")
        assert len(eng._review_buffer) == 30

    def test_rewrite_of_a_buffered_row_does_not_grow_the_buffer(self):
        eng = _engine(max_lines=5, max_buffer=5)
        eng._review_record(1, "原始")
        eng._review_record(1, "refine 改过的")
        assert len(eng._review_buffer) == 1
        assert eng._review_buffer[0]["text"] == "refine 改过的"

    def test_trim_returns_empty_when_under_cap(self):
        eng = _engine(max_lines=1, max_buffer=5)
        eng._review_record(1, "一句")
        with eng._review_lock:
            dropped, depth = eng._trim_review_buffer()
        assert dropped == [] and depth == 1


# ── P1: read-only context ────────────────────────────────────────────────────

class TestContextFormatting:
    def test_first_batch_says_there_is_none(self):
        assert "第一批" in ms._format_caption_review_context([], 12)

    def test_disabled_by_zero(self):
        rows = [{"text": "有内容", "dst": "has content"}]
        assert "第一批" in ms._format_caption_review_context(rows, 0)

    def test_pairs_source_with_translation(self):
        rows = [{"text": "用 ChatOps 部署", "dst": "Deploy via ChatOps"}]
        out = ms._format_caption_review_context(rows, 12)
        assert out == "- 用 ChatOps 部署（Deploy via ChatOps）"

    def test_untranslated_row_still_carries_its_source(self):
        out = ms._format_caption_review_context([{"text": "还没翻译", "dst": ""}], 12)
        assert out == "- 还没翻译"

    def test_only_the_tail_is_kept(self):
        rows = [{"text": f"第{i}句", "dst": ""} for i in range(10)]
        out = ms._format_caption_review_context(rows, 3)
        assert out.splitlines() == ["- 第7句", "- 第8句", "- 第9句"]

    def test_never_uses_the_numbered_batch_format(self):
        """Numbering in the reply is POSITIONAL into the batch, so a context
        line echoed in `N ||| … ||| …` shape would land its text on an
        unrelated segment. Giving the model no such template is the guard."""
        rows = [{"text": "上一批的内容", "dst": "previous"}]
        out = ms._format_caption_review_context(rows, 12)
        assert ms._REVIEW_SEP not in out

    def test_newlines_are_flattened(self):
        out = ms._format_caption_review_context(
            [{"text": "两\n行", "dst": "two\nlines"}], 12)
        assert "\n" not in out.lstrip("- ")

    def test_blank_sources_are_skipped(self):
        out = ms._format_caption_review_context(
            [{"text": "   ", "dst": "x"}, {"text": "真内容", "dst": ""}], 12)
        assert out == "- 真内容"


class _FakeReviewLLM:
    """Captures the prompt and replies with the batch echoed back."""

    def __init__(self, reply_from=None):
        self.prompts: list = []
        self._reply_from = reply_from

    def __call__(self, prompt):
        self.prompts.append(prompt)
        # Rebuild a valid reply from the 【字幕】 section the engine sent.
        body = prompt.split("【字幕】")[-1]
        out = []
        for line in body.splitlines():
            if ms._REVIEW_SEP not in line:
                continue
            parts = [p.strip() for p in line.split(ms._REVIEW_SEP)]
            if len(parts) < 3 or not parts[0].strip().isdigit():
                continue
            src = self._reply_from(parts[1]) if self._reply_from else parts[1]
            out.append(f"{parts[0]} {ms._REVIEW_SEP} {src} "
                       f"{ms._REVIEW_SEP} translated")
        return "\n".join(out)


def _run_one_batch(eng):
    """Drive exactly one review cycle on the worker thread."""
    eng._review_interval = 0.01
    eng._running = True
    t = threading.Thread(target=eng._review_worker, daemon=True)
    t.start()
    import time
    deadline = time.time() + 5
    while time.time() < deadline and not eng._review_llm.prompts:
        time.sleep(0.01)
    eng._running = False
    t.join(timeout=2)


class TestContextInPrompt:
    def test_first_batch_has_no_context_and_later_ones_do(self):
        eng = _engine(context_lines=12, max_lines=10)
        eng._review_llm = _FakeReviewLLM()
        eng._review_record(1, "用 ChatOps 部署")
        _run_one_batch(eng)
        assert "第一批" in eng._review_llm.prompts[0]

        eng._review_record(2, "下一句")
        eng._review_llm.prompts.clear()
        _run_one_batch(eng)
        # The settled text of batch 1 now precedes batch 2's lines.
        prompt = eng._review_llm.prompts[0]
        # split from the END: rule 6 mentions 【上文】 literally, so the first
        # occurrence is in the rules, not the section itself.
        ctx = prompt.split("【上文】")[-1].split("【字幕】")[0]
        assert "用 ChatOps 部署" in ctx
        assert "下一句" not in ctx          # the batch itself is not context

    def test_correction_not_the_original_carries_forward(self):
        """Terminology decided by the pass is what the next batch must see —
        that is the whole point of the window."""
        eng = _engine(context_lines=12, max_lines=10)
        eng._review_llm = _FakeReviewLLM(
            reply_from=lambda s: s.replace("拆特ops", "ChatOps"))
        eng._review_record(1, "我们用拆特ops 发布")
        _run_one_batch(eng)
        assert eng._review_context[-1]["text"] == "我们用ChatOps 发布"

    def test_context_is_capped_by_config(self):
        eng = _engine(context_lines=2, max_lines=10)
        assert eng._review_context.maxlen == 2

    def test_zero_context_lines_still_substitutes_the_slot(self):
        eng = _engine(context_lines=0, max_lines=10)
        eng._review_llm = _FakeReviewLLM()
        eng._review_record(1, "一句话")
        _run_one_batch(eng)
        prompt = eng._review_llm.prompts[0]
        assert "{context}" not in prompt      # slot must never leak literally

    def test_prompt_slot_is_always_filled(self):
        eng = _engine(max_lines=10)
        eng._review_llm = _FakeReviewLLM()
        eng._review_record(1, "一句话")
        _run_one_batch(eng)
        for slot in ("{context}", "{lines}", "{hotwords}"):
            assert slot not in eng._review_llm.prompts[0]


class TestPromptParity:
    def test_config_jsonc_matches_the_builtin_default(self):
        """Project rule: every prompt ships as an editable block in
        config.jsonc, and the two must not drift."""
        cfg = ms.load_config()
        got = ms._resolve_prompt(cfg, "caption_review")
        assert got == ms._PROMPT_DEFAULTS["caption_review"].rstrip("\n")

    def test_default_prompt_declares_the_context_section(self):
        text = ms._PROMPT_DEFAULTS["caption_review"]
        assert "{context}" in text and "【上文】" in text
        # It must tell the model not to re-emit that section, or the
        # positional index mapping breaks.
        assert "不要输出这部分" in text
