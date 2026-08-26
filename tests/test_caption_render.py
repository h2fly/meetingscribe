"""Tests for _CaptionDocRenderer — the incremental caption-pane renderer.

Runs against a REAL QTextBrowser (offscreen): the whole point of the class is
that patching a tail produces the same document Qt would have built from a
full `setHtml`, and only Qt can answer that. Skipped when PyQt6 is absent.

The core invariants, asserted after every operation:
  1. plain text identical to a full-rebuild reference document;
  2. blockCount == 2 × committed groups (the bookkeeping the patch relies on);
  3. the common paths (append / tail edit) do NOT fall back to "full".
"""

import html
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import meetingscribe as ms

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication, QTextBrowser  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def pane(qapp):
    view = QTextBrowser()
    return view, ms._CaptionDocRenderer(view)


def group(i, translated=True, role=None):
    """One caption group's HTML — exactly two paragraphs, as the GUI emits."""
    tag = (f'<span style="color:#0a84ff; font-weight:600;">[{role}]</span> '
           if role else "")
    src = f"这是第 {i} 句字幕，长度接近真实一行"
    dst = f"Caption line {i}." if translated else "…"
    return (f'<p style="margin:6px 0 0 0; color:#1f1f1f;">'
            f'{tag}{html.escape(src)}</p>'
            f'<p style="margin:1px 0 0 0; color:#0066d6;">{html.escape(dst)}</p>')


def assert_matches_full_rebuild(view, groups):
    """Invariants 1 + 2."""
    reference = QTextBrowser()
    reference.setHtml("".join(groups))
    assert view.document().toPlainText() == reference.document().toPlainText()
    if groups:
        assert view.document().blockCount() == 2 * len(groups)


class TestIncrementalPaths:
    def test_first_paint_is_full(self, pane):
        view, r = pane
        assert r.render([group(0)]) == "full"
        assert_matches_full_rebuild(view, [group(0)])

    def test_appending_a_line_patches(self, pane):
        view, r = pane
        # Seeded with 2 groups: a patch must leave at least one untouched
        # leading group, so a 1-group document legitimately rebuilds.
        r.render([group(0), group(1)])
        extended = [group(0), group(1), group(2, translated=False)]
        assert r.render(extended) == "append"
        assert_matches_full_rebuild(view, extended)

    def test_appending_to_one_group_document_rebuilds(self, pane):
        view, r = pane
        r.render([group(0)])
        extended = [group(0), group(1)]
        assert r.render(extended) == "full"      # nothing to keep untouched
        assert_matches_full_rebuild(view, extended)

    def test_translation_landing_patches(self, pane):
        view, r = pane
        groups = [group(0), group(1), group(2, translated=False)]
        r.render(groups)
        done = [group(0), group(1), group(2)]
        assert r.render(done) == "tail"
        assert_matches_full_rebuild(view, done)

    def test_refine_rewriting_a_middle_group_patches(self, pane):
        view, r = pane
        r.render([group(0)])
        r.render([group(0), group(1), group(2)])
        edited = [group(0), group(1, translated=False), group(2)]
        assert r.render(edited) == "tail"
        assert_matches_full_rebuild(view, edited)

    def test_identical_render_is_a_noop(self, pane):
        view, r = pane
        groups = [group(0), group(1)]
        r.render(groups)
        assert r.render(list(groups)) == "unchanged"
        assert_matches_full_rebuild(view, groups)

    def test_long_session_append_stays_incremental(self, pane):
        """The regression that matters: with 3 h of scroll-back retained, a
        new line must not drag the whole document through a re-layout."""
        view, r = pane
        groups = [group(i) for i in range(1200)]
        assert r.render(groups) == "full"          # seeding is one full paint
        for i in range(1200, 1210):
            groups = groups + [group(i, translated=False)]
            assert r.render(groups) == "append"
            groups[-1] = group(i)                  # its translation lands
            assert r.render(groups) == "tail"
        assert_matches_full_rebuild(view, groups)

    def test_sequence_of_mixed_edits_never_diverges(self, pane):
        view, r = pane
        groups = []
        for i in range(25):
            groups.append(group(i, translated=False))
            r.render(list(groups))
            assert_matches_full_rebuild(view, groups)
            groups[-1] = group(i)
            r.render(list(groups))
            assert_matches_full_rebuild(view, groups)
            if i >= 3:                              # a late refine, 3 back
                groups[i - 3] = group(i - 3, role="mic")
                r.render(list(groups))
                assert_matches_full_rebuild(view, groups)


class TestFallbacks:
    def test_divergence_at_index_zero_rebuilds(self, pane):
        view, r = pane
        r.render([group(0), group(1)])
        replaced = [group(9), group(1)]
        assert r.render(replaced) == "full"
        assert_matches_full_rebuild(view, replaced)

    def test_history_eviction_rebuilds(self, pane):
        # Retention dropping the oldest rows shifts every group → index 0
        # differs → full rebuild. Rare (once per evicted line) and correct.
        view, r = pane
        r.render([group(0), group(1), group(2)])
        evicted = [group(1), group(2)]
        assert r.render(evicted) == "full"
        assert_matches_full_rebuild(view, evicted)

    def test_reset_forces_next_render_full(self, pane):
        view, r = pane
        groups = [group(0), group(1)]
        r.render(groups)
        r.reset()
        assert r.render(groups) == "full"
        assert_matches_full_rebuild(view, groups)

    def test_external_document_change_is_survived(self, pane):
        """If anything else touched the document the block invariant breaks;
        the renderer must notice and rebuild instead of corrupting the pane."""
        view, r = pane
        groups = [group(0), group(1)]
        r.render(groups)
        view.setHtml("<p>something else entirely</p>")   # invariant broken
        extended = groups + [group(2)]
        assert r.render(extended) == "full"
        assert_matches_full_rebuild(view, extended)

    def test_empty_shows_placeholder(self, pane):
        view, r = pane
        r.render([group(0)])
        assert r.render([], '<p style="color:#8a8a8a;">还没有字幕</p>') == "full"
        assert "还没有字幕" in view.document().toPlainText()

    def test_placeholder_render_is_idempotent(self, pane):
        view, r = pane
        placeholder = '<p style="color:#8a8a8a;">还没有字幕</p>'
        assert r.render([], placeholder) == "full"
        assert r.render([], placeholder) == "unchanged"

    def test_rows_after_placeholder(self, pane):
        view, r = pane
        r.render([], '<p style="color:#8a8a8a;">还没有字幕</p>')
        groups = [group(0)]
        assert r.render(groups) == "full"
        assert_matches_full_rebuild(view, groups)


class TestScrollBehaviour:
    def test_tail_following_when_pinned_at_bottom(self, pane):
        view, r = pane
        view.resize(400, 120)
        r.render([group(i) for i in range(200)])
        sb = view.verticalScrollBar()
        sb.setValue(sb.maximum())
        groups = [group(i) for i in range(201)]
        r.render(groups)
        assert sb.value() >= sb.maximum() - 8

    def test_scroll_position_kept_when_reading_back(self, pane):
        view, r = pane
        view.resize(400, 120)
        r.render([group(i) for i in range(200)])
        sb = view.verticalScrollBar()
        sb.setValue(0)                       # user scrolled to the top
        r.render([group(i) for i in range(201)])
        assert sb.value() == 0


def test_first_divergence_helper():
    f = ms._CaptionDocRenderer._first_divergence
    assert f([], []) == 0
    assert f(["a"], ["a"]) == 1
    assert f(["a", "b"], ["a", "c"]) == 1
    assert f(["a"], ["a", "b"]) == 1
    assert f(["a", "b"], ["a"]) == 1
    assert f(["x"], ["y"]) == 0
