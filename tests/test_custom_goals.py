"""
Tests for custom goals (自定义目标): the goal store, the AI prompt expansion,
generation over a transcript, and how goal outputs show up in the artifact scan.

The LLM is never called — `_llm_run` is monkeypatched.
Run: pytest tests/
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import meetingscribe as ms


@pytest.fixture(autouse=True)
def isolated_goals(monkeypatch, tmp_path):
    """Never touch the developer's real ~/Documents/meetingscribe/goals.json."""
    monkeypatch.setattr(ms, "GOALS_FILE", tmp_path / "goals.json")


def test_empty_when_no_file():
    assert ms._load_goals() == []


def test_upsert_creates_then_updates_in_place():
    g = ms._upsert_goal("需求清单", "整理客户需求", "你是……")
    assert ms._GOAL_ID_RE.match(g["id"])
    assert ms._load_goals() == [g]

    g2 = ms._upsert_goal("需求清单 v2", "整理客户需求", "新的 prompt", goal_id=g["id"])
    goals = ms._load_goals()
    assert len(goals) == 1
    assert goals[0] == g2 and goals[0]["name"] == "需求清单 v2"


def test_upsert_keeps_creation_order():
    a = ms._upsert_goal("A", "a", "")
    b = ms._upsert_goal("B", "b", "")
    ms._upsert_goal("A2", "a", "", goal_id=a["id"])
    assert [g["id"] for g in ms._load_goals()] == [a["id"], b["id"]]


@pytest.mark.parametrize("name,goal,prompt", [
    ("", "x", "y"),          # no title
    ("   ", "x", "y"),       # whitespace title
    ("T", "", ""),           # nothing to run
])
def test_upsert_rejects_invalid(name, goal, prompt):
    with pytest.raises(ValueError):
        ms._upsert_goal(name, goal, prompt)
    assert ms._load_goals() == []


def test_upsert_truncates_overlong_fields():
    g = ms._upsert_goal("名" * 100, "x", "p" * (ms._GOAL_PROMPT_MAX + 50))
    assert len(g["name"]) == ms._GOAL_NAME_MAX
    assert len(g["prompt"]) == ms._GOAL_PROMPT_MAX


def test_delete_goal():
    a = ms._upsert_goal("A", "a", "")
    b = ms._upsert_goal("B", "b", "")
    assert ms._delete_goal(a["id"]) is True
    assert [g["id"] for g in ms._load_goals()] == [b["id"]]
    assert ms._delete_goal(a["id"]) is False


def test_load_drops_malformed_entries(tmp_path):
    ms.GOALS_FILE.write_text(json.dumps({"goals": [
        {"id": "0123abcd", "name": "ok", "goal": "g", "prompt": ""},
        {"id": "../../etc", "name": "bad id", "goal": "g"},
        {"id": "0123abcd", "name": "duplicate", "goal": "g"},
        {"id": "deadbeef", "name": 5, "goal": "g"},
        "not a dict",
    ]}), encoding="utf-8")
    goals = ms._load_goals()
    assert [g["name"] for g in goals] == ["ok"]


def test_load_tolerates_broken_json():
    ms.GOALS_FILE.write_text("{not json", encoding="utf-8")
    assert ms._load_goals() == []


def test_goal_output_path_rejects_bad_id(tmp_path):
    wav = tmp_path / "20260512_090120.wav"
    assert ms._goal_output_path(wav, "0123abcd").name == "20260512_090120.goal-0123abcd.md"
    with pytest.raises(ValueError):
        ms._goal_output_path(wav, "../x")


def test_enrich_substitutes_and_strips_fence(monkeypatch):
    seen = {}

    def fake_run(prompt, provider, cfg, label):
        seen["prompt"] = prompt
        return "```markdown\n你是一位需求分析师。\n## 需求\n```"

    monkeypatch.setattr(ms, "_llm_run", fake_run)
    out = ms.enrich_goal_prompt("需求清单", "整理客户需求", "claude", {})
    assert out == "你是一位需求分析师。\n## 需求"
    assert "需求清单" in seen["prompt"] and "整理客户需求" in seen["prompt"]
    assert "{goal}" not in seen["prompt"] and "{name}" not in seen["prompt"]


def test_enrich_requires_goal_text(monkeypatch):
    monkeypatch.setattr(ms, "_llm_run", lambda *a: pytest.fail("LLM called"))
    with pytest.raises(ValueError):
        ms.enrich_goal_prompt("x", "   ", "claude", {})


def test_generate_goal_notes_wraps_prompt(monkeypatch):
    seen = {}
    monkeypatch.setattr(ms, "_llm_run",
                        lambda prompt, *a: seen.setdefault("prompt", prompt) and "OUT")
    goal = {"id": "0123abcd", "name": "需求", "goal": "g",
            "prompt": "列出所有需求。原文见 {transcript}"}
    out = ms.generate_goal_notes("TRANSCRIPT-BODY", "claude", {}, goal,
                                 background="BG", roster="张三")
    assert out == "OUT"
    p = seen["prompt"]
    assert "列出所有需求" in p
    # The user's own {transcript} must not paste the transcript in twice.
    assert p.count("TRANSCRIPT-BODY") == 1
    assert "BG" in p and "张三" in p
    for ph in ("{goal_prompt}", "{hotwords}", "{roster}", "{context}", "{transcript}"):
        assert ph not in p


def test_generate_goal_notes_falls_back_to_description(monkeypatch):
    seen = {}
    monkeypatch.setattr(ms, "_llm_run",
                        lambda prompt, *a: seen.setdefault("prompt", prompt) and "OUT")
    ms.generate_goal_notes("T", "claude", {},
                           {"id": "0123abcd", "name": "n", "goal": "只要三句话总结", "prompt": ""})
    assert "只要三句话总结" in seen["prompt"]


def test_transcript_cannot_expand_placeholders(monkeypatch):
    seen = {}
    monkeypatch.setattr(ms, "_llm_run",
                        lambda prompt, *a: seen.setdefault("prompt", prompt) and "OUT")
    ms.generate_goal_notes("说了 {roster} 和 {context}", "claude", {},
                           {"id": "0123abcd", "name": "n", "goal": "g", "prompt": "p"},
                           roster="SECRET-ROSTER")
    assert seen["prompt"].count("SECRET-ROSTER") == 1


def test_save_goal_output_heads_with_title(tmp_path):
    wav = tmp_path / "20260512_090120.wav"
    path = ms.save_goal_output("body", wav, {"id": "0123abcd", "name": "需求清单"})
    assert path.read_text(encoding="utf-8") == "# 需求清单\n\nbody"


def test_scan_attaches_goal_output_without_phantom_meeting(tmp_path):
    (tmp_path / "20260512_090120.客户访谈.wav").write_bytes(b"")
    (tmp_path / "20260512_090120.客户访谈.polish.txt").write_text("x")
    (tmp_path / "20260512_090120.客户访谈.goal-0123abcd.md").write_text("# g")
    items = ms._scan_meeting_artifacts(tmp_path)
    assert len(items) == 1
    m = items[0]
    assert m["stem"] == "20260512_090120.客户访谈"
    assert set(m["goal_paths"]) == {"0123abcd"}
    assert m["has_md"] is True
    assert m["md_path"] is None


def test_scan_goal_only_meeting_survives_deleted_audio(tmp_path):
    (tmp_path / "20260512_090120.goal-0123abcd.md").write_text("# g")
    items = ms._scan_meeting_artifacts(tmp_path)
    assert len(items) == 1 and items[0]["audio_exists"] is False
    assert "0123abcd" in items[0]["goal_paths"]


def test_rename_and_delete_carry_goal_outputs(tmp_path):
    wav = tmp_path / "20260512_090120.wav"
    wav.write_bytes(b"")
    (tmp_path / "20260512_090120.goal-0123abcd.md").write_text("# g")
    new_wav = ms._rename_meeting_files(wav, "复盘")
    assert (tmp_path / "20260512_090120.复盘.goal-0123abcd.md").exists()
    deleted, errors = ms._delete_meeting_files(new_wav)
    assert deleted == 2 and errors == []


def test_config_jsonc_mirrors_goal_prompt_defaults():
    cfg = ms.load_config()
    for key in ("goal_enrich", "goal_notes"):
        assert ms._resolve_prompt(cfg, key) == ms._PROMPT_DEFAULTS[key]
