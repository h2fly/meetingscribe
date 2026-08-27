"""Tests for `hotwords --notion` — the one-off Notion import.

The HTTP layer is faked; what matters here is that Notion's shapes are read
tolerantly, that a token is required and never leaked, and that titles are
MINED rather than added verbatim (a title is a sentence, a hotword is a term).
"""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import meetingscribe as ms


# ── candidate extraction ─────────────────────────────────────────────────────

class TestCandidates:
    def test_cjk_name_kept_whole(self):
        out = ms._notion_hotword_candidates(["李雷", "王小美"], [])
        assert out == ["李雷", "王小美"]

    def test_latin_name_split_into_spoken_tokens(self):
        """Nobody says the full name mid-sentence, so both halves are useful."""
        out = ms._notion_hotword_candidates(["Aaron Chen"], [])
        assert out == ["Aaron", "Chen"]

    def test_bilingual_name_keeps_both_scripts(self):
        """Measured on a real 212-person engineering directory against a real
        transcript of the same team: the English given name was the spoken form
        for Aaron / Peter / Oliver — their Chinese names never appeared in the
        transcript at all — while 李雷 and 韩梅 were spoken in Chinese. Keeping
        only the CJK run threw away the useful half 3 times out of 5."""
        assert ms._notion_hotword_candidates(
            ["王小明 Aaron Wang (ENG - CN)"], []) == ["王小明", "Aaron", "Wang"]
        assert ms._notion_hotword_candidates(
            ["李雷 Ethan Zhang (ENG - SRE - CN)"], []) == ["李雷", "Ethan", "Zhang"]

    def test_two_letter_latin_tokens_dropped(self):
        """Romanised surname syllables (Li / Wu / Ma / Su / Ou / Ni) collide
        with ordinary speech and nobody says them alone."""
        assert ms._notion_hotword_candidates(["Bob Li"], []) == ["Bob"]
        assert ms._notion_hotword_candidates(["Kate Wu"], []) == ["Kate"]

    def test_digit_bearing_tokens_are_dropped(self):
        """Display names have no digits; digit-bearing terms like N4D come
        from the TITLE miner, not from the name path."""
        assert ms._notion_hotword_candidates(["Deploy 12345"], []) == ["Deploy"]
        assert ms._notion_hotword_candidates(["Agent007 Smith"], []) == ["Smith"]

    def test_parenthetical_alias_dropped(self):
        out = ms._notion_hotword_candidates(["黄河 (Kevin)"], [])
        assert out == ["黄河"]

    def test_email_login_fragments_dropped(self):
        """`he.huang@…` would yield the common syllables "he" and "huang",
        which bias real speech. Display names are capitalised; logins aren't."""
        assert ms._notion_hotword_candidates(["he.huang@aftership.com"], []) == []

    def test_titles_are_mined_not_taken_verbatim(self):
        titles = ["2026 Q3 运维体系规划", "Acme 推理迁移方案",
                  "GKE 集群 gVisor 兼容性测试"]
        out = ms._notion_hotword_candidates([], titles)
        assert "Acme" in out and "GKE" in out and "gVisor" in out
        # the whole sentence never becomes a hotword
        assert not any(len(t) > 12 for t in out)
        assert "2026 Q3 运维体系规划" not in out

    def test_deduplicated_case_insensitively(self):
        out = ms._notion_hotword_candidates(["Aaron", "aaron", "AARON"], [])
        assert out == ["Aaron"]

    def test_limit_is_respected(self):
        # Alphabetic on purpose: a digit in a display name is noise and is
        # dropped, so "Name0" would never become a candidate.
        names = [f"Person{chr(65 + i % 26)}{chr(97 + i // 26)}" for i in range(50)]
        assert len(ms._notion_hotword_candidates(names, [], limit=10)) == 10

    def test_empty_inputs(self):
        assert ms._notion_hotword_candidates([], []) == []


# ── API shape tolerance ──────────────────────────────────────────────────────

def _fake_api(monkeypatch, pages: dict):
    """Route _notion_request by path prefix to canned payloads."""
    seen = []

    def fake(path, token, payload=None):
        seen.append((path, token, payload))
        for prefix, body in pages.items():
            if path.startswith(prefix):
                return body
        return {"results": []}

    monkeypatch.setattr(ms, "_notion_request", fake)
    return seen


class TestMemberNames:
    def test_bots_skipped(self, monkeypatch):
        _fake_api(monkeypatch, {"/users": {"results": [
            {"type": "person", "name": "李雷"},
            {"type": "bot", "name": "My Integration"},
            {"type": "person", "name": "Aaron Chen"},
        ]}})
        assert ms._notion_member_names("tok") == ["李雷", "Aaron Chen"]

    def test_blank_names_skipped(self, monkeypatch):
        _fake_api(monkeypatch, {"/users": {"results": [
            {"type": "person", "name": ""},
            {"type": "person"},
            {"type": "person", "name": "  Bella  "},
        ]}})
        assert ms._notion_member_names("tok") == ["Bella"]


class TestTitles:
    def test_page_and_database_shapes_both_read(self, monkeypatch):
        """Page titles live under properties.<name>.title, database titles at
        the top level. Walking for title-typed rich text covers both without
        special-casing an API version."""
        _fake_api(monkeypatch, {"/search": {"results": [
            {"object": "page", "properties": {
                "Name": {"type": "title",
                         "title": [{"plain_text": "Acme 迁移"}]}}},
            {"object": "database",
             "title": [{"plain_text": "SRE 项目台账"}]},
        ]}})
        out = ms._notion_titles("tok")
        assert "Acme 迁移" in out and "SRE 项目台账" in out

    def test_multi_part_rich_text_joined(self, monkeypatch):
        _fake_api(monkeypatch, {"/search": {"results": [
            {"object": "page", "properties": {"Name": {
                "type": "title",
                "title": [{"plain_text": "N4D "}, {"plain_text": "机型迁移"}]}}},
        ]}})
        assert "N4D 机型迁移" in ms._notion_titles("tok")

    def test_unexpected_shape_is_survived(self, monkeypatch):
        _fake_api(monkeypatch, {"/search": {"results": [
            {"object": "page", "properties": {"Name": None}},
            "not-a-dict",
            {"object": "page"},
        ]}})
        assert ms._notion_titles("tok") == []


class TestPagination:
    def test_follows_the_cursor(self, monkeypatch):
        calls = []

        def fake(path, token, payload=None):
            calls.append((path, payload))
            if len(calls) == 1:
                return {"results": [{"a": 1}], "has_more": True,
                        "next_cursor": "c1"}
            return {"results": [{"a": 2}], "has_more": False}

        monkeypatch.setattr(ms, "_notion_request", fake)
        out = ms._notion_paginate("/search", "tok", {"page_size": 100})
        assert out == [{"a": 1}, {"a": 2}]
        assert calls[1][1]["start_cursor"] == "c1"

    def test_page_budget_bounds_a_huge_workspace(self, monkeypatch):
        monkeypatch.setattr(
            ms, "_notion_request",
            lambda path, token, payload=None: {
                "results": [{}], "has_more": True, "next_cursor": "x"})
        assert len(ms._notion_paginate("/users", "tok", max_pages=3)) == 3

    def test_stops_when_cursor_missing(self, monkeypatch):
        monkeypatch.setattr(
            ms, "_notion_request",
            lambda path, token, payload=None: {"results": [{}], "has_more": True})
        assert len(ms._notion_paginate("/users", "tok")) == 1


# ── command wiring ───────────────────────────────────────────────────────────

def _args(**kw):
    base = dict(show=False, no_llm=True, notion=True, notion_limit=400,
                min_cjk=0, dry_run=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


class TestCommand:
    def test_missing_token_exits_with_instructions(self, monkeypatch, capsys):
        monkeypatch.delenv("NOTION_TOKEN", raising=False)
        monkeypatch.delenv("NOTION_API_KEY", raising=False)
        with pytest.raises(SystemExit):
            ms._hotwords_from_notion(_args(), {}, {"max_count": 1000})
        out = capsys.readouterr().out
        assert "NOTION_TOKEN" in out and "integration" in out

    def test_dry_run_writes_nothing(self, monkeypatch, capsys):
        monkeypatch.setenv("NOTION_TOKEN", "secret-token")
        monkeypatch.setattr(ms, "_notion_member_names", lambda t: ["李雷"])
        monkeypatch.setattr(ms, "_notion_titles", lambda t: ["Acme 迁移"])
        monkeypatch.setattr(
            ms, "_persist_hotwords",
            lambda *a, **k: pytest.fail("dry-run must not persist"))
        ms._hotwords_from_notion(_args(dry_run=True), {}, {"max_count": 1000})
        assert "李雷" in capsys.readouterr().out

    def test_persists_and_reports(self, monkeypatch, capsys):
        monkeypatch.setenv("NOTION_TOKEN", "secret-token")
        monkeypatch.setattr(ms, "_notion_member_names", lambda t: ["Aaron Chen"])
        monkeypatch.setattr(ms, "_notion_titles", lambda t: [])
        seen = {}

        def fake_persist(terms, cfg, max_count, **kw):
            seen.update(terms=terms, max_count=max_count, kw=kw)
            cfg.setdefault("stt", {}).setdefault("funasr", {})["hotword"] = \
                " ".join(terms)
            return terms

        monkeypatch.setattr(ms, "_persist_hotwords", fake_persist)
        cfg = {}
        ms._hotwords_from_notion(_args(), cfg, {"max_count": 1000})
        assert seen["terms"] == ["Aaron", "Chen"]
        assert seen["max_count"] == 1000
        # Workspace names are authoritative, not guesses mined from one
        # meeting, so they must be pinned against rolling eviction.
        assert seen["kw"].get("pin") is True
        assert "新增 2 个" in capsys.readouterr().out

    def test_token_never_printed(self, monkeypatch, capsys):
        monkeypatch.setenv("NOTION_TOKEN", "ntn_supersecret")
        monkeypatch.setattr(ms, "_notion_member_names", lambda t: ["李雷"])
        monkeypatch.setattr(ms, "_notion_titles", lambda t: [])
        monkeypatch.setattr(ms, "_persist_hotwords", lambda *a, **k: ["李雷"])
        ms._hotwords_from_notion(_args(), {}, {"max_count": 1000})
        assert "ntn_supersecret" not in capsys.readouterr().out

    def test_api_failure_exits_without_leaking_the_token(self, monkeypatch, capsys):
        monkeypatch.setenv("NOTION_TOKEN", "ntn_supersecret")

        def boom(token):
            raise RuntimeError("Notion API /users 返回 401: API token is invalid.")

        monkeypatch.setattr(ms, "_notion_member_names", boom)
        with pytest.raises(SystemExit):
            ms._hotwords_from_notion(_args(), {}, {"max_count": 1000})
        out = capsys.readouterr().out
        assert "401" in out and "ntn_supersecret" not in out

    def test_min_cjk_drops_short_chinese_terms(self, monkeypatch, capsys):
        """Measured: at scale, two-character Chinese hotwords steal ordinary
        speech (「分眼都在那边」→「分眼都郑娜边」)."""
        monkeypatch.setenv("NOTION_TOKEN", "tok")
        monkeypatch.setattr(ms, "_notion_member_names",
                            lambda t: ["李雷", "王小美", "Aaron"])
        monkeypatch.setattr(ms, "_notion_titles", lambda t: [])
        got = {}
        monkeypatch.setattr(ms, "_persist_hotwords",
                            lambda terms, cfg, mc, **kw: got.setdefault("t", terms))
        ms._hotwords_from_notion(_args(min_cjk=3), {}, {"max_count": 1000})
        assert "李雷" not in got["t"]          # 2 chars, dropped
        assert "王小美" in got["t"] and "Aaron" in got["t"]

    def test_risky_terms_are_reported_not_hidden(self, monkeypatch, capsys):
        monkeypatch.setenv("NOTION_TOKEN", "tok")
        monkeypatch.setattr(ms, "_notion_member_names", lambda t: ["李雷", "赵小雨"])
        monkeypatch.setattr(ms, "_notion_titles", lambda t: [])
        monkeypatch.setattr(ms, "_persist_hotwords", lambda *a, **k: [])
        ms._hotwords_from_notion(_args(), {}, {"max_count": 1000})
        assert "2 字纯中文词" in capsys.readouterr().out


class TestRequestHeaders:
    def test_sends_bearer_token_and_api_version(self, monkeypatch):
        captured = {}

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return json.dumps({"results": []}).encode()

        def fake_urlopen(req, timeout=None, context=None):
            captured["headers"] = dict(req.headers)
            captured["method"] = req.get_method()
            captured["url"] = req.full_url
            return _Resp()

        import urllib.request
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        ms._notion_request("/users", "tok123")
        assert captured["headers"]["Authorization"] == "Bearer tok123"
        assert captured["headers"]["Notion-version"] == ms._NOTION_VERSION
        assert captured["method"] == "GET"
        assert captured["url"].endswith("/users")

    def test_payload_makes_it_a_post(self, monkeypatch):
        captured = {}

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return json.dumps({"results": []}).encode()

        def fake_urlopen(req, timeout=None, context=None):
            captured["method"] = req.get_method()
            captured["body"] = req.data
            return _Resp()

        import urllib.request
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        ms._notion_request("/search", "tok", {"page_size": 100})
        assert captured["method"] == "POST"
        assert json.loads(captured["body"])["page_size"] == 100
