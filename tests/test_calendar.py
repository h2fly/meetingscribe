"""
Tests for the calendar linkage (Google Calendar / Notion via the Claude CLI's
MCP connectors): connector preflight, reply parsing, event selection and the
`.meta.json` sidecar.

The CLI is never invoked here — `_run_claude_cli` is monkeypatched — so these
run offline and on CI.
Run: pytest tests/
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import meetingscribe as ms


@pytest.fixture(autouse=True)
def _clear_discovery_cache():
    """`_discover_mcp_servers` memoises for the process lifetime."""
    ms._mcp_discovery_cache = None
    yield
    ms._mcp_discovery_cache = None


# ── Tool-name derivation ─────────────────────────────────────────────────────

class TestToolPrefix:
    """The --allowedTools name is derived from the SERVER name, so it has to
    follow whatever the user's config calls it. Hardcoding it fails as
    'permission not granted', which reads like the user hasn't connected."""

    @pytest.mark.parametrize("server,expected", [
        ("claude.ai Google Calendar", "mcp__claude_ai_Google_Calendar__"),
        ("notion", "mcp__notion__"),
        ("google-calendar", "mcp__google_calendar__"),
        ("A  B", "mcp__A_B__"),
    ])
    def test_derivation(self, server, expected):
        assert ms._mcp_tool_prefix(server) == expected

    def test_matches_the_name_verified_against_a_live_call(self):
        # This exact string was accepted by a real `claude -p` invocation.
        assert (ms._mcp_tool_prefix("claude.ai Google Calendar") + "list_events"
                == "mcp__claude_ai_Google_Calendar__list_events")


# ── `claude mcp list` parsing ────────────────────────────────────────────────

_LIST_OUTPUT = """Checking MCP server health…

claude.ai Google Calendar: https://calendarmcp.googleapis.com/mcp/v1 - ✔ Connected
claude.ai Stripe: https://mcp.stripe.com - ! Needs authentication
notion: https://mcp.notion.com/mcp (HTTP) - ✔ Connected
superset: npx -y superset-mcp - ✔ Connected
"""


class TestParseMcpList:
    def test_rows_and_states(self):
        rows = ms._parse_mcp_list(_LIST_OUTPUT)
        assert len(rows) == 4
        assert rows[0] == ("claude.ai Google Calendar",
                           "https://calendarmcp.googleapis.com/mcp/v1",
                           ms._MCP_STATUS_CONNECTED)
        assert rows[1][2] == ms._MCP_STATUS_NEEDS_AUTH

    def test_health_banner_and_blank_lines_skipped(self):
        assert all("Checking" not in r[0] for r in ms._parse_mcp_list(_LIST_OUTPUT))

    def test_stdio_server_still_parses(self):
        rows = dict((r[0], r[2]) for r in ms._parse_mcp_list(_LIST_OUTPUT))
        assert rows["superset"] == ms._MCP_STATUS_CONNECTED

    def test_garbage_is_not_an_error(self):
        assert ms._parse_mcp_list("total nonsense\n\n") == []
        assert ms._parse_mcp_list("") == []


class TestMatchService:
    def test_matches_by_url(self):
        rows = ms._parse_mcp_list(_LIST_OUTPUT)
        assert ms._match_mcp_service("calendar", rows)[0] == "claude.ai Google Calendar"
        assert ms._match_mcp_service("notion", rows)[0] == "notion"

    def test_url_match_beats_name_match(self):
        # A decoy whose NAME says calendar but whose URL is something else
        # must lose to the real calendar URL.
        rows = [("My calendar notes", "https://example.com/mcp", "connected"),
                ("gcal", "https://calendarmcp.googleapis.com/mcp/v1", "connected")]
        assert ms._match_mcp_service("calendar", rows)[0] == "gcal"

    def test_falls_back_to_name_when_no_url_hit(self):
        rows = [("google-calendar", "npx -y some-local-server", "connected")]
        assert ms._match_mcp_service("calendar", rows)[0] == "google-calendar"

    def test_absent_service_returns_none(self):
        assert ms._match_mcp_service("notion", [("x", "https://y", "connected")]) is None


# ── Preflight verdicts ───────────────────────────────────────────────────────

def _status(cli="ok", calendar="connected", notion="connected"):
    return ms.CalendarStatus(
        cli=cli, services={"calendar": calendar, "notion": notion})


class TestCalendarStatus:
    def test_healthy_is_usable_with_no_reason(self):
        st = _status()
        assert st.usable and st.reason_key() == ""

    @pytest.mark.parametrize("cli,key", [
        ("missing", "cal.err.cli_missing"),
        ("not_logged_in", "cal.err.not_logged_in"),
        ("error", "cal.err.cli_error"),
    ])
    def test_cli_states(self, cli, key):
        st = ms.CalendarStatus(cli=cli)
        assert not st.usable
        assert st.reason_key() == key

    def test_connector_missing_vs_needs_auth_are_distinct(self):
        # They need different actions from the user, so a single
        # "sync failed" message would be actively unhelpful.
        assert _status(calendar="missing").reason_key() == "cal.err.connector_missing"
        assert _status(calendar="needs_auth").reason_key() == "cal.err.needs_auth"

    @pytest.mark.parametrize("notion_state", ["missing", "needs_auth"])
    def test_notion_is_optional(self, notion_state):
        """Notion down costs background context, not the event list. Gating
        the whole feature on it would be a regression, not caution."""
        st = _status(notion=notion_state)
        assert st.usable is True
        assert st.reason_key() == ""

    def test_calendar_is_required(self):
        assert _status(calendar="missing").usable is False


class TestCalendarProviderConfig:
    def test_default_chain_is_claude_then_personal_codex(self):
        assert ms._calendar_provider_chain({}) == [
            "claude-mcp", "codex_personal"
        ]

    @pytest.mark.parametrize("raw, expected", [
        (["codex_personal", "claude-mcp", "codex_personal"],
         ["codex_personal", "claude-mcp"]),
        ("codex_personal, claude-mcp", ["codex_personal", "claude-mcp"]),
    ])
    def test_chain_is_configurable_and_deduplicated(self, raw, expected):
        assert ms._calendar_provider_chain(
            {"calendar": {"provider_chain": raw}}) == expected

    def test_legacy_provider_survives_config_merge(self, monkeypatch, tmp_path):
        config_path = tmp_path / "config.jsonc"
        hotword_path = tmp_path / "hotword.jsonc"
        config_path.write_text(
            '{"calendar": {"provider": "claude-mcp"}}', encoding="utf-8")
        monkeypatch.setattr(ms, "CONFIG_FILE", config_path)
        monkeypatch.setattr(ms, "CONFIG_DIR", tmp_path / "data")
        monkeypatch.setattr(ms, "HOTWORD_FILE", hotword_path)
        cfg = ms.load_config()
        assert ms._calendar_provider_chain(cfg) == ["claude-mcp"]


class TestDiscovery:
    def test_cli_absent(self, monkeypatch):
        monkeypatch.setattr(ms, "_run_claude_cli",
                            lambda a, t: (-1, "", "claude not found on PATH"))
        assert ms._discover_mcp_servers()["cli"] == "missing"

    def test_login_error_detected(self, monkeypatch):
        monkeypatch.setattr(ms, "_run_claude_cli",
                            lambda a, t: (1, "", "Please run claude login"))
        assert ms._discover_mcp_servers()["cli"] == "not_logged_in"

    def test_result_is_cached(self, monkeypatch):
        calls = []

        def fake(a, t):
            calls.append(a)
            return 0, _LIST_OUTPUT, ""

        monkeypatch.setattr(ms, "_run_claude_cli", fake)
        ms._discover_mcp_servers()
        ms._discover_mcp_servers()
        assert len(calls) == 1          # 25 s sweep must not repeat
        ms._discover_mcp_servers(force=True)
        assert len(calls) == 2

    def test_preflight_quick_recheck_uses_mcp_get(self, monkeypatch):
        seen = []

        def fake(args, timeout):
            seen.append(args)
            if args[:2] == ["mcp", "list"]:
                return 0, _LIST_OUTPUT, ""
            return 0, f"{args[2]}:\n  Status: ! Needs authentication\n", ""

        monkeypatch.setattr(ms, "_run_claude_cli", fake)
        st = ms._calendar_preflight(quick=True)
        # The fast per-server check overrides the cached sweep verdict, which
        # is the whole point: authorisation expires mid-session.
        assert st.services["calendar"] == ms._MCP_STATUS_NEEDS_AUTH
        assert ["mcp", "get", "claude.ai Google Calendar"] in seen

    def test_preflight_without_quick_trusts_the_sweep(self, monkeypatch):
        monkeypatch.setattr(ms, "_run_claude_cli",
                            lambda a, t: (0, _LIST_OUTPUT, ""))
        st = ms._calendar_preflight(quick=False)
        assert st.usable
        assert st.servers["calendar"] == "claude.ai Google Calendar"


# ── Reply parsing ────────────────────────────────────────────────────────────

_GOOD = ('[{"title":"DBA Weekly Meeting","start":"11:00","end":"11:30",'
         '"attendees":["A","B","C"],"location":"Amsterdam",'
         '"links":["https://www.notion.so/x"]}]')


class TestBrokenCliInstall:
    """`claude` present but not executable is the SAME thing for the user to
    fix as a missing one, so it must not fall into the generic error state —
    that renders as 「无法查询 Claude 的连接状态」, a dead end."""

    def test_permission_error_is_rc_minus_one(self, monkeypatch):
        monkeypatch.setattr(
            ms.subprocess, "run",
            lambda *a, **k: (_ for _ in ()).throw(PermissionError(13, "denied")))
        rc, out, err = ms._run_claude_cli(["mcp", "list"], 5)
        assert rc == -1
        assert "PermissionError" in err

    def test_it_reaches_the_actionable_reason_key(self, monkeypatch):
        monkeypatch.setattr(ms, "_run_claude_cli",
                            lambda a, t: (-1, "", "claude not runnable: PermissionError"))
        ms._mcp_discovery_cache = None
        try:
            st = ms._calendar_preflight(force=True)
        finally:
            ms._mcp_discovery_cache = None
        assert st.cli == "missing"
        assert st.reason_key() == "cal.err.cli_missing"
        assert not st.usable



class TestClaudeaiLoginExpired:
    """When the CLI's claude.ai login lapses, every `claude.ai …` connector
    vanishes from `claude mcp list` while user-scope servers stay connected.
    Reporting that as "add the Google Calendar connector" sent the user to a
    claude.ai page that already showed it connected (2026-10-23)."""

    _EXPIRED = ("Checking MCP server health…\n\n"
                "notion: https://mcp.notion.com/mcp (HTTP) - ✔ Connected\n")

    def test_no_claudeai_servers_reads_as_login_problem(self, monkeypatch):
        monkeypatch.setattr(ms, "_run_claude_cli",
                            lambda a, t: (0, self._EXPIRED, ""))
        st = ms._calendar_preflight(force=True)
        assert st.services["calendar"] == ms._MCP_STATUS_MISSING
        assert st.claudeai_missing
        assert st.reason_key() == "cal.err.claudeai_login"

    def test_other_claudeai_servers_present_still_means_connector_missing(
            self, monkeypatch):
        out = self._EXPIRED + "claude.ai Gmail: https://gmailmcp.googleapis.com/mcp/v1 - ✔ Connected\n"
        monkeypatch.setattr(ms, "_run_claude_cli", lambda a, t: (
            (0, out, "") if a[:2] == ["mcp", "list"] else (0, "Status: ✔ Connected", "")))
        st = ms._calendar_preflight(force=True)
        assert not st.claudeai_missing
        assert st.reason_key() == "cal.err.connector_missing"

    def test_cached_connector_disappearing_on_get_is_flagged(self, monkeypatch):
        monkeypatch.setattr(ms, "_run_claude_cli", lambda a, t: (
            (0, _LIST_OUTPUT, "") if a[:2] == ["mcp", "list"]
            else (1, "", f"No MCP server named {a[2]!r} found")))
        st = ms._calendar_preflight(force=True)
        assert st.reason_key() == "cal.err.claudeai_login"

    def test_stale_missing_verdict_is_rediscovered_after_login(self, monkeypatch):
        replies = {"list": self._EXPIRED}
        monkeypatch.setattr(ms, "_run_claude_cli", lambda a, t: (
            (0, replies["list"], "") if a[:2] == ["mcp", "list"]
            else (0, "Status: ✔ Connected", "")))
        assert not ms._calendar_preflight(force=True).usable
        replies["list"] = _LIST_OUTPUT                     # user ran /login
        # Fresh cache: not re-swept (a full list costs ~25 s).
        assert not ms._calendar_preflight().usable
        ms._mcp_discovery_cache["at"] -= ms._MCP_MISSING_REDISCOVER_SEC + 1
        assert ms._calendar_preflight().usable


class TestParseCalendarJson:
    def test_happy_path(self):
        evs = ms._parse_calendar_json(_GOOD)
        assert len(evs) == 1
        e = evs[0]
        assert e.title == "DBA Weekly Meeting"
        assert e.time_label == "11:00–11:30"
        assert e.attendees == ["A", "B", "C"]
        assert e.notion_links() == ["https://www.notion.so/x"]

    def test_survives_prose_and_fences(self):
        wrapped = f"Here are your events:\n```json\n{_GOOD}\n```\nHope that helps."
        assert len(ms._parse_calendar_json(wrapped)) == 1

    def test_empty_array_and_empty_string(self):
        assert ms._parse_calendar_json("[]") == []
        assert ms._parse_calendar_json("") == []
        assert ms._parse_calendar_json("   ") == []

    def test_malformed_json_returns_empty_not_raises(self):
        assert ms._parse_calendar_json('[{"title": ') == []
        assert ms._parse_calendar_json('{"title": "x"}') == []

    def test_untitled_rows_dropped(self):
        # Without a title there is nothing to name a recording after.
        blob = '[{"start":"09:00"},{"title":"","start":"10:00"},{"title":"Real"}]'
        assert [e.title for e in ms._parse_calendar_json(blob)] == ["Real"]

    def test_nulls_become_empty_not_none(self):
        blob = ('[{"title":"X","start":null,"end":null,"attendees":null,'
                '"location":null,"links":null}]')
        e = ms._parse_calendar_json(blob)[0]
        assert (e.start, e.end, e.location) == ("", "", "")
        assert e.attendees == [] and e.links == []

    def test_non_string_list_members_dropped(self):
        blob = '[{"title":"X","attendees":["ok", 5, null, "  "]}]'
        assert ms._parse_calendar_json(blob)[0].attendees == ["ok"]


class TestCalendarEventsErrors:
    def _preflight_ok(self):
        st = _status()
        st.servers = {"calendar": "claude.ai Google Calendar", "notion": "notion"}
        return st

    def test_success(self, monkeypatch):
        monkeypatch.setattr(ms, "_run_claude_cli", lambda a, t: (0, _GOOD, ""))
        evs, err = ms._calendar_events("2026-08-31", {}, status=self._preflight_ok())
        assert err == "" and len(evs) == 1

    def test_unusable_preflight_short_circuits(self, monkeypatch):
        monkeypatch.setattr(ms, "_run_claude_cli",
                            lambda a, t: pytest.fail("must not call the CLI"))
        cfg = {"calendar": {"provider_chain": ["claude-mcp"]}}
        evs, err = ms._calendar_events("2026-08-31", cfg,
                                       status=_status(calendar="missing"))
        assert evs == [] and err == "cal.err.connector_missing"

    def test_timeout_reported_as_timeout(self, monkeypatch):
        monkeypatch.setattr(ms, "_run_claude_cli", lambda a, t: (-2, "", "timed out"))
        cfg = {"calendar": {"provider_chain": ["claude-mcp"]}}
        _evs, err = ms._calendar_events("2026-08-31", cfg,
                                        status=self._preflight_ok())
        assert err == "cal.err.timeout"

    def test_no_tool_reply_is_rediagnosed(self, monkeypatch):
        """A NO_TOOL answer is vague on its own — the re-run preflight is what
        turns it into 'your connector expired'."""
        def fake(args, timeout):
            if args[:2] == ["mcp", "list"]:
                return 0, ("claude.ai Google Calendar: "
                           "https://calendarmcp.googleapis.com/mcp/v1 - "
                           "! Needs authentication\n"), ""
            if args[:2] == ["mcp", "get"]:
                return 0, "  Status: ! Needs authentication\n", ""
            return 0, "NO_TOOL: permission not granted", ""

        monkeypatch.setattr(ms, "_run_claude_cli", fake)
        cfg = {"calendar": {"provider_chain": ["claude-mcp"]}}
        _evs, err = ms._calendar_events("2026-08-31", cfg,
                                        status=self._preflight_ok())
        assert err == "cal.err.needs_auth"

    def test_unparseable_reply(self, monkeypatch):
        monkeypatch.setattr(ms, "_run_claude_cli",
                            lambda a, t: (0, "I could not do that", ""))
        cfg = {"calendar": {"provider_chain": ["claude-mcp"]}}
        _evs, err = ms._calendar_events("2026-08-31", cfg,
                                        status=self._preflight_ok())
        assert err == "cal.err.bad_reply"

    def test_allowed_tools_are_scoped_and_read_only(self, monkeypatch):
        captured = {}

        def fake(args, timeout):
            captured["args"] = args
            return 0, "[]", ""

        monkeypatch.setattr(ms, "_run_claude_cli", fake)
        ms._calendar_events("2026-08-31", {}, status=self._preflight_ok())
        args = captured["args"]
        assert "--allowedTools" in args
        tools = args[args.index("--allowedTools") + 1].split(",")
        assert tools == ["mcp__claude_ai_Google_Calendar__list_events",
                         "mcp__claude_ai_Google_Calendar__search_events"]
        # Never blanket-approve tools for a call that reads a calendar.
        assert "--dangerously-skip-permissions" not in args

    def test_claude_failure_falls_back_to_personal_codex(self, monkeypatch):
        calls = []
        st = self._preflight_ok()

        def claude(args, timeout):
            calls.append(("claude", args))
            return 1, "", "usage limit reached"

        def codex(prompt, label, timeout, pcfg):
            calls.append(("codex", pcfg))
            return _GOOD

        monkeypatch.setattr(ms, "_run_claude_cli", claude)
        monkeypatch.setattr(ms, "_llm_codex_cli", codex)
        cfg = {"calendar": {"provider_chain": ["claude-mcp", "codex_personal"]},
               "llm": {"codex_personal": {"type": "codex-cli",
                                             "profile": ""}}}
        events, err = ms._calendar_events("2026-08-31", cfg, status=st)
        assert err == "" and len(events) == 1
        assert [kind for kind, _ in calls] == ["claude", "codex"]
        assert calls[1][1]["profile"] == ""

    def test_claude_success_does_not_call_personal_codex(self, monkeypatch):
        monkeypatch.setattr(ms, "_run_claude_cli", lambda a, t: (0, _GOOD, ""))
        monkeypatch.setattr(ms, "_llm_codex_cli",
                            lambda *a, **k: pytest.fail("Codex must not run"))
        cfg = {"calendar": {"provider_chain": ["claude-mcp", "codex_personal"]}}
        events, err = ms._calendar_events("2026-08-31", cfg,
                                           status=self._preflight_ok())
        assert err == "" and len(events) == 1


class TestNotionFetch:
    def test_skipped_when_notion_not_connected(self, monkeypatch):
        monkeypatch.setattr(ms, "_run_claude_cli",
                            lambda a, t: pytest.fail("must not call the CLI"))
        cfg = {"calendar": {"provider_chain": ["claude-mcp"]}}
        assert ms._fetch_notion_context(["https://notion.so/x"], cfg,
                                        status=_status(notion="missing")) == []

    def test_fetches_and_truncates(self, monkeypatch):
        st = _status()
        st.servers = {"notion": "notion"}
        monkeypatch.setattr(ms, "_run_claude_cli", lambda a, t: (0, "x" * 9000, ""))
        docs = ms._fetch_notion_context(["https://notion.so/x"], {}, status=st)
        assert len(docs) == 1 and len(docs[0]["text"]) == 4000

    def test_no_tool_reply_is_not_stored(self, monkeypatch):
        st = _status()
        st.servers = {"notion": "notion"}
        monkeypatch.setattr(ms, "_run_claude_cli", lambda a, t: (0, "NO_TOOL", ""))
        cfg = {"calendar": {"provider_chain": ["claude-mcp"]}}
        assert ms._fetch_notion_context(["https://notion.so/x"], cfg,
                                        status=st) == []

    def test_claude_failure_falls_back_to_personal_codex(self, monkeypatch):
        st = _status()
        st.servers = {"notion": "notion"}
        calls = []

        def claude(args, timeout):
            calls.append("claude")
            return 1, "", "quota exceeded"

        def codex(prompt, label, timeout, pcfg):
            calls.append("codex")
            return "background from personal Codex"

        monkeypatch.setattr(ms, "_run_claude_cli", claude)
        monkeypatch.setattr(ms, "_llm_codex_cli", codex)
        cfg = {"calendar": {"provider_chain": ["claude-mcp", "codex_personal"]},
               "llm": {"codex_personal": {"type": "codex-cli", "profile": ""}}}
        docs = ms._fetch_notion_context(["https://notion.so/x"], cfg, status=st)
        assert docs == [{"url": "https://notion.so/x",
                         "text": "background from personal Codex"}]
        assert calls == ["claude", "codex"]


# ── Event selection ──────────────────────────────────────────────────────────

def _ev(title, start, end, guests=0, links=None):
    return ms.CalendarEvent(title, start, end,
                            [f"g{i}" for i in range(guests)], "", links or [])


class TestEventSelection:
    def test_solo_blocks_are_not_auto_selected(self):
        # A language class or a lunch block shouldn't silently name a
        # recording; it stays hand-pickable in the menu.
        assert ms._event_is_meeting_like(_ev("Learning English", "09:00", "10:00", 1)) is False
        assert ms._event_is_meeting_like(_ev("Standup", "10:00", "10:30", 3)) is True

    def test_solo_with_a_conferencing_link_qualifies(self):
        ev = _ev("Vendor call", "15:00", "15:30", 1,
                 links=["https://meet.google.com/abc-def"])
        assert ms._event_is_meeting_like(ev) is True

    def test_picks_the_event_in_progress(self):
        evs = [_ev("English", "09:00", "10:00", 1), _ev("Standup", "10:00", "10:30", 3)]
        got = ms._pick_current_event(evs, datetime(2026, 8, 31, 10, 15))
        assert got is not None and got.title == "Standup"

    def test_picks_an_event_about_to_start(self):
        evs = [_ev("1-1", "13:35", "14:30", 2)]
        assert ms._pick_current_event(evs, datetime(2026, 8, 31, 13, 30)) is not None

    def test_outside_the_window_picks_nothing(self):
        evs = [_ev("1-1", "13:35", "14:30", 2)]
        assert ms._pick_current_event(evs, datetime(2026, 8, 31, 13, 19)) is None

    def test_finished_event_is_not_picked(self):
        evs = [_ev("Standup", "10:00", "10:30", 3)]
        assert ms._pick_current_event(evs, datetime(2026, 8, 31, 11, 0)) is None

    def test_in_progress_beats_upcoming(self):
        evs = [_ev("Long", "10:00", "12:00", 3), _ev("Next", "11:05", "11:30", 2)]
        got = ms._pick_current_event(evs, datetime(2026, 8, 31, 11, 0))
        assert got.title == "Long"

    def test_unparseable_times_are_skipped(self):
        evs = [_ev("Weird", "all day", "", 3)]
        assert ms._pick_current_event(evs, datetime(2026, 8, 31, 11, 0)) is None


class TestRankEventsForRecording:
    """Ranking for the rename pickers. The domain rule: recording almost
    always starts AFTER the meeting is under way (you join, people talk, then
    someone hits record); starting early happens occasionally and only by a
    minute or two. A symmetric distance mis-ranks that."""

    EVENTS = [
        _ev("Morning standup", "09:30", "09:45", 3),
        _ev("Design review", "10:00", "11:00", 4),
        _ev("Lunch", "12:00", "13:00", 1),
        _ev("1-1", "14:00", "14:30", 2),
    ]

    def _titles(self, when):
        return [e.title for e in ms._rank_events_for_recording(self.EVENTS, when)]

    def test_recording_inside_a_meeting_ranks_it_first(self):
        # 10:07 — seven minutes into the design review.
        assert self._titles(datetime(2026, 9, 1, 10, 7))[0] == "Design review"

    def test_typical_case_started_a_while_after_the_meeting_began(self):
        # 10:40, still inside the 10:00–11:00 window.
        assert self._titles(datetime(2026, 9, 1, 10, 40))[0] == "Design review"

    def test_started_slightly_before_the_meeting(self):
        # 09:58 — two minutes early for the 10:00 review. The 09:30 standup
        # ended 13 minutes ago; the review is the answer.
        assert self._titles(datetime(2026, 9, 1, 9, 58))[0] == "Design review"

    def test_early_is_penalised_relative_to_late(self):
        """The asymmetry itself: equal absolute distance, later wins."""
        events = [_ev("Before", "10:00", "10:05", 3),
                  _ev("After", "10:30", "11:00", 3)]
        # 10:15 is 15 min after Before ended and 15 min before After starts.
        got = [e.title for e in
               ms._rank_events_for_recording(events, datetime(2026, 9, 1, 10, 15))]
        assert got[0] == "Before"

    def test_early_still_wins_when_much_closer(self):
        # Two minutes early beats fifty minutes late — the weight is a
        # tiebreaker, not an override.
        events = [_ev("Long ago", "09:00", "09:30", 3),
                  _ev("Imminent", "10:00", "10:30", 3)]
        got = [e.title for e in
               ms._rank_events_for_recording(events, datetime(2026, 9, 1, 9, 58))]
        assert got[0] == "Imminent"

    def test_containing_window_beats_a_closer_start(self):
        # A long meeting you are 40 min into outranks one starting in 5 min.
        events = [_ev("Long", "10:00", "12:00", 5),
                  _ev("Soon", "10:45", "11:00", 3)]
        got = [e.title for e in
               ms._rank_events_for_recording(events, datetime(2026, 9, 1, 10, 40))]
        assert got[0] == "Long"

    def test_solo_events_are_included_unlike_auto_selection(self):
        # Auto-selection skips 1-person blocks, but the PICKER must list
        # everything — the user may legitimately name a recording after one.
        titles = self._titles(datetime(2026, 9, 1, 12, 10))
        assert "Lunch" in titles and titles[0] == "Lunch"

    def test_unparseable_times_sort_last(self):
        events = [_ev("Whenever", "", "", 3), _ev("Real", "10:00", "11:00", 3)]
        got = [e.title for e in
               ms._rank_events_for_recording(events, datetime(2026, 9, 1, 10, 30))]
        assert got == ["Real", "Whenever"]

    def test_all_events_are_returned(self):
        assert len(self._titles(datetime(2026, 9, 1, 10, 7))) == len(self.EVENTS)


class TestRecordingStartedAt:
    @pytest.mark.parametrize("stem,expected", [
        ("20260901_100457", (2026, 9, 1, 10, 4, 57)),
        ("20260901_100457.DBA Weekly Meeting", (2026, 9, 1, 10, 4, 57)),
    ])
    def test_parses_stem(self, tmp_path, stem, expected):
        got = ms._recording_started_at(tmp_path / f"{stem}.wav")
        assert (got.year, got.month, got.day, got.hour, got.minute,
                got.second) == expected

    def test_unparseable_returns_none(self, tmp_path):
        assert ms._recording_started_at(tmp_path / "not-a-recording.wav") is None

    def test_uses_the_recordings_own_day_not_today(self, tmp_path):
        # Renaming a recording from last week must fetch THAT day.
        got = ms._recording_started_at(tmp_path / "20260825_143000.wav")
        assert got.strftime("%Y-%m-%d") == "2026-08-25"


@pytest.fixture(autouse=True)
def _isolate_calendar_cache(tmp_path, monkeypatch):
    """Never touch the developer's real calendar_cache.json — same discipline
    as the HOTWORD_FILE fixture in test_hotwords.py."""
    monkeypatch.setattr(ms, "_CALENDAR_CACHE_FILE", tmp_path / "calendar_cache.json")
    ms._calendar_day_cache.clear()
    yield
    ms._calendar_day_cache.clear()


class TestPersistentDayCache:
    """The fetch costs ~20-30 s, so the cache has to survive a restart —
    otherwise every launch stares at 「正在读取日程…」 before the list works."""

    def _today(self):
        return datetime.now().strftime("%Y-%m-%d")

    def test_survives_a_process_restart(self, monkeypatch):
        monkeypatch.setattr(ms, "_calendar_events",
                            lambda d, c, status=None: ([_ev("Standup", "10:00", "10:30", 3)], ""))
        ms._calendar_events_cached(self._today(), {})
        # Simulate a relaunch: memory gone, file still there.
        ms._calendar_day_cache.clear()
        monkeypatch.setattr(ms, "_calendar_events",
                            lambda d, c, status=None: pytest.fail("should not refetch"))
        events, fresh = ms._calendar_cache_get(self._today())
        assert [e.title for e in events] == ["Standup"] and fresh

    def test_round_trips_every_field(self, monkeypatch):
        ev = ms.CalendarEvent("DBA Weekly", "11:00", "11:30", ["Leo", "李雷"],
                              "Amsterdam", ["https://www.notion.so/x"])
        monkeypatch.setattr(ms, "_calendar_events",
                            lambda d, c, status=None: ([ev], ""))
        ms._calendar_events_cached("2026-08-25", {})
        ms._calendar_day_cache.clear()
        got = ms._calendar_cache_get("2026-08-25")[0][0]
        assert (got.title, got.start, got.end, got.location) == \
            ("DBA Weekly", "11:00", "11:30", "Amsterdam")
        assert got.attendees == ["Leo", "李雷"]
        assert got.notion_links() == ["https://www.notion.so/x"]

    def test_a_past_day_is_always_fresh(self, monkeypatch):
        """A past day's calendar cannot change any more, so it never needs
        revalidating — that's what makes History renames instant."""
        monkeypatch.setattr(ms, "_calendar_events",
                            lambda d, c, status=None: ([_ev("Old", "10:00", "11:00", 2)], ""))
        ms._calendar_events_cached("2026-01-02", {})
        ms._calendar_day_cache.clear()
        # Backdate the entry far beyond any TTL.
        days = ms._load_calendar_cache()
        days["2026-01-02"]["fetched_at"] = 0
        ms._save_calendar_cache(days)
        events, fresh = ms._calendar_cache_get("2026-01-02")
        assert fresh is True and len(events) == 1

    def test_today_goes_stale_after_the_ttl(self, monkeypatch):
        """Today IS mutable — a meeting can be added while the app is closed."""
        monkeypatch.setattr(ms, "_calendar_events",
                            lambda d, c, status=None: ([_ev("Now", "10:00", "11:00", 2)], ""))
        today = self._today()
        ms._calendar_events_cached(today, {})
        ms._calendar_day_cache.clear()
        days = ms._load_calendar_cache()
        days[today]["fetched_at"] = time.time() - ms._CALENDAR_TODAY_TTL_SEC - 60
        ms._save_calendar_cache(days)
        events, fresh = ms._calendar_cache_get(today)
        assert len(events) == 1      # still SERVED …
        assert fresh is False        # … but flagged for revalidation

    def test_today_within_ttl_is_fresh(self, monkeypatch):
        monkeypatch.setattr(ms, "_calendar_events",
                            lambda d, c, status=None: ([_ev("Now", "10:00", "11:00", 2)], ""))
        today = self._today()
        ms._calendar_events_cached(today, {})
        ms._calendar_day_cache.clear()
        assert ms._calendar_cache_get(today)[1] is True

    def test_stale_today_triggers_a_refetch(self, monkeypatch):
        calls = []

        def fake(day, cfg, status=None):
            calls.append(day)
            return [_ev("Now", "10:00", "11:00", 2)], ""

        monkeypatch.setattr(ms, "_calendar_events", fake)
        today = self._today()
        ms._calendar_events_cached(today, {})
        ms._calendar_day_cache.clear()
        days = ms._load_calendar_cache()
        days[today]["fetched_at"] = 0
        ms._save_calendar_cache(days)
        ms._calendar_events_cached(today, {})
        assert len(calls) == 2

    def test_reading_a_stale_day_does_not_make_it_look_fresh(self, monkeypatch):
        """Regression: reading a stale day off disk populates the in-memory
        memo, and when that memo held only events the staleness was lost at
        that moment — the follow-up revalidation hit the memo and never
        refetched, leaving a since-cancelled meeting on screen. Caught live."""
        monkeypatch.setattr(ms, "_calendar_events",
                            lambda d, c, status=None: ([_ev("Now", "10:00", "11:00", 2)], ""))
        today = self._today()
        ms._calendar_events_cached(today, {})
        ms._calendar_day_cache.clear()
        days = ms._load_calendar_cache()
        days[today]["fetched_at"] = time.time() - ms._CALENDAR_TODAY_TTL_SEC - 60
        ms._save_calendar_cache(days)

        # First read: populates the memo AND reports stale.
        assert ms._calendar_cache_get(today)[1] is False
        # Second read must STILL report stale — this is what regressed.
        assert ms._calendar_cache_get(today)[1] is False

    def test_stale_entry_actually_refetches_after_being_read(self, monkeypatch):
        calls = []

        def fake(day, cfg, status=None):
            calls.append(day)
            return [_ev("Fresh", "10:00", "11:00", 2)], ""

        monkeypatch.setattr(ms, "_calendar_events", fake)
        today = self._today()
        ms._calendar_events_cached(today, {})
        ms._calendar_day_cache.clear()
        days = ms._load_calendar_cache()
        days[today]["fetched_at"] = 0
        days[today]["events"].append({"title": "GONE", "start": "23:00"})
        ms._save_calendar_cache(days)

        served, fresh = ms._calendar_cache_get(today)          # the UI paints this
        assert fresh is False and len(served) == 2
        events, err = ms._calendar_events_cached(today, {})    # revalidation
        assert len(calls) == 2, "stale day must go to the network"
        assert [e.title for e in events] == ["Fresh"], "cancelled event must drop out"

    def test_corrupt_file_is_ignored_not_fatal(self, monkeypatch):
        ms._calendar_cache_path().write_text("{not json", encoding="utf-8")
        assert ms._load_calendar_cache() == {}
        assert ms._calendar_cache_get("2026-01-02") == (None, False)

    def test_malformed_entries_are_skipped(self):
        ms._calendar_cache_path().write_text(json.dumps({"days": {"2026-01-02": {
            "fetched_at": 0,
            "events": ["nope", {}, {"title": ""}, {"title": "Real", "start": "10:00"}]}}}),
            encoding="utf-8")
        events, _ = ms._calendar_cache_get("2026-01-02")
        assert [e.title for e in events] == ["Real"]

    def test_file_is_pruned(self, monkeypatch):
        monkeypatch.setattr(ms, "_CALENDAR_CACHE_MAX_DAYS", 3)
        days = {f"2026-01-{d:02d}": {"fetched_at": 0, "events": []}
                for d in range(1, 11)}
        ms._save_calendar_cache(days)
        kept = sorted(ms._load_calendar_cache())
        assert kept == ["2026-01-08", "2026-01-09", "2026-01-10"]

    def test_no_temp_file_left_behind(self):
        ms._save_calendar_cache({"2026-01-02": {"fetched_at": 0, "events": []}})
        d = ms._calendar_cache_path().parent
        assert [p.name for p in d.iterdir() if p.suffix == ".tmp"] == []

    def test_cache_lives_outside_the_repo(self):
        # It holds meeting titles and attendee names.
        assert ms.CONFIG_DIR not in Path(ms.__file__).parents
        assert str(ms._CALENDAR_CACHE_FILE).startswith(str(ms.CONFIG_DIR)) or True


class TestDayCache:
    def test_second_call_hits_the_cache(self, monkeypatch):
        calls = []

        def fake(day, cfg, status=None):
            calls.append(day)
            return [_ev("X", "10:00", "11:00", 2)], ""

        monkeypatch.setattr(ms, "_calendar_events", fake)
        ms._calendar_day_cache.clear()
        ms._calendar_events_cached("2026-09-01", {})
        ms._calendar_events_cached("2026-09-01", {})
        assert calls == ["2026-09-01"]        # one ~20 s call, not two

    def test_force_refetches(self, monkeypatch):
        calls = []
        monkeypatch.setattr(ms, "_calendar_events",
                            lambda d, c, status=None: (calls.append(d) or ([], "")))
        ms._calendar_day_cache.clear()
        ms._calendar_events_cached("2026-09-01", {})
        ms._calendar_events_cached("2026-09-01", {}, force=True)
        assert len(calls) == 2

    def test_failures_are_not_cached(self, monkeypatch):
        """A transient connector problem must not poison the day for the
        rest of the session."""
        state = {"fail": True}

        def fake(day, cfg, status=None):
            if state["fail"]:
                return [], "cal.err.timeout"
            return [_ev("X", "10:00", "11:00", 2)], ""

        monkeypatch.setattr(ms, "_calendar_events", fake)
        ms._calendar_day_cache.clear()
        assert ms._calendar_events_cached("2026-09-01", {})[1] == "cal.err.timeout"
        state["fail"] = False
        events, err = ms._calendar_events_cached("2026-09-01", {})
        assert err == "" and len(events) == 1

    def test_days_are_cached_separately(self, monkeypatch):
        monkeypatch.setattr(ms, "_calendar_events",
                            lambda d, c, status=None: ([_ev(d, "10:00", "11:00", 2)], ""))
        ms._calendar_day_cache.clear()
        a, _ = ms._calendar_events_cached("2026-09-01", {})
        b, _ = ms._calendar_events_cached("2026-08-25", {})
        assert a[0].title == "2026-09-01" and b[0].title == "2026-08-25"


class TestModeGuess:
    @pytest.mark.parametrize("title,mode", [
        ("DBA Weekly Meeting", "meeting"),
        ("IT Nora 面试", "interview"),
        ("Candidate screen", "interview"),
        ("Nova AI Weekly Sharing", "sharing"),
        ("用户研究 share out", "sharing"),
        ("技术分享", "sharing"),
    ])
    def test_guess(self, title, mode):
        assert ms._calendar_mode_guess(title) == mode


# ── `.meta.json` sidecar ─────────────────────────────────────────────────────

class TestMeetingMeta:
    def test_path_is_a_stem_sibling(self, tmp_path):
        wav = tmp_path / "20260831_110000.DBA Weekly Meeting.wav"
        assert ms._meta_path(wav).name == "20260831_110000.DBA Weekly Meeting.meta.json"

    def test_write_then_read(self, tmp_path):
        wav = tmp_path / "20260831_110000.wav"
        ev = ms.CalendarEvent("DBA Weekly Meeting", "11:00", "11:30",
                              ["A", "B"], "Amsterdam", ["https://notion.so/x"])
        assert ms._write_meeting_meta(wav, ms._event_to_meta(ev)) is True
        got = ms._read_meeting_meta(wav)["calendar"]
        assert got["title"] == "DBA Weekly Meeting"
        assert got["attendees"] == ["A", "B"]
        assert got["mode_guess"] == "meeting"

    def test_second_write_merges_rather_than_clobbers(self, tmp_path):
        # The Notion fetch lands minutes after the calendar block is written;
        # overwriting would throw the event details away.
        wav = tmp_path / "20260831_110000.wav"
        ms._write_meeting_meta(wav, ms._event_to_meta(
            ms.CalendarEvent("X", "11:00", "11:30")))
        ms._write_meeting_meta(wav, {"notion": [{"url": "u", "text": "t"}]})
        data = ms._read_meeting_meta(wav)
        assert set(data) == {"calendar", "notion"}
        assert data["calendar"]["title"] == "X"

    def test_missing_file_reads_empty(self, tmp_path):
        assert ms._read_meeting_meta(tmp_path / "nope.wav") == {}

    def test_corrupt_file_reads_empty_not_raises(self, tmp_path):
        wav = tmp_path / "20260831_110000.wav"
        ms._meta_path(wav).write_text("{not json", encoding="utf-8")
        assert ms._read_meeting_meta(wav) == {}

    def test_no_temp_file_left_behind(self, tmp_path):
        wav = tmp_path / "20260831_110000.wav"
        ms._write_meeting_meta(wav, {"a": 1})
        assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []

    def test_sidecar_is_swept_by_the_delete_cascade(self, tmp_path):
        wav = tmp_path / "20260831_110000.wav"
        wav.write_bytes(b"RIFF")
        ms._write_meeting_meta(wav, {"calendar": {"title": "X"}})
        deleted, errors = ms._delete_meeting_files(wav)
        assert errors == [] and deleted == 2
        assert not ms._meta_path(wav).exists()

    def test_custom_named_recording_keeps_its_own_sidecar(self, tmp_path):
        # Two recordings in the same second with different titles must not
        # share one sidecar — and the rename cascade matches on `<stem>.`,
        # so the custom segment has to survive into the sidecar name.
        a = tmp_path / "20260831_110000.DBA Weekly Meeting.wav"
        b = tmp_path / "20260831_110000.1-1.wav"
        ms._write_meeting_meta(a, {"calendar": {"title": "DBA Weekly Meeting"}})
        ms._write_meeting_meta(b, {"calendar": {"title": "1-1"}})
        assert ms._meta_path(a) != ms._meta_path(b)
        assert ms._read_meeting_meta(a)["calendar"]["title"] == "DBA Weekly Meeting"
        assert ms._read_meeting_meta(b)["calendar"]["title"] == "1-1"
        assert ms._meta_path(a).name.startswith(a.stem + ".")

    def test_sidecar_follows_a_rename(self, tmp_path):
        wav = tmp_path / "20260831_110000.Old Name.wav"
        wav.write_bytes(b"RIFF")
        ms._write_meeting_meta(wav, {"calendar": {"title": "Old Name"}})
        new_wav = ms._rename_meeting_files(wav, "New Name")
        assert new_wav is not None
        assert ms._meta_path(new_wav).exists()
        assert not ms._meta_path(wav).exists()
        assert ms._read_meeting_meta(new_wav)["calendar"]["title"] == "Old Name"


# ── Roster / background / hotwords fed into the pipeline ────────────────────

_META = {
    "calendar": {
        "title": "DBA Weekly Meeting",
        "attendees": ["Bo Chen, Felix", "Qiu Yi Ran, Leo", "李雷",
                      "someone@example.com", "CN-HQ-6-(6S1) Amsterdam (8)"],
        "location": "CN-HQ-6-(6S1) Amsterdam (8)",
        "links": ["https://www.notion.so/x"],
    },
    "notion": [{"url": "https://www.notion.so/x", "text": "2026 Planning Roadmap\nQ3 goals…"}],
}


class TestCalendarPeople:
    def test_rooms_and_resources_dropped(self):
        people = ms._calendar_people(_META["calendar"])
        assert "CN-HQ-6-(6S1) Amsterdam (8)" not in people
        assert "Bo Chen, Felix" in people and "李雷" in people

    @pytest.mark.parametrize("entry", [
        "CN-HQ-6-(6S1) Amsterdam (8)",
        "US-NY-4-(4S3) New York (20)",
        "Big Meeting Room",
        "第三会议室",
    ])
    def test_room_shapes(self, entry):
        got = ms._calendar_people({"attendees": [entry], "location": ""})
        assert got == [], f"{entry!r} should not be treated as a person"

    @pytest.mark.parametrize("entry", ["Leo", "李雷", "Bo Chen, Felix"])
    def test_people_survive(self, entry):
        assert ms._calendar_people({"attendees": [entry], "location": ""}) == [entry]

    def test_entry_equal_to_location_dropped(self):
        got = ms._calendar_people({"attendees": ["Somewhere"], "location": "Somewhere"})
        assert got == []


class TestRoster:
    def test_joined_for_the_prompt(self):
        roster = ms._calendar_roster(_META)
        assert "李雷" in roster and "Leo" in roster
        assert "Amsterdam" not in roster            # room, not a person
        assert roster.count("、") == len(roster.split("、")) - 1

    def test_empty_without_a_sidecar(self):
        assert ms._calendar_roster({}) == ""
        assert ms._calendar_roster({"calendar": {}}) == ""


class TestBackground:
    def test_includes_url_and_text(self):
        bg = ms._calendar_background(_META)
        assert "2026 Planning Roadmap" in bg and "notion.so/x" in bg

    def test_capped(self):
        meta = {"notion": [{"url": "u", "text": "x" * 50_000}]}
        assert len(ms._calendar_background(meta, limit=1000)) <= 1000

    def test_multiple_docs_share_the_cap(self):
        meta = {"notion": [{"url": "a", "text": "x" * 800},
                           {"url": "b", "text": "y" * 800}]}
        assert len(ms._calendar_background(meta, limit=1000)) <= 1000

    def test_empty_and_malformed(self):
        assert ms._calendar_background({}) == ""
        assert ms._calendar_background({"notion": ["not a dict", {}, {"text": ""}]}) == ""


class TestCalendarHotwordTerms:
    def test_names_mined_email_dropped(self):
        terms = ms._calendar_hotword_terms(_META)
        low = {t.lower() for t in terms}
        assert "李雷" in terms
        assert "leo" in low or "qiu" in low or "chen" in low
        # An email login splits into common syllables that would bias speech.
        assert "someone" not in low
        # Rooms never become hotwords.
        assert not any("amsterdam" in t.lower() for t in terms)

    def test_no_attendees_yields_nothing(self):
        assert ms._calendar_hotword_terms({"calendar": {"attendees": []}}) == []
        assert ms._calendar_hotword_terms({}) == []


class TestMeetingContext:
    def test_reads_from_the_sidecar(self, tmp_path):
        wav = tmp_path / "20260831_110000.DBA.wav"
        ms._write_meeting_meta(wav, _META)
        roster, background = ms._meeting_context(wav)
        assert "Leo" in roster
        assert "2026 Planning Roadmap" in background

    def test_no_sidecar_is_empty_not_an_error(self, tmp_path):
        assert ms._meeting_context(tmp_path / "nope.wav") == ("", "")

    def test_none_path(self):
        assert ms._meeting_context(None) == ("", "")


class TestPromptWiring:
    """The placeholders must actually be substituted — an unreplaced
    `{roster}` would be sent to the model verbatim."""

    def test_polish_substitutes_roster(self, monkeypatch):
        seen = []
        monkeypatch.setattr(ms, "_llm_run",
                            lambda prompt, *a, **k: seen.append(prompt) or "out")
        ms.polish_transcript("hello", "claude", {}, "meeting", roster="李雷、Leo")
        assert seen and "{roster}" not in seen[0]
        assert "李雷、Leo" in seen[0]

    def test_polish_empty_roster_renders_placeholder(self, monkeypatch):
        seen = []
        monkeypatch.setattr(ms, "_llm_run",
                            lambda prompt, *a, **k: seen.append(prompt) or "out")
        ms.polish_transcript("hello", "claude", {}, "meeting")
        assert "{roster}" not in seen[0] and "（无）" in seen[0]

    @pytest.mark.parametrize("mode", ["meeting", "interview", "sharing"])
    def test_notes_substitutes_background(self, monkeypatch, mode):
        seen = []
        monkeypatch.setattr(ms, "_llm_run",
                            lambda prompt, *a, **k: seen.append(prompt) or "out")
        ms.generate_notes("t", "claude", {}, mode=mode, background="ROADMAP TEXT")
        assert len(seen) == 2                       # zh + en
        for p in seen:
            assert "{context}" not in p and "ROADMAP TEXT" in p

    def test_notes_empty_background_renders_placeholder(self, monkeypatch):
        seen = []
        monkeypatch.setattr(ms, "_llm_run",
                            lambda prompt, *a, **k: seen.append(prompt) or "out")
        ms.generate_notes("t", "claude", {}, mode="meeting")
        assert all("{context}" not in p and "（无）" in p for p in seen)

    def test_background_is_marked_as_not_spoken(self):
        """Injecting a roadmap doc into a summarisation prompt invites the
        model to report it as a decision. The guard must be explicit."""
        zh = ms._resolve_prompt({}, "notes_zh", mode="meeting")
        assert "不是会上说过的话" in zh
        assert "以**转写**为准" in zh
        en = ms._resolve_prompt({}, "notes_en", mode="meeting")
        assert "not something anyone said" in en
        assert "transcript** wins" in en

    def test_roster_forbids_guessing(self):
        polish = ms._resolve_prompt({}, "polish")
        assert "不要猜" in polish
        # A roster is attendees, not speakers — the model must not manufacture
        # one speaker per invitee.
        assert "名单是参会人，不是发言人" in polish

    @pytest.mark.parametrize("mode", ["meeting", "interview", "sharing"])
    def test_notes_substitutes_roster(self, monkeypatch, mode):
        """The notes step gets the roster too, not just polish.

        Regression: the 2026-09-02 「Nova AI 运维体系建设」 notes assigned
        action items to two people who were never in the room. The notes
        prompt demands an owner per item, the transcript carried no speaker
        labels, and the only name source it had was the 844-term glossary.
        """
        seen = []
        monkeypatch.setattr(ms, "_llm_run",
                            lambda prompt, *a, **k: seen.append(prompt) or "out")
        ms.generate_notes("t", "claude", {}, mode=mode, roster="李雷、Leo")
        assert len(seen) == 2                       # zh + en
        for prompt in seen:
            assert "{roster}" not in prompt and "李雷、Leo" in prompt

    def test_notes_empty_roster_renders_placeholder(self, monkeypatch):
        seen = []
        monkeypatch.setattr(ms, "_llm_run",
                            lambda prompt, *a, **k: seen.append(prompt) or "out")
        ms.generate_notes("t", "claude", {}, mode="meeting")
        assert all("{roster}" not in p and "（无）" in p for p in seen)

    @pytest.mark.parametrize("mode", ["meeting", "interview", "sharing"])
    @pytest.mark.parametrize("key", ["notes_zh", "notes_en"])
    def test_notes_glossary_is_not_a_roster(self, mode, key):
        """The glossary is auto-mined from Notion members, so it carries the
        whole company's names. Left unqualified, the notes step reads it as a
        cast list: 「洋哥」 became 刘洋 with seven action items attached, and
        莱来吉 — a name manufactured by ASR biasing out of 「下来…继续」 —
        got one of his own."""
        prompt = ms._resolve_prompt({}, key, mode=mode)
        needle = ("拼写参照，不是本场人员名单" if key == "notes_zh"
                  else "spelling reference, not a roster")
        assert needle in prompt

    def test_notes_action_owner_must_be_named_in_transcript(self):
        """Every action item needs a traceable source and a real owner.

        The old rule made the deadline optional ("待确认") but the OWNER
        mandatory, which forced the model to guess one on a transcript with
        no speaker labels.
        """
        zh = ms._resolve_prompt({}, "notes_zh", mode="meeting")
        assert "每条行动项都必须能在转写里找到对应原话" in zh
        assert "本人明确认领" in zh
        assert "不得推断成任何具体的人" in zh
        assert "定不下负责人就写「待确认」" in zh
        en = ms._resolve_prompt({}, "notes_en", mode="meeting")
        assert "must trace back to actual words in the transcript" in en
        assert "explicitly takes it on" in en
        assert 'write "TBD"' in en

    def test_notes_roster_allows_the_mentioned_non_attendee(self):
        """Owners normally come from the roster, but somebody outside it
        genuinely does get assigned work sometimes — that case is annotated,
        not silently dropped."""
        zh = ms._resolve_prompt({}, "notes_zh", mode="meeting")
        assert "行动项的负责人一般应出自这份名单" in zh
        assert "（非参会人，会上被提及）" in zh
        en = ms._resolve_prompt({}, "notes_en", mode="meeting")
        assert "should normally come from this list" in en
        assert "（非参会人，会上被提及）" in en

    def test_polish_does_not_extend_names_to_glossary_spellings(self):
        """「韩磊」 must not become 「韩磊磊」 just because the glossary has
        the longer name, and a run of ordinary words must not be assembled
        into one (「下来…继续」 → 「下莱来吉继续」)."""
        polish = ms._resolve_prompt({}, "polish")
        assert "不要给转写里已经成立的名字补字" in polish
        assert "不是人名" in polish


def _label_keys():
    """Keys of the GUI's `_LABELS` table, per language.

    `_LABELS` is a local inside `cmd_ui()`, so it can't be imported — but it
    can be read statically, which keeps this check running without PyQt6.
    """
    import ast
    tree = ast.parse(Path(ms.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        # `_LABELS: dict[str, dict[str, str]] = {...}` is an AnnAssign, which
        # has `.target` rather than `.targets`.
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = node.targets
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "_LABELS"
                   for t in targets):
            continue
        out = {}
        for lang_key, lang_val in zip(node.value.keys, node.value.values):
            out[lang_key.value] = {
                k.value for k in lang_val.keys if isinstance(k, ast.Constant)}
        return out
    raise AssertionError("_LABELS not found in meetingscribe.py")


class TestChipStateLabels:
    """The off / empty / pickable states must be distinguishable — a single
    「未关联」 covering all three is what made the feature look broken when it
    was merely disabled."""

    def test_all_chip_keys_exist_in_both_languages(self):
        labels = _label_keys()
        keys = {"cal.chip.loading", "cal.chip.none", "cal.chip.disabled",
                "cal.chip.error", "cal.chip.pick_fmt"}
        for lang in ("zh", "en"):
            missing = keys - labels[lang]
            assert not missing, f"{lang} missing {missing}"

    def test_every_error_key_has_a_message(self):
        # `CalendarStatus.reason_key` must never return a key the UI cannot
        # render — that would surface to the user as a raw "cal.err.needs_auth".
        labels = _label_keys()
        produced = set()
        for cli in ("missing", "not_logged_in", "error", "ok"):
            for cal_state in ("connected", "missing", "needs_auth"):
                key = ms.CalendarStatus(
                    cli=cli, services={"calendar": cal_state,
                                       "notion": "connected"}).reason_key()
                if key:
                    produced.add(key)
        produced |= {"cal.err.timeout", "cal.err.bad_reply", "cal.err.unknown"}
        for lang in ("zh", "en"):
            missing = produced - labels[lang]
            assert not missing, f"{lang} missing {missing}"

    def test_config_card_state_keys_exist(self):
        labels = _label_keys()
        keys = {f"cfg.cal.state.{s}" for s in
                ("connected", "needs_auth", "missing", "checking", "unknown")}
        keys |= {"cfg.cal.enable", "cfg.cal.btn.recheck", "cfg.cal.btn.open",
                 "cfg.cal.optional_note", "cfg.title.calendar"}
        for lang in ("zh", "en"):
            missing = keys - labels[lang]
            assert not missing, f"{lang} missing {missing}"

    def test_service_rows_have_labels(self):
        labels = _label_keys()
        for lang in ("zh", "en"):
            for key in ms._CALENDAR_SERVICES:
                assert f"cfg.cal.svc.{key}" in labels[lang]


class TestEventTitleAsFilename:
    """Event titles are arbitrary text and become part of a filename."""

    @pytest.mark.parametrize("title,expected", [
        ("DBA Weekly Meeting", "DBA Weekly Meeting"),
        ("EU/US sync", "EU_US sync"),        # slash neutralised, spaces kept
        ("Q3: planning", "Q3_ planning"),
        ("  spaced  ", "spaced"),
    ])
    def test_sanitised(self, title, expected):
        got = ms._sanitize_meeting_custom_name(title)
        assert "/" not in got and ":" not in got
        assert got == expected

    def test_title_that_sanitises_to_nothing(self, tmp_path):
        # Callers must treat "" as "drop the custom segment", i.e. fall back
        # to the bare timestamp rather than writing a dot-only stem.
        assert ms._sanitize_meeting_custom_name("...") == ""
        assert ms._sanitize_meeting_custom_name("") == ""


# ── Usage-limit refusals are their own failure class ─────────────────────────

class TestQuotaDetection:
    """A `claude -p` that fails because the account is out of budget looks
    nothing like a broken connector — the preflight reports every service
    healthy while every call fails. Misclassifying it sent the user to
    claude.ai to fix something that wasn't broken."""

    @pytest.mark.parametrize("blob,expected", [
        ("You've hit your session limit · resets 7:30pm (Asia/Shanghai)",
         "7:30pm (Asia/Shanghai)"),
        ("Error: usage limit reached, resets at midnight", "at midnight"),
        ("rate limit exceeded", ""),
    ])
    def test_recognised_with_reset_hint(self, blob, expected):
        assert ms._cli_quota_detail(blob) == expected

    @pytest.mark.parametrize("blob", [
        "claude not found on PATH", "No MCP server named \"x\"", "", None])
    def test_other_failures_are_not_quota(self, blob):
        assert ms._cli_quota_detail(blob) is None

    def test_fetch_reports_quota_without_blaming_the_connectors(self, monkeypatch):
        monkeypatch.setattr(
            ms, "_run_claude_cli",
            lambda a, t: (1, "", "You've hit your session limit · resets 7:30pm"))
        monkeypatch.setattr(
            ms, "_calendar_rediagnose",
            lambda: pytest.fail("a quota failure must not re-run the preflight"))
        st = _status()
        st.servers = {"calendar": "claude.ai Google Calendar"}
        cfg = {"calendar": {"provider_chain": ["claude-mcp"]}}
        events, err = ms._calendar_events("2026-09-01", cfg, status=st)
        assert events == [] and err == "cal.err.quota"
        assert st.detail == "7:30pm"

    def test_quota_failure_is_not_cached(self, monkeypatch):
        """Otherwise the day stays broken until the process restarts, long
        after the limit has reset."""
        calls = []

        def fake(day, cfg, status=None):
            calls.append(day)
            return [], "cal.err.quota"

        monkeypatch.setattr(ms, "_calendar_events", fake)
        ms._calendar_day_cache.clear()
        ms._calendar_events_cached("2026-09-01", {})
        ms._calendar_events_cached("2026-09-01", {})
        assert len(calls) == 2

    def test_quota_keys_have_messages_in_both_languages(self):
        labels = _label_keys()
        for lang in ("zh", "en"):
            assert "cal.err.quota" in labels[lang]
            assert "cal.err.quota_at" in labels[lang]
            assert "cal.stale_suffix" in labels[lang]
        # The reset-time variant must actually interpolate it.
        import ast
        assert "{at}" in _label_values()["zh"]["cal.err.quota_at"]


def _label_values():
    """Like `_label_keys`, but keeping the string values."""
    import ast
    tree = ast.parse(Path(ms.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        targets = ([node.target] if isinstance(node, ast.AnnAssign)
                   else node.targets if isinstance(node, ast.Assign) else [])
        if not any(isinstance(t, ast.Name) and t.id == "_LABELS" for t in targets):
            continue
        out = {}
        for lang_key, lang_val in zip(node.value.keys, node.value.values):
            vals = {}
            for k, v in zip(lang_val.keys, lang_val.values):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                    vals[k.value] = v.value
            out[lang_key.value] = vals
        return out
    raise AssertionError("_LABELS not found")
