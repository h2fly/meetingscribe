"""
Tests for the environment check (运行环境): which external dependencies the
app reports as missing, and the LLM login probe. CoreAudio, PATH and the LLM
are all faked — nothing here touches devices, the shell or the network.
Run: pytest tests/
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import meetingscribe as ms


@pytest.fixture
def env(monkeypatch):
    """Fake world: `which` answers from a set, the plan and the Multi-Output
    composition are set per test."""
    state = {"cmds": {"claude", "ffmpeg"}, "plan": ms.AudioPlan(
        sys_source_name="BlackHole 2ch", multi_output_name="多输出设备"),
        "subs": ["MacBook Air Speakers"]}
    import shutil
    monkeypatch.setattr(shutil, "which",
                        lambda c: f"/bin/{c}" if c in state["cmds"] else None)
    monkeypatch.setattr(ms, "resolve_audio_devices",
                        lambda query_fresh=False: state["plan"])
    monkeypatch.setattr(ms, "_get_multi_output_physical_subs",
                        lambda name: state["subs"])
    monkeypatch.setattr(ms.sys, "platform", "darwin")
    return state


CFG = {"polish_provider": "claude", "meeting_notes_provider": "claude",
       "llm_fallback": []}


def _by_key(items):
    return {i.key: i for i in items}


def test_all_present(env):
    items = _by_key(ms._environment_check(CFG))
    assert [k for k, i in items.items() if not i.ok] == []
    assert items["multi_output"].detail == "多输出设备"
    assert items["llm"].detail == "claude"


def test_missing_blackhole_and_multi_output(env):
    env["plan"] = ms.AudioPlan()
    items = _by_key(ms._environment_check(CFG))
    assert items["blackhole"].state == "missing" and items["blackhole"].required
    assert items["multi_output"].state == "missing"


def test_multi_output_without_physical_output(env):
    env["subs"] = []
    item = _by_key(ms._environment_check(CFG))["multi_output"]
    assert item.state == "no_physical" and not item.ok


def test_llm_fallback_counts_as_usable(env):
    env["cmds"] = {"claude"}
    cfg = dict(CFG, polish_provider="codex_proxy",
               meeting_notes_provider="codex_proxy", llm_fallback=["claude"])
    item = _by_key(ms._environment_check(cfg))["llm"]
    assert item.ok and item.detail == "claude"


def test_llm_missing_names_the_command(env):
    env["cmds"] = set()
    cfg = dict(CFG, polish_provider="codex_proxy", llm_fallback=["claude"])
    item = _by_key(ms._environment_check(cfg))["llm"]
    assert item.state == "missing"
    assert item.missing_cmds == ["claude", "codex"]


def test_api_provider_needs_key_not_command(env):
    env["cmds"] = set()
    cfg = dict(CFG, polish_provider="openai", meeting_notes_provider="openai",
               llm={"openai": {"api_key": ""}})
    item = _by_key(ms._environment_check(cfg))["llm"]
    assert item.state == "missing" and item.detail == "openai"
    cfg["llm"]["openai"]["api_key"] = "set-in-test"
    assert _by_key(ms._environment_check(cfg))["llm"].ok


def test_ffmpeg_is_optional(env):
    env["cmds"] = {"claude"}
    item = _by_key(ms._environment_check(CFG))["ffmpeg"]
    assert item.state == "missing" and item.required is False


def test_audio_rows_skipped_off_macos(env, monkeypatch):
    monkeypatch.setattr(ms.sys, "platform", "linux")
    assert [i.key for i in ms._environment_check(CFG)] == ["llm", "ffmpeg"]


def test_probe_success(env, monkeypatch):
    monkeypatch.setattr(ms, "_llm_call_one", lambda *a: "OK")
    assert ms._llm_probe(CFG) == (True, "claude")


def test_probe_reports_cli_error_and_never_exits(env, monkeypatch):
    def boom(*a):
        raise ms._LLMCallFailed("error", "Invalid API key · Please run /login")
    monkeypatch.setattr(ms, "_llm_call_one", boom)
    ok, msg = ms._llm_probe(CFG)
    assert not ok and "run /login" in msg and msg.startswith("claude:")


def test_probe_skips_unavailable_and_survives_sys_exit(env, monkeypatch):
    env["cmds"] = {"codex"}
    cfg = dict(CFG, polish_provider="claude", llm_fallback=["codex_proxy"])
    seen = []

    def call(prompt, name, *a):
        seen.append(name)
        raise SystemExit(1)
    monkeypatch.setattr(ms, "_llm_call_one", call)
    ok, msg = ms._llm_probe(cfg)
    assert seen == ["codex_proxy"] and not ok and "SystemExit" in msg
