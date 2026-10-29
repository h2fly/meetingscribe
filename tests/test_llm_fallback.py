"""
Tests for the LLM provider fallback chain: when the primary provider times
out, exits non-zero, or reports a usage limit, the next provider is tried.

The CLIs are never invoked here — the per-provider runners are monkeypatched.
Run: pytest tests/
"""

import io
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import meetingscribe as ms


CFG = {"llm_fallback": ["codex"], "llm_timeout": 60}


@pytest.fixture(autouse=True)
def isolated_provider_state(monkeypatch, tmp_path):
    """Keep the circuit-breaker state out of the user's real config dir."""
    monkeypatch.setattr(ms, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        ms, "_LLM_PROVIDER_STATE_FILE", tmp_path / ".llm-provider-state.json")


def _priority_cfg():
    return {
        "llm_fallback": ["codex"],
        "llm_priority": ["claude", "codex"],
        "llm_failover_cooldown_hours": 3,
        "llm_timeout": 60,
        "llm": {
            "claude": {"type": "claude-cli"},
            "codex": {"type": "codex-cli", "timeout": 30},
        },
    }


class TestProviderChain:
    def test_primary_then_fallbacks(self):
        assert ms._llm_provider_chain("claude", CFG) == ["claude", "codex"]

    def test_no_duplicate_when_primary_is_the_fallback(self):
        assert ms._llm_provider_chain("codex", CFG) == ["codex"]

    def test_empty_fallback_restores_single_provider(self):
        assert ms._llm_provider_chain("claude", {"llm_fallback": []}) == ["claude"]

    def test_missing_key_is_tolerated(self):
        assert ms._llm_provider_chain("claude", {}) == ["claude"]

    def test_blank_entries_ignored(self):
        cfg = {"llm_fallback": ["", "  ", "codex"]}
        assert ms._llm_provider_chain("claude", cfg) == ["claude", "codex"]

    def test_new_defaults_use_codex_proxy(self):
        assert ms.DEFAULT_CONFIG["polish_provider"] == "codex_proxy"
        assert ms.DEFAULT_CONFIG["meeting_notes_provider"] == "codex_proxy"
        assert ms._llm_priority({}) == ["codex_proxy", "claude"]

    def test_legacy_codex_config_is_an_alias_for_proxy(self):
        cfg = {"llm": {"codex": {"type": "codex-cli", "profile": "proxy",
                                  "model": "legacy-model"}}}
        resolved = ms._llm_provider_config("codex", cfg)
        assert resolved["type"] == "codex-cli"
        assert resolved["profile"] == "proxy"
        assert resolved["model"] == "legacy-model"


class TestProviderPriorityCache:
    def test_failed_claude_is_skipped_during_cooldown(self, monkeypatch):
        calls = []
        now = [1000.0]
        monkeypatch.setattr(ms.time, "time", lambda: now[0])

        def claude(prompt, label, timeout, model=""):
            calls.append("claude")
            raise ms._LLMCallFailed("quota", "上限")

        def codex(prompt, label, timeout, pcfg):
            calls.append("codex")
            return "ok"

        monkeypatch.setattr(ms, "_llm_claude_cli", claude)
        monkeypatch.setattr(ms, "_llm_codex_cli", codex)
        cfg = _priority_cfg()

        assert ms._llm_run("p1", "claude", cfg, "标签") == "ok"
        assert ms._llm_run("p2", "claude", cfg, "标签") == "ok"
        assert calls == ["claude", "codex", "codex"]

        state = json.loads(ms._LLM_PROVIDER_STATE_FILE.read_text())
        assert state["cooldown_until"] == 1000.0 + 3 * 3600

    def test_expired_cooldown_probes_claude_and_restores_it(self, monkeypatch):
        calls = []
        now = [1000.0]
        monkeypatch.setattr(ms.time, "time", lambda: now[0])
        cfg = _priority_cfg()

        def claude(prompt, label, timeout, model=""):
            calls.append("claude")
            if len(calls) == 1:
                raise ms._LLMCallFailed("quota", "上限")
            return "claude answer"

        monkeypatch.setattr(ms, "_llm_claude_cli", claude)
        monkeypatch.setattr(ms, "_llm_codex_cli",
                            lambda *a, **k: calls.append("codex") or "ok")

        assert ms._llm_run("p1", "claude", cfg, "标签") == "ok"
        now[0] += 3 * 3600 + 1
        assert ms._llm_run("p2", "claude", cfg, "标签") == "claude answer"
        assert calls == ["claude", "codex", "claude"]
        assert not ms._LLM_PROVIDER_STATE_FILE.exists()

    def test_failed_probe_restarts_the_cooldown(self, monkeypatch):
        calls = []
        now = [1000.0]
        monkeypatch.setattr(ms.time, "time", lambda: now[0])
        cfg = _priority_cfg()

        def claude(prompt, label, timeout, model=""):
            calls.append("claude")
            raise ms._LLMCallFailed("quota", "上限")

        monkeypatch.setattr(ms, "_llm_claude_cli", claude)
        monkeypatch.setattr(ms, "_llm_codex_cli",
                            lambda *a, **k: calls.append("codex") or "ok")

        assert ms._llm_run("p1", "claude", cfg, "标签") == "ok"
        now[0] += 3 * 3600 + 1
        assert ms._llm_run("p2", "claude", cfg, "标签") == "ok"
        assert calls == ["claude", "codex", "claude", "codex"]

        state = json.loads(ms._LLM_PROVIDER_STATE_FILE.read_text())
        assert state["cooldown_until"] == now[0] + 3 * 3600

    def test_expired_cooldown_has_one_probe_lease(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr(ms.time, "time", lambda: now[0])
        cfg = _priority_cfg()
        state = {
            "primary": "claude",
            "fallback": "codex",
            "cooldown_until": 999.0,
            "probe_until": 0,
        }
        ms._LLM_PROVIDER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        ms._LLM_PROVIDER_STATE_FILE.write_text(json.dumps(state))

        assert ms._llm_select_provider("claude", cfg) == ("claude", True)
        assert ms._llm_select_provider("claude", cfg) == ("codex", False)

    def test_priority_and_cooldown_are_configurable(self):
        cfg = {
            "llm_priority": "codex, claude",
            "llm_failover_cooldown_hours": 0.5,
        }
        assert ms._llm_priority(cfg) == ["codex", "claude"]
        assert ms._llm_failover_cooldown_seconds(cfg) == 1800


class TestFallbackDispatch:
    def _cfg(self):
        return {"llm_fallback": ["codex"], "llm_timeout": 60,
                "llm": {"claude": {"type": "claude-cli"},
                        "codex": {"type": "codex-cli", "timeout": 30}}}

    @pytest.mark.parametrize("kind", ["quota", "timeout", "error", "missing"])
    def test_every_failure_kind_falls_through(self, monkeypatch, kind):
        calls = []

        def bad_claude(prompt, label, timeout, model=""):
            calls.append("claude")
            raise ms._LLMCallFailed(kind, "boom")

        def good_codex(prompt, label, timeout, pcfg):
            calls.append("codex")
            return "codex answer"

        monkeypatch.setattr(ms, "_llm_claude_cli", bad_claude)
        monkeypatch.setattr(ms, "_llm_codex_cli", good_codex)
        out = ms._llm_run("p", "claude", self._cfg(), "标签")
        assert out == "codex answer"
        assert calls == ["claude", "codex"]

    def test_primary_success_never_touches_the_fallback(self, monkeypatch):
        monkeypatch.setattr(ms, "_llm_claude_cli",
                            lambda p, l, t, model="": "claude answer")
        monkeypatch.setattr(
            ms, "_llm_codex_cli",
            lambda *a, **k: pytest.fail("fallback must not run on success"))
        assert ms._llm_run("p", "claude", self._cfg(), "标签") == "claude answer"

    def test_cancellation_never_falls_back(self, monkeypatch):
        """停止任务 is not a provider failure — retrying elsewhere would keep
        burning tokens after the user asked to stop."""
        def cancelled(prompt, label, timeout, model=""):
            raise ms._PipelineCancelled()

        monkeypatch.setattr(ms, "_llm_claude_cli", cancelled)
        monkeypatch.setattr(
            ms, "_llm_codex_cli",
            lambda *a, **k: pytest.fail("cancel must not trigger a fallback"))
        with pytest.raises(ms._PipelineCancelled):
            ms._llm_run("p", "claude", self._cfg(), "标签")

    def test_all_failing_exits_with_every_reason(self, monkeypatch, capsys):
        monkeypatch.setattr(ms, "_llm_claude_cli", lambda p, l, t, model="":
                            (_ for _ in ()).throw(ms._LLMCallFailed("quota", "上限")))
        monkeypatch.setattr(ms, "_llm_codex_cli", lambda *a, **k:
                            (_ for _ in ()).throw(ms._LLMCallFailed("timeout", "慢")))
        with pytest.raises(SystemExit):
            ms._llm_run("p", "claude", self._cfg(), "标签")
        out = capsys.readouterr().out
        assert "claude" in out and "codex" in out
        assert "上限" in out and "慢" in out

    def test_unknown_fallback_name_is_skipped_not_fatal(self, monkeypatch):
        cfg = self._cfg()
        cfg["llm_fallback"] = ["nope", "codex"]
        monkeypatch.setattr(ms, "_llm_claude_cli", lambda p, l, t, model="":
                            (_ for _ in ()).throw(ms._LLMCallFailed("quota", "上限")))
        monkeypatch.setattr(ms, "_llm_codex_cli", lambda *a, **k: "ok")
        assert ms._llm_run("p", "claude", cfg, "标签") == "ok"

    def test_fallback_uses_its_own_timeout(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(ms, "_llm_claude_cli", lambda p, l, t, model="":
                            (_ for _ in ()).throw(ms._LLMCallFailed("quota", "上限")))

        def codex(prompt, label, timeout, pcfg):
            seen["timeout"] = timeout
            return "ok"

        monkeypatch.setattr(ms, "_llm_codex_cli", codex)
        ms._llm_run("p", "claude", self._cfg(), "标签")
        # 30 from the provider block, NOT the global 60.
        assert seen["timeout"] == 30

    def test_codex_proxy_is_primary_and_claude_is_fallback(self, monkeypatch):
        calls = []

        def codex(prompt, label, timeout, pcfg):
            calls.append(("codex_proxy", pcfg["profile"], pcfg["model"]))
            raise ms._LLMCallFailed("error", "proxy down")

        def claude(prompt, label, timeout, model=""):
            calls.append(("claude", model))
            return "claude answer"

        monkeypatch.setattr(ms, "_llm_codex_cli", codex)
        monkeypatch.setattr(ms, "_llm_claude_cli", claude)
        cfg = {
            "llm_priority": ["codex_proxy", "claude"],
            "llm_fallback": ["claude"],
            "llm_failover_cooldown_hours": 0,
        }
        assert ms._llm_run("p", "codex_proxy", cfg, "标签") == "claude answer"
        assert calls[0][0] == "codex_proxy"
        assert calls[0][1] == "proxy"
        assert calls[1][0] == "claude"


class TestClaudeRunnerRaisesInsteadOfExiting:
    """Before the fallback existed these paths called sys.exit(1), which made
    a second provider structurally impossible."""

    def test_quota_is_classified(self, monkeypatch):
        class P:
            returncode = 1

            def communicate(self, timeout=None):
                return "", "You've hit your session limit · resets 7:30pm"

        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: P())
        with pytest.raises(ms._LLMCallFailed) as e:
            ms._llm_claude_cli("p", "标签", 5)
        assert e.value.kind == "quota"
        assert "7:30pm" in e.value.message

    def test_other_nonzero_exit_is_error_not_quota(self, monkeypatch):
        class P:
            returncode = 2

            def communicate(self, timeout=None):
                return "", "some unrelated explosion"

        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: P())
        with pytest.raises(ms._LLMCallFailed) as e:
            ms._llm_claude_cli("p", "标签", 5)
        assert e.value.kind == "error"

    def test_missing_binary_is_classified(self, monkeypatch):
        def boom(*a, **k):
            raise FileNotFoundError()

        monkeypatch.setattr(subprocess, "Popen", boom)
        with pytest.raises(ms._LLMCallFailed) as e:
            ms._llm_claude_cli("p", "标签", 5)
        assert e.value.kind == "missing"


class TestCodexRunner:
    def test_reads_the_answer_from_the_output_file(self, monkeypatch, tmp_path):
        captured = {}

        class P:
            returncode = 0

            def communicate(self, timeout=None):
                # `-o <path>` is the argument right after "-o".
                args = captured["cmd"]
                Path(args[args.index("-o") + 1]).write_text(
                    "  codex 的回答  ", encoding="utf-8")
                return "agent chatter on stdout", ""

        def fake_popen(cmd, **kw):
            captured["cmd"] = cmd
            return P()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        out = ms._llm_codex_cli("prompt", "标签", 30, {})
        # stdout carries agent progress, so the answer must come from the file.
        assert out == "codex 的回答"
        assert "--skip-git-repo-check" in captured["cmd"]
        assert captured["cmd"][:3] == ["codex", "-p", "proxy"]
        assert captured["cmd"][3] == "exec"
        assert captured["cmd"][-1] == "prompt"

    def test_profile_model_and_extra_args_are_forwarded(self, monkeypatch):
        captured = {}

        class P:
            returncode = 0

            def communicate(self, timeout=None):
                args = captured["cmd"]
                Path(args[args.index("-o") + 1]).write_text("x", encoding="utf-8")
                return "", ""

        monkeypatch.setattr(subprocess, "Popen",
                            lambda cmd, **kw: (captured.update(cmd=cmd), P())[1])
        ms._llm_codex_cli("p", "标签", 30,
                          {"profile": "proxy", "model": "m1", "args": ["--json"]})
        cmd = captured["cmd"]
        assert cmd[:3] == ["codex", "-p", "proxy"]
        assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "m1"
        assert "--json" in cmd

    def test_personal_profile_does_not_use_proxy(self, monkeypatch):
        captured = {}

        class P:
            returncode = 0

            def communicate(self, timeout=None):
                out = Path(captured["cmd"][captured["cmd"].index("-o") + 1])
                out.write_text("x", encoding="utf-8")
                return "", ""

        monkeypatch.setattr(subprocess, "Popen",
                            lambda cmd, **kw: (captured.update(cmd=cmd), P())[1])
        ms._llm_codex_cli("p", "标签", 30,
                          {"profile": "", "model": "", "models": []})
        cmd = captured["cmd"]
        assert cmd[0:2] == ["codex", "exec"]
        assert "proxy" not in cmd

    def test_capacity_switches_to_the_next_model(self, monkeypatch):
        seen = []

        def fake_body(prompt, label, timeout, pcfg, tmpdir):
            seen.append(pcfg["model"])
            if pcfg["model"] == "first":
                raise ms._LLMCallFailed("capacity", "模型容量不足")
            return "answer"

        monkeypatch.setattr(ms, "_llm_codex_cli_body", fake_body)
        out = ms._llm_codex_cli(
            "p", "标签", 30,
            {"profile": "proxy", "model": "first",
             "models": ["second"]})
        assert out == "answer"
        assert seen == ["first", "second"]

    def test_model_quota_switches_to_the_next_model(self, monkeypatch):
        seen = []

        def fake_body(prompt, label, timeout, pcfg, tmpdir):
            seen.append(pcfg["model"])
            if pcfg["model"] == "gpt-5.5":
                raise ms._LLMCallFailed("model_quota", "Codex 用量已达上限")
            return "answer"

        monkeypatch.setattr(ms, "_llm_codex_cli_body", fake_body)
        assert ms._llm_codex_cli(
            "p", "标签", 30,
            {"profile": "proxy", "model": "gpt-5.5",
             "models": ["gpt-5.6-sol"]}) == "answer"
        assert seen == ["gpt-5.5", "gpt-5.6-sol"]

    def test_model_quota_walks_all_configured_models(self, monkeypatch):
        seen = []

        def fake_body(prompt, label, timeout, pcfg, tmpdir):
            seen.append(pcfg["model"])
            if pcfg["model"] != "gpt-5.5":
                raise ms._LLMCallFailed("model_quota", "Codex 用量已达上限")
            return "answer"

        monkeypatch.setattr(ms, "_llm_codex_cli_body", fake_body)
        assert ms._llm_codex_cli(
            "p", "标签", 30,
            {"profile": "proxy", "model": "gpt-5.6-sol",
             "models": ["gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5"]}
        ) == "answer"
        assert seen == [
            "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5"
        ]

    def test_empty_answer_is_a_failure_not_an_empty_string(self, monkeypatch):
        class P:
            returncode = 0

            def communicate(self, timeout=None):
                return "", ""      # no output file written

        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: P())
        with pytest.raises(ms._LLMCallFailed) as e:
            ms._llm_codex_cli("p", "标签", 30, {})
        assert e.value.kind == "error"

    def test_quota_from_codex_is_classified(self, monkeypatch):
        class P:
            returncode = 1

            def communicate(self, timeout=None):
                return "", "usage limit reached, resets at 9pm"

        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: P())
        with pytest.raises(ms._LLMCallFailed) as e:
            ms._llm_codex_cli("p", "标签", 30, {})
        assert e.value.kind == "model_quota" and "9pm" in e.value.message

    def test_model_capacity_from_codex_is_classified(self, monkeypatch):
        class P:
            returncode = 1

            def communicate(self, timeout=None):
                return "", "Selected model is at capacity"

        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: P())
        with pytest.raises(ms._LLMCallFailed) as e:
            ms._llm_codex_cli("p", "标签", 30, {})
        assert e.value.kind == "capacity"

    def test_missing_binary(self, monkeypatch):
        monkeypatch.setattr(subprocess, "Popen",
                            lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
        with pytest.raises(ms._LLMCallFailed) as e:
            ms._llm_codex_cli("p", "标签", 30, {})
        assert e.value.kind == "missing"


class TestHttpProvidersJoinTheChain:
    """An HTTP provider that exits the process can never be fallen back FROM.

    `_llm_openai` / `_llm_gemini` used to `sys.exit(1)`, so a 429 on the
    primary killed the pipeline even with `llm_fallback` configured.
    """

    def _cfg(self):
        return {"llm_fallback": ["codex"], "llm_timeout": 60,
                "llm": {"openai": {"type": "openai", "api_key": "k"},
                        "codex": {"type": "codex-cli", "timeout": 30}}}

    def test_openai_429_falls_through_to_codex(self, monkeypatch):
        err = urllib.error.HTTPError(
            None, 429, "Too Many Requests", {},
            io.BytesIO(b"rate limit exceeded"))
        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda *a, **k: (_ for _ in ()).throw(err))
        monkeypatch.setattr(ms, "_llm_codex_cli",
                            lambda prompt, label, timeout, pcfg: "codex answer")
        assert ms._llm_run("p", "openai", self._cfg(), "标签") == "codex answer"

    def test_gemini_timeout_falls_through_to_codex(self, monkeypatch):
        err = urllib.error.URLError(TimeoutError("timed out"))
        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda *a, **k: (_ for _ in ()).throw(err))
        monkeypatch.setattr(ms, "_llm_codex_cli",
                            lambda prompt, label, timeout, pcfg: "codex answer")
        cfg = self._cfg()
        cfg["llm"]["gemini"] = {"type": "gemini", "api_key": "k"}
        assert ms._llm_run("p", "gemini", cfg, "标签") == "codex answer"

    def test_a_bad_reply_shape_also_falls_through(self, monkeypatch):
        # A 200 whose JSON has no choices is a provider failure too — the
        # fallback is the only thing that can still produce an answer.
        class R:
            def read(self):
                return b'{"choices": []}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: R())
        monkeypatch.setattr(ms, "_llm_codex_cli",
                            lambda prompt, label, timeout, pcfg: "codex answer")
        assert ms._llm_run("p", "openai", self._cfg(), "标签") == "codex answer"


class TestCodexTempDirIsCleanedUp:
    """One temp dir per call, so a long session must not leak one each time."""

    def _run(self, monkeypatch, returncode=0, write=True):
        seen = {}

        class P:
            def __init__(self):
                self.returncode = returncode

            def communicate(self, timeout=None):
                args = seen["cmd"]
                out = Path(args[args.index("-o") + 1])
                seen["out"] = out
                if write:
                    out.write_text("answer", encoding="utf-8")
                return "", ""

        monkeypatch.setattr(subprocess, "Popen",
                            lambda cmd, **kw: (seen.update(cmd=cmd), P())[1])
        return seen

    def test_removed_after_success(self, monkeypatch):
        seen = self._run(monkeypatch)
        assert ms._llm_codex_cli("p", "标签", 30, {}) == "answer"
        assert not seen["out"].parent.exists()

    def test_removed_after_failure(self, monkeypatch):
        seen = self._run(monkeypatch, returncode=1, write=False)
        with pytest.raises(ms._LLMCallFailed):
            ms._llm_codex_cli("p", "标签", 30, {})
        assert not seen["out"].parent.exists()


class TestStdinIsClosed:
    """`codex exec` (and `claude -p`) also accept a PIPED prompt, so an
    inherited stdin that never reaches EOF makes them wait for input forever
    even though the prompt was passed as an argument. Measured on codex with
    the same trivial prompt: inherited stdin → no reply in 90 s; DEVNULL →
    9.3 s. This is the difference between a working fallback and one that only
    ever reports a timeout."""

    def _spawn_kwargs(self, monkeypatch, fn):
        seen = {}

        class P:
            returncode = 0

            def communicate(self, timeout=None):
                args = seen["cmd"]
                if "-o" in args:
                    Path(args[args.index("-o") + 1]).write_text("x", encoding="utf-8")
                return "answer", ""

        monkeypatch.setattr(
            subprocess, "Popen",
            lambda cmd, **kw: (seen.update(cmd=cmd, kw=kw), P())[1])
        fn()
        return seen["kw"]

    def test_codex_gets_devnull(self, monkeypatch):
        kw = self._spawn_kwargs(
            monkeypatch, lambda: ms._llm_codex_cli("p", "标签", 30, {}))
        assert kw["stdin"] is subprocess.DEVNULL

    def test_claude_gets_devnull(self, monkeypatch):
        ms._PIPELINE_CANCEL.clear()
        kw = self._spawn_kwargs(
            monkeypatch, lambda: ms._llm_claude_cli("p", "标签", 30))
        assert kw["stdin"] is subprocess.DEVNULL


class TestSwitchIsVisibleToTheUI:
    """A run served by the fallback must not look identical to one served by
    the primary — the GUI's run-log pane is where that is said."""

    def _cfg(self):
        return {"llm_fallback": ["codex"], "llm_timeout": 60,
                "llm": {"claude": {"type": "claude-cli"},
                        "codex": {"type": "codex-cli"}}}

    def _patch(self, monkeypatch):
        monkeypatch.setattr(
            ms, "_llm_claude_cli",
            lambda prompt, label, timeout, model="": (_ for _ in ()).throw(
                ms._LLMCallFailed("quota", "用量已达上限")))
        monkeypatch.setattr(ms, "_llm_codex_cli",
                            lambda prompt, label, timeout, pcfg: "codex answer")

    def test_notify_hook_receives_the_switch(self, monkeypatch):
        self._patch(monkeypatch)
        seen = []
        monkeypatch.setattr(ms, "_llm_notify", seen.append)
        assert ms._llm_run("p", "claude", self._cfg(), "标签") == "codex answer"
        assert len(seen) == 1
        assert "claude" in seen[0] and "codex" in seen[0]

    def test_no_hook_no_crash(self, monkeypatch):
        self._patch(monkeypatch)
        monkeypatch.setattr(ms, "_llm_notify", None)
        assert ms._llm_run("p", "claude", self._cfg(), "标签") == "codex answer"

    def test_a_broken_hook_never_breaks_the_pipeline(self, monkeypatch):
        # The hook is UI code observing the pipeline; a dead Qt object on the
        # other end must not take the run down with it.
        self._patch(monkeypatch)
        monkeypatch.setattr(
            ms, "_llm_notify",
            lambda msg: (_ for _ in ()).throw(RuntimeError("wrapped C/C++ deleted")))
        assert ms._llm_run("p", "claude", self._cfg(), "标签") == "codex answer"


class TestUnrunnableBinaryIsAMissing:
    """A CLI that EXISTS but cannot be executed must reach the fallback.

    Observed live: npm had installed @anthropic-ai/claude-code but the native
    binary never downloaded, leaving a 0644 stub at `bin/claude.exe`. Exec
    then fails with `PermissionError: [Errno 13] ... 'claude'` — which, being
    caught nowhere, escaped `_llm_run` entirely, so codex never got its turn
    in exactly the situation the fallback was built for.
    """

    @pytest.mark.parametrize("exc", [FileNotFoundError(), PermissionError(13, "denied"),
                                     OSError(8, "Exec format error")])
    def test_claude(self, monkeypatch, exc):
        ms._PIPELINE_CANCEL.clear()
        monkeypatch.setattr(subprocess, "Popen",
                            lambda *a, **k: (_ for _ in ()).throw(exc))
        with pytest.raises(ms._LLMCallFailed) as e:
            ms._llm_claude_cli("p", "标签", 30)
        assert e.value.kind == "missing"

    @pytest.mark.parametrize("exc", [FileNotFoundError(), PermissionError(13, "denied")])
    def test_codex(self, monkeypatch, exc):
        monkeypatch.setattr(subprocess, "Popen",
                            lambda *a, **k: (_ for _ in ()).throw(exc))
        with pytest.raises(ms._LLMCallFailed) as e:
            ms._llm_codex_cli("p", "标签", 30, {})
        assert e.value.kind == "missing"

    def test_the_chain_actually_switches(self, monkeypatch):
        ms._PIPELINE_CANCEL.clear()
        cfg = {"llm_fallback": ["codex"], "llm_timeout": 60,
               "llm": {"claude": {"type": "claude-cli"},
                       "codex": {"type": "codex-cli"}}}
        monkeypatch.setattr(
            subprocess, "Popen",
            lambda *a, **k: (_ for _ in ()).throw(PermissionError(13, "denied")))
        monkeypatch.setattr(ms, "_llm_codex_cli",
                            lambda prompt, label, timeout, pcfg: "codex answer")
        assert ms._llm_run("p", "claude", cfg, "标签") == "codex answer"


class TestCalendarPathIsExcluded:
    """The calendar / Notion fetch must never fall back: it needs Claude's own
    MCP connectors, so another CLI could not serve it even in principle."""

    def test_calendar_uses_run_claude_cli_not_llm_run(self, monkeypatch):
        monkeypatch.setattr(
            ms, "_llm_run",
            lambda *a, **k: pytest.fail("calendar must not go through _llm_run"))
        monkeypatch.setattr(ms, "_run_claude_cli", lambda a, t: (0, "[]", ""))
        st = ms.CalendarStatus(
            cli="ok", services={"calendar": "connected", "notion": "connected"})
        st.servers = {"calendar": "claude.ai Google Calendar"}
        events, err = ms._calendar_events("2026-09-01", {}, status=st)
        assert err == "" and events == []

    def test_notion_fetch_uses_run_claude_cli_not_llm_run(self, monkeypatch):
        monkeypatch.setattr(
            ms, "_llm_run",
            lambda *a, **k: pytest.fail("notion must not go through _llm_run"))
        monkeypatch.setattr(ms, "_run_claude_cli", lambda a, t: (0, "page text", ""))
        st = ms.CalendarStatus(
            cli="ok", services={"calendar": "connected", "notion": "connected"})
        st.servers = {"notion": "notion"}
        docs = ms._fetch_notion_context(["https://notion.so/x"], {}, status=st)
        assert docs and docs[0]["text"] == "page text"
