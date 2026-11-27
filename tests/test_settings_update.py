"""
Tests for `update_config_values` — the writer behind the settings form. It
must change only the given leaves, keep every comment, update the running
config too, and never overwrite a config.jsonc it cannot parse.
Run: pytest tests/
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import meetingscribe as ms

SAMPLE = """{
  // 校对用哪个模型
  "polish_provider": "claude",
  "llm_fallback": ["claude"],   // 备选
  "live_captions": {
    "mt_provider": "hymt",  // 翻译引擎
    "review": {
      "enabled": true
    }
  },
  "stt": {
    "funasr": {
      "spk_model": ""
    }
  }
}
"""


@pytest.fixture
def cfg_file(monkeypatch, tmp_path):
    f = tmp_path / "config.jsonc"
    f.write_text(SAMPLE, encoding="utf-8")
    monkeypatch.setattr(ms, "CONFIG_FILE", f)
    monkeypatch.setattr(ms, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(ms, "HOTWORD_FILE", tmp_path / "hotword.jsonc")
    return f


def _disk(f):
    return json.loads(ms._strip_jsonc_comments(f.read_text(encoding="utf-8")))


def test_changes_only_the_given_leaves_and_keeps_comments(cfg_file):
    cfg = {"polish_provider": "claude", "unrelated_default": 1}
    ms.update_config_values(cfg, {
        ("live_captions", "review", "enabled"): False,
        ("llm_fallback",): [],
        ("stt", "funasr", "spk_model"): "cam++",
    })
    text = cfg_file.read_text(encoding="utf-8")
    assert "// 校对用哪个模型" in text and "// 翻译引擎" in text and "// 备选" in text
    d = _disk(cfg_file)
    assert d["live_captions"]["review"]["enabled"] is False
    assert d["llm_fallback"] == [] and d["stt"]["funasr"]["spk_model"] == "cam++"
    # Runtime defaults never leak into the file.
    assert "unrelated_default" not in d
    # Only the three lines changed.
    changed = [a for a, b in zip(SAMPLE.splitlines(), text.splitlines()) if a != b]
    assert len(changed) == 3


def test_updates_running_config(cfg_file):
    cfg = {"live_captions": {"mt_provider": "hymt"}}
    ms.update_config_values(cfg, {("live_captions", "mt_provider"): "marian"})
    assert cfg["live_captions"]["mt_provider"] == "marian"
    assert _disk(cfg_file)["live_captions"]["mt_provider"] == "marian"


def test_unparseable_file_is_never_overwritten(cfg_file):
    cfg_file.write_text("{ broken", encoding="utf-8")
    with pytest.raises(ValueError):
        ms.update_config_values({}, {("polish_provider",): "codex_proxy"})
    assert cfg_file.read_text(encoding="utf-8") == "{ broken"


def test_missing_file_gets_just_the_override(cfg_file):
    cfg_file.unlink()
    ms.update_config_values({}, {("hotwords", "auto_update"): False})
    assert _disk(cfg_file) == {"hotwords": {"auto_update": False}}


def test_shipped_config_has_every_form_setting():
    # Each control must hit an existing line, or saving it would fall back
    # to a full rewrite that drops the user's comments.
    d = json.loads(ms._strip_jsonc_comments(
        (Path(ms.__file__).parent / "config.jsonc").read_text(encoding="utf-8")))
    for path in ms._SETTINGS_PATHS:
        node = d
        for k in path:
            assert k in node, ".".join(path)
            node = node[k]


def test_calendar_ships_disabled():
    d = json.loads(ms._strip_jsonc_comments(
        (Path(ms.__file__).parent / "config.jsonc").read_text(encoding="utf-8")))
    assert d["calendar"]["enabled"] is False
