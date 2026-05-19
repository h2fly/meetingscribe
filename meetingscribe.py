#!/usr/bin/env python3
"""
meetingscribe.py — 录音 → 转写 → 校对 → 纪要/面试总结

━━━ 模式 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  meeting   会议纪要模式（默认）
  interview 面试总结模式

━━━ 录音 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # 默认模式录音（Ctrl+C 停止），自动完成转写 → 校对 → 纪要
  python3 meetingscribe.py record

  # 面试模式
  python3 meetingscribe.py record --mode interview
  python3 meetingscribe.py record --mode interview --title "后端工程师面试"

  # 指定各环节 provider（临时覆盖，优先级高于 config）
  python3 meetingscribe.py record --transcribe-provider openai
  python3 meetingscribe.py record --polish-provider gemini
  python3 meetingscribe.py record --meeting-notes-provider openai
  python3 meetingscribe.py record --transcribe-provider openai --polish-provider gemini --meeting-notes-provider claude
  python3 meetingscribe.py record --mode interview --transcribe-provider gemini --meeting-notes-provider openai

━━━ 转写已有文件 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # 传 .wav：若同目录已有 .raw.txt 则自动跳过转写，直接校对 + 纪要
  python3 meetingscribe.py transcribe audio.wav
  python3 meetingscribe.py transcribe audio.wav --mode interview
  python3 meetingscribe.py transcribe audio.wav --transcribe-provider openai
  python3 meetingscribe.py transcribe audio.wav --polish-provider gemini
  python3 meetingscribe.py transcribe audio.wav --meeting-notes-provider openai
  python3 meetingscribe.py transcribe audio.wav --transcribe-provider openai --polish-provider gemini --meeting-notes-provider claude
  python3 meetingscribe.py transcribe audio.wav --mode interview --meeting-notes-provider gemini

  # 传 .raw.txt：直接跳过转写，从校对步骤开始
  python3 meetingscribe.py transcribe audio.raw.txt
  python3 meetingscribe.py transcribe audio.raw.txt --mode interview
  python3 meetingscribe.py transcribe audio.raw.txt --polish-provider gemini --meeting-notes-provider openai

━━━ 桌面 UI ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python3 meetingscribe.py ui

━━━ 设备 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python3 meetingscribe.py devices

━━━ 配置 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # 查看当前配置
  python3 meetingscribe.py config

  # 修改配置（永久生效，可被命令行参数临时覆盖）
  python3 meetingscribe.py config --set mode=interview
  python3 meetingscribe.py config --set transcribe_provider=gemini
  python3 meetingscribe.py config --set polish_provider=openai
  python3 meetingscribe.py config --set meeting_notes_provider=claude
  python3 meetingscribe.py config --set llm_timeout=1200
  python3 meetingscribe.py config --set polish_chunk_size=3000
  python3 meetingscribe.py config --set polish_max_workers=5   # 校对并发数，0=自动(max(4, cpu//2))

━━━ 各环节 provider 可选值 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  --transcribe-provider    funasr（默认，本地）| whisper | openai | gemini
  --polish-provider        claude（默认）| openai | gemini
  --meeting-notes-provider claude（默认）| openai | gemini

━━━ 输出文件（与录音 / 输入文件同目录）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  <stem>.wav            录音文件
  <stem>.raw.txt        原始转写
  <stem>.polish.txt     校对后转写（会议纪要/面试总结的输入）
  <stem>.md             会议纪要 / 面试总结
"""

import argparse
import atexit
import copy
import json
import math
import os
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import subprocess

import numpy as np
import sounddevice as sd
# faster_whisper 按需懒加载（仅在 transcribe_provider=whisper 时导入）

# ── 日志 ──────────────────────────────────────────────────────────────────────
#
# Two channels of output:
#   • print() / print(..., file=sys.stderr)  → console + log file (via tee)
#                                              — for messages the USER needs to see
#   • _log(category, message)                → log file only (never console)
#                                              — for diagnostic detail
#
# Canonical categories (use these literals — keeps `grep '[CATEGORY]' ...` predictable):
#
#   REC       recording lifecycle (start / stop / abort / save)
#   DEVICE    resolver decisions, device enumeration
#   AUDIO     output switches, dOut / sOut transitions, PortAudio refresh
#   HOTPLUG   CoreAudio listener fires, monitor reconcile passes
#   RESTORE   _restore_output_if_needed decision branches
#   STREAM    MultiStreamRecorder per-stream open / close / error
#   STT       transcription progress, model loading, chunk completion
#   POLISH    polish stage progress
#   LLM       LLM call lifecycle, timeouts
#   PIPELINE  high-level transcribe→polish→notes flow
#   CONFIG    config load / merge / overrides
#   ERR       caught exceptions (replaces silent except: pass)
#   WARN      non-fatal but user-actionable

_DEBUG = False              # legacy global, kept for any code reading it; --debug also sets _LOG_TO_CONSOLE
_LOG_TO_CONSOLE = False     # if True, _log() additionally mirrors lines to stderr (set by --debug)
_log_file_handle = None     # opened by _setup_log_file(); cleared by its restore()
_log_file_lock = threading.Lock()


def _log(category: str, message: str):
    """Write a timestamped, categorised line to the daily log file ONLY.

    Never touches stdout/stderr in the default configuration — call print()
    for user-facing output. With --debug enabled, the same line is also
    mirrored to stderr so live triage still works.

    Safe to call from any thread; safe to call before _setup_log_file() runs
    (silent no-op until the handle exists). Logging itself never raises into
    the caller.
    """
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [{category}] {message}\n"
    try:
        with _log_file_lock:
            if _log_file_handle is not None:
                _log_file_handle.write(line)
                _log_file_handle.flush()
    except Exception:
        pass
    if _LOG_TO_CONSOLE:
        try:
            sys.stderr.write(line)
        except Exception:
            pass


def _dbg(msg: str):
    """Deprecated: prefer _log(category, message). Kept as a thin shim that
    forwards uncategorised diagnostic messages to the log file (and to stderr
    when --debug is enabled). Will be removed once all call sites migrate."""
    _log("DEBUG", msg)


class _QuietCapture:
    """Context manager: redirect stdout+stderr into in-memory buffers, then
    forward the captured lines to _log(category, ...) on exit. Used around
    third-party libraries (FunASR / faster-whisper) whose tqdm progress bars
    and per-frame timing dicts would otherwise drown the console. The captured
    text still lands in the daily log file, just not in front of the user.
    """
    def __init__(self, category: str):
        self._category = category
        self._buf_out: list[str] = []
        self._buf_err: list[str] = []
        self._saved_out = None
        self._saved_err = None

    def __enter__(self):
        import io
        self._saved_out, self._saved_err = sys.stdout, sys.stderr
        self._sio_out, self._sio_err = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = self._sio_out, self._sio_err
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout, sys.stderr = self._saved_out, self._saved_err
        for buf in (self._sio_out, self._sio_err):
            text = buf.getvalue()
            for raw in text.splitlines():
                line = raw.strip()
                if line:
                    _log(self._category, line)
        return False  # don't swallow exceptions


class _TimestampedStdout:
    """Wraps stdout to prepend [HH:MM:SS] to every printed line, optionally tee-ing to a log file."""
    def __init__(self, orig, log_file=None):
        self._orig = orig
        self._log_file = log_file
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, s: str):
        with self._lock:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                ts = datetime.now().strftime("%H:%M:%S")
                out = f"[{ts}] {line}\n" if line else "\n"
                self._orig.write(out)
                if self._log_file:
                    self._log_file.write(out)
                    self._log_file.flush()

    def flush(self):
        self._orig.flush()

    def fileno(self):
        return self._orig.fileno()


def _purge_old_logs(log_dir: Path, days: int = 7) -> None:
    if not log_dir.exists():
        return
    cutoff = datetime.now().timestamp() - days * 86400
    for f in log_dir.glob("*.log"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError as e:
            _log("ERR", f"purge_old_logs {f}: {type(e).__name__}: {e}")

# ── 配置 ──────────────────────────────────────────────────────────────────────

CONFIG_DIR = Path.home() / "Documents" / "meetingscribe"
LOG_DIR    = CONFIG_DIR / "logs"
CONFIG_FILE = Path(__file__).parent / "config.jsonc"


def _setup_log_file():
    """Open today's log file (one file per day, appended) and tee stdout+stderr to it.

    Files older than 7 days are purged on startup. A session header is appended
    so multiple runs within the same day are easy to distinguish.

    Returns (file_handle, restore_callable). Call restore_callable() in a finally
    block to undo the wrapping and close the file.
    """
    global _log_file_handle
    _purge_old_logs(LOG_DIR, days=7)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{datetime.now().strftime('%Y%m%d')}.log"
    fh = open(log_path, "a", encoding="utf-8")
    fh.write(f"\n========== session start {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==========\n")
    fh.flush()
    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout = _TimestampedStdout(saved_out, log_file=fh)
    sys.stderr = _TimestampedStdout(saved_err, log_file=fh)
    # Expose the same handle to _log() so file-only diagnostic lines land in the
    # same file as the stdout/stderr tee — chronologically interleaved.
    _log_file_handle = fh

    def _restore():
        global _log_file_handle
        sys.stdout, sys.stderr = saved_out, saved_err
        with _log_file_lock:
            _log_file_handle = None
        try:
            fh.close()
        except Exception:
            pass

    return fh, _restore

_funasr_model_cache: dict = {}  # (asr_model, vad_model, punc_model) -> AutoModel instance

DEFAULT_CONFIG = {
    # ── 并发控制 / 性能 ────────────────────────────────────────────────────
    # LLM 调用超时（秒），长会议建议调大
    "llm_timeout": 600,
    # 校对时单块最大字符数，超出则分块处理
    "polish_chunk_size": 12000,
    # 校对并发数（同时调用 LLM 的块数），0 = 自动(max(4, cpu核数/2))
    "polish_max_workers": 0,
    # ── 运行模式 ──────────────────────────────────────────────────────────
    # meeting（会议纪要）| interview（面试总结）
    "mode": "meeting",
    # ── 各环节 provider ───────────────────────────────────────────────────
    "transcribe_provider": "funasr",
    "polish_provider": "claude",
    "meeting_notes_provider": "claude",
    # ── 录音设备 ──────────────────────────────────────────────────────────
    "sample_rate": 48000,
    "channels": 3,
    "output_record": None,
    "output_restore": None,
    "device_system_audio": None,
    "device_mic": None,
    # ── 语音转文字 provider 配置（含并发参数）─────────────────────────────
    "stt": {
        "funasr": {
            "workers": 0,               # 并发实例数；0 = 自动（max(2, CPU核数/2)）
            "chunk_secs": 300,          # 超过此时长自动分块并发（秒），0 = 始终串行
            "model": "paraformer-zh",   # ASR 模型（首次运行自动下载）
            "vad_model": "fsmn-vad",    # VAD 分句模型，支持长音频
            "punc_model": "ct-punc",    # 标点恢复模型
            "hotword": "",              # 热词（空格分隔），提升专有名词识别率
        },
        "whisper": {
            "workers": 2,           # 并行实例数；内存占用 = workers × 模型大小
            "chunk_secs": 300,      # 超过此时长自动分块并行（秒），0 = 始终串行
            "cpu_threads": 0,       # 每个实例的内部线程数；0 = 自动（CPU 核数 / 2）
            "model": "base",        # tiny / base / small / medium / large-v3
        },
        "openai": {
            "api_key": "",
            "model": "whisper-1",
            "base_url": "https://api.openai.com/v1",
        },
        "gemini": {
            "api_key": "",
            "model": "gemini-2.0-flash",
        },
    },
    # 文本处理 provider 配置
    "llm": {
        "claude": {
            "type": "claude-cli",
            "model": "",        # e.g. "claude-haiku-4-5-20251001" for faster polish
        },
        "openai": {
            "type": "openai",
            "api_key": "",
            "model": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
        },
        "gemini": {
            "type": "gemini",
            "api_key": "",
            "model": "gemini-1.5-pro",
        },
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """override 覆盖 base，嵌套 dict 递归合并。"""
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _strip_jsonc_comments(text: str) -> str:
    """剥离 // 行注释，使 JSONC 文件可被标准 json 解析。"""
    import re
    # 匹配字符串内容（跳过）或 // 注释（替换为空）
    pattern = re.compile(r'"(?:[^"\\]|\\.)*"|//[^\n]*')
    return pattern.sub(lambda m: m.group(0) if m.group(0).startswith('"') else "", text)


def load_config() -> dict:
    CONFIG_DIR.mkdir(exist_ok=True)
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, encoding="utf-8") as f:
            on_disk = json.loads(_strip_jsonc_comments(f.read()))
        cfg = _deep_merge(DEFAULT_CONFIG, on_disk)
    else:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
    return cfg


def save_config(cfg: dict):
    """Save cfg to CONFIG_FILE.

    When the existing file is a parseable JSONC document, edit it in place
    so that comments, blank lines and key ordering are preserved — only the
    leaf values that actually differ between `cfg` and disk are rewritten.
    Falls back to a plain `json.dump` (comments dropped) when there is no
    existing file, the existing file is unparseable, or any diff can't be
    applied line-locally (e.g. a brand-new key was introduced).
    """
    CONFIG_DIR.mkdir(exist_ok=True)
    if _save_config_preserving_comments(cfg):
        return
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def _save_config_preserving_comments(cfg: dict) -> bool:
    """Patch `CONFIG_FILE` in place so that JSONC comments survive a
    ``config --set`` operation.

    Algorithm:
      1. Read existing file as text + parse the comment-stripped JSON to
         know each key's current value.
      2. Diff the parsed existing dict against `cfg` and collect a list of
         ``(path, new_value)`` leaf changes. ``path`` is a tuple of keys
         from root, e.g. ``("stt", "funasr", "workers")``.
      3. Walk the file line by line. Track the scope (a stack of dict
         keys) by recognising ``"key": {`` as a scope-open and a leading
         ``}`` on a stripped line as a scope-close.
      4. When a line ``"key": <value>...`` matches a pending diff, splice
         in the new value at the front of `<value>` and keep the trailing
         comma / whitespace / inline ``// comment`` exactly as written.

    Returns True on a successful in-place write. Returns False (with a
    diagnostic ``[CONFIG]`` log line) when the in-place strategy can't
    handle the change — the caller then falls back to ``json.dump``.
    """
    if not CONFIG_FILE.exists():
        return False
    try:
        text = CONFIG_FILE.read_text(encoding="utf-8")
        existing = json.loads(_strip_jsonc_comments(text))
    except (OSError, json.JSONDecodeError, ValueError) as e:
        _log("CONFIG", f"in-place save: parse failed: {type(e).__name__}: {e}")
        return False

    diffs: list[tuple[tuple[str, ...], object]] = []

    def _collect(old, new, prefix):
        if isinstance(new, dict) and isinstance(old, dict):
            for k in new:
                if k not in old:
                    diffs.append((prefix + (k,), new[k]))
                else:
                    _collect(old[k], new[k], prefix + (k,))
            return
        if new != old:
            diffs.append((prefix, new))

    _collect(existing, cfg, ())
    if not diffs:
        return True

    import re
    KEY_RE = re.compile(
        r'^(?P<indent>\s*)"(?P<key>(?:\\.|[^"\\])*)"\s*:\s*(?P<rest>.*)$'
    )

    lines = text.splitlines(keepends=True)
    pending: dict[tuple[str, ...], object] = {p: v for p, v in diffs}
    scope: list[str] = []

    for i, raw in enumerate(lines):
        # Strip the line terminator so KEY_RE works regardless of CRLF / LF;
        # rebuild it from the original `raw` after the substitution.
        line = raw.rstrip("\n").rstrip("\r")
        eol = raw[len(line):]
        m = KEY_RE.match(line)
        if m:
            key = m.group("key")
            rest = m.group("rest").rstrip()
            current_path = tuple(scope) + (key,)

            if rest == "{":
                scope.append(key)
                continue
            if rest.startswith("{") and rest != "{":
                _log("CONFIG", f"in-place save: inline object at {current_path}; "
                               f"falling back to json.dump")
                return False

            if current_path in pending:
                new_val = pending.pop(current_path)
                # Find old value via the parsed existing dict.
                cursor: object = existing
                try:
                    for k in current_path:
                        cursor = cursor[k]  # type: ignore[index]
                except (KeyError, TypeError):
                    return False
                old_json = json.dumps(cursor, ensure_ascii=False)
                new_json = json.dumps(new_val, ensure_ascii=False)
                if not rest.startswith(old_json):
                    _log("CONFIG", f"in-place save: value mismatch at {current_path}; "
                                   f"expected start={old_json!r} got={rest[:40]!r}; "
                                   f"falling back to json.dump")
                    return False
                trailer = rest[len(old_json):]  # ',' + ' // comment' or ''
                lines[i] = f"{m.group('indent')}\"{key}\": {new_json}{trailer}{eol}"
            continue

        # Non-key line: detect scope close on a stripped, comment-less view.
        commentless = re.sub(r'//.*$', '', line)
        if commentless.strip().startswith("}") and scope:
            scope.pop()

    if pending:
        _log("CONFIG", f"in-place save: {len(pending)} diff(s) unapplied "
                       f"(keys: {list(pending)}); falling back to json.dump")
        return False

    CONFIG_FILE.write_text("".join(lines), encoding="utf-8")
    return True


# ── 音频输出切换 ──────────────────────────────────────────────────────────────

def _set_default_output_for(selector: str, device_name: str):
    """Internal: write `device_name` to the macOS HAL property identified by
    `selector` ('dOut' = media default, 'sOut' = system default / volume-key target).
    No-op on non-macOS. Prints a warning if no device with that name is found.
    """
    if sys.platform != "darwin":
        return
    import ctypes, ctypes.util, struct

    ca = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
    cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))

    class _Addr(ctypes.Structure):
        _fields_ = [("sel", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("elem", ctypes.c_uint32)]

    def _fcc(s):
        return struct.unpack(">I", s.encode())[0]

    kSystem = 1
    kGlobal = _fcc("glob")

    addr = _Addr(_fcc("dev#"), kGlobal, 0)
    sz = ctypes.c_uint32(0)
    ca.AudioObjectGetPropertyDataSize(kSystem, ctypes.byref(addr), 0, None, ctypes.byref(sz))
    ids = (ctypes.c_uint32 * (sz.value // 4))()
    ca.AudioObjectGetPropertyData(kSystem, ctypes.byref(addr), 0, None, ctypes.byref(sz), ids)

    kUTF8 = 0x08000100
    for dev_id in ids:
        cf_str = ctypes.c_void_p(0)
        sz2 = ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
        ca.AudioObjectGetPropertyData(
            dev_id, ctypes.byref(_Addr(_fcc("lnam"), kGlobal, 0)),
            0, None, ctypes.byref(sz2), ctypes.byref(cf_str),
        )
        if not cf_str.value:
            continue
        buf = ctypes.create_string_buffer(512)
        if cf.CFStringGetCString(cf_str, buf, 512, kUTF8):
            if buf.value.decode("utf-8") == device_name:
                val = ctypes.c_uint32(dev_id)
                ca.AudioObjectSetPropertyData(
                    kSystem, ctypes.byref(_Addr(_fcc(selector), kGlobal, 0)),
                    0, None, ctypes.c_uint32(4), ctypes.byref(val),
                )
                return
    _log("WARN", f"set_default_output_for({selector}): device {device_name!r} not found")


def switch_output(device_name: str):
    """Switch the macOS *media* default output (kAudioHardwarePropertyDefaultOutputDevice).
    This is where most applications route audio. macOS only; no-op on other platforms.
    """
    _set_default_output_for("dOut", device_name)


def switch_system_output(device_name: str):
    """Switch the macOS *system* default output (kAudioHardwarePropertyDefaultSystemOutputDevice).
    This is the device the hardware volume keys (F11/F12, Touch Bar, menu-bar slider) target.
    Distinct from switch_output() so recording-start can leave the volume-key device alone
    while routing media through the Multi-Output Device. macOS only.
    """
    _set_default_output_for("sOut", device_name)


def _get_output_device_for_selector(selector: str) -> str | None:
    """Return the name of the macOS device pointed to by a CoreAudio HAL selector.

    selector="dOut" → kAudioHardwarePropertyDefaultOutputDevice
        — the device media (Apple Music, Safari, our own playback) currently
          targets. When we switch_output() this is what we change.

    selector="sOut" → kAudioHardwarePropertyDefaultSystemOutputDevice
        — the physical device macOS uses for system sounds. macOS keeps this
          synchronized with the actual hardware path (e.g., switches to
          'External Headphones' when headphones are plugged in) even if the
          user / our app has overridden the regular default to some aggregate.
          This is what we want for "restore at exit": the user's real device.
    """
    if sys.platform != "darwin":
        return None
    import ctypes, ctypes.util, struct

    ca = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
    cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))

    class _Addr(ctypes.Structure):
        _fields_ = [("sel", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("elem", ctypes.c_uint32)]

    def _fcc(s):
        return struct.unpack(">I", s.encode())[0]

    kSystem, kGlobal, kUTF8 = 1, _fcc("glob"), 0x08000100

    dev_id = ctypes.c_uint32(0)
    sz = ctypes.c_uint32(4)
    ca.AudioObjectGetPropertyData(
        kSystem, ctypes.byref(_Addr(_fcc(selector), kGlobal, 0)),
        0, None, ctypes.byref(sz), ctypes.byref(dev_id),
    )
    if not dev_id.value:
        return None

    cf_str = ctypes.c_void_p(0)
    sz2 = ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
    ca.AudioObjectGetPropertyData(
        dev_id.value, ctypes.byref(_Addr(_fcc("lnam"), kGlobal, 0)),
        0, None, ctypes.byref(sz2), ctypes.byref(cf_str),
    )
    if not cf_str.value:
        return None
    buf = ctypes.create_string_buffer(512)
    return buf.value.decode("utf-8") if cf.CFStringGetCString(cf_str, buf, 512, kUTF8) else None


def _get_current_output_device() -> str | None:
    """Name of the current media output target (kAudioHardwarePropertyDefaultOutputDevice).

    This is what switch_output() writes, and can be the aggregate when we're
    capturing. For the user's actual physical device use _get_system_output_device().
    """
    return _get_output_device_for_selector("dOut")


def _get_system_output_device() -> str | None:
    """Name of the user's current physical output device.

    macOS keeps kAudioHardwarePropertyDefaultSystemOutputDevice tracking the
    real hardware path independently of any override on the regular default —
    so plugging in headphones updates this even while our aggregate is the
    media default.
    """
    return _get_output_device_for_selector("sOut")


def get_device_volume(device_name: str) -> float | None:
    """Get master output volume (0.0–1.0) for a named device. macOS only."""
    if sys.platform != "darwin":
        return None
    import ctypes, ctypes.util, struct

    ca = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
    cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))

    class _Addr(ctypes.Structure):
        _fields_ = [("sel", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("elem", ctypes.c_uint32)]

    def _fcc(s):
        return struct.unpack(">I", s.encode())[0]

    kSystem, kGlobal, kUTF8 = 1, _fcc("glob"), 0x08000100
    kScopeOutput = _fcc("outp")

    sz = ctypes.c_uint32(0)
    ca.AudioObjectGetPropertyDataSize(kSystem, ctypes.byref(_Addr(_fcc("dev#"), kGlobal, 0)), 0, None, ctypes.byref(sz))
    ids = (ctypes.c_uint32 * (sz.value // 4))()
    ca.AudioObjectGetPropertyData(kSystem, ctypes.byref(_Addr(_fcc("dev#"), kGlobal, 0)), 0, None, ctypes.byref(sz), ids)

    for dev_id in ids:
        cf_str = ctypes.c_void_p(0)
        sz2 = ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
        ca.AudioObjectGetPropertyData(dev_id, ctypes.byref(_Addr(_fcc("lnam"), kGlobal, 0)), 0, None, ctypes.byref(sz2), ctypes.byref(cf_str))
        if not cf_str.value:
            continue
        buf = ctypes.create_string_buffer(512)
        if not cf.CFStringGetCString(cf_str, buf, 512, kUTF8):
            continue
        if buf.value.decode("utf-8") != device_name:
            continue
        # 'vmvc' = virtual master volume (Mac built-in audio); 'volu' = per-channel (USB/BT)
        for (prop, elems) in ((_fcc("vmvc"), (0,)), (_fcc("volu"), (0, 1))):
            for elem in elems:
                vol = ctypes.c_float(0.0)
                sz3 = ctypes.c_uint32(4)
                ret = ca.AudioObjectGetPropertyData(
                    dev_id, ctypes.byref(_Addr(prop, kScopeOutput, elem)),
                    0, None, ctypes.byref(sz3), ctypes.byref(vol),
                )
                if ret == 0:
                    return vol.value
    return None


def set_device_volume(device_name: str, volume: float):
    """Set output volume (0.0–1.0) on a named device. macOS only; no-op elsewhere."""
    if sys.platform != "darwin":
        return
    import ctypes, ctypes.util, struct

    ca = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
    cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))

    class _Addr(ctypes.Structure):
        _fields_ = [("sel", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("elem", ctypes.c_uint32)]

    def _fcc(s):
        return struct.unpack(">I", s.encode())[0]

    kSystem, kGlobal, kUTF8 = 1, _fcc("glob"), 0x08000100
    kScopeOutput = _fcc("outp")
    volume = max(0.0, min(1.0, volume))

    sz = ctypes.c_uint32(0)
    ca.AudioObjectGetPropertyDataSize(kSystem, ctypes.byref(_Addr(_fcc("dev#"), kGlobal, 0)), 0, None, ctypes.byref(sz))
    ids = (ctypes.c_uint32 * (sz.value // 4))()
    ca.AudioObjectGetPropertyData(kSystem, ctypes.byref(_Addr(_fcc("dev#"), kGlobal, 0)), 0, None, ctypes.byref(sz), ids)

    for dev_id in ids:
        cf_str = ctypes.c_void_p(0)
        sz2 = ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
        ca.AudioObjectGetPropertyData(dev_id, ctypes.byref(_Addr(_fcc("lnam"), kGlobal, 0)), 0, None, ctypes.byref(sz2), ctypes.byref(cf_str))
        if not cf_str.value:
            continue
        buf = ctypes.create_string_buffer(512)
        if not cf.CFStringGetCString(cf_str, buf, 512, kUTF8):
            continue
        if buf.value.decode("utf-8") != device_name:
            continue
        vol = ctypes.c_float(volume)
        # Try 'vmvc' first (virtual master — Mac built-in / headphone jack)
        ret = ca.AudioObjectSetPropertyData(
            dev_id, ctypes.byref(_Addr(_fcc("vmvc"), kScopeOutput, 0)),
            0, None, ctypes.c_uint32(4), ctypes.byref(vol),
        )
        if ret != 0:
            # Fallback: per-channel 'volu' (USB audio, some external DACs)
            for elem in (0, 1, 2):
                ca.AudioObjectSetPropertyData(
                    dev_id, ctypes.byref(_Addr(_fcc("volu"), kScopeOutput, elem)),
                    0, None, ctypes.c_uint32(4), ctypes.byref(vol),
                )
        return


def _ca_get_device_mute(device_name: str) -> bool | None:
    """Read the per-device mute flag (kAudioDevicePropertyMute, output scope,
    master element). Returns True/False if the property is implemented for the
    device, None if the device is missing or doesn't support the property.
    macOS only; returns None on other platforms.
    """
    if sys.platform != "darwin":
        return None
    import ctypes, ctypes.util, struct

    ca = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
    cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))

    class _Addr(ctypes.Structure):
        _fields_ = [("sel", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("elem", ctypes.c_uint32)]

    def _fcc(s):
        return struct.unpack(">I", s.encode())[0]

    kSystem, kGlobal, kUTF8 = 1, _fcc("glob"), 0x08000100
    kScopeOutput = _fcc("outp")

    sz = ctypes.c_uint32(0)
    ca.AudioObjectGetPropertyDataSize(kSystem, ctypes.byref(_Addr(_fcc("dev#"), kGlobal, 0)), 0, None, ctypes.byref(sz))
    ids = (ctypes.c_uint32 * (sz.value // 4))()
    ca.AudioObjectGetPropertyData(kSystem, ctypes.byref(_Addr(_fcc("dev#"), kGlobal, 0)), 0, None, ctypes.byref(sz), ids)

    for dev_id in ids:
        cf_str = ctypes.c_void_p(0)
        sz2 = ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
        ca.AudioObjectGetPropertyData(dev_id, ctypes.byref(_Addr(_fcc("lnam"), kGlobal, 0)), 0, None, ctypes.byref(sz2), ctypes.byref(cf_str))
        if not cf_str.value:
            continue
        buf = ctypes.create_string_buffer(512)
        if not cf.CFStringGetCString(cf_str, buf, 512, kUTF8):
            continue
        if buf.value.decode("utf-8") != device_name:
            continue
        muted = ctypes.c_uint32(0)
        sz3 = ctypes.c_uint32(4)
        ret = ca.AudioObjectGetPropertyData(
            dev_id, ctypes.byref(_Addr(_fcc("mute"), kScopeOutput, 0)),
            0, None, ctypes.byref(sz3), ctypes.byref(muted),
        )
        return bool(muted.value) if ret == 0 else None
    return None


def _ca_set_device_mute(device_name: str, muted: bool) -> bool:
    """Set the per-device mute flag (kAudioDevicePropertyMute, output scope,
    master element). Returns True on success, False on any failure (device not
    found, property not settable, OSStatus non-zero). macOS only; returns
    False on other platforms.
    """
    if sys.platform != "darwin":
        return False
    import ctypes, ctypes.util, struct

    ca = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
    cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))

    class _Addr(ctypes.Structure):
        _fields_ = [("sel", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("elem", ctypes.c_uint32)]

    def _fcc(s):
        return struct.unpack(">I", s.encode())[0]

    kSystem, kGlobal, kUTF8 = 1, _fcc("glob"), 0x08000100
    kScopeOutput = _fcc("outp")

    sz = ctypes.c_uint32(0)
    ca.AudioObjectGetPropertyDataSize(kSystem, ctypes.byref(_Addr(_fcc("dev#"), kGlobal, 0)), 0, None, ctypes.byref(sz))
    ids = (ctypes.c_uint32 * (sz.value // 4))()
    ca.AudioObjectGetPropertyData(kSystem, ctypes.byref(_Addr(_fcc("dev#"), kGlobal, 0)), 0, None, ctypes.byref(sz), ids)

    for dev_id in ids:
        cf_str = ctypes.c_void_p(0)
        sz2 = ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
        ca.AudioObjectGetPropertyData(dev_id, ctypes.byref(_Addr(_fcc("lnam"), kGlobal, 0)), 0, None, ctypes.byref(sz2), ctypes.byref(cf_str))
        if not cf_str.value:
            continue
        buf = ctypes.create_string_buffer(512)
        if not cf.CFStringGetCString(cf_str, buf, 512, kUTF8):
            continue
        if buf.value.decode("utf-8") != device_name:
            continue
        val = ctypes.c_uint32(1 if muted else 0)
        ret = ca.AudioObjectSetPropertyData(
            dev_id, ctypes.byref(_Addr(_fcc("mute"), kScopeOutput, 0)),
            0, None, ctypes.c_uint32(4), ctypes.byref(val),
        )
        return ret == 0
    return False


def _coreaudio_device_info() -> dict[str, str]:
    """Return {device_name: transport_type_fourcc} for all CoreAudio devices. macOS only.

    Common transport types:
      'bltn' built-in  |  'virt' virtual (BlackHole)  |  'aggt' aggregate (Multi-Output)
      'usb ' USB       |  'blue' Bluetooth             |  'blea' Bluetooth LE
    """
    if sys.platform != "darwin":
        return {}
    import ctypes, ctypes.util, struct

    ca = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
    cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))

    class _Addr(ctypes.Structure):
        _fields_ = [("sel", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("elem", ctypes.c_uint32)]

    def _fcc(s):
        return struct.unpack(">I", s.encode())[0]

    kSystem, kGlobal, kUTF8 = 1, _fcc("glob"), 0x08000100

    sz = ctypes.c_uint32(0)
    ca.AudioObjectGetPropertyDataSize(
        kSystem, ctypes.byref(_Addr(_fcc("dev#"), kGlobal, 0)), 0, None, ctypes.byref(sz)
    )
    ids = (ctypes.c_uint32 * (sz.value // 4))()
    ca.AudioObjectGetPropertyData(
        kSystem, ctypes.byref(_Addr(_fcc("dev#"), kGlobal, 0)), 0, None, ctypes.byref(sz), ids
    )

    result = {}
    for dev_id in ids:
        cf_str = ctypes.c_void_p(0)
        sz2 = ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
        ca.AudioObjectGetPropertyData(
            dev_id, ctypes.byref(_Addr(_fcc("lnam"), kGlobal, 0)),
            0, None, ctypes.byref(sz2), ctypes.byref(cf_str),
        )
        if not cf_str.value:
            continue
        buf = ctypes.create_string_buffer(512)
        if not cf.CFStringGetCString(cf_str, buf, 512, kUTF8):
            continue
        name = buf.value.decode("utf-8")

        transport = ctypes.c_uint32(0)
        sz3 = ctypes.c_uint32(4)
        ca.AudioObjectGetPropertyData(
            dev_id, ctypes.byref(_Addr(_fcc("tran"), kGlobal, 0)),
            0, None, ctypes.byref(sz3), ctypes.byref(transport),
        )
        result[name] = struct.pack(">I", transport.value).decode("latin-1")

    return result


def _coreaudio_device_raw_dump() -> list[dict]:
    """Enumerate every CoreAudio device with full diagnostic detail. macOS only.

    Each entry contains:
      id               AudioObjectID
      name             device name
      uid              kAudioDevicePropertyDeviceUID string ('' if not readable)
      class_id         4-char fourcc ('adev' regular, 'aagg' aggregate/multi-output, 'asub' sub-device) or ''
      class_status     OSStatus from the class fetch (0 = OK)
      transport        4-char fourcc ('bltn'/'usb '/'blue'/'blea'/'hdmi'/'virt'/'grup' ...) or ''
      transport_status OSStatus from the transport fetch
      sub_uids         list[str] of sub-device UIDs (only for class_id == 'aagg')

    Used by `devices --raw` and the recording/restore log injection points to
    diagnose mis-classified aggregates that bypass the name/transport
    heuristics in _is_physical_output / _transport_priority.
    """
    if sys.platform != "darwin":
        return []
    import ctypes, ctypes.util, struct

    ca = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
    cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
    cf.CFArrayGetCount.restype = ctypes.c_long
    cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
    cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
    cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    cf.CFRelease.restype = None

    class _Addr(ctypes.Structure):
        _fields_ = [("sel", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("elem", ctypes.c_uint32)]

    def _fcc(s):
        return struct.unpack(">I", s.encode())[0]

    def _fcc_str(value: int) -> str:
        return struct.pack(">I", value).decode("latin-1") if value else ""

    kSystem, kGlobal, kUTF8 = 1, _fcc("glob"), 0x08000100

    sz = ctypes.c_uint32(0)
    ca.AudioObjectGetPropertyDataSize(
        kSystem, ctypes.byref(_Addr(_fcc("dev#"), kGlobal, 0)), 0, None, ctypes.byref(sz)
    )
    ids = (ctypes.c_uint32 * (sz.value // 4))()
    ca.AudioObjectGetPropertyData(
        kSystem, ctypes.byref(_Addr(_fcc("dev#"), kGlobal, 0)), 0, None, ctypes.byref(sz), ids
    )

    result: list[dict] = []
    for dev_id in ids:
        cf_str = ctypes.c_void_p(0)
        sz2 = ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
        name_status = ca.AudioObjectGetPropertyData(
            dev_id, ctypes.byref(_Addr(_fcc("lnam"), kGlobal, 0)),
            0, None, ctypes.byref(sz2), ctypes.byref(cf_str),
        )
        if name_status != 0 or not cf_str.value:
            continue
        buf = ctypes.create_string_buffer(512)
        if not cf.CFStringGetCString(cf_str, buf, 512, kUTF8):
            continue
        name = buf.value.decode("utf-8")

        cf_uid = ctypes.c_void_p(0)
        sz_uid = ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
        uid_status = ca.AudioObjectGetPropertyData(
            dev_id, ctypes.byref(_Addr(_fcc("uid "), kGlobal, 0)),
            0, None, ctypes.byref(sz_uid), ctypes.byref(cf_uid),
        )
        uid = ""
        if uid_status == 0 and cf_uid.value:
            buf_uid = ctypes.create_string_buffer(512)
            if cf.CFStringGetCString(cf_uid, buf_uid, 512, kUTF8):
                uid = buf_uid.value.decode("utf-8")

        cls = ctypes.c_uint32(0)
        sz3 = ctypes.c_uint32(4)
        class_status = ca.AudioObjectGetPropertyData(
            dev_id, ctypes.byref(_Addr(_fcc("clas"), kGlobal, 0)),
            0, None, ctypes.byref(sz3), ctypes.byref(cls),
        )

        tran = ctypes.c_uint32(0)
        sz4 = ctypes.c_uint32(4)
        tran_status = ca.AudioObjectGetPropertyData(
            dev_id, ctypes.byref(_Addr(_fcc("tran"), kGlobal, 0)),
            0, None, ctypes.byref(sz4), ctypes.byref(tran),
        )

        class_fourcc = _fcc_str(cls.value) if class_status == 0 else ""
        tran_fourcc = _fcc_str(tran.value) if tran_status == 0 else ""

        sub_uids: list[str] = []
        if class_fourcc == "aagg":
            # Use kAudioAggregateDevicePropertyComposition ('acom') — a CFDictionary
            # containing a 'subdevices' CFArray. The narrower 'agdv' /
            # kAudioAggregateDevicePropertyFullSubDeviceList selector returns
            # kAudioHardwareUnknownPropertyError ('who?') on at least some macOS
            # versions, so we read the composition dict directly.
            cf.CFStringCreateWithCString.restype = ctypes.c_void_p
            cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
            cf.CFDictionaryGetValue.restype = ctypes.c_void_p
            cf.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            cf.CFStringGetCString.restype = ctypes.c_bool
            cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
            kASCII = 0x600

            dict_ref = ctypes.c_void_p(0)
            dict_sz = ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
            comp_status = ca.AudioObjectGetPropertyData(
                dev_id, ctypes.byref(_Addr(_fcc("acom"), kGlobal, 0)),
                0, None, ctypes.byref(dict_sz), ctypes.byref(dict_ref),
            )
            if comp_status == 0 and dict_ref.value:
                key_sub = cf.CFStringCreateWithCString(None, b"subdevices", kASCII)
                key_uid = cf.CFStringCreateWithCString(None, b"uid", kASCII)
                try:
                    arr_ref = cf.CFDictionaryGetValue(dict_ref, key_sub) if key_sub else None
                    if arr_ref:
                        count = cf.CFArrayGetCount(arr_ref)
                        for i in range(count):
                            sub_dict = cf.CFArrayGetValueAtIndex(arr_ref, i)
                            if not sub_dict or not key_uid:
                                continue
                            uid_cfstr = cf.CFDictionaryGetValue(sub_dict, key_uid)
                            if not uid_cfstr:
                                continue
                            buf2 = ctypes.create_string_buffer(512)
                            if cf.CFStringGetCString(uid_cfstr, buf2, 512, kUTF8):
                                sub_uids.append(buf2.value.decode("utf-8"))
                finally:
                    if key_sub:
                        cf.CFRelease(key_sub)
                    if key_uid:
                        cf.CFRelease(key_uid)
                    cf.CFRelease(dict_ref)

        result.append({
            "id": int(dev_id),
            "name": name,
            "uid": uid,
            "class_id": class_fourcc,
            "class_status": int(class_status),
            "transport": tran_fourcc,
            "transport_status": int(tran_status),
            "sub_uids": sub_uids,
        })

    return result


def _format_device_raw_entry(entry: dict) -> str:
    """Render one _coreaudio_device_raw_dump entry as a single compact line.

    Format:
      id=NN name='...' class='aagg'[!s=N] tran='bltn'[!s=N] subs=['uid', ...]
    The '!s=N' suffix is only emitted when the OSStatus from that property
    fetch was non-zero — making mis-classified aggregates visually obvious.
    """
    cls = entry["class_id"] or "''"
    tran = entry["transport"] or "''"
    cls_warn = f" !s={entry['class_status']}" if entry["class_status"] else ""
    tran_warn = f" !s={entry['transport_status']}" if entry["transport_status"] else ""
    base = (
        f"id={entry['id']} name={entry['name']!r} "
        f"class={cls!r}{cls_warn} tran={tran!r}{tran_warn}"
    )
    if entry["sub_uids"]:
        base += f" subs={entry['sub_uids']}"
    return base


def _log_device_raw_dump(reason: str) -> None:
    """Write the full CoreAudio raw device dump to the log file under [DEVICE-RAW].

    Called at every dOut-decision boundary (recording start, restore entry) so
    a post-mortem can reconstruct the exact device topology and class/transport
    tags at the moment of the decision. macOS only; no-op elsewhere.
    """
    if sys.platform != "darwin":
        return
    try:
        dump = _coreaudio_device_raw_dump()
    except Exception as e:
        _log("ERR", f"device raw dump failed reason={reason}: {type(e).__name__}: {e}")
        return
    _log("DEVICE-RAW", f"begin reason={reason} count={len(dump)}")
    for entry in dump:
        _log("DEVICE-RAW", _format_device_raw_entry(entry))
    _log("DEVICE-RAW", f"end reason={reason}")


def _get_multi_output_physical_subs(multi_output_name: str | None) -> list[str]:
    """Return the names of the physical (non-virtual, non-aggregate) sub-devices
    of the named Multi-Output Device. Empty list if the device is missing, has
    no sub-devices, or has only virtual sub-devices. macOS only.

    Used by the per-device mute lifecycle to decide which Multi-Output member
    to silence when the user is listening through a different output.
    """
    if not multi_output_name or sys.platform != "darwin":
        return []
    try:
        dump = _coreaudio_device_raw_dump()
    except Exception as e:
        _log("ERR", f"multi-output sub enumeration failed: {type(e).__name__}: {e}")
        return []
    multi = next((e for e in dump if e.get("name") == multi_output_name), None)
    if not multi or not multi.get("sub_uids"):
        return []
    uid_to_entry = {e["uid"]: e for e in dump if e.get("uid")}
    rejected = {"virt", "aggt", "grup"}
    result: list[str] = []
    for sub_uid in multi["sub_uids"]:
        entry = uid_to_entry.get(sub_uid)
        if not entry:
            continue
        if entry.get("class_id") == "aagg":
            continue
        if entry.get("transport") in rejected:
            continue
        result.append(entry["name"])
    return result


# ── 音频设备解析（统一入口）& 热插拔 ─────────────────────────────────────────

# Names that identify the user's pre-configured Multi-Output Device across macOS locales.
_MULTI_OUT_NAMES = ("Multi-Output Device", "多输出设备", "多重輸出裝置")

# Transport-type tags returned by _coreaudio_device_info():
#   external → preferred for mic / output  ('usb ', 'blue', 'blea', 'hdmi', 'thnd', ...)
#   built-in → second choice                ('bltn')
#   virtual  → BlackHole, only used as recording sink
#   aggregate → Multi-Output Devices, only used as recording sink
_TRANSPORT_BUILTIN = "bltn"
_TRANSPORT_VIRTUAL_OR_AGGREGATE = frozenset({"virt", "aggt", "grup"})


@dataclass
class AudioPlan:
    """Single answer for which devices to use, computed by resolve_audio_devices().

    Consumed at three lifecycle points:
      • recording start  → mic_name + sys_source_name drive the recorder's `wanted`
      • hotplug during   → re-resolved; recorder reconciles open streams
      • recording stop   → restore_output_name drives _restore_output_if_needed()
    """
    mic_index: int | None = None
    mic_name: str | None = None
    sys_source_index: int | None = None
    sys_source_name: str | None = None
    multi_output_name: str | None = None
    restore_output_name: str | None = None
    is_external_output: bool = False
    warnings: list[str] = field(default_factory=list)


# Serializes PortAudio (de)init with stream-open calls so neither operation
# happens while the other is mutating PortAudio's internal device cache.
_portaudio_lock = threading.RLock()

# Set by the CoreAudio property listener (macOS) when the device list or the
# system-default-output changes. The MultiStreamRecorder's monitor thread waits
# on this event in place of time.sleep(), so hotplug response is bounded by
# whichever fires first: the listener (~100 ms) or the 1 s fallback timeout.
_hotplug_event = threading.Event()

# Set by recorder start paths and cleared on stop / abort. The
# AudioDeviceMonitor inspects this to choose between the idle branch (run
# restore on first tick and on every device-set change) and the recording
# branch (signal _hotplug_event on change; never mutate streams).
_recording_active = threading.Event()

# Set ONLY when the recording-start path actually called
# switch_output(multi_output_name). _restore_output_if_needed uses this as a
# gate for post-recording restores: if no start-time switch happened (because
# the user has the Multi-Output Device as their permanent macOS default),
# there is nothing to restore and the stop path skips the switch_output call —
# avoiding a second music-app pause. Cleared at the top of every new
# MultiStreamRecorder.start() so each session starts from a clean slate. The
# idle-state restore path (AudioDeviceMonitor) bypasses this gate by passing
# reason="idle-event".
_recording_did_switch = threading.Event()

# AudioDeviceMonitor safety-net timeout. The monitor is event-driven (blocks on
# _hotplug_event) — this timeout is the maximum interval between forced
# self-heal ticks if the HAL listener ever silently stops firing. Long enough
# to keep the log quiet in normal operation; short enough that a lost event
# resyncs within half a minute.
_AUDIO_MONITOR_SAFETY_TIMEOUT_SEC = 30.0

# Process-wide singleton for the audio-device monitor thread. Constructed
# lazily by _get_audio_monitor(); used by cmd_ui and cmd_record.
_audio_monitor: "AudioDeviceMonitor | None" = None

# Strong references to ctypes structures and CFUNCTYPE callbacks for the
# CoreAudio HAL property listeners. Kept here so the callback isn't GC'd
# while CoreAudio still holds a pointer to it.
_listener_state: dict = {"installed": False}


# ── Per-device mute lifecycle for the recording session ────────────────────────
#
# Maps device-name → original (pre-recording) mute state. Populated when we
# first mute a device; the entry is removed when we restore the original.
# Persisted to .active_mutes.json after every change so a process crash mid-
# recording can be cleaned up by the next launch's _recover_persisted_mutes().
_active_mutes: dict[str, bool] = {}
_mutes_lock = threading.RLock()
_MUTE_STATE_FILE = CONFIG_DIR / ".active_mutes.json"


def _persist_mutes() -> None:
    """Atomically write _active_mutes to disk (or delete the file when empty).
    Caller must hold _mutes_lock.
    """
    if sys.platform != "darwin":
        return
    try:
        if not _active_mutes:
            if _MUTE_STATE_FILE.exists():
                _MUTE_STATE_FILE.unlink(missing_ok=True)
                _log("MUTE", "persist: deleted (no active mutes)")
            return
        payload = {
            "schema_version": 1,
            "pid": os.getpid(),
            "muted": dict(_active_mutes),
        }
        tmp = _MUTE_STATE_FILE.with_suffix(_MUTE_STATE_FILE.suffix + ".tmp")
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False))
        os.replace(tmp, _MUTE_STATE_FILE)
        _log("MUTE", f"persist: muted={dict(_active_mutes)}")
    except Exception as e:
        _log("MUTE", f"persist failed: {type(e).__name__}: {e}")


def _reconcile_recording_mutes(plan: "AudioPlan") -> None:
    """Ensure each physical sub-device of the Multi-Output is in the desired
    mute state: muted if it is NOT the active listening target, unmuted if it
    IS. Idempotent and re-callable on hotplug. macOS only; no-op elsewhere.

    "Active listening target" is plan.restore_output_name (the resolver's
    external-over-built-in pick). Virtual/aggregate sub-devices are skipped.

    Mute-state changes are tracked ASYMMETRICALLY:

      - When we MUTE a previously-unmuted device (False→True), we record its
        original False state in `_active_mutes`, so restore_all puts it back
        to False at stop — the user's speakers come back online.

      - When we UNMUTE a previously-muted device (True→False), we DO NOT
        track it. A pre-existing device-level mute on the user's active
        listening target is almost always a stale state (e.g. macOS leaves a
        headphone device muted across plug cycles); restoring it would just
        re-silence the device the user is currently using. So we clear the
        mute and leave it cleared.

    This makes the rule: "we put devices INTO silence so the recording is
    clean; we never put devices BACK into silence at stop". If the user truly
    wanted a device muted, they can re-mute it manually after stop.
    """
    if sys.platform != "darwin":
        return
    with _mutes_lock:
        if not plan.multi_output_name:
            return
        physical_subs = _get_multi_output_physical_subs(plan.multi_output_name)
        if not physical_subs:
            return
        active = plan.restore_output_name
        if not active:
            _log("MUTE", "reconcile: no restore target; refusing to change mute state")
            return
        for sub in physical_subs:
            desired = (sub != active)
            current = _ca_get_device_mute(sub)
            if current is None:
                _log("MUTE", f"device={sub!r} mute-property unsupported; skipping")
                continue
            if current == desired:
                continue
            ok = _ca_set_device_mute(sub, desired)
            if desired:
                # False → True: we silenced the device. Track for restore.
                if sub not in _active_mutes:
                    _active_mutes[sub] = current  # always False at this branch
                _log("MUTE", f"device={sub!r} muted (original={_active_mutes[sub]}) ok={ok}")
            else:
                # True → False: we cleared a mute. Do NOT keep tracking —
                # restore_all SHALL NOT re-apply this silence at stop.
                # Two sub-cases (distinguished by whether the device was in our
                # tracking dict before this call):
                #   - reverting our own earlier reconcile-mute (e.g. user just
                #     became the active listener for this device): pop the
                #     entry so restore_all doesn't redundantly write again
                #   - clearing a pre-existing stale mute on the active sub:
                #     no tracking entry to pop; we just leave it unmuted
                was_our_mute = sub in _active_mutes
                _active_mutes.pop(sub, None)
                reason = "reverting our prior mute" if was_our_mute else "clearing stale pre-existing mute"
                _log("MUTE", f"device={sub!r} unmuted ({reason}; not restoring at stop) ok={ok}")
        _log(
            "MUTE",
            f"reconcile: active={active!r} multi_subs={physical_subs} tracked={list(_active_mutes.keys())}",
        )
        _persist_mutes()


def _restore_all_recording_mutes() -> None:
    """Restore each muted-by-us device to its pre-recording mute state. Removes
    entries from _active_mutes on success and deletes the persistence file when
    the dict ends up empty. macOS only.
    """
    if sys.platform != "darwin":
        return
    with _mutes_lock:
        if not _active_mutes:
            return
        snapshot = list(_active_mutes.items())
        _log("MUTE", f"restore-all: count={len(snapshot)}")
        for name, original in snapshot:
            ok = _ca_set_device_mute(name, original)
            _log("MUTE", f"device={name!r} → original={original} ok={ok}")
            if ok:
                _active_mutes.pop(name, None)
        _persist_mutes()


def _pid_is_alive(pid: int) -> bool:
    """Return True if `pid` corresponds to a running process (any user).
    Uses os.kill(pid, 0) which raises ProcessLookupError when the pid is gone
    and PermissionError when the process exists in a different user namespace
    (still "alive" for our purposes — don't touch its state).
    """
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _recover_persisted_mutes() -> None:
    """At process startup, restore mute state left behind by a dead prior
    MeetingScribe process. Skips recovery if the owning pid is still alive
    (concurrent instance) or if the state file is corrupted. macOS only.
    """
    if sys.platform != "darwin":
        return
    if not _MUTE_STATE_FILE.exists():
        return
    try:
        data = json.loads(_MUTE_STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError) as e:
        _log("MUTE", f"recover: cannot read state file: {type(e).__name__}: {e}; deleting")
        _MUTE_STATE_FILE.unlink(missing_ok=True)
        return
    pid = data.get("pid")
    if pid and pid != os.getpid() and _pid_is_alive(pid):
        _log("MUTE", f"recover: pid={pid} is alive; refusing to touch its mute state")
        return
    muted = data.get("muted") or {}
    for name, original in muted.items():
        ok = _ca_set_device_mute(name, bool(original))
        _log("MUTE", f"recover: device={name!r} → original={original} ok={ok}")
    _MUTE_STATE_FILE.unlink(missing_ok=True)


def _atexit_restore_mutes() -> None:
    """atexit handler — best-effort restore. Never raises."""
    try:
        _restore_all_recording_mutes()
    except Exception as e:
        try:
            _log("MUTE", f"atexit cleanup failed: {type(e).__name__}: {e}")
        except Exception:
            pass


if sys.platform == "darwin":
    try:
        _recover_persisted_mutes()
    except Exception as _e:
        try:
            _log("MUTE", f"startup recover failed: {type(_e).__name__}: {_e}")
        except Exception:
            pass
    atexit.register(_atexit_restore_mutes)


def _refresh_portaudio():
    """Drop PortAudio's internal device cache and re-enumerate. macOS-safe only
    when no sounddevice streams are currently open — Pa_Terminate closes them.
    Callers must guarantee that. Called at recording start/stop boundaries only.
    """
    with _portaudio_lock:
        try:
            sd._terminate()
            sd._initialize()
            _log("AUDIO", "PortAudio cache refreshed")
        except Exception as e:
            _log("ERR", f"PortAudio refresh: {type(e).__name__}: {e}")


def _transport_priority(name: str, transport: dict[str, str]) -> int:
    """Lower is better. 9999 = reject for raw mic/output picks."""
    t = transport.get(name, "")
    if t in _TRANSPORT_VIRTUAL_OR_AGGREGATE:
        return 9999
    if "BlackHole" in name or "Aggregate" in name or name in _MULTI_OUT_NAMES:
        return 9999
    if t == _TRANSPORT_BUILTIN:
        return 1
    return 0  # USB / Bluetooth / HDMI / Thunderbolt / unknown-but-external


def _is_physical_output(name: str | None, transport: dict[str, str]) -> bool:
    if not name:
        return False
    if transport.get(name) in _TRANSPORT_VIRTUAL_OR_AGGREGATE:
        return False
    if name in _MULTI_OUT_NAMES or "BlackHole" in name or "Aggregate" in name:
        return False
    return True


def resolve_audio_devices(query_fresh: bool = False) -> AudioPlan:
    """Single source of truth for audio device selection across the recording lifecycle.

    Picks the mic + system-audio capture source + restore target by transport
    priority (external > built-in; aggregates/virtual rejected for raw picks).

    Set query_fresh=True at lifecycle boundaries (start/stop) so PortAudio
    re-enumerates first. Do NOT pass True while any sounddevice stream is open —
    Pa_Terminate closes streams.
    """
    if query_fresh:
        _refresh_portaudio()

    transport = _coreaudio_device_info()  # {} on non-macOS
    with _portaudio_lock:
        all_devices = list(sd.query_devices())

    inputs = [(i, d["name"]) for i, d in enumerate(all_devices) if d["max_input_channels"] >= 1]
    outputs = [(i, d["name"]) for i, d in enumerate(all_devices) if d["max_output_channels"] >= 1]

    warnings: list[str] = []

    # ── Microphone: external > built-in; aggregates/virtual rejected
    mic_valid = sorted(
        ((i, n) for (i, n) in inputs if _transport_priority(n, transport) < 9999),
        key=lambda x: (_transport_priority(x[1], transport), x[0]),
    )
    if mic_valid:
        mic_index, mic_name = mic_valid[0]
    else:
        mic_index, mic_name = None, None
        warnings.append("no-mic")

    # ── System-audio capture: BlackHole (the virtual sub-device of the
    # Multi-Output Device). The Multi-Output Device itself is an OUTPUT path —
    # in stacked mode it exposes no input channels — so we capture from BlackHole.
    blackhole_pair = next(
        ((i, n) for (i, n) in inputs if transport.get(n) == "virt" or "BlackHole" in n),
        None,
    )
    if blackhole_pair:
        sys_source_index, sys_source_name = blackhole_pair
    else:
        sys_source_index, sys_source_name = None, None
        warnings.append("no-system-audio-source")

    multi_out = next((n for (_, n) in outputs if n in _MULTI_OUT_NAMES), None)
    if multi_out is None:
        warnings.append("no-multi-output-device")

    # ── Restore target: re-query sOut (tracks hardware), reject if it's an
    # aggregate/virtual — restoration must always land on a real physical device.
    restore = _get_system_output_device() or _get_current_output_device()
    if not _is_physical_output(restore, transport):
        out_priority = sorted(
            ((i, n) for (i, n) in outputs if _is_physical_output(n, transport)),
            key=lambda x: (_transport_priority(x[1], transport), x[0]),
        )
        restore = out_priority[0][1] if out_priority else None
    if restore is None:
        warnings.append("no-restore-target")

    is_external = bool(restore) and transport.get(restore, "") != _TRANSPORT_BUILTIN

    plan = AudioPlan(
        mic_index=mic_index,
        mic_name=mic_name,
        sys_source_index=sys_source_index,
        sys_source_name=sys_source_name,
        multi_output_name=multi_out,
        restore_output_name=restore,
        is_external_output=is_external,
        warnings=warnings,
    )
    _log(
        "DEVICE",
        f"plan: mic={mic_name!r} sys={sys_source_name!r} "
        f"multi={multi_out!r} restore={restore!r} "
        f"external={is_external} warnings={warnings}",
    )
    return plan


def _restore_output_if_needed(plan: AudioPlan, *, reason: str = "post-recording") -> str | None:
    """Stop-time / idle-event restore policy. Returns the device we ended on,
    or None on no-op.

    `reason` controls the post-recording gate:
      • "post-recording" (default) — runs only when `_recording_did_switch` is
        set (i.e. the start path actually performed switch_output(multi)).
        When the user has the Multi-Output Device as their permanent macOS
        default, start performed no switch → nothing to restore → no second
        music-app pause on stop.
      • "idle-event" — bypasses the gate; runs unconditionally. Used by the
        AudioDeviceMonitor on hotplug events so dOut+sOut alignment still
        follows the resolver between recordings.

    Evaluates the macOS media-default (dOut) and system-default (sOut) properties
    INDEPENDENTLY. Either half is written only when its current value is
    non-physical (aggregate, BlackHole, or other virtual device) — so a music app
    whose stream is already pointing at a physical dOut never sees a device
    change. Writing dOut here is what makes the hardware volume keys (F11/F12,
    Touch Bar, menu-bar slider) functional again after recording: macOS routes
    the volume keys to whichever device dOut points at, and the Multi-Output
    Device has no master volume control of its own.

    NOTE: writing dOut may briefly pause some music apps (Apple Music, Spotify)
    that watch for default-device changes. The user explicitly chose "volume
    keys work after stop" over "music never pauses" — see the
    fix-volume-keys-after-restore change.
    """
    # Snapshot the raw CoreAudio device topology BEFORE any gate / decision —
    # gives post-mortem visibility into class/transport tags at the moment we
    # picked (or skipped) a restore target. Cheap (~10 devices, file-only).
    _log_device_raw_dump(reason=f"restore-entry:{reason}")

    if reason == "post-recording":
        if not _recording_did_switch.is_set():
            _log("RESTORE", "post-recording gate: did_switch=False; skipping")
            return None
        _log("RESTORE", "post-recording gate: did_switch=True; running")
    elif reason == "idle-event":
        _log("RESTORE", "idle-event gate: bypassing did_switch check; running")
    else:
        _log("WARN", f"_restore_output_if_needed: unknown reason={reason!r}; treating as post-recording")
        if not _recording_did_switch.is_set():
            _log("RESTORE", "post-recording gate: did_switch=False; skipping")
            return None

    transport = _coreaudio_device_info()
    current_dout = _get_current_output_device()
    current_sout = _get_system_output_device()

    dout_physical = _is_physical_output(current_dout, transport)
    sout_physical = _is_physical_output(current_sout, transport)
    need_dout = not dout_physical
    need_sout = not sout_physical

    _log(
        "RESTORE",
        f"dOut={current_dout!r} (physical={dout_physical}) "
        f"sOut={current_sout!r} (physical={sout_physical}) "
        f"need_dout={need_dout} need_sout={need_sout}",
    )

    if not need_dout and not need_sout:
        _log("RESTORE", "both already physical; no-op")
        return current_dout

    with _portaudio_lock:
        try:
            live_outputs = [
                d["name"] for d in sd.query_devices() if d["max_output_channels"] >= 1
            ]
        except Exception as e:
            _log("ERR", f"restore live-outputs query: {type(e).__name__}: {e}")
            live_outputs = []

    # Build a priority-ordered candidate list. Plan's target wins if it's still attached.
    candidates: list[str] = []
    if plan.restore_output_name and plan.restore_output_name in live_outputs:
        candidates.append(plan.restore_output_name)
    elif plan.restore_output_name:
        _log("RESTORE", f"plan target {plan.restore_output_name!r} not in live outputs; falling back")
    for name in sorted(live_outputs, key=lambda n: _transport_priority(n, transport)):
        if _is_physical_output(name, transport) and name not in candidates:
            candidates.append(name)

    for target in candidates:
        try:
            if need_dout:
                switch_output(target)
                _log("AUDIO", f"dOut → {target!r}")
            if need_sout:
                switch_system_output(target)
                _log("AUDIO", f"sOut → {target!r}")
            return target
        except Exception as e:
            _log("ERR", f"restore switch to {target!r} failed: {type(e).__name__}: {e}")
    _log("RESTORE", "no candidate succeeded; default left unchanged")
    return None


def _install_device_listeners() -> bool:
    """Install CoreAudio HAL property listeners so hotplug events wake the recorder
    within ~100 ms instead of waiting for the 1 s polling tick. Idempotent.
    Returns True on success; no-op on non-macOS.
    """
    if sys.platform != "darwin":
        return False
    if _listener_state.get("installed"):
        return True
    import ctypes, ctypes.util, struct

    ca = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))

    class _Addr(ctypes.Structure):
        _fields_ = [("sel", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("elem", ctypes.c_uint32)]

    def _fcc(s):
        return struct.unpack(">I", s.encode())[0]

    LISTENER_PROC = ctypes.CFUNCTYPE(
        ctypes.c_int32,    # OSStatus
        ctypes.c_uint32,   # inObjectID
        ctypes.c_uint32,   # inNumberAddresses
        ctypes.c_void_p,   # inAddresses
        ctypes.c_void_p,   # inClientData
    )

    def _on_event(_obj_id, _n_addrs, _addrs, _client):
        # Keep callback minimal: log + set the event. The recorder thread does
        # the real reconcile work off the CoreAudio callback thread.
        try:
            _log("HOTPLUG", "event fired")
        except Exception:
            pass
        _hotplug_event.set()
        return 0  # noErr

    cb = LISTENER_PROC(_on_event)
    ca.AudioObjectAddPropertyListener.argtypes = [
        ctypes.c_uint32, ctypes.POINTER(_Addr), LISTENER_PROC, ctypes.c_void_p,
    ]
    ca.AudioObjectAddPropertyListener.restype = ctypes.c_int32
    ca.AudioObjectRemovePropertyListener.argtypes = [
        ctypes.c_uint32, ctypes.POINTER(_Addr), LISTENER_PROC, ctypes.c_void_p,
    ]
    ca.AudioObjectRemovePropertyListener.restype = ctypes.c_int32

    kSystem, kGlobal = 1, _fcc("glob")
    addrs = []
    for sel in (_fcc("dev#"), _fcc("sOut")):  # device-list, system-default-output
        a = _Addr(sel, kGlobal, 0)
        try:
            ret = ca.AudioObjectAddPropertyListener(kSystem, ctypes.byref(a), cb, None)
        except Exception:
            ret = -1
        if ret == 0:
            addrs.append(a)
    if not addrs:
        _log("ERR", "install_device_listeners: no selectors accepted listener")
        return False
    _listener_state.update({
        "installed": True,
        "callback": cb,        # keep CFUNCTYPE instance alive
        "proc_type": LISTENER_PROC,
        "addrs": addrs,        # keep _Addr instances alive
        "addr_type": _Addr,
        "ca": ca,
    })
    _log("HOTPLUG", f"installed listeners on {len(addrs)} selectors (dev#, sOut)")
    return True


def _remove_device_listeners():
    """Symmetric teardown for _install_device_listeners(). Idempotent."""
    if sys.platform != "darwin" or not _listener_state.get("installed"):
        return
    import ctypes
    ca = _listener_state.get("ca")
    cb = _listener_state.get("callback")
    for a in _listener_state.get("addrs", []):
        try:
            ca.AudioObjectRemovePropertyListener(1, ctypes.byref(a), cb, None)
        except Exception as e:
            _log("ERR", f"remove_device_listener: {type(e).__name__}: {e}")
    _listener_state.clear()
    _listener_state["installed"] = False
    _log("HOTPLUG", "listeners removed")


class AudioDeviceMonitor(threading.Thread):
    """Event-driven device-watcher thread.

    Blocks on `_hotplug_event` (set by the macOS CoreAudio HAL listener within
    ~100 ms of any plug / unplug / default-output change) with a 30 s safety
    timeout as a backstop. Owns the idle-state "restore to best device"
    decision; during a recording the recorder's own `_monitor` thread is the
    authoritative reconciler.

    Idle branch (when _recording_active is NOT set):
      • First tick (synchronous on start()) always runs
        `_restore_output_if_needed`.
      • Subsequent ticks restore only when the resolved triple
        `(restore_output_name, mic_name, sys_source_name)` changed.

    Recording branch (when _recording_active IS set):
      • Log-only. The recorder's `_monitor` is also woken by the same
        `_hotplug_event` and does the actual reconciliation.
    """

    def __init__(self):
        super().__init__(daemon=True, name="AudioDeviceMonitor")
        self._stop_event = threading.Event()
        self._prev_triple: tuple | None = None
        # Memoizes the mute-relevant inputs of the last recording-branch
        # reconciliation so we only call _reconcile_recording_mutes when those
        # inputs actually change (e.g. user plugs/unplugs a physical output).
        self._prev_mute_triple: tuple | None = None
        # Optional callback fired (from this monitor thread) every time the
        # recording-branch observes a plan change. GUI code sets this to keep
        # widgets like the volume slider bound to the current physical output
        # when the user hotplugs mid-recording. None outside the GUI (CLI).
        # The callback is responsible for its own thread-safety — it MUST NOT
        # block; common pattern is to schedule a Tk update via root.after(0).
        self.on_recording_plan_change: "Callable[[AudioPlan], None] | None" = None

    def start(self):
        if self.is_alive():
            return
        super().start()

    def stop(self, timeout: float = 1.5):
        self._stop_event.set()
        # Nudge any internal wait so the thread exits promptly.
        _hotplug_event.set()
        self.join(timeout)
        if self.is_alive():
            _log("WARN", f"monitor stop join timeout after {timeout}s")

    def run(self):
        # Initial synchronous tick so the process always syncs to current best
        # device on launch, preserving the first-tick-restore semantics.
        self._safe_tick()
        while not self._stop_event.is_set():
            _hotplug_event.wait(timeout=_AUDIO_MONITOR_SAFETY_TIMEOUT_SEC)
            if self._stop_event.is_set():
                break
            self._safe_tick()

    def _safe_tick(self):
        try:
            self._tick_once()
        except Exception as e:
            _log("ERR", f"monitor tick: {type(e).__name__}: {e}")
        # Clear the event after consuming it. The recorder's _monitor also
        # clears, but Event.clear() is idempotent — a double-clear is a no-op,
        # and one missed wake is recovered by the recorder's own 1 s fallback
        # timeout or by the next real event.
        _hotplug_event.clear()

    def _tick_once(self):
        is_recording = _recording_active.is_set()
        if is_recording:
            self._recording_branch()
            return
        plan = resolve_audio_devices(query_fresh=True)
        triple = (plan.restore_output_name, plan.mic_name, plan.sys_source_name)
        self._idle_branch(plan, triple)
        self._prev_triple = triple

    # Public test seam.
    def tick_once(self):
        self._tick_once()

    def _idle_branch(self, plan: "AudioPlan", triple: tuple):
        # Drop the mute-baseline so the next recording starts with a clean
        # comparison baseline (the prior recording's mutes have been restored).
        self._prev_mute_triple = None
        if self._prev_triple is None or triple != self._prev_triple:
            # Idle-event restores bypass the post-recording did-switch gate so
            # hotplug-driven dOut+sOut alignment still works between recordings.
            restored = _restore_output_if_needed(plan, reason="idle-event")
            _log("MONITOR", f"restoring; restored={restored!r}; triple={triple!r}")
        else:
            _log("MONITOR", "idle no-change")

    def _recording_branch(self):
        # The recorder's _monitor thread is the authoritative reconciler for
        # the input streams — we don't touch those here. But we DO own the
        # mute-state reconciliation for the Multi-Output's output sub-devices,
        # so we re-resolve the plan and re-evaluate which sub-device should be
        # muted when the user plugs / unplugs a physical output mid-recording.
        _log("MONITOR", "recording event observed")
        try:
            # query_fresh=False: streams are open; do NOT terminate PortAudio
            # (CoreAudio enumeration used by _get_multi_output_physical_subs is
            # independent of PortAudio).
            plan = resolve_audio_devices(query_fresh=False)
        except Exception as e:
            _log("ERR", f"recording-branch resolve: {type(e).__name__}: {e}")
            return
        mute_triple = (plan.multi_output_name, plan.restore_output_name)
        if mute_triple == self._prev_mute_triple:
            return
        self._prev_mute_triple = mute_triple
        try:
            _reconcile_recording_mutes(plan)
        except Exception as e:
            _log("ERR", f"recording-branch reconcile_mutes: {type(e).__name__}: {e}")
        cb = self.on_recording_plan_change
        if cb is not None:
            try:
                cb(plan)
            except Exception as e:
                _log("ERR", f"recording-branch plan-change callback: {type(e).__name__}: {e}")


def _get_audio_monitor() -> AudioDeviceMonitor:
    """Lazy-construct the process-wide AudioDeviceMonitor singleton."""
    global _audio_monitor
    if _audio_monitor is None:
        _audio_monitor = AudioDeviceMonitor()
    return _audio_monitor


# ── Legacy adapters: existing call sites pass through resolve_audio_devices().
# These keep the old dict-shaped contracts (used by cmd_devices and CLI/UI flow)
# while the rest of the file is migrated to AudioPlan.

def _auto_detect_devices() -> dict:
    """Legacy adapter — delegates to resolve_audio_devices()."""
    plan = resolve_audio_devices()
    return {
        "device_system_audio": plan.sys_source_name,
        "device_mic": plan.mic_name,
        "output_record": plan.multi_output_name or plan.sys_source_name,
        "output_restore": plan.restore_output_name,
    }


def _resolve_devices(cfg: dict) -> dict:
    """Legacy adapter — auto-detects via the unified resolver, lets cfg override."""
    keys = ("device_system_audio", "device_mic", "output_record", "output_restore")
    if all(cfg.get(k) for k in keys):
        return {k: cfg[k] for k in keys}
    detected = _auto_detect_devices()
    return {k: cfg.get(k) or detected.get(k) for k in keys}


def _prepare_recording_devices(cfg: dict) -> dict:
    """Resolve devices for recording. Never creates audio devices automatically.

    Returns the same dict-shaped contract as the legacy adapter plus:
      'output_record_id' (always None — kept for backwards compatibility)
      'warnings'         (list[str])
    """
    devs = _resolve_devices(cfg)
    warnings: list[str] = []
    blackhole = devs.get("device_system_audio")
    out_record = devs.get("output_record")

    if blackhole and out_record == blackhole:
        warnings.append(
            "[提示] 未检测到聚合输出设备。录音时将切换默认输出到 BlackHole，"
            "系统音频会被录到但扬声器/耳机不再出声。"
            "若需同时监听，请在『音频 MIDI 设置』里手动创建一个 Multi-Output Device "
            "（包含 BlackHole 和你的扬声器/耳机）。"
        )

    return {**devs, "output_record_id": None, "warnings": warnings}


# ── 录音 ──────────────────────────────────────────────────────────────────────

class AudioRecorder:
    def __init__(self, device, sample_rate: int, channels: int):
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self._frames: list[np.ndarray] = []
        self._stream = None
        self.recording = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"  [audio warn] {status}", file=sys.stderr)
        if self.recording:
            self._frames.append(indata.copy())

    def start(self):
        self.recording = True
        self._frames = []
        self._stream = sd.InputStream(
            device=self.device,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
            blocksize=int(self.sample_rate * 0.1),
        )
        self._stream.start()

    def stop(self):
        self.recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def save(self, path: Path) -> bool:
        if not self._frames:
            return False
        audio = np.concatenate(self._frames, axis=0)  # shape: (frames, channels)

        # Drop silent channels; always write stereo (all STT providers handle stereo)
        if audio.ndim > 1 and audio.shape[1] > 2:
            channel_rms = np.sqrt(np.mean(audio ** 2, axis=0))
            active = [i for i, rms in enumerate(channel_rms) if rms > 1e-5]
            if not active:
                active = list(range(audio.shape[1]))
            if len(active) == 1:
                # mono → duplicate to stereo
                audio = np.column_stack([audio[:, active[0]]] * 2)
            else:
                audio = audio[:, active[:2]]

        out_channels = 1 if audio.ndim == 1 else audio.shape[1]
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(out_channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return True


# ── 多路录音 ──────────────────────────────────────────────────────────────────

class MultiStreamRecorder:
    """从任意多个输入设备同时录制，混合为立体声输出。

    热插拔：后台线程在 _hotplug_event（CoreAudio 监听器触发，~100 ms 延迟）
    或 1 秒轮询超时之间取最先到达者，重新检测设备列表，
      • 把 wanted 中新出现的设备加进录制
      • 把已消失的设备的流优雅关闭（不中断整体录音）
    """

    on_device_added: "callable | None" = None  # callback(device_name: str)
    on_warning: "callable | None" = None       # callback(code: str)

    def __init__(self, wanted: list, sample_rate: int, role_labels: list[str] | None = None):
        # wanted: 期望录制的设备名列表（保序，决定 L/R 声道分配）
        # role_labels: 与 wanted 一一对应的语义标签 ('system' / 'mic' / 其他)，
        #              用于在某路录不到时生成正确的 warning code。
        self.wanted = [d for d in wanted if d]
        self.sample_rate = sample_rate
        if role_labels:
            self.role_labels = [
                role_labels[i] if i < len(role_labels) else d
                for i, d in enumerate(self.wanted)
            ]
        else:
            self.role_labels = list(self.wanted)
        self._frames: dict[str, list] = {}   # device → frame list
        self._streams: dict[str, object] = {}  # device → InputStream
        self._rms: dict[str, float] = {}     # device → latest RMS
        self._lock = threading.Lock()        # protects _streams / _frames dict ops
        self.recording = False
        self.skipped: list[str] = []
        self.warnings: list[str] = []        # de-duped warning codes for sidecar
        # Tracked so stop() can join the monitor thread before clearing
        # _streams — eliminates the race where _monitor_iteration's _try_open
        # resets _frames mid-stop, discarding captured audio.
        self._monitor_thread: "threading.Thread | None" = None

    def _emit_warning(self, code: str):
        if code in self.warnings:
            return
        self.warnings.append(code)
        _log("WARN", f"recorder: {code}")
        if self.on_warning:
            try:
                self.on_warning(code)
            except Exception as e:
                _log("ERR", f"on_warning callback: {type(e).__name__}: {e}")

    def _role_warning_code(self, device: str, kind: str) -> str | None:
        """Map (device, 'not-opened'|'disappeared') to a stable sidecar code, or None
        if the role is generic and shouldn't produce a labelled warning."""
        try:
            role = self.role_labels[self.wanted.index(device)]
        except (ValueError, IndexError):
            return None
        if role == "mic":
            return f"mic-{kind}"
        if role == "system":
            return f"system-audio-{kind}"
        return None

    def _make_cb(self, device: str):
        def _cb(indata, frames, time_info, status):
            if self.recording:
                self._frames[device].append(indata.copy())  # GIL 保护 list.append
                if _DEBUG:
                    self._rms[device] = float(np.sqrt(np.mean(indata ** 2)))
        return _cb

    def _try_open(self, device: str) -> bool:
        """尝试打开设备的 InputStream。成功返回 True。

        Serialized with PortAudio refresh via _portaudio_lock so a hotplug-triggered
        sd._terminate()/_initialize() cycle can't run mid-open.
        """
        block = int(self.sample_rate * 0.1)
        try:
            with _portaudio_lock:
                stream = sd.InputStream(
                    device=device, samplerate=self.sample_rate,
                    channels=1, dtype="float32",
                    callback=self._make_cb(device), blocksize=block,
                )
                stream.start()
            with self._lock:
                # Defensive guard: if stop() ran concurrently while we were
                # opening the stream, do NOT install it — installing would
                # reset self._frames[device] to [] and wipe any captured
                # audio from a previous stream on the same device.
                if not self.recording:
                    _log(
                        "STREAM",
                        f"discarded post-stop open of device={device!r}",
                    )
                    # Release the orphan stream resources outside this lock
                    # block via the finally pattern: stop+close immediately
                    # below, then return False. Stop/close are safe to call
                    # outside the lock since the stream isn't tracked anywhere.
                    orphan = stream
                else:
                    self._streams[device] = stream
                    self._frames[device] = []
                    self._rms[device] = 0.0
                    orphan = None
            if orphan is not None:
                try:
                    orphan.stop()
                except Exception as e:
                    _log("STREAM", f"orphan stop device={device!r}: {type(e).__name__}: {e}")
                try:
                    orphan.close()
                except Exception as e:
                    _log("STREAM", f"orphan close device={device!r}: {type(e).__name__}: {e}")
                return False
            _log("STREAM", f"opened device={device!r}")
            return True
        except Exception as e:
            _log("STREAM", f"open failed device={device!r} reason={type(e).__name__}: {e}")
            return False

    def _close_one(self, device: str, reason: str = "normal"):
        """Stop+close a single stream and drop it from the tracking dict.
        `reason` is logged for traceability (normal | disappeared | recorder-stop)."""
        with self._lock:
            stream = self._streams.pop(device, None)
        if stream is None:
            return
        try:
            stream.stop()
        except Exception as e:
            _log("ERR", f"stream stop device={device!r}: {type(e).__name__}: {e}")
        try:
            stream.close()
        except Exception as e:
            _log("ERR", f"stream close device={device!r}: {type(e).__name__}: {e}")
        _log("STREAM", f"closed device={device!r} reason={reason}")

    def _monitor(self):
        """后台线程：热插拔检测 + reconcile + debug 日志。

        Wakes on _hotplug_event (CoreAudio listener) or after 1 s timeout —
        whichever comes first. On each wake re-derives the target device set
        from the resolver and reconciles open streams: opens newly-wanted /
        appearing devices, closes disappeared / no-longer-wanted ones. Mic and
        system-audio swaps mid-recording are honoured (USB headset plugged in
        → mic swaps from built-in to USB with a brief gap on the mic channel).
        """
        while self.recording:
            woken = _hotplug_event.wait(timeout=1.0)
            if woken:
                _hotplug_event.clear()
            if not self.recording:
                break
            self._monitor_iteration()

    def _monitor_iteration(self):
        """One pass of the monitor loop: query devices, re-resolve plan,
        reconcile streams. Extracted from `_monitor` so unit tests can drive
        it deterministically without spinning the wait loop."""
        try:
            with _portaudio_lock:
                avail = {
                    d["name"] for d in sd.query_devices()
                    if d["max_input_channels"] >= 1
                }
        except Exception as e:
            _log("ERR", f"monitor query_devices: {type(e).__name__}: {e}")
            avail = set()

        # Re-derive `wanted` from the resolver on every wake so the recorder
        # follows the latest plan (e.g. USB mic plugged in mid-recording
        # promotes from built-in to USB). query_fresh=False so PortAudio is
        # NOT terminated while our streams are open.
        try:
            plan = resolve_audio_devices(query_fresh=False)
        except Exception as e:
            _log("ERR", f"monitor resolve: {type(e).__name__}: {e}")
            plan = None

        # role index → (old_device, new_device) for any role swap this tick.
        # Used after open to prepend pre-swap frames so the recorded WAV keeps
        # audio continuity across the swap.
        role_swaps: dict[int, tuple[str | None, str | None]] = {}
        if plan is not None:
            new_wanted = [n for n in (plan.sys_source_name, plan.mic_name) if n]
            if new_wanted and new_wanted != self.wanted:
                # Log per-role swaps so the WAV's mid-recording boundary is
                # traceable. Index 0 = system-audio source, index 1 = mic.
                roles = ("system", "mic")
                for i, role in enumerate(roles):
                    old = self.wanted[i] if i < len(self.wanted) else None
                    new = new_wanted[i] if i < len(new_wanted) else None
                    if old != new:
                        _log("STREAM", f"{role} swap from={old!r} to={new!r} reason=hotplug")
                        role_swaps[i] = (old, new)
                self.wanted = new_wanted
                # Keep role_labels aligned so warning codes stay accurate.
                self.role_labels = list(roles[: len(new_wanted)])

        # Close streams that are either no longer wanted (mic swap: built-in
        # mic still attached but the resolver picked a USB mic instead) or
        # whose device disappeared from the live device list. Close BEFORE
        # opening so a role swap drains the old callback first.
        wanted_set = set(self.wanted)
        with self._lock:
            disappeared = [d for d in list(self._streams) if d not in avail]
            no_longer_wanted = [d for d in list(self._streams) if d in avail and d not in wanted_set]
        for dev in disappeared:
            self._close_one(dev, reason="disappeared")
            code = self._role_warning_code(dev, "disappeared")
            if code:
                self._emit_warning(code)
        for dev in no_longer_wanted:
            self._close_one(dev, reason="role-swap")

        # Open streams for wanted devices that just appeared (or just joined
        # `wanted` via a role swap).
        pending = [d for d in self.wanted if d not in self._streams]
        opened_this_tick: list[str] = []
        for dev in pending:
            if dev in avail and self._try_open(dev):
                opened_this_tick.append(dev)
                if self.on_device_added:
                    try:
                        self.on_device_added(dev)
                    except Exception as e:
                        _log("ERR", f"on_device_added callback: {type(e).__name__}: {e}")

        # Preserve audio continuity across role swaps: prepend the pre-swap
        # device's frames into the post-swap device's frame list. Must happen
        # AFTER _try_open (which reset _frames[new] to []) and AFTER the old
        # stream's _close_one (which drained its callback).
        for i, (old, new) in role_swaps.items():
            if old and new and new in self._frames:
                with self._lock:
                    carried = self._frames.pop(old, [])
                    self._frames[new] = carried + self._frames[new]
                if carried:
                    _log(
                        "STREAM",
                        f"role-swap carry: {len(carried)} frames from {old!r} into {new!r}",
                    )

        if opened_this_tick or disappeared or no_longer_wanted:
            _log(
                "HOTPLUG",
                f"reconcile: opened={opened_this_tick} "
                f"closed_disappeared={disappeared} closed_role_swap={no_longer_wanted}",
            )

        if _DEBUG:
            parts = []
            for dev in self.wanted:
                if dev in self._streams:
                    rms = self._rms.get(dev, 0.0)
                    tag = "有声" if rms > 0.001 else "静音"
                    parts.append(f"{dev}: {rms:.4f}({tag})")
                else:
                    parts.append(f"{dev}: 等待中")
            _log("STREAM", " | ".join(parts))

    def start(self):
        if not self.wanted:
            raise ValueError("没有选择任何录音设备")
        self.recording = True
        self._frames = {}
        self._rms = {}
        self.skipped = []
        self.warnings = []
        _hotplug_event.clear()  # don't fire on stale events from before start
        _recording_active.set()  # tell AudioDeviceMonitor to use the recording branch
        # NOTE: _recording_did_switch is NOT cleared here. The flag is cleared
        # at the entry of the recording-start lifecycle (before the
        # switch_output decision) so that the start-path's switch decision
        # drives the flag for the entire session. Clearing here would wipe
        # the bit just set by the start-path switch.

        try:
            with _portaudio_lock:
                avail = {
                    d["name"] for d in sd.query_devices()
                    if d["max_input_channels"] >= 1
                }
        except Exception as e:
            _log("ERR", f"start query_devices: {type(e).__name__}: {e}")
            avail = set()

        for device in self.wanted:
            if device in avail:
                if not self._try_open(device):
                    self.skipped.append(device)
            # 不在 avail 里：monitor 线程会在下一秒重试
        _log(
            "REC",
            f"start: wanted={self.wanted} opened={list(self._streams)} "
            f"skipped={self.skipped}",
        )
        # 无论开了几路，都启动监控线程（热插拔 + debug）。
        # Track the thread so stop() can join it before clearing _streams,
        # eliminating the race where _monitor_iteration's _try_open resets
        # _frames mid-stop and discards captured audio.
        self._monitor_thread = threading.Thread(
            target=self._monitor, daemon=True, name="RecorderMonitor"
        )
        self._monitor_thread.start()
        _log("REC", f"monitor thread started: name={self._monitor_thread.name!r}")

    def stop(self):
        self.recording = False
        _hotplug_event.set()  # wake monitor thread immediately so it exits
        # Join the monitor thread BEFORE clearing _streams. Without this,
        # _monitor_iteration could run concurrently with the stream cleanup
        # below: its `pending = [d for d in self.wanted if d not in
        # self._streams]` would see the cleared dict and call _try_open, which
        # resets self._frames[device] = [] — wiping captured audio.
        thread = self._monitor_thread
        if thread is not None and thread.is_alive():
            _log("REC", f"stop: joining monitor thread (timeout=2.0s)")
            thread.join(timeout=2.0)
            if thread.is_alive():
                _log("WARN", "recorder monitor join timeout after 2.0s")
            else:
                _log("REC", "stop: monitor thread joined cleanly")
        self._monitor_thread = None
        try:
            with self._lock:
                streams_to_close = dict(self._streams)
                self._streams.clear()
            for device, stream in streams_to_close.items():
                try:
                    stream.stop()
                except Exception as e:
                    _log("ERR", f"stream stop device={device!r}: {type(e).__name__}: {e}")
                try:
                    stream.close()
                except Exception as e:
                    _log("ERR", f"stream close device={device!r}: {type(e).__name__}: {e}")
                _log("STREAM", f"closed device={device!r} reason=recorder-stop")
            _log("REC", f"stop: closed {len(streams_to_close)} streams")
        finally:
            _recording_active.clear()  # AudioDeviceMonitor returns to idle branch

    def save(self, path: Path) -> bool:
        # 按 wanted 顺序收集各路音频（保证 L/R 声道语义）
        channels = []
        for device in self.wanted:
            frames = self._frames.get(device, [])
            if frames:
                audio = np.concatenate(frames)
                channels.append(audio[:, 0] if audio.ndim > 1 else audio)
            else:
                channels.append(None)  # 该设备未录到任何数据

        # Surface partial-capture as labelled warnings (mic / system not opened).
        for device, ch in zip(self.wanted, channels):
            if ch is None:
                code = self._role_warning_code(device, "not-opened")
                if code:
                    self._emit_warning(code)

        valid = [c for c in channels if c is not None]
        if not valid:
            return False

        if len(valid) == 1:
            mixed = np.column_stack([valid[0], valid[0]])
        else:
            c0, c1 = channels[0], channels[1]
            if c0 is None:
                mixed = np.column_stack([valid[0], valid[0]])
            elif c1 is None:
                mixed = np.column_stack([c0, c0])
            else:
                n = min(len(c0), len(c1))
                mixed = np.column_stack([c0[:n], c1[:n]])

        audio_int16 = (np.clip(mixed, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

        # Sidecar warnings file — only created when warnings exist; one code per line.
        if self.warnings:
            sidecar = path.with_name(path.stem + ".warnings.txt")
            try:
                sidecar.write_text("\n".join(self.warnings) + "\n", encoding="utf-8")
                _log("REC", f"sidecar written: {sidecar} ({len(self.warnings)} warnings)")
            except Exception as e:
                _log("ERR", f"sidecar write {sidecar}: {type(e).__name__}: {e}")
        _log(
            "REC",
            f"save: path={path.name} channels={2} frames={mixed.shape[0]} "
            f"warnings={self.warnings}",
        )
        return True


# ── 转写 ──────────────────────────────────────────────────────────────────────

def transcribe(audio_path: Path, provider: str, cfg: dict, on_progress=None, on_chunk_done=None) -> str:
    stt_cfgs = _deep_merge(DEFAULT_CONFIG["stt"], cfg.get("stt", {}))
    pcfg = stt_cfgs.get(provider)
    if pcfg is None:
        print(f"[错误] 未知转写 provider '{provider}'，请在 config stt 中配置")
        sys.exit(1)

    print(f"[转写] 使用 {provider} 转写 {audio_path.name} ...")
    if provider == "funasr":
        return _transcribe_funasr(audio_path, pcfg, on_progress=on_progress, on_chunk_done=on_chunk_done)
    elif provider == "whisper":
        return _transcribe_whisper(audio_path, pcfg, on_progress=on_progress)
    elif provider == "openai":
        return _transcribe_openai(audio_path, pcfg)
    elif provider == "gemini":
        return _transcribe_gemini(audio_path, pcfg)
    else:
        print(f"[错误] 不支持的转写 provider: {provider}")
        sys.exit(1)


def _transcribe_funasr(audio_path: Path, pcfg: dict, on_progress=None, on_chunk_done=None) -> str:
    import concurrent.futures, tempfile

    asr_model   = pcfg.get("model", "paraformer-zh")
    vad_model   = pcfg.get("vad_model", "fsmn-vad")
    punc_model  = pcfg.get("punc_model", "ct-punc")
    hotword     = pcfg.get("hotword", "")
    chunk_secs       = int(pcfg.get("chunk_secs", 300))
    _workers_cfg     = int(pcfg.get("workers", 0))
    max_workers      = _workers_cfg if _workers_cfg > 0 else max(6, (os.cpu_count() or 4) // 2)

    with wave.open(str(audio_path), "rb") as wf:
        total_frames = wf.getnframes()
        framerate    = wf.getframerate()
        n_channels   = wf.getnchannels()
        sampwidth    = wf.getsampwidth()
    total_secs = total_frames / framerate

    def _load_model():
        key = (asr_model, vad_model, punc_model)
        if key not in _funasr_model_cache:
            from funasr import AutoModel
            _funasr_model_cache[key] = AutoModel(
                model=asr_model, vad_model=vad_model, punc_model=punc_model,
                disable_update=True,
            )
        return _funasr_model_cache[key]

    def _items_to_lines(items, offset_s=0.0):
        """Turn FunASR's generate() output into [HH.Hs] prefixed lines. Defensive:
        FunASR's timestamp shape varies across versions / punc-model paths
        (list-of-pairs, numpy array, mixed punctuation tokens with no timestamp),
        so any unexpected shape just drops the timestamp prefix instead of
        crashing the whole transcription pipeline.
        """
        lines = []
        if not items:
            return lines
        for item in items:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            start_s = None
            ts = item.get("timestamp")
            try:
                if ts is not None and len(ts) > 0:
                    first = ts[0]
                    if first is not None and len(first) > 0:
                        start_s = float(first[0]) / 1000.0 + offset_s
            except (TypeError, ValueError, IndexError):
                start_s = None
            if start_s is not None:
                lines.append(f"[{start_s:05.1f}s] {text}")
            else:
                lines.append(text)
        return lines

    # ── 短录音：直接串行转写 ──────────────────────────────────────────────────
    if chunk_secs <= 0 or total_secs <= chunk_secs:
        print(f"[转写] 加载 FunASR {asr_model}（首次运行会自动下载模型）...")
        m = _load_model()
        if on_progress:
            on_progress(10)
        print(f"[转写] 开始转写 {audio_path.name} ...")
        kwargs: dict = dict(input=str(audio_path), batch_size_s=300)
        if hotword:
            kwargs["hotword"] = hotword
        # Capture FunASR's tqdm bars and timing dicts into the log file only
        # — the console stays clean and the user sees only our [转写]/[校对]/[纪要] lines.
        with _QuietCapture("STT"):
            results = m.generate(**kwargs)
        if on_progress:
            on_progress(38)
        result_text = "\n".join(_items_to_lines(results))
        if on_chunk_done:
            on_chunk_done(result_text, 0)
        return result_text

    # ── 长录音：分块并发转写 ──────────────────────────────────────────────────
    n_chunks       = math.ceil(total_secs / chunk_secs)
    actual_workers = min(max_workers, n_chunks)
    chunk_label = f"{chunk_secs // 60} 分钟" if chunk_secs >= 60 else f"{chunk_secs} 秒"
    print(
        f"[转写] 录音时长 {total_secs / 60:.1f} 分钟，分 {n_chunks} 块并发转写"
        f"（每块 {chunk_label}，并发 {actual_workers}）"
        f"，加载 FunASR {asr_model}..."
    )

    m = _load_model()

    def _run_chunk(args):
        chunk_path_str, offset_s, idx = args
        kwargs: dict = dict(input=chunk_path_str, batch_size_s=300)
        if hotword:
            kwargs["hotword"] = hotword
        # Per-chunk failure is non-fatal — one bad chunk shouldn't sacrifice the
        # other 10 chunks of a 50-minute recording. The error is logged and the
        # chunk's lines come back as an empty list.
        try:
            with _QuietCapture("STT"):
                items = m.generate(**kwargs)
            return idx, _items_to_lines(items, offset_s)
        except Exception as e:
            import traceback
            _log("ERR", f"funasr chunk {idx + 1} failed: {type(e).__name__}: {e}")
            _log("ERR", "traceback:\n" + traceback.format_exc())
            print(f"[转写] 第 {idx + 1} 块失败: {e}", file=sys.stderr)
            return idx, [f"[块{idx + 1}失败: {e}]"]

    with tempfile.TemporaryDirectory(prefix="meetingscribe_") as tmpdir:
        chunk_args = []
        with wave.open(str(audio_path), "rb") as wf:
            for i in range(n_chunks):
                start_f = int(i * chunk_secs * framerate)
                end_f   = min(int((i + 1) * chunk_secs * framerate), total_frames)
                wf.setpos(start_f)
                chunk_data = wf.readframes(end_f - start_f)
                chunk_path = Path(tmpdir) / f"chunk_{i:04d}.wav"
                with wave.open(str(chunk_path), "wb") as cw:
                    cw.setnchannels(n_channels)
                    cw.setsampwidth(sampwidth)
                    cw.setframerate(framerate)
                    cw.writeframes(chunk_data)
                chunk_args.append((str(chunk_path), float(i * chunk_secs), i))

        chunk_results = [None] * n_chunks
        done_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {executor.submit(_run_chunk, args): args[2] for args in chunk_args}
            for future in concurrent.futures.as_completed(futures):
                idx, lines = future.result()
                chunk_results[idx] = lines
                done_count += 1
                print(f"[转写] 第 {idx + 1}/{n_chunks} 块完成")
                if on_progress:
                    on_progress(5 + int(done_count / n_chunks * 33))
                if on_chunk_done:
                    on_chunk_done("\n".join(lines), idx)

    all_lines: list[str] = []
    for chunk_lines in chunk_results:
        if chunk_lines:
            all_lines.extend(chunk_lines)
    return "\n".join(all_lines)


def _transcribe_whisper(audio_path: Path, pcfg: dict, on_progress=None) -> str:
    import concurrent.futures, tempfile
    from faster_whisper import WhisperModel

    model_size  = pcfg.get("model", "base")
    chunk_secs  = int(pcfg.get("chunk_secs", 300))
    max_workers     = max(1, int(pcfg.get("workers", 2)))
    _cpu_threads_cfg = int(pcfg.get("cpu_threads", 0))
    cpu_threads      = _cpu_threads_cfg if _cpu_threads_cfg > 0 else max(1, (os.cpu_count() or 2) // 2)

    # 读取 WAV 元数据
    with wave.open(str(audio_path), "rb") as wf:
        total_frames = wf.getnframes()
        framerate    = wf.getframerate()
        n_channels   = wf.getnchannels()
        sampwidth    = wf.getsampwidth()
    total_secs = total_frames / framerate

    # ── 短录音：直接串行转写 ──────────────────────────────────────────────────
    if chunk_secs <= 0 or total_secs <= chunk_secs:
        print(f"[转写] 加载 Whisper {model_size}（首次运行会下载模型）...")
        model = WhisperModel(model_size, device="cpu", compute_type="int8",
                             cpu_threads=cpu_threads)
        segments, info = model.transcribe(
            str(audio_path), language="zh", beam_size=5, vad_filter=True
        )
        print(f"[转写] 语言: {info.language}（置信度 {info.language_probability:.0%}）")
        lines = []
        for seg in segments:
            if seg.text.strip():
                lines.append(f"[{seg.start:05.1f}s] {seg.text.strip()}")
        return "\n".join(lines)

    # ── 长录音：分块并行转写 ──────────────────────────────────────────────────
    n_chunks       = math.ceil(total_secs / chunk_secs)
    actual_workers = min(max_workers, n_chunks)
    chunk_label = f"{chunk_secs // 60} 分钟" if chunk_secs >= 60 else f"{chunk_secs} 秒"
    print(
        f"[转写] 录音时长 {total_secs / 60:.1f} 分钟，分 {n_chunks} 块并发转写"
        f"（每块 {chunk_label}，并发 {actual_workers}）"
        f"，加载 Whisper {model_size}..."
    )

    with tempfile.TemporaryDirectory(prefix="meetingscribe_") as tmpdir:
        # 切块并写入临时 WAV 文件
        chunk_args = []
        with wave.open(str(audio_path), "rb") as wf:
            for i in range(n_chunks):
                start_f = int(i * chunk_secs * framerate)
                end_f   = min(int((i + 1) * chunk_secs * framerate), total_frames)
                wf.setpos(start_f)
                chunk_data = wf.readframes(end_f - start_f)

                chunk_path = Path(tmpdir) / f"chunk_{i:04d}.wav"
                with wave.open(str(chunk_path), "wb") as cw:
                    cw.setnchannels(n_channels)
                    cw.setsampwidth(sampwidth)
                    cw.setframerate(framerate)
                    cw.writeframes(chunk_data)

                chunk_args.append((str(chunk_path), float(i * chunk_secs), i))

        # 各线程独立加载模型实例，CTranslate2 推理期间释放 GIL，实现真正并行
        def _run_chunk(args):
            chunk_path_str, offset, idx = args
            m = WhisperModel(model_size, device="cpu", compute_type="int8",
                             cpu_threads=cpu_threads)
            segs, _ = m.transcribe(
                chunk_path_str, language="zh", beam_size=5, vad_filter=True
            )
            lines = [
                f"[{seg.start + offset:05.1f}s] {seg.text.strip()}"
                for seg in segs if seg.text.strip()
            ]
            return idx, lines

        results = [None] * n_chunks
        done_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {executor.submit(_run_chunk, args): args[2] for args in chunk_args}
            for future in concurrent.futures.as_completed(futures):
                idx, lines = future.result()
                results[idx] = lines
                done_count += 1
                print(f"[转写] 第 {idx + 1}/{n_chunks} 块完成")
                if on_progress:
                    on_progress(5 + int(done_count / n_chunks * 35))

    all_lines: list[str] = []
    for chunk_lines in results:
        all_lines.extend(chunk_lines)
    return "\n".join(all_lines)


def _transcribe_openai(audio_path: Path, pcfg: dict) -> str:
    import urllib.request, urllib.error, json as _json, uuid

    api_key = pcfg.get("api_key", "")
    model = pcfg.get("model", "whisper-1")
    url = pcfg.get("base_url", "https://api.openai.com/v1").rstrip("/") + "/audio/transcriptions"
    boundary = "----Boundary" + uuid.uuid4().hex

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="model"\r\n\r\n'
        f"{model}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="response_format"\r\n\r\ntext\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{audio_path.name}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + audio_data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    })
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.read().decode().strip()
    except urllib.error.HTTPError as e:
        print(f"[错误] OpenAI STT HTTP {e.code}: {e.read().decode()}")
        sys.exit(1)
    except urllib.error.URLError as e:
        if isinstance(e.reason, (TimeoutError, OSError)) and "timed out" in str(e.reason).lower():
            print("[错误] OpenAI STT 请求超时，录音文件可能过大")
        else:
            print(f"[错误] OpenAI STT 网络错误: {e.reason}")
        sys.exit(1)


def _transcribe_gemini(audio_path: Path, pcfg: dict) -> str:
    import base64, urllib.request, urllib.error, json as _json

    api_key = pcfg.get("api_key", "")
    model = pcfg.get("model", "gemini-2.0-flash")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models"
        f"/{model}:generateContent?key={api_key}"
    )
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    payload = _json.dumps({
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
                {"text": "请将这段音频完整转写为文字。直接输出转写内容，不要添加任何解释。"},
            ]
        }]
    }).encode()

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = _json.loads(resp.read())
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except urllib.error.HTTPError as e:
        print(f"[错误] Gemini STT HTTP {e.code}: {e.read().decode()}")
        sys.exit(1)
    except urllib.error.URLError as e:
        if isinstance(e.reason, (TimeoutError, OSError)) and "timed out" in str(e.reason).lower():
            print("[错误] Gemini STT 请求超时，录音文件可能过大")
        else:
            print(f"[错误] Gemini STT 网络错误: {e.reason}")
        sys.exit(1)
    except (ValueError, KeyError, IndexError) as e:
        print(f"[错误] Gemini STT 返回格式异常: {e}")
        sys.exit(1)


# ── Prompts ───────────────────────────────────────────────────────────────────

_POLISH_BASE = """\
以下是语音识别自动转写的文本，带有时间戳，可能存在错别字、同音字混淆、断句不当等问题。

请在**不改变原意**的前提下：
1. 去掉所有时间戳（如 [00.0s]）
2. 将所有片段合并为连贯的自然段落，按语义分段
3. 纠正明显的错别字和同音字错误
4. 修复错误的断句和标点
5. 删除重复内容——语音识别可能对相邻片段重复识别同一句话，检查前后句子，去掉重复的短语或句子
6. 无法确定的内容用【？】标注

只输出整理后的正文，不要解释修改内容。

---
【原始转写】
{transcript}
"""

PROMPTS = {
    "meeting": {
        "polish": "你是一位专业的文字校对助手，正在处理一段会议录音的转写文本。\n\n" + _POLISH_BASE,
        "notes_zh": """\
你是一位专业的会议纪要助手。请根据以下转写文本生成结构化会议纪要。

要求：
1. **会议概要** — 2~3 句话概括核心内容
2. **主要议题** — 逐条列出讨论的关键议题
3. **决策事项** — 明确达成的决定或共识
4. **行动项** — 格式：负责人 · 事项 · 截止时间（无明确信息则标"待确认"）
5. **关键洞察** — 值得记录的重要观点

用中文输出，格式为 Markdown，简洁清晰。若内容较短或不完整，如实说明。

---
【会议转写】
{transcript}
""",
        "notes_en": """\
You are a professional meeting notes assistant. Based on the following meeting transcript, generate structured meeting notes in English.

Requirements:
1. **Meeting Summary** — 2–3 sentences summarizing the core content
2. **Key Topics** — list each major topic discussed
3. **Decisions Made** — explicit decisions or consensus reached
4. **Action Items** — format: Owner · Task · Due Date (mark "TBD" if unknown)
5. **Key Insights** — notable observations worth recording

Output in English, Markdown format, concise and clear. If the content is short or incomplete, state so honestly.

---
[Meeting Transcript]
{transcript}
""",
    },
    "interview": {
        "polish": "你是一位专业的文字校对助手，正在处理一段面试录音的转写文本。如能区分面试官与候选人，请在段落前标注「面试官：」或「候选人：」。\n\n" + _POLISH_BASE,
        "notes_zh": """\
你是一位专业的面试评估助手。请根据以下面试转写文本生成结构化的面试总结。

要求：
1. **候选人概况** — 姓名（如提及）、应聘岗位、整体印象（2~3 句）
2. **核心问答摘要** — 按主题归纳关键问题与候选人的回答要点
3. **技术 / 专业能力** — 具体技能掌握程度、深度、广度
4. **综合素质** — 沟通表达、逻辑思维、学习能力、团队意识等
5. **亮点** — 突出表现或印象深刻的回答
6. **不足 / 待确认** — 回答模糊、经验欠缺或需进一步了解的方面
7. **专业能力评估** — 从专业知识、方案设计、项目管理、数据分析等维度逐项评估
8. **价值观评估** — 从客户成功、极客精神、快速交付、简单直接、多元兼容等维度逐项评估
9. **综合评价与建议** — 是否推荐进入下一轮，及理由

用中文及英文输出，格式为 Markdown，客观专业。若内容较短或不完整，如实说明。

---
【面试转写】
{transcript}
""",
        "notes_en": """\
You are a professional interview evaluation assistant. Based on the following interview transcript, generate a structured interview summary in English.

Requirements:
1. **Candidate Overview** — name (if mentioned), role applied for, overall impression (2–3 sentences)
2. **Q&A Summary** — key questions and candidate's responses, grouped by theme
3. **Technical / Professional Skills** — depth and breadth of specific skills demonstrated
4. **Soft Skills** — communication, logical thinking, learning ability, teamwork, etc.
5. **Highlights** — standout moments or particularly impressive answers
6. **Gaps / To Verify** — vague answers, lacking experience, or areas needing follow-up
7. **Professional Competency Assessment** — evaluate across dimensions such as domain knowledge, solution design, project management, and data analysis
8. **Values Assessment** — evaluate across dimensions such as customer success, geek spirit, fast delivery, simple & direct, and diversity & inclusion
9. **Overall Assessment & Recommendation** — whether to advance to next round, with reasoning

Output in English, Markdown format, objective and professional. If the content is short or incomplete, state so honestly.

---
[Interview Transcript]
{transcript}
""",
    },
}


def _llm_run(prompt: str, provider_name: str, cfg: dict, label: str) -> str:
    llm_cfgs = {**DEFAULT_CONFIG["llm"], **cfg.get("llm", {})}
    pcfg = llm_cfgs.get(provider_name)
    if pcfg is None:
        print(f"[错误] 未知 provider '{provider_name}'，请在 config llm 中配置")
        sys.exit(1)

    timeout = cfg.get("llm_timeout", DEFAULT_CONFIG["llm_timeout"])
    ptype = pcfg.get("type", provider_name)

    if ptype == "claude-cli":
        return _llm_claude_cli(prompt, label, timeout, model=pcfg.get("model", ""))
    elif ptype == "openai":
        return _llm_openai(prompt, pcfg, label, timeout)
    elif ptype == "gemini":
        return _llm_gemini(prompt, pcfg, label, timeout)
    else:
        print(f"[错误] 不支持的 provider type: {ptype}")
        sys.exit(1)


def _llm_claude_cli(prompt: str, label: str, timeout: int, model: str = "") -> str:
    cmd = ["claude", "-p", prompt]
    if model:
        cmd += ["--model", model]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        print("[错误] 找不到 claude 命令，请确认 Claude Code CLI 已安装且在 PATH 中")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"[错误] claude-cli 超时（{label}，{timeout}s）；可通过 config --set llm_timeout=900 调大")
        sys.exit(1)
    if result.returncode != 0:
        print(f"[错误] claude-cli 非零退出（{label}）:\n{result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def _llm_openai(prompt: str, pcfg: dict, label: str, timeout: int) -> str:
    import json as _json
    import urllib.request, urllib.error

    url = pcfg.get("base_url", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    api_key = pcfg.get("api_key", "")
    model = pcfg.get("model", "gpt-4o")

    payload = _json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = _json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        print(f"[错误] OpenAI HTTP {e.code}（{label}）: {e.read().decode()}")
        sys.exit(1)
    except urllib.error.URLError as e:
        if isinstance(e.reason, (TimeoutError, OSError)) and "timed out" in str(e.reason).lower():
            print(f"[错误] OpenAI 超时（{label}，{timeout}s）；可通过 config --set llm_timeout=900 调大")
        else:
            print(f"[错误] OpenAI 网络错误（{label}）: {e.reason}")
        sys.exit(1)
    except (ValueError, KeyError, IndexError) as e:
        print(f"[错误] OpenAI 返回格式异常（{label}）: {e}")
        sys.exit(1)


def _llm_gemini(prompt: str, pcfg: dict, label: str, timeout: int) -> str:
    import json as _json
    import urllib.request, urllib.error

    api_key = pcfg.get("api_key", "")
    model = pcfg.get("model", "gemini-1.5-pro")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models"
        f"/{model}:generateContent?key={api_key}"
    )
    payload = _json.dumps({
        "contents": [{"parts": [{"text": prompt}]}]
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = _json.loads(resp.read())
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except urllib.error.HTTPError as e:
        print(f"[错误] Gemini HTTP {e.code}（{label}）: {e.read().decode()}")
        sys.exit(1)
    except urllib.error.URLError as e:
        if isinstance(e.reason, (TimeoutError, OSError)) and "timed out" in str(e.reason).lower():
            print(f"[错误] Gemini 超时（{label}，{timeout}s）；可通过 config --set llm_timeout=900 调大")
        else:
            print(f"[错误] Gemini 网络错误（{label}）: {e.reason}")
        sys.exit(1)
    except (ValueError, KeyError, IndexError) as e:
        print(f"[错误] Gemini 返回格式异常（{label}）: {e}")
        sys.exit(1)


def polish_transcript(transcript: str, provider: str, cfg: dict, mode: str = "meeting") -> str:
    import concurrent.futures

    chunk_size = cfg.get("polish_chunk_size", DEFAULT_CONFIG["polish_chunk_size"])
    lines = transcript.splitlines()

    chunks, current, current_len = [], [], 0
    for line in lines:
        if current and current_len + len(line) > chunk_size:
            chunks.append("\n".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append("\n".join(current))

    total = len(chunks)
    if total == 0:
        return ""
    _w = cfg.get("polish_max_workers", DEFAULT_CONFIG["polish_max_workers"])
    max_workers = _w if _w > 0 else max(4, (os.cpu_count() or 8) // 2)
    print(f"[校对] 并行调用 {provider}（{mode} 模式，共 {total} 块，并发 {min(max_workers, total)}）...")

    def _run(i_chunk):
        i, chunk = i_chunk
        prompt = PROMPTS[mode]["polish"].format(transcript=chunk)
        result = _llm_run(prompt, provider, cfg, f"校对[{i}/{total}]")
        print(f"[校对] 第 {i}/{total} 块完成")
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_run, enumerate(chunks, 1)))

    return "\n\n".join(results)


def generate_notes(transcript: str, provider: str, cfg: dict, mode: str = "meeting") -> str:
    import concurrent.futures
    label = "面试总结" if mode == "interview" else "会议纪要"
    print(f"[{label}] 并行生成中英文版本（{provider}）...")
    prompt_zh = PROMPTS[mode]["notes_zh"].format(transcript=transcript)
    prompt_en = PROMPTS[mode]["notes_en"].format(transcript=transcript)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fut_zh = ex.submit(_llm_run, prompt_zh, provider, cfg, f"{label}(中文)")
        fut_en = ex.submit(_llm_run, prompt_en, provider, cfg, f"{label}(English)")
        notes_zh = fut_zh.result()
        notes_en = fut_en.result()
    divider = "\n\n---\n\n"
    return notes_zh + divider + notes_en


def transcribe_and_polish(
    audio_path: Path,
    transcribe_provider: str,
    polish_provider: str,
    cfg: dict,
    mode: str,
    on_progress=None,
) -> tuple[str, str]:
    """Overlap FunASR transcription and polish: submit each completed chunk for polishing immediately.
    Falls back to sequential for non-funasr providers.
    Returns (raw_transcript, polished_transcript).
    """
    import concurrent.futures

    if transcribe_provider != "funasr":
        raw = transcribe(audio_path, transcribe_provider, cfg, on_progress=on_progress)
        polished = polish_transcript(raw, polish_provider, cfg, mode)
        return raw, polished

    _w = cfg.get("polish_max_workers", DEFAULT_CONFIG["polish_max_workers"])
    max_polish_workers = _w if _w > 0 else max(4, (os.cpu_count() or 8) // 2)

    pending: list[tuple[int, concurrent.futures.Future]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_polish_workers + 1) as polish_ex:
        # Warm-up: send a tiny request concurrently with transcription so the first
        # real polish batch doesn't suffer the API cold-start penalty (~150s → ~60s).
        print(f"[预热] 并发预热 {polish_provider} API...")
        _warmup_fut = polish_ex.submit(_llm_run, "x", polish_provider, cfg, "预热")

        def on_chunk_done(text: str, idx: int):
            prompt = PROMPTS[mode]["polish"].format(transcript=text)
            print(f"[校对] 第 {idx + 1} 块转写完成，已提交校对...")
            fut = polish_ex.submit(_llm_run, prompt, polish_provider, cfg, f"校对[块{idx + 1}]")
            pending.append((idx, fut))

        raw_all = transcribe(audio_path, transcribe_provider, cfg,
                             on_progress=on_progress, on_chunk_done=on_chunk_done)

        try:
            _warmup_fut.result(timeout=30)
            print("[预热] 完成")
        except Exception as e:
            _log("WARN", f"polish warmup did not complete: {type(e).__name__}: {e}")

        total = len(pending)
        results_by_idx: dict[int, str] = {}
        for idx, fut in pending:
            results_by_idx[idx] = fut.result()
            print(f"[校对] 第 {idx + 1}/{total} 块完成")

    if not results_by_idx:
        polished = polish_transcript(raw_all, polish_provider, cfg, mode)
    else:
        polished = "\n\n".join(results_by_idx[i] for i in sorted(results_by_idx))

    return raw_all, polished


# ── 保存纪要 ──────────────────────────────────────────────────────────────────

def save_minutes(minutes: str, audio_path: Path) -> Path:
    note_path = audio_path.with_suffix(".md")
    note_path.write_text(minutes, encoding="utf-8")
    return note_path


# ── 子命令 ────────────────────────────────────────────────────────────────────

def _cmd_devices_raw(cfg):
    """Print full CoreAudio diagnostic dump + current resolver decision.

    Same data is also mirrored into the daily log under [DEVICE-RAW] so future
    bug reports can correlate the user's complaint with the device topology at
    that moment.
    """
    if sys.platform != "darwin":
        print("[--raw] CoreAudio 原始转储仅在 macOS 可用")
        return

    dump = _coreaudio_device_raw_dump()
    pa_channels: dict[str, tuple[int, int]] = {}
    try:
        for d in sd.query_devices():
            pa_channels[d["name"]] = (d["max_input_channels"], d["max_output_channels"])
    except Exception as e:
        print(f"[--raw] PortAudio 枚举失败: {type(e).__name__}: {e}")

    plan = resolve_audio_devices(query_fresh=True)
    transport = _coreaudio_device_info()

    sep = "=" * 70
    print(f"\n{sep}\n原始 CoreAudio 设备转储\n{sep}\n")
    for i, e in enumerate(dump, 1):
        in_ch, out_ch = pa_channels.get(e["name"], (0, 0))
        cls = e["class_id"] or "''"
        tran = e["transport"] or "''"
        is_agg = e["class_id"] == "aagg"
        is_phys = _is_physical_output(e["name"], transport)
        prio = _transport_priority(e["name"], transport)

        cls_warn = f"  ⚠ status={e['class_status']}" if e["class_status"] else ""
        tran_warn = f"  ⚠ status={e['transport_status']}" if e["transport_status"] else ""
        agg_tag = "  ⚠ AGGREGATE" if is_agg else ""

        print(f"[#{i}] {e['name']}")
        print(f"     AudioObjectID:        {e['id']}")
        print(f"     Class:                {cls!r}{cls_warn}{agg_tag}")
        print(f"     Transport:            {tran!r}{tran_warn}")
        print(f"     Input/Output ch:      {in_ch} in / {out_ch} out")
        if e["sub_uids"]:
            print(f"     Sub-device UIDs ({len(e['sub_uids'])}):")
            for uid in e["sub_uids"]:
                print(f"       - {uid}")
        print(f"     → _is_physical_output: {is_phys}")
        print(f"     → _transport_priority: {prio}"
              f"{'  (next/built-in)' if prio == 1 else '  (rejected)' if prio == 9999 else '  (preferred/external)'}")
        print()

    print(f"{sep}\n当前 resolver 决策 (resolve_audio_devices)\n{sep}")
    print(f"  mic_name:             {plan.mic_name}")
    print(f"  sys_source_name:      {plan.sys_source_name}")
    print(f"  multi_output_name:    {plan.multi_output_name}")
    print(f"  restore_output_name:  {plan.restore_output_name}")
    print(f"  is_external_output:   {plan.is_external_output}")
    print(f"  warnings:             {plan.warnings}")
    print()

    # Mirror to the daily log so the same snapshot is available for retro analysis.
    _log_device_raw_dump(reason="devices-cli-raw")
    _log(
        "DEVICE",
        f"--raw plan: mic={plan.mic_name!r} sys={plan.sys_source_name!r} "
        f"multi={plan.multi_output_name!r} restore={plan.restore_output_name!r} "
        f"warnings={plan.warnings}",
    )


def cmd_devices(args, cfg):
    if getattr(args, "raw", False):
        _, restore = _setup_log_file()
        try:
            _cmd_devices_raw(cfg)
        finally:
            restore()
        return

    transport = _coreaudio_device_info()
    devs = _resolve_devices(cfg)
    auto_sys = devs["device_system_audio"]
    auto_mic = devs["device_mic"]
    auto_out_rec = devs["output_record"]
    auto_out_rest = devs["output_restore"]

    print("\n可用音频输入设备：\n")
    for dev in sd.query_devices():
        if dev["max_input_channels"] < 1:
            continue
        name = dev["name"]
        t = transport.get(name, "?")
        tags = []
        if name == auto_sys:
            tags.append("系统音频✓")
        if name == auto_mic:
            tags.append("麦克风✓")
        label = f"  [{'/'.join(tags)}]" if tags else ""
        print(f"  {name}  ({t}){label}")

    print("\n可用音频输出设备：\n")
    for dev in sd.query_devices():
        if dev["max_output_channels"] < 1:
            continue
        name = dev["name"]
        t = transport.get(name, "?")
        tags = []
        if name == auto_out_rec:
            tags.append("录音时切换✓")
        if name == auto_out_rest:
            tags.append("录音后还原✓")
        label = f"  [{'/'.join(tags)}]" if tags else ""
        print(f"  {name}  ({t}){label}")
    print()


def cmd_record(args, cfg):
    _, restore = _setup_log_file()
    try:
        _cmd_record_body(args, cfg)
    finally:
        restore()


def _cmd_record_body(args, cfg):
    recordings_dir = CONFIG_DIR / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_path = recordings_dir / f"{ts}.wav"

    # Install hotplug listeners so plug/unplug during recording wakes the recorder
    # within ~100 ms instead of the 1 s polling fallback.
    _install_device_listeners()
    # Run the 1 Hz audio-device monitor for the lifetime of this CLI command
    # so idle-window device changes (pre-recording, post-recording before
    # process exit) trigger restores just like in GUI mode.
    _get_audio_monitor().start()
    try:
        # Fresh resolve at the lifecycle boundary (no streams open yet → safe to
        # terminate/initialize PortAudio for a clean device snapshot).
        plan = resolve_audio_devices(query_fresh=True)

        if not plan.sys_source_name:
            print("[警告] 未找到 BlackHole 设备，系统音频将无法录制。请安装 BlackHole 2ch。", file=sys.stderr)
        if not plan.multi_output_name:
            print(
                "[提示] 未检测到 Multi-Output Device。请在『音频 MIDI 设置』里手动创建一个 "
                "包含 BlackHole 和你的扬声器/耳机的多输出设备，以便录音时仍能听到声音。",
                file=sys.stderr,
            )

        # Reset per-session gate at start-of-lifecycle so the start-time
        # switch decision below drives the _recording_did_switch flag.
        _recording_did_switch.clear()
        _log("AUDIO", "session gate cleared (did_switch=False at start-of-lifecycle)")

        # Snapshot raw CoreAudio topology before the dOut decision so logs
        # carry the exact class/transport state if a wrong restore later fires.
        _log_device_raw_dump(reason="recording-start:cli")

        # Switch the macOS default output to the Multi-Output Device so playing
        # apps route through both BlackHole and the user's speakers/headphones.
        # If dOut is already the Multi-Output Device (user has it as permanent
        # default), skip the switch entirely — no event delivered to audio
        # apps, no music pause, and the stop path will also be a no-op
        # (gated by _recording_did_switch).
        _prev_dout = _get_current_output_device()
        if plan.multi_output_name and _prev_dout != plan.multi_output_name:
            switch_output(plan.multi_output_name)
            _recording_did_switch.set()
            _log("AUDIO", f"start switch: from={_prev_dout!r} to={plan.multi_output_name!r} performed=True")
            print(f"[音频] 输出已切换至: {plan.multi_output_name}")
            print("[音频] 等待 1 秒让播放器重新路由...")
            time.sleep(1)
        else:
            _log("AUDIO", f"start switch: from={_prev_dout!r} to={plan.multi_output_name!r} performed=False")

        # Silence the Multi-Output's inactive physical sub-devices so the user
        # hears audio only through plan.restore_output_name (e.g. headphones
        # when plugged in). dOut is unchanged; BlackHole capture is unaffected.
        _reconcile_recording_mutes(plan)

        wanted = [n for n in (plan.sys_source_name, plan.mic_name) if n]
        role_labels = []
        if plan.sys_source_name:
            role_labels.append("system")
        if plan.mic_name:
            role_labels.append("mic")
        if not wanted:
            print("[错误] 没有可用的录音设备", file=sys.stderr)
            sys.exit(1)

        recorder = MultiStreamRecorder(wanted, cfg["sample_rate"], role_labels=role_labels)
        recorder.on_warning = lambda code: print(f"[警告] 录音异常: {code}", file=sys.stderr)
        recorder.on_device_added = lambda dev: print(f"[REC] 热插拔加入: {dev}", file=sys.stderr)

        print(f"\n[录音] 系统音频={plan.sys_source_name or '未找到'} | 麦克风={plan.mic_name or '未找到'}")
        print("[录音] 开始录音，按 Ctrl+C 停止...\n")

        recorder.start()
        start_time = time.time()

        try:
            while True:
                elapsed = int(time.time() - start_time)
                m, s = divmod(elapsed, 60)
                print(f"\r  ● 录音中  {m:02d}:{s:02d}", end="", flush=True)
                time.sleep(0.5)
        except KeyboardInterrupt:
            print()

        recorder.stop()
        duration = time.time() - start_time

        # Restore mute state on Multi-Output sub-devices BEFORE the dOut
        # restore, so any speakers we silenced come back at the moment dOut
        # returns to point at the user's physical output.
        _restore_all_recording_mutes()

        # Restore policy: skip switch_output if the current default is already a
        # physical device — switching pauses music apps. Re-detect current
        # physical device (the headphones the user might have plugged in /
        # unplugged mid-recording) for the actual restore target.
        stop_plan = resolve_audio_devices(query_fresh=True)
        restored = _restore_output_if_needed(stop_plan, reason="post-recording")
        if restored:
            print(f"[音频] 输出已还原至: {restored}")

        if not recorder.save(audio_path):
            print("[错误] 未录到任何音频")
            sys.exit(1)

        print(f"[录音] 完成，时长 {duration:.0f}s → {audio_path}\n")
    finally:
        # Defensive: idempotent if the normal stop path already restored mutes;
        # critical if an exception interrupted the recording before that ran.
        try:
            _restore_all_recording_mutes()
        except Exception as e:
            _log("ERR", f"cmd_record restore_mutes: {type(e).__name__}: {e}")
        try:
            _get_audio_monitor().stop()
        except Exception as e:
            _log("ERR", f"cmd_record monitor.stop: {type(e).__name__}: {e}")
        _remove_device_listeners()

    mode = getattr(args, "mode", None) or cfg.get("mode", "meeting")
    if mode not in PROMPTS:
        print(f"[错误] 未知模式 '{mode}'，可选：{list(PROMPTS)}")
        sys.exit(1)

    transcribe_provider  = getattr(args, "transcribe_provider", None)  or cfg.get("transcribe_provider", "funasr")
    polish_provider      = getattr(args, "polish_provider", None)       or cfg.get("polish_provider", "claude")
    notes_provider       = getattr(args, "meeting_notes_provider", None) or cfg.get("meeting_notes_provider", "claude")

    raw_txt_path = audio_path.with_name(audio_path.stem + ".raw.txt")
    polish_path = audio_path.with_name(audio_path.stem + ".polish.txt")

    need_transcribe = not raw_txt_path.exists()
    need_polish = not polish_path.exists()

    if need_transcribe and need_polish:
        transcript_raw, transcript_polished = transcribe_and_polish(
            audio_path, transcribe_provider, polish_provider, cfg, mode)
        raw_txt_path.write_text(transcript_raw, encoding="utf-8")
        polish_path.write_text(transcript_polished, encoding="utf-8")
    elif need_transcribe:
        print(f"[校对] 检测到已有校对文件 {polish_path.name}，跳过校对")
        transcript_raw = transcribe(audio_path, transcribe_provider, cfg)
        raw_txt_path.write_text(transcript_raw, encoding="utf-8")
        transcript_polished = polish_path.read_text(encoding="utf-8")
    elif need_polish:
        print(f"[转写] 检测到已有转写文件 {raw_txt_path.name}，跳过转写")
        transcript_raw = raw_txt_path.read_text(encoding="utf-8")
        transcript_polished = polish_transcript(transcript_raw, polish_provider, cfg, mode)
        polish_path.write_text(transcript_polished, encoding="utf-8")
    else:
        print(f"[转写] 检测到已有转写文件 {raw_txt_path.name}，跳过转写")
        print(f"[校对] 检测到已有校对文件 {polish_path.name}，跳过校对")
        transcript_raw = raw_txt_path.read_text(encoding="utf-8")
        transcript_polished = polish_path.read_text(encoding="utf-8")

    print(f"\n[校对] 已保存: {polish_path}")
    print("\n── 校对后转写 " + "─" * 46)
    print(transcript_polished)
    print("─" * 60)

    notes_label = "面试总结" if mode == "interview" else "会议纪要"
    notes = generate_notes(transcript_polished, notes_provider, cfg, mode)
    print(f"\n── {notes_label} " + "─" * (58 - len(notes_label)))
    print(notes)
    print("─" * 60)
    note_path = save_minutes(notes, audio_path)
    print(f"\n✅ 完成！{notes_label}已保存: {note_path}")


def cmd_transcribe(args, cfg):
    _, restore = _setup_log_file()
    try:
        _cmd_transcribe_body(args, cfg)
    finally:
        restore()


def _cmd_transcribe_body(args, cfg):
    input_path = Path(args.file)
    if not input_path.exists():
        print(f"[错误] 文件不存在: {input_path}")
        sys.exit(1)

    mode = getattr(args, "mode", None) or cfg.get("mode", "meeting")
    if mode not in PROMPTS:
        print(f"[错误] 未知模式 '{mode}'，可选：{list(PROMPTS)}")
        sys.exit(1)

    transcribe_provider  = getattr(args, "transcribe_provider", None)   or cfg.get("transcribe_provider", "funasr")
    polish_provider      = getattr(args, "polish_provider", None)        or cfg.get("polish_provider", "claude")
    notes_provider       = getattr(args, "meeting_notes_provider", None) or cfg.get("meeting_notes_provider", "claude")

    # 支持直接传入 .raw.txt 或 .polish.txt，或传 .wav 自动检测跳过已完成步骤
    if input_path.suffix == ".txt" and input_path.stem.endswith(".polish"):
        # 直接传入 .polish.txt，跳过转写和校对
        audio_path = input_path.with_name(input_path.stem[:-10] + ".wav")
        polish_path = input_path
        print(f"[校对] 使用已有校对文件: {polish_path.name}（跳过转写和校对）")
        transcript_polished = polish_path.read_text(encoding="utf-8")
        transcript_raw = None
    elif input_path.suffix == ".txt" and input_path.stem.endswith(".raw"):
        # 直接传入 .raw.txt，跳过转写
        audio_path = input_path.with_name(input_path.stem[:-4] + ".wav")
        raw_txt_path = input_path
        polish_path = audio_path.with_name(audio_path.stem + ".polish.txt")
        print(f"[转写] 使用已有转写文件: {raw_txt_path.name}（跳过转写）")
        transcript_raw = raw_txt_path.read_text(encoding="utf-8")
        transcript_polished = None
    else:
        audio_path = input_path
        raw_txt_path = audio_path.with_name(audio_path.stem + ".raw.txt")
        polish_path = audio_path.with_name(audio_path.stem + ".polish.txt")
        need_transcribe = not raw_txt_path.exists()
        need_polish = not polish_path.exists()
        if need_transcribe and need_polish:
            transcript_raw, transcript_polished = transcribe_and_polish(
                audio_path, transcribe_provider, polish_provider, cfg, mode)
            raw_txt_path.write_text(transcript_raw, encoding="utf-8")
            polish_path.write_text(transcript_polished, encoding="utf-8")
        elif need_transcribe:
            print(f"[校对] 检测到已有校对文件 {polish_path.name}，跳过校对")
            transcript_raw = transcribe(audio_path, transcribe_provider, cfg)
            raw_txt_path.write_text(transcript_raw, encoding="utf-8")
            transcript_polished = polish_path.read_text(encoding="utf-8")
        else:
            print(f"[转写] 检测到已有转写文件 {raw_txt_path.name}，跳过转写")
            transcript_raw = raw_txt_path.read_text(encoding="utf-8")
            transcript_polished = None

    if transcript_polished is None:
        if polish_path.exists():
            print(f"[校对] 检测到已有校对文件 {polish_path.name}，跳过校对")
            transcript_polished = polish_path.read_text(encoding="utf-8")
        else:
            transcript_polished = polish_transcript(transcript_raw, polish_provider, cfg, mode)
            polish_path.write_text(transcript_polished, encoding="utf-8")

    notes_label = "面试总结" if mode == "interview" else "会议纪要"
    print(f"\n[校对] 已保存: {polish_path}")
    print("\n── 校对后转写 " + "─" * 46)
    print(transcript_polished)
    print("─" * 60)

    notes = generate_notes(transcript_polished, notes_provider, cfg, mode)
    print(f"\n── {notes_label} " + "─" * (58 - len(notes_label)))
    print(notes)
    print("─" * 60)
    note_path = save_minutes(notes, audio_path)
    print(f"\n✅ 完成！{notes_label}已保存: {note_path}")


def cmd_config(args, cfg):
    if args.set:
        key, _, value = args.set.partition("=")
        key = key.strip()
        value = value.strip()
        # 尝试转成数字
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                pass
        cfg[key] = value
        save_config(cfg)
        print(f"✅ {key} = {value}")
    else:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))


# ── 桌面 UI (Tkinter) ────────────────────────────────────────────────────────

def cmd_ui(args, cfg):
    _, restore = _setup_log_file()
    try:
        _cmd_ui_body(args, cfg)
    finally:
        restore()


def _cmd_ui_body(args, cfg):  # noqa: C901
    import queue as _q
    import threading as _t
    try:
        import tkinter as tk
        from tkinter import ttk, scrolledtext, filedialog, messagebox
    except ImportError:
        print("[错误] tkinter 不可用，请确认 Python 安装包含 tkinter")
        sys.exit(1)

    # ── 配色（灰蓝深色主题）────────────────────────────────────────────────────
    BG      = "#1c2b3a"
    CARD    = "#233447"
    BORDER  = "#334d68"
    ACCENT  = "#5b9fd6"
    SUCCESS = "#4aad8a"
    DANGER  = "#d9534f"
    TEXT    = "#c8d8eb"
    MUTED   = "#6888a8"
    BTN     = "#2a3f58"
    LANG_ON = "#8bafc8"   # 语言切换按钮激活背景

    # ── 状态 ──────────────────────────────────────────────────────────────────
    st = {
        "status": "idle",   # idle | recording | processing | done | error
        "recorder": None,
        "audio_path": None,
        "chosen_path": None,
        "result_path": None,
        "lang": "zh",
    }
    log_q = _q.Queue()
    timer_job = [None]
    timer_secs = [0]
    cancel_flag = [False]
    pipeline_running = [False]
    vol_device = [None]   # physical output device to target for volume

    # Install CoreAudio HAL listeners so headphone plug/unplug fires the recorder
    # monitor's hotplug event within ~100 ms. Removed in _on_close. No-op on
    # non-macOS, where the recorder's 1 s polling remains the only mechanism.
    _install_device_listeners()

    # Start the dedicated 1 Hz audio-device monitor. Handles idle-state restore
    # (first tick + on every device-set change) and signals _hotplug_event on
    # device changes during a recording. Stopped in _on_close.
    _audio_monitor_instance = _get_audio_monitor()

    def _on_recording_plan_change(plan: "AudioPlan"):
        """Monitor-thread callback: rebinds the volume slider to the new
        listening target when the user hotplugs mid-recording. Mutates the
        vol_device cell (GIL-safe) then schedules a slider refresh on the Tk
        main thread via root.after(0, ...) — the established cross-thread
        pattern in this app."""
        new_dev = plan.restore_output_name
        if not new_dev or new_dev == vol_device[0]:
            return
        vol_device[0] = new_dev
        try:
            root.after(0, _sync_vol_slider)
        except Exception as e:
            _log("ERR", f"vol slider hotplug refresh schedule: {type(e).__name__}: {e}")

    _audio_monitor_instance.on_recording_plan_change = _on_recording_plan_change
    _audio_monitor_instance.start()

    # Safety net for Ctrl+C / terminal close: window-close handler may not fire.
    # We can't undo switch_output() cross-process, but we can at least re-detect
    # the current physical device and restore the media default to it if the
    # current default is still an aggregate/virtual device.
    def _atexit_restore():
        _log("AUDIO", "atexit restore invoked")
        try:
            _restore_all_recording_mutes()
        except Exception as e:
            _log("ERR", f"atexit restore_mutes: {type(e).__name__}: {e}")
        try:
            _restore_output_if_needed(resolve_audio_devices(query_fresh=False), reason="post-recording")
        except Exception as e:
            _log("ERR", f"atexit restore: {type(e).__name__}: {e}")
        try:
            _remove_device_listeners()
        except Exception as e:
            _log("ERR", f"atexit remove_listeners: {type(e).__name__}: {e}")

    atexit.register(_atexit_restore)
    vol_updating = [False]  # suppress feedback loop when slider is set programmatically

    # ── i18n ──────────────────────────────────────────────────────────────────
    TR = {
        "zh": dict(
            start="▶   开始录音", stop="◼   停止录音",
            choose="选择录音文件", chosen_none="未选择",
            chosen_prefix="当前选择文件：",
            action_meeting="开始整理会议纪要",
            action_interview="开始整理面试记录",
            stop_task="◼   停止任务",
            ready="就绪", recording="录音中…",
            processing="处理中…", done="✓  完成",
            error="✕  出错", open_result="打开结果文件",
            log_title="LOG",
            devices_label="监听设备",
            no_device="请至少选择一个录音设备",
            no_file="请先选择录音文件",
            open_audio_midi="打开 Audio MIDI 设置",
            open_sound_settings="打开声音设置",
        ),
        "en": dict(
            start="▶   Start Recording", stop="◼   Stop Recording",
            choose="Choose .wav File", chosen_none="None selected",
            chosen_prefix="Selected: ",
            action_meeting="Generate Meeting Notes",
            action_interview="Generate Interview Summary",
            stop_task="◼   Stop Task",
            ready="Ready", recording="Recording…",
            processing="Processing…", done="✓  Done",
            error="✕  Error", open_result="Open Result",
            log_title="LOG",
            devices_label="Input Devices",
            no_device="Please select at least one input device",
            no_file="Please select a recording file first",
            open_audio_midi="Audio MIDI Setup",
            open_sound_settings="Sound Settings",
        ),
    }

    def t(key):
        return TR[st["lang"]][key]

    # ── 根窗口 ─────────────────────────────────────────────────────────────────
    root = tk.Tk()
    root.title("MeetingScribe")
    root.geometry("580x820")
    root.resizable(False, False)
    root.configure(bg=BG)

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TProgressbar",
                    troughcolor=BORDER, background=ACCENT,
                    darkcolor=ACCENT, lightcolor=ACCENT, thickness=4)

    def sep(parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=20, pady=6)

    def card(parent, pady=(0, 0), expand=False):
        outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
        outer.pack(fill="both" if expand else "x", padx=20, pady=pady, expand=expand)
        inner = tk.Frame(outer, bg=CARD, padx=18, pady=14)
        inner.pack(fill="both" if expand else "x", expand=expand)
        return inner

    # ── 辅助函数 ───────────────────────────────────────────────────────────────

    def _show_log_line(msg):
        """Display in UI log box only — for messages already routed through sys.stdout
        (which the _setup_log_file tee has persisted to the log file)."""
        ts = datetime.now().strftime("%H:%M:%S")
        log_box.configure(state="normal")
        log_box.insert("end", f"[{ts}] {msg}\n")
        log_box.see("end")
        log_box.configure(state="disabled")

    def add_log(msg):
        """Display in UI log box AND emit via stdout so the log file captures it."""
        _show_log_line(msg)
        print(msg)

    pbar_pct = [0]

    def _draw_progress(pct=None):
        if pct is not None:
            pbar_pct[0] = max(0, min(100, pct))
        pbar_canvas.update_idletasks()
        w = pbar_canvas.winfo_width()
        h = pbar_canvas.winfo_height()
        if w <= 1:
            root.after(50, lambda: _draw_progress())
            return
        pbar_canvas.delete("all")
        pbar_canvas.create_rectangle(0, 0, w, h, fill=BORDER, outline="")
        fill_w = int(w * pbar_pct[0] / 100)
        if fill_w > 0:
            color = DANGER if pbar_pct[0] < 100 else SUCCESS
            pbar_canvas.create_rectangle(0, 0, fill_w, h, fill=color, outline="")
        pbar_canvas.create_text(w // 2, h // 2, text=f"{pbar_pct[0]:.0f}%",
                                fill=TEXT, font=("Menlo", 10, "bold"))

    def _set_action_btns(enabled: bool):
        state = "normal" if enabled else "disabled"
        btn_meeting.configure(state=state, fg=ACCENT, activeforeground=ACCENT)
        btn_interview.configure(state=state, fg=ACCENT, activeforeground=ACCENT)

    def set_lang(lang):
        st["lang"] = lang
        is_rec = st["status"] == "recording"
        rec_btn.configure(text=t("stop") if is_rec else t("start"))
        choose_btn.configure(text=t("choose"))
        btn_meeting.configure(text=t("action_meeting"))
        btn_interview.configure(text=t("action_interview"))
        stop_btn.configure(text=t("stop_task"))
        open_btn.configure(text=t("open_result"))
        if _btn_midi:
            _btn_midi.configure(text=t(_btn_midi_key))
        if st.get("chosen_path"):
            chosen_var.set(t("chosen_prefix") + Path(st["chosen_path"]).name)
        else:
            chosen_var.set(t("chosen_none"))

    def toggle_record():
        if st["status"] == "idle":
            _start_recording()
        elif st["status"] == "recording":
            _stop_recording()

    def _vol_icon(val):
        if val == 0:   return "🔇"
        if val <= 40:  return "🔉"
        if val <= 70:  return "🔊"
        return "📢"

    def _sync_vol_slider():
        if sys.platform != "darwin":
            return
        d = vol_device[0]
        if not d:
            return
        v = get_device_volume(d)
        if v is not None:
            try:
                vol_updating[0] = True
                pct = int(v * 100)
                vol_slider.set(pct)
                vol_pct_var.set(f"{pct}%")
                vol_icon_var.set(_vol_icon(pct))
            finally:
                vol_updating[0] = False

    def _on_vol_change(val_str):
        if vol_updating[0]:
            return
        val = int(float(val_str))
        vol_pct_var.set(f"{val}%")
        vol_icon_var.set(_vol_icon(val))
        if vol_device[0]:
            try:
                set_device_volume(vol_device[0], val / 100.0)
            except Exception as e:
                _log("ERR", f"set_device_volume({vol_device[0]!r}, {val}): {type(e).__name__}: {e}")

    def _tick():
        timer_secs[0] += 1
        s = timer_secs[0]
        timer_var.set(f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}")
        timer_job[0] = root.after(1000, _tick)

    def _start_recording():
        # Fresh resolve at the lifecycle boundary — no streams open yet, so it's
        # safe to terminate/initialize PortAudio for an accurate device snapshot.
        plan = resolve_audio_devices(query_fresh=True)
        add_log(
            f"[设备] 系统音频={plan.sys_source_name or '未找到'} | "
            f"麦克风={plan.mic_name or '未找到'} | "
            f"录音输出={plan.multi_output_name or '无 (用 BlackHole)'} | "
            f"还原至={plan.restore_output_name or '未知'} "
            f"({'external' if plan.is_external_output else 'built-in'})"
        )
        if not plan.multi_output_name:
            add_log(
                "[提示] 未检测到 Multi-Output Device。请在『音频 MIDI 设置』里手动创建一个 "
                "包含 BlackHole 和你的扬声器/耳机的多输出设备，否则录音时听不到声音。"
            )

        # Reset per-session gate at start-of-lifecycle so the start-time
        # switch decision below drives the _recording_did_switch flag.
        _recording_did_switch.clear()
        _log("AUDIO", "session gate cleared (did_switch=False at start-of-lifecycle)")

        # Snapshot raw CoreAudio topology before the dOut decision so logs
        # carry the exact class/transport state if a wrong restore later fires.
        _log_device_raw_dump(reason="recording-start:ui")

        # Switch to the Multi-Output Device only if we're not already on it.
        # If we're already on it (back-to-back recordings, or user has Multi-
        # Output as their permanent macOS default), skipping the switch means
        # the music app sees no device change → no playback pause. The stop
        # path will also be a no-op because _recording_did_switch stays clear.
        _prev_dout = _get_current_output_device()
        if plan.multi_output_name and _prev_dout != plan.multi_output_name:
            switch_output(plan.multi_output_name)
            _recording_did_switch.set()
            _log("AUDIO", f"start switch: from={_prev_dout!r} to={plan.multi_output_name!r} performed=True")
        else:
            _log("AUDIO", f"start switch: from={_prev_dout!r} to={plan.multi_output_name!r} performed=False")

        # Silence the Multi-Output's inactive physical sub-devices so the user
        # hears audio only through plan.restore_output_name (e.g. headphones
        # when plugged in). dOut is unchanged; BlackHole capture is unaffected.
        _reconcile_recording_mutes(plan)

        recordings_dir = CONFIG_DIR / "recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = recordings_dir / f"{ts}.wav"

        # Mark recording immediately (recorder=None until stream opens after settle delay)
        # so a second click can't call _start_recording again.
        st.update(status="recording", recorder=None, audio_path=audio_path,
                  plan=plan)
        rec_btn.configure(text=t("stop"), fg=DANGER, activeforeground=DANGER)
        timer_secs[0] = 0
        timer_var.set("00:00:00")
        timer_lbl.configure(fg=DANGER)
        timer_job[0] = root.after(1000, _tick)
        _set_action_btns(False)
        _draw_progress(0)
        add_log(f"[REC] {audio_path.name}")
        if sys.platform == "darwin":
            vol_device[0] = plan.restore_output_name
            root.after(1200, _sync_vol_slider)

        def _abort(msg: str):
            add_log(msg)
            if timer_job[0]:
                root.after_cancel(timer_job[0])
                timer_job[0] = None
            # If recorder.start() partially succeeded then raised, _recording_active
            # may be left set. Idempotent clear here returns the AudioDeviceMonitor
            # to its idle branch.
            _recording_active.clear()
            # Defensive: idempotent. If reconcile happened before the failure,
            # restore mutes so the user doesn't end up with a silenced speaker
            # while no recording is running.
            try:
                _restore_all_recording_mutes()
            except Exception as e:
                _log("ERR", f"abort restore_mutes: {type(e).__name__}: {e}")
            # Don't switch back here — _restore_output_if_needed in _on_close will
            # handle that. Switching mid-flow risks pausing music apps that may
            # still be migrating their stream off the Multi-Output Device.
            rec_btn.configure(text=t("start"), fg=ACCENT, activeforeground=ACCENT)
            timer_lbl.configure(fg=TEXT)
            timer_var.set("00:00:00")
            _set_action_btns(True)
            st.update(status="idle")

        def _do_start():
            if st.get("status") != "recording":
                return  # user cancelled before this fired

            # Build wanted from the plan; rebuild liveness from a fresh device list
            # (no PortAudio terminate needed — settle delay above already let macOS
            # propagate the Multi-Output Device's input channels).
            try:
                with _portaudio_lock:
                    _avail = {
                        d["name"] for d in sd.query_devices()
                        if d["max_input_channels"] >= 1
                    }
            except Exception:
                _avail = set()

            wanted = [n for n in (plan.sys_source_name, plan.mic_name) if n]
            role_labels = []
            if plan.sys_source_name:
                role_labels.append("system")
            if plan.mic_name:
                role_labels.append("mic")

            if not any(n in _avail for n in wanted):
                _abort(f"[ERR] {t('no_device')}")
                return

            recorder = MultiStreamRecorder(wanted, cfg["sample_rate"], role_labels=role_labels)
            recorder.on_device_added = lambda dev: add_log(f"[REC] 热插拔：{dev} 已加入录音")
            recorder.on_warning = lambda code: add_log(f"[警告] 录音异常: {code}")
            try:
                recorder.start()
            except Exception as e:
                _abort(f"[ERR] 录音设备启动失败: {e}")
                return
            for msg in recorder.skipped:
                add_log(f"[警告] 跳过设备: {msg}")
            opened = list(recorder._streams)
            add_log(f"[REC] 已开流: {' + '.join(opened) if opened else '等待设备...'}")
            st["recorder"] = recorder

        # 立即启动：麦克风通常马上可用；聚合设备若未出现，monitor 线程每秒重试
        root.after(0, _do_start)

    def _stop_recording():
        if timer_job[0]:
            root.after_cancel(timer_job[0])
        recorder = st.get("recorder")
        if recorder:
            recorder.stop()
        # Restore mute state on Multi-Output sub-devices BEFORE the dOut
        # restore, so any silenced speakers come back at the moment dOut
        # returns to the user's physical output.
        try:
            _restore_all_recording_mutes()
        except Exception as e:
            _log("ERR", f"stop restore_mutes: {type(e).__name__}: {e}")
        # Restore dOut (and sOut, if also non-physical) to the user's actual
        # physical device. Writing dOut is what brings the hardware volume keys
        # (F11/F12, Touch Bar, menu-bar slider) back to life — they follow dOut,
        # and the Multi-Output Device has no master volume control.
        #
        # Side-effect: music apps that watch for default-device changes (Apple
        # Music, Spotify) may pause briefly. _restore_output_if_needed skips
        # the dOut write when dOut is already physical, so back-to-back
        # recordings (which leave dOut on the Multi-Output Device) still
        # trigger the switch once per stop — accepted trade-off.
        plan: AudioPlan | None = st.get("plan")
        if sys.platform == "darwin":
            try:
                restored = _restore_output_if_needed(
                    resolve_audio_devices(query_fresh=True),
                    reason="post-recording",
                )
                if restored:
                    add_log(f"[音频] 输出已还原至: {restored}")
            except Exception as e:
                _log("ERR", f"stop output restore: {type(e).__name__}: {e}")
            target = plan.restore_output_name if plan else None
            vol_device[0] = target or _get_current_output_device()
            root.after(100, _sync_vol_slider)
        audio_path = st.get("audio_path")
        saved = recorder and audio_path and recorder.save(audio_path)
        if not saved:
            add_log("[ERR] 未录到任何音频（可能在设备初始化完成前停止）" if not recorder else "[ERR] 未录到任何音频")
            st.update(status="idle", recorder=None)
            rec_btn.configure(text=t("start"), fg=ACCENT, activeforeground=ACCENT)
            timer_lbl.configure(fg=TEXT)
            _set_action_btns(True)
            return
        add_log(f"[REC] 完成 → {audio_path.name}")
        st.update(status="idle", recorder=None, chosen_path=audio_path)
        chosen_var.set(t("chosen_prefix") + audio_path.name)
        rec_btn.configure(text=t("start"), fg=ACCENT, activeforeground=ACCENT)
        timer_lbl.configure(fg=TEXT)
        _set_action_btns(True)

    def choose_file():
        path = filedialog.askopenfilename(
            title=t("choose"),
            filetypes=[("WAV", "*.wav")],
        )
        if path:
            st["chosen_path"] = Path(path)
            chosen_var.set(t("chosen_prefix") + Path(path).name)
            _set_action_btns(True)

    def stop_pipeline():
        cancel_flag[0] = True
        pipeline_running[0] = False
        stop_btn.pack_forget()
        action_row.pack(fill="x")
        _draw_progress(0)
        st["status"] = "idle"
        rec_btn.configure(state="normal")
        _set_action_btns(True)
        add_log("[STOP] 已停止任务")

    def _start_pipeline(mode: str):
        path = st.get("chosen_path")
        if not path:
            messagebox.showwarning(t("choose"), t("no_file"))
            return
        if pipeline_running[0]:
            return
        cancel_flag[0] = False
        pipeline_running[0] = True
        rec_btn.configure(state="disabled")
        _set_action_btns(False)
        st["status"] = "processing"
        _draw_progress(0)
        open_btn.pack_forget()
        action_row.pack_forget()
        stop_btn.pack(fill="x")
        _t.Thread(target=_run_pipeline, args=(path, mode), daemon=True).start()

    def _run_pipeline(input_path: Path, mode: str):
        class _Tee:
            def __init__(self, orig):
                # Unwrap nested _Tee so repeated pipelines never double-log, but
                # preserve the _TimestampedStdout wrap from _setup_log_file so its
                # log file keeps receiving pipeline output.
                real = orig
                while isinstance(real, _Tee):
                    real = real._orig
                self._orig, self._buf = real, ""
                self._lock = _t.Lock()
            def write(self, s):
                self._orig.write(s)
                with self._lock:
                    self._buf += s
                    while "\n" in self._buf:
                        line, self._buf = self._buf.split("\n", 1)
                        if line.strip():
                            log_q.put(("log", line))
            def flush(self): self._orig.flush()
            def fileno(self): return self._orig.fileno()

        old_out = sys.stdout
        sys.stdout = _Tee(old_out)
        try:
            tp  = cfg.get("transcribe_provider", "funasr")
            pp  = cfg.get("polish_provider", "claude")
            np_ = cfg.get("meeting_notes_provider", "claude")

            audio_path = input_path
            raw_txt   = audio_path.with_name(audio_path.stem + ".raw.txt")
            polish_path = audio_path.with_name(audio_path.stem + ".polish.txt")

            log_q.put(("progress", 5))
            need_transcribe = not raw_txt.exists()
            need_polish = not polish_path.exists()

            def _transcribe_progress(pct):
                log_q.put(("progress", pct))

            if need_transcribe and need_polish:
                transcript_raw, transcript_polished = transcribe_and_polish(
                    audio_path, tp, pp, cfg, mode, on_progress=_transcribe_progress)
                raw_txt.write_text(transcript_raw, encoding="utf-8")
                polish_path.write_text(transcript_polished, encoding="utf-8")
            elif need_transcribe:
                print(f"[校对] 检测到 {polish_path.name}，跳过校对")
                transcript_raw = transcribe(audio_path, tp, cfg, on_progress=_transcribe_progress)
                raw_txt.write_text(transcript_raw, encoding="utf-8")
                transcript_polished = polish_path.read_text(encoding="utf-8")
            elif need_polish:
                print(f"[转写] 检测到 {raw_txt.name}，跳过转写")
                transcript_raw = raw_txt.read_text(encoding="utf-8")
                transcript_polished = polish_transcript(transcript_raw, pp, cfg, mode)
                polish_path.write_text(transcript_polished, encoding="utf-8")
            else:
                print(f"[转写] 检测到 {raw_txt.name}，跳过转写")
                print(f"[校对] 检测到 {polish_path.name}，跳过校对")
                transcript_raw = raw_txt.read_text(encoding="utf-8")
                transcript_polished = polish_path.read_text(encoding="utf-8")

            log_q.put(("progress", 85))
            notes = generate_notes(transcript_polished, np_, cfg, mode)
            note_path = save_minutes(notes, audio_path)
            print(f"✓ 完成 → {note_path}")
            log_q.put(("done", str(note_path)))
        except SystemExit:
            log_q.put(("error", ""))  # actual error already in log via _Tee; just trigger UI reset
        except Exception as e:
            # Print the full traceback to stdout (captured by the _Tee + log file)
            # so the user can pinpoint the failure on the next run.
            import traceback
            traceback.print_exc()
            log_q.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            sys.stdout = old_out
            pipeline_running[0] = False

    def open_result():
        path = st.get("result_path")
        if path:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.run(["open", path])
            else:
                subprocess.run(["xdg-open", path])

    def _poll():
        try:
            while True:
                kind, msg = log_q.get_nowait()
                if kind == "log":
                    _show_log_line(msg)
                elif kind == "progress":
                    _draw_progress(msg)
                elif kind == "done":
                    stop_btn.pack_forget()
                    action_row.pack(fill="x")
                    rec_btn.configure(state="normal")
                    _set_action_btns(True)
                    if not cancel_flag[0]:
                        _draw_progress(100)
                        st.update(status="idle", result_path=msg)
                        open_btn.pack(padx=20, pady=(8, 4))
                elif kind == "error":
                    stop_btn.pack_forget()
                    action_row.pack(fill="x")
                    rec_btn.configure(state="normal")
                    _set_action_btns(True)
                    if not cancel_flag[0]:
                        if msg:
                            add_log(f"[ERR] {msg}")
                        _draw_progress(0)
                        st["status"] = "idle"
        except _q.Empty:
            pass
        root.after(100, _poll)

    # ── Widgets ───────────────────────────────────────────────────────────────

    # ① Header
    hdr = tk.Frame(root, bg=BG)
    hdr.pack(fill="x", padx=20, pady=(18, 10))
    tk.Label(hdr, text="MEETINGSCRIBE",
             font=("Menlo", 14, "bold"), bg=BG, fg=ACCENT).pack(side="left")
    lang_row = tk.Frame(hdr, bg=BG)
    lang_row.pack(side="right")
    if sys.platform == "darwin":
        _btn_midi_key = "open_audio_midi"
        _btn_midi = tk.Button(
            lang_row, text="", relief="flat", bd=0, padx=8, pady=3,
            font=("Menlo", 11), cursor="hand2", bg=BTN, fg=MUTED,
            activebackground=BTN, activeforeground=ACCENT,
            command=lambda: subprocess.run(
                ["open", "/System/Applications/Utilities/Audio MIDI Setup.app"]
            ),
        )
        _btn_midi.pack(side="left", padx=(0, 8))
    elif sys.platform == "win32":
        _btn_midi_key = "open_sound_settings"
        _btn_midi = tk.Button(
            lang_row, text="", relief="flat", bd=0, padx=8, pady=3,
            font=("Menlo", 11), cursor="hand2", bg=BTN, fg=MUTED,
            activebackground=BTN, activeforeground=ACCENT,
            command=lambda: subprocess.run(["control", "mmsys.cpl"]),
        )
        _btn_midi.pack(side="left", padx=(0, 8))
    else:
        _btn_midi_key = None
        _btn_midi = None
    btn_lang = tk.Button(lang_row, text="中文/EN", relief="flat", bd=0, padx=8, pady=3,
                         font=("Menlo", 11), cursor="hand2", bg=BTN, fg=MUTED,
                         activebackground=LANG_ON, activeforeground="#1c2b3a",
                         command=lambda: set_lang("en" if st["lang"] == "zh" else "zh"))
    btn_lang.pack(side="left")

    sep(root)

    # ② 录音
    rc = card(root, pady=(0, 0))
    timer_var = tk.StringVar(value="00:00:00")
    timer_lbl = tk.Label(rc, textvariable=timer_var, bg=CARD, fg=TEXT,
                         font=("Menlo", 48, "bold"))
    timer_lbl.pack(pady=(4, 12))
    rec_btn = tk.Button(rc, text="", font=("Menlo", 13, "bold"),
                        bg=BTN, fg=ACCENT, activebackground=BTN, activeforeground=ACCENT,
                        relief="flat", bd=0, cursor="hand2",
                        padx=32, pady=11, command=toggle_record)
    rec_btn.pack(pady=(0, 4))

    def _make_vol_slider(parent):
        TRACK_H, THUMB_R, H = 6, 9, 24
        PAD = THUMB_R + 2
        _val = [50]

        cv = tk.Canvas(parent, height=H, bg=CARD, highlightthickness=0, bd=0,
                       cursor="hand2")

        def _draw():
            cv.delete("all")
            w = cv.winfo_width() or 200
            cy = H // 2
            x0, x1 = PAD, w - PAD
            fx = x0 + (_val[0] / 100.0) * max(x1 - x0, 1)
            cv.create_rectangle(x0, cy - TRACK_H // 2, x1, cy + TRACK_H // 2,
                                 fill=BORDER, outline="")
            if fx > x0:
                cv.create_rectangle(x0, cy - TRACK_H // 2, fx, cy + TRACK_H // 2,
                                     fill=ACCENT, outline="")
            cv.create_oval(fx - THUMB_R, cy - THUMB_R,
                           fx + THUMB_R, cy + THUMB_R,
                           fill=ACCENT, outline=BG, width=2)

        def _set_val(x):
            w = cv.winfo_width() or 200
            frac = (x - PAD) / max(w - 2 * PAD, 1)
            new_val = int(max(0.0, min(1.0, frac)) * 100)
            _val[0] = new_val
            _draw()
            _on_vol_change(str(new_val))

        cv.bind("<Button-1>", lambda e: _set_val(e.x))
        cv.bind("<B1-Motion>", lambda e: _set_val(e.x))
        cv.bind("<Configure>", lambda e: _draw())

        class _Slider:
            def get(self): return _val[0]
            def set(self, v): _val[0] = int(v); _draw()
            def pack(self, **kw): cv.pack(**kw)

        return _Slider()

    if sys.platform == "darwin":
        vol_row = tk.Frame(rc, bg=CARD)
        vol_row.pack(fill="x", pady=(0, 10))
        vol_icon_var = tk.StringVar(value="🔊")
        tk.Label(vol_row, textvariable=vol_icon_var, bg=CARD, fg=TEXT,
                 font=("Menlo", 12)).pack(side="left")
        vol_slider = _make_vol_slider(vol_row)
        vol_slider.pack(side="left", expand=True, fill="x", padx=6)
        vol_pct_var = tk.StringVar(value="—")
        tk.Label(vol_row, textvariable=vol_pct_var, bg=CARD, fg=MUTED,
                 font=("Menlo", 11), width=4, anchor="e").pack(side="left")
        vol_device[0] = _get_current_output_device()
        root.after(300, _sync_vol_slider)

    sep(root)

    # ③ 选择录音文件
    fc = card(root, pady=(0, 0))
    file_row = tk.Frame(fc, bg=CARD)
    file_row.pack(fill="x")
    choose_btn = tk.Button(file_row, text="", font=("Menlo", 11),
                           bg=BTN, fg=ACCENT, activebackground=BTN, activeforeground=ACCENT,
                           relief="flat", bd=0, cursor="hand2",
                           padx=8, pady=5, command=choose_file)
    choose_btn.pack(side="left")
    chosen_var = tk.StringVar()
    tk.Label(file_row, textvariable=chosen_var, bg=CARD, fg=ACCENT,
             font=("Menlo", 11), wraplength=320).pack(side="left", padx=12)

    sep(root)

    # ④ 整理模式（action 按钮 左右排列）
    ac = card(root, pady=(0, 0))
    action_row = tk.Frame(ac, bg=CARD)
    action_row.pack(fill="x")
    btn_meeting = tk.Button(action_row, text="", font=("Menlo", 12, "bold"),
                            bg=BTN, fg=ACCENT, activebackground=BTN,
                            activeforeground=ACCENT, disabledforeground=MUTED,
                            relief="flat", bd=0, cursor="hand2",
                            padx=12, pady=11, state="normal",
                            command=lambda: _start_pipeline("meeting"))
    btn_meeting.pack(side="left", expand=True, fill="x", padx=(0, 4))
    btn_interview = tk.Button(action_row, text="", font=("Menlo", 12, "bold"),
                              bg=BTN, fg=ACCENT, activebackground=BTN,
                              activeforeground=ACCENT, disabledforeground=MUTED,
                              relief="flat", bd=0, cursor="hand2",
                              padx=12, pady=11, state="normal",
                              command=lambda: _start_pipeline("interview"))
    btn_interview.pack(side="left", expand=True, fill="x")
    stop_btn = tk.Button(ac, text="停止任务", font=("Menlo", 12, "bold"),
                         bg=BTN, fg=DANGER, activebackground=BTN,
                         activeforeground=DANGER,
                         relief="flat", bd=0, cursor="hand2",
                         padx=20, pady=11, command=stop_pipeline)
    # stop_btn intentionally not packed — shown only while pipeline runs

    # ⑤ 进度条（Canvas，窄条，显示百分比）
    sep(root)
    pbar_frame = tk.Frame(root, bg=BG)
    pbar_frame.pack(fill="x", padx=20, pady=(4, 4))
    pbar_canvas = tk.Canvas(pbar_frame, height=20, bg=CARD,
                            highlightthickness=1, highlightbackground=BORDER)
    pbar_canvas.pack(fill="x")
    pbar_canvas.bind("<Configure>", lambda e: _draw_progress())
    root.after(100, lambda: _draw_progress(0))

    sep(root)

    # ⑧ 结果（完成后显示）— 必须在日志卡片之前创建，否则 expand=True 的日志会把它挤到窗口外
    result_frame = tk.Frame(root, bg=BG)
    result_frame.pack(fill="x")
    open_btn = tk.Button(result_frame, text="",
                         font=("Menlo", 12, "bold"),
                         bg=BTN, fg=ACCENT,
                         activebackground=BTN, activeforeground=ACCENT,
                         relief="flat", bd=0, cursor="hand2",
                         padx=28, pady=11, command=open_result)
    # open_btn intentionally not packed — shown only after pipeline completes

    # ⑦ 日志
    lc = card(root, pady=(0, 0), expand=True)
    log_box = scrolledtext.ScrolledText(
        lc, height=8, font=("Menlo", 11),
        bg="#111d2a", fg=MUTED, insertbackground=ACCENT,
        relief="flat", bd=0, state="disabled", wrap="word",
        selectbackground=BORDER,
    )
    log_box.pack(fill="both", expand=True)
    log_box.configure(state="normal")
    log_box.insert("end", "LOG\n")
    log_box.configure(state="disabled")

    # ── 关闭时清理 ─────────────────────────────────────────────────────────────
    def _on_close():
        # Best-effort: stop any in-flight recording, then restore the media
        # output to the user's actual physical device. Re-resolves at this exact
        # moment so headphones plugged in or unplugged mid-session are honoured.
        _log("REC", "GUI _on_close invoked")
        recorder = st.get("recorder")
        if recorder:
            try:
                recorder.stop()
            except Exception as e:
                _log("ERR", f"on_close recorder.stop: {type(e).__name__}: {e}")
        # Stop the audio-device monitor BEFORE _restore_output_if_needed so its
        # next tick can't race with the final restore call.
        try:
            _get_audio_monitor().stop()
        except Exception as e:
            _log("ERR", f"on_close monitor.stop: {type(e).__name__}: {e}")
        try:
            _restore_all_recording_mutes()
        except Exception as e:
            _log("ERR", f"on_close restore_mutes: {type(e).__name__}: {e}")
        try:
            _restore_output_if_needed(resolve_audio_devices(query_fresh=True), reason="post-recording")
        except Exception as e:
            _log("ERR", f"on_close restore: {type(e).__name__}: {e}")
        try:
            _remove_device_listeners()
        except Exception as e:
            _log("ERR", f"on_close remove_listeners: {type(e).__name__}: {e}")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)

    # ── 初始化并启动 ───────────────────────────────────────────────────────────
    set_lang("zh")
    _poll()
    root.mainloop()


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="一行命令：录音 + 转写 + 纪要",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", metavar="<命令>")

    stt_help = "可选：funasr（默认）/ whisper / openai / gemini，或 stt 配置中的任意 key"
    llm_help = "可选：claude / openai / gemini，或 llm 配置中的任意 key"

    mode_help = "运行模式：meeting（会议纪要，默认）| interview（面试总结）"

    # --mode / --debug 可放在子命令之前（全局）或之后（子命令级），两种写法均有效
    parser.add_argument("--mode", metavar="MODE", help=mode_help)
    parser.add_argument("--debug", action="store_true", help="启用调试日志（每秒打印各路音频电平到 stderr）")

    p_rec = sub.add_parser("record", help="开始录音（Ctrl+C 停止）")
    p_rec.add_argument("--mode", metavar="MODE", help=mode_help, default=argparse.SUPPRESS)
    p_rec.add_argument("--transcribe-provider", metavar="PROVIDER", help=f"语音转文字模型，{stt_help}")
    p_rec.add_argument("--polish-provider", metavar="PROVIDER", help=f"转写校对模型，{llm_help}")
    p_rec.add_argument("--meeting-notes-provider", metavar="PROVIDER", help=f"纪要/总结模型，{llm_help}")

    p_tr = sub.add_parser("transcribe", help="转写已有音频文件")
    p_tr.add_argument("file", help="音频文件路径")
    p_tr.add_argument("--mode", metavar="MODE", help=mode_help, default=argparse.SUPPRESS)
    p_tr.add_argument("--transcribe-provider", metavar="PROVIDER", help=f"语音转文字模型，{stt_help}")
    p_tr.add_argument("--polish-provider", metavar="PROVIDER", help=f"转写校对模型，{llm_help}")
    p_tr.add_argument("--meeting-notes-provider", metavar="PROVIDER", help=f"纪要/总结模型，{llm_help}")

    sub.add_parser("ui", help="打开桌面图形界面")

    p_dev = sub.add_parser("devices", help="列出可用音频设备")
    p_dev.add_argument(
        "--raw",
        action="store_true",
        help="打印每个设备的 ClassID / transport / sub-device UID 等原始诊断信息，并写入日志",
    )

    p_cfg = sub.add_parser("config", help="查看或修改配置")
    p_cfg.add_argument("--set", metavar="key=value", help="设置配置项")

    args = parser.parse_args()

    global _DEBUG, _LOG_TO_CONSOLE
    _DEBUG = getattr(args, "debug", False)
    _LOG_TO_CONSOLE = _DEBUG

    cfg = load_config()

    dispatch = {
        "record": cmd_record,
        "transcribe": cmd_transcribe,
        "ui": cmd_ui,
        "devices": cmd_devices,
        "config": cmd_config,
    }

    if args.cmd in dispatch:
        dispatch[args.cmd](args, cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
