#!/usr/bin/env python3
"""
meetingscribe.py — 录音 → 转写 → 校对 → 纪要/面试总结/分享总结

━━━ 模式 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  meeting   会议纪要模式（默认）
  interview 面试总结模式
  sharing   分享总结模式（知识分享 / 技术 talk / 最佳实践讲座）

━━━ 录音 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # 默认模式录音（Ctrl+C 停止），自动完成转写 → 校对 → 纪要
  python3 meetingscribe.py record

  # 面试模式
  python3 meetingscribe.py record --mode interview
  python3 meetingscribe.py record --mode interview --title "后端工程师面试"

  # 分享总结模式（多主讲人 + 问答）
  python3 meetingscribe.py record --mode sharing

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
  python3 meetingscribe.py transcribe audio.wav --mode sharing
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
  --transcribe-provider    funasr（默认，本地）| openai | gemini
  --polish-provider        claude（默认）| openai | gemini
  --meeting-notes-provider claude（默认）| openai | gemini

━━━ 输出文件（与录音 / 输入文件同目录）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  <stem>.wav            录音文件
  <stem>.raw.txt        原始转写
  <stem>.polish.txt     校对后转写（会议纪要/面试总结/分享总结的输入）
  <stem>.meeting.md     会议纪要（mode=meeting）
  <stem>.interview.md   面试报告（mode=interview）
  <stem>.sharing.md     分享总结（mode=sharing）
"""

import argparse
import atexit
import collections
import copy
import html
import json
import math
import os
import queue
import re
import signal
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


class _LogFileHandler:
    """Bridge Python ``logging`` records into the shared daily log file.

    Third-party libraries (modelscope, jieba, funasr, ...) emit through the
    ``logging`` module. Their default StreamHandlers grabbed the original
    sys.stderr at import time, so our sys.stderr swap in _setup_log_file()
    doesn't catch them. This handler is installed on the root logger inside
    _setup_log_file() and writes WARNING+ records into the same file as
    ``_log()`` — sharing _log_file_lock so writes interleave cleanly.

    Note: not a logging.Handler subclass to avoid importing ``logging`` at
    module load. Duck-typed: only ``level``, ``handle()``, ``createLock()``,
    ``acquire()``, ``release()`` are exercised by logging.Logger.callHandlers.
    """

    def __init__(self, level):
        self.level = level
        self._lock = threading.RLock()

    def createLock(self):
        pass  # already created in __init__

    def acquire(self):
        self._lock.acquire()

    def release(self):
        self._lock.release()

    def handle(self, record):
        if record.levelno < self.level:
            return
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            try:
                msg = record.getMessage()
            except Exception:
                msg = str(record.msg)
            line = f"[{ts}] [LOG-{record.levelname}] {record.name}: {msg}\n"
            with _log_file_lock:
                if _log_file_handle is not None:
                    _log_file_handle.write(line)
                    _log_file_handle.flush()
        except Exception:
            pass


def _dbg(msg: str):
    """Deprecated: prefer _log(category, message). Kept as a thin shim that
    forwards uncategorised diagnostic messages to the log file (and to stderr
    when --debug is enabled). Will be removed once all call sites migrate."""
    _log("DEBUG", msg)


# ANSI colour codes plus tqdm's bar / rate syntax. tqdm redraws with \r, and
# str.splitlines() splits on \r too, so a single 60-second FunASR call lands
# as hundreds of near-identical bar frames in the log (measured: ~5 000 of
# the 8 583 [CAPTION] lines in one 30-minute caption session, 1.3 MB/day).
# The frames carry no diagnostic value — timings we actually care about are
# logged explicitly by their call sites — so _QuietCapture drops them.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
_PROGRESS_NOISE_RE = re.compile(
    r"^\s*\d{1,3}%\s*\|"            # "  0%|" / "100%|" bar prefix
    r"|\d+/\d+\s*\[\d{2}:\d{2}"     # " 1/1 [00:01<00:00, …"
    r"|\?it/s|\bit/s\]|\bs/it\]"    # rate suffixes, incl. the "?it/s" start
)


def _is_progress_noise(line: str) -> bool:
    """True for tqdm bar frames / bare terminal control leftovers."""
    return not line or bool(_PROGRESS_NOISE_RE.search(line))


class _QuietCapture:
    """Context manager: redirect stdout+stderr into in-memory buffers, then
    forward the captured lines to _log(category, ...) on exit. Used around
    third-party libraries (FunASR) whose tqdm progress bars
    and per-frame timing dicts would otherwise drown the console. The captured
    text still lands in the daily log file, just not in front of the user —
    except tqdm progress frames, which are dropped entirely (see
    `_is_progress_noise`): they made the caption log unreadable and unsearchable.
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
        # Restore ONLY what we installed. sys.stdout is process-global, so a
        # concurrent capture on another thread may have replaced ours in the
        # meantime; overwriting it here would strand that capture's buffer as
        # the process's stdout and silently swallow everything printed
        # afterwards. Anything that must reach the terminal regardless should
        # hold its own stream handle (see `_cmd_captions_body`) rather than
        # trust sys.stdout — this class cannot make a global safe for threads.
        if sys.stdout is self._sio_out:
            sys.stdout = self._saved_out
        if sys.stderr is self._sio_err:
            sys.stderr = self._saved_err
        for buf in (self._sio_out, self._sio_err):
            text = buf.getvalue()
            for raw in text.splitlines():
                line = _ANSI_RE.sub("", raw).strip()
                if _is_progress_noise(line):
                    continue
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
# ASR hotwords live in their OWN file, gitignored. They are auto-mined from
# every meeting, so the list fills up with colleague names, customer names and
# internal codenames — personal data that has no business in a config file
# that ships with the project (this repo is public). config.jsonc keeps the
# key as an empty default; hotword.jsonc overrides it when present.
HOTWORD_FILE = Path(__file__).parent / "hotword.jsonc"


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

    # Capture Python `logging` records (modelscope / jieba / funasr / urllib3
    # ...) into the same file. Their existing StreamHandlers retain their
    # references to the original stderr — those still print to the terminal,
    # so the user-visible behavior is unchanged; we just additionally tee
    # WARNING+ records into the log file. INFO/DEBUG are filtered out to keep
    # the file readable; bump to logging.INFO here if more verbosity is
    # needed during diagnostics.
    import logging as _logging
    log_handler = _LogFileHandler(level=_logging.WARNING)
    _logging.root.addHandler(log_handler)

    def _restore():
        global _log_file_handle
        try:
            _logging.root.removeHandler(log_handler)
        except Exception:
            pass
        sys.stdout, sys.stderr = saved_out, saved_err
        with _log_file_lock:
            _log_file_handle = None
        try:
            fh.close()
        except Exception:
            pass

    return fh, _restore

_funasr_model_cache: dict = {}  # (asr_model, vad_model, punc_model) -> AutoModel instance


# ── FunASR jieba dict resilience ─────────────────────────────────────────────
#
# FunASR's punctuation model (iic/punc_ct-transformer_*) ships a jieba_usr_dict
# whose lines are <word> only (no frequency column). jieba 0.42.1's module-level
# load_userdict (jieba/__init__.py:307) crashes on such lines with
# `IndexError: list index out of range` because it executes
# `word, freq = tup[0], tup[1]` after splitting on " ". jieba is already at
# its latest release, so we cannot fix this by upgrading. Instead we self-heal:
# when AutoModel(...) raises an IndexError that came through jieba, we append
# " 1" to every malformed line in every jieba_usr_dict under the modelscope
# cache, then retry AutoModel once. The repair is idempotent (sentinel file)
# and crash-safe (atomic tmp + os.replace).


def _modelscope_cache_root() -> Path | None:
    """Honour MODELSCOPE_CACHE env var if set and existing, else ~/.cache/modelscope."""
    env = os.environ.get("MODELSCOPE_CACHE")
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p
    default = Path.home() / ".cache" / "modelscope"
    if default.is_dir():
        return default
    return None


def _patch_one_jieba_dict(path: Path) -> tuple[bool, int, int]:
    """Append ' 1' to every non-empty whitespace-free line in *path*.

    Idempotent (sentinel `.jieba_usr_dict.patched` next to the file) and
    crash-safe (atomic `tmp + os.replace`). On OSError, removes the partial
    .tmp and re-raises so the caller can decide whether to continue.

    Returns ``(modified, lines_fixed, lines_total)`` where ``modified`` is
    True iff the file's contents on disk actually changed.
    """
    sentinel = path.parent / ".jieba_usr_dict.patched"
    try:
        path_mtime = path.stat().st_mtime
    except OSError:
        return (False, 0, 0)
    if sentinel.exists():
        try:
            if sentinel.stat().st_mtime >= path_mtime:
                return (False, 0, 0)
        except OSError as e:
            _log("STT", f"jieba dict sentinel stat failed: {sentinel} err={e!r}; proceeding with re-patch")

    content = path.read_text(encoding="utf-8")
    lines = content.split("\n")
    fixed = 0
    total = 0
    for i, raw in enumerate(lines):
        stripped = raw.rstrip()
        if not stripped:
            continue
        total += 1
        if " " not in stripped and "\t" not in stripped:
            lines[i] = f"{stripped} 1"
            fixed += 1
    new_text = "\n".join(lines)

    if fixed == 0:
        try:
            sentinel.write_text("v1\n", encoding="utf-8")
        except OSError as e:
            _log("STT", f"jieba dict sentinel write failed: {sentinel} err={e!r}")
        return (False, 0, total)

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(new_text, encoding="utf-8")
        os.replace(str(tmp), str(path))
    except OSError as e:
        _log("STT", f"jieba dict patch failed: {path} err={e!r}")
        try:
            tmp.unlink(missing_ok=True)
        except OSError as cleanup_err:
            _log("STT", f"jieba dict tmp cleanup failed: {tmp} err={cleanup_err!r}")
        raise
    try:
        sentinel.write_text("v1\n", encoding="utf-8")
    except OSError as e:
        _log("STT", f"jieba dict sentinel write failed: {sentinel} err={e!r}")
    return (True, fixed, total)


def _patch_funasr_jieba_dicts() -> int:
    """Discover and repair every malformed jieba_usr_dict under the modelscope cache.

    Discovery is bounded to ``<cache>/hub/models/**/jieba_usr_dict`` and skips
    symlinks plus any file whose resolved path escapes the cache root.
    Returns the number of files actually modified (0 if cache is missing,
    nothing was malformed, or every per-file patch failed).
    """
    root = _modelscope_cache_root()
    if root is None:
        _log("STT", "jieba dict patch: no modelscope cache found; nothing to do")
        return 0
    models_root = root / "hub" / "models"
    if not models_root.is_dir():
        _log("STT", f"jieba dict patch: {models_root} does not exist; nothing to do")
        return 0

    try:
        cache_resolved = root.resolve()
    except OSError:
        cache_resolved = root

    candidates: list[Path] = []
    for p in models_root.glob("**/jieba_usr_dict"):
        if not p.is_file():
            continue
        if p.is_symlink():
            continue
        try:
            real = p.resolve()
        except OSError:
            continue
        try:
            real.relative_to(cache_resolved)
        except ValueError:
            continue
        candidates.append(p)

    _log(
        "STT",
        f"FunASR AutoModel raised IndexError from jieba; attempting "
        f"jieba_usr_dict repair on {len(candidates)} candidate(s) under {root}",
    )
    modified_count = 0
    already_count = 0
    for p in candidates:
        try:
            modified, fixed, total = _patch_one_jieba_dict(p)
        except OSError:
            continue
        if modified:
            _log("STT", f"jieba dict patch: {p} (added freq=1 to {fixed}/{total} lines)")
            modified_count += 1
        else:
            _log("STT", f"jieba dict already patched: {p}")
            already_count += 1
    _log(
        "STT",
        f"jieba dict patch complete: {modified_count} file(s) modified, "
        f"{already_count} already up to date",
    )
    return modified_count


def _indexerror_came_from_jieba(exc: BaseException) -> bool:
    """True iff *exc*'s traceback contains a frame from ``jieba/__init__.py``."""
    needle = os.sep + "jieba" + os.sep + "__init__.py"
    tb = exc.__traceback__
    while tb is not None:
        filename = tb.tb_frame.f_code.co_filename or ""
        if filename.endswith(needle):
            return True
        tb = tb.tb_next
    return False


def _load_funasr_automodel(asr_model: str, vad_model: str, punc_model: str,
                           spk_model: str = ""):
    """Construct ``funasr.AutoModel`` with self-healing for the jieba dict bug.

    On the first ``IndexError`` whose traceback originates in jieba, patch
    malformed ``jieba_usr_dict`` files under the modelscope cache and retry
    once with identical kwargs. If the retry fails, raise the *original*
    IndexError with the new failure chained via ``raise … from …``.

    ``spk_model`` (e.g. ``"cam++"``) adds speaker diarization: results then
    carry a ``sentence_info`` list whose entries have a ``spk`` cluster id.
    """
    from funasr import AutoModel
    kwargs = dict(
        model=asr_model, vad_model=vad_model, punc_model=punc_model,
        disable_update=True,
    )
    if spk_model:
        kwargs["spk_model"] = spk_model
    try:
        return AutoModel(**kwargs)
    except IndexError as original_err:
        if not _indexerror_came_from_jieba(original_err):
            raise
        n_patched = _patch_funasr_jieba_dicts()
        if n_patched == 0:
            _log("STT", "jieba dict patch produced 0 changes; re-raising original IndexError")
            raise
        _log("STT", f"Retrying FunASR AutoModel after patching {n_patched} jieba dict(s)")
        try:
            return AutoModel(**kwargs)
        except Exception as retry_err:
            raise original_err from retry_err


def _funasr_cache_key(asr_model: str, vad_model: str, punc_model: str,
                      spk_model: str = ""):
    """Key for `_funasr_model_cache`. Stays a 3-tuple without diarization so
    the batch pipeline and the live-caption refine pass keep SHARING one
    paraformer-large instance (~1 GB) instead of loading it twice."""
    base = (asr_model, vad_model, punc_model)
    return base + (spk_model,) if spk_model else base


def _spk_label(spk, order: dict) -> str:
    """Stable 1-based display label for a FunASR speaker cluster id.

    FunASR hands back ints (0-based) on some versions and strings on others,
    and clusters are numbered in the order the model happened to find them —
    so labels are assigned by first appearance in the transcript instead.
    """
    key = str(spk)
    if key not in order:
        order[key] = len(order) + 1
    return f"说话人{order[key]}"


def _spk_lines(items, offset_s: float = 0.0) -> "list[str]":
    """Format FunASR diarization output into `[12.3s] [说话人1] …` lines.

    Reads the ``sentence_info`` payload that appears when AutoModel was built
    with a ``spk_model``. Consecutive sentences from the same speaker merge
    into one line — one line per sentence would bury the turn structure the
    notes prompts are supposed to read. Returns ``[]`` when no usable
    sentence_info is present, so callers can fall back to unlabelled lines
    rather than losing the transcript to a FunASR version difference.
    """
    if not items:
        return []
    order: dict = {}
    lines: list[str] = []
    cur_label = None
    cur_start = None
    cur_texts: list[str] = []

    def _flush():
        if not cur_texts:
            return
        body = "".join(cur_texts).strip()
        if not body:
            return
        if cur_start is not None:
            lines.append(f"[{cur_start:05.1f}s] [{cur_label}] {body}")
        else:
            lines.append(f"[{cur_label}] {body}")

    for item in items:
        if not isinstance(item, dict):
            continue
        for sent in (item.get("sentence_info") or []):
            if not isinstance(sent, dict):
                continue
            text = str(sent.get("text") or "").strip()
            if not text:
                continue
            label = _spk_label(sent.get("spk", 0), order)
            start_s = None
            try:
                raw = sent.get("start")
                if raw is None:
                    ts = sent.get("timestamp")
                    raw = ts[0][0] if ts and len(ts[0]) else None
                if raw is not None:
                    start_s = float(raw) / 1000.0 + offset_s
            except (TypeError, ValueError, IndexError):
                start_s = None
            if label != cur_label:
                _flush()
                cur_label, cur_start, cur_texts = label, start_s, [text]
            else:
                if cur_start is None:
                    cur_start = start_s
                cur_texts.append(text)
    _flush()
    return lines


# ── Prompt defaults ──────────────────────────────────────────────────────────
#
# These are the built-in pipeline prompts used by `polish_transcript` and
# `generate_notes`. Users can override any subset via `config.jsonc`'s
# top-level ``"prompts"`` block; missing keys fall through to these
# defaults (see `_resolve_prompt`).
#
# Each prompt is a single Python string here. In `config.jsonc` the same
# content is also written as an **array of strings** (joined with ``"\n"``
# at load time) so multi-line prompts stay readable in JSONC — JSON itself
# forbids literal newlines inside a string.
#
# The literal token ``{transcript}`` is substituted with the chunk /
# transcript at runtime via ``str.replace`` (not ``str.format``), so other
# braces in the prompt do NOT need to be escaped as ``{{`` / ``}}``.

_PROMPT_DEFAULTS: dict = {
    # ``polish`` is mode-agnostic: the cleanup rules are the same whether
    # the transcript is from a meeting or an interview. The optional
    # speaker-labelling instruction is phrased generally so the LLM can
    # decide on a case-by-case basis (interviews → 面试官 / 候选人;
    # meetings → usually no labels because speakers blur).
    "polish": """\
你是一位专业的文字校对助手，正在处理一段录音的自动转写文本（可能来自会议、面试或其他场景）。

以下是语音识别自动转写的文本，带有时间戳，可能存在错别字、同音字混淆、断句不当等问题。

请在**不改变原意**的前提下：
1. 去掉所有时间戳（如 [00.0s]）
2. 将所有片段合并为连贯的自然段落，按语义分段
3. 纠正明显的错别字和同音字错误
4. 修复错误的断句和标点
5. 删除重复内容——语音识别可能对相邻片段重复识别同一句话，检查前后句子，去掉重复的短语或句子
6. 无法确定的内容用【？】标注
7. 如能可靠区分不同发言者，请在段落前标注角色或姓名（如「面试官：」/「候选人：」/「主持人：」/具体姓名）；若无法可靠区分则不要强行标注
8. 若某行带 [说话人N] 前缀（声纹聚类结果），必须保留发言人区分：同一个 N 视为同一个人，可结合上下文把它替换成姓名或角色（如「说话人2」→「候选人」）；不要把不同 N 合并成一个人，也不要因为整理段落而丢掉发言人标注

【术语表】以下是本场景常出现的专有名词（转写中与其发音相近的错误词请优先修正为表内写法；表内没有的不要强行替换）：
{hotwords}

只输出整理后的正文，不要解释修改内容。

---
【原始转写】
{transcript}
""",
    # ``hotwords_extract`` is mode-agnostic. Scheme B of the hotword
    # automation: after notes generation the pipeline mines the polished
    # transcript for ASR hotwords and merges them into
    # `stt.funasr.hotword` in config.jsonc, so the NEXT recording already
    # benefits. (Scheme A is the rule-based miner
    # `_extract_hotword_candidates` — no LLM involved.)
    "hotwords_extract": """\
你是一个语音识别热词提取助手。请从下面的文本中提取值得加入 ASR 热词表的词条，用于提升后续录音转写对专有名词的识别准确率。

提取范围：产品名、项目名、公司名、团队名、人名、技术术语、英文缩写、中英混合词（如 API网关）。

要求：
1. 只输出热词本身，全部放在同一行，词与词之间用单个空格分隔
2. 每个热词内部不能含空格；英文多词术语拆成单个单词分别输出
3. 优先选择语音识别容易写错的词，不要输出常见普通词汇
4. 最多 30 个；一个都没有时输出空行
5. 不要输出任何解释、编号、标点或其他内容

---
【文本】
{transcript}
""",
    # ``caption_mt`` drives the live captions' Qwen translation backend
    # (live_captions.mt_provider = "qwen"). Placeholders (literal
    # str.replace, same as {transcript} elsewhere): {src_lang} /
    # {dst_lang} = 中文/英文, {glossary} = 命中的热词术语表指令（可为空）,
    # {text} = 原文。The Marian / NLLB backends are not prompt-driven and
    # ignore this template.
    "caption_mt": """\
你是专业的同声传译引擎。请把下面这句{src_lang}口语翻译成{dst_lang}，保持口语化、简洁、自然；只输出译文，不要任何解释或标注。{glossary}

【原文】
{text}
""",
    # ``caption_fix_mt`` is the combined ASR-correction + translation
    # prompt for the Qwen caption backend (used for FINALIZED lines when
    # live_captions.qwen.correct is on; partials keep using caption_mt).
    # Placeholders: {src_lang} / {dst_lang} (中文/英文), {glossary}（术语表
    # 指令，可为空）, {context}（前几句定稿字幕，语境用）, {text}（原文）。
    # Output contract parsed by `_parse_fix_mt`: two lines 修正：/翻译：.
    "caption_fix_mt": """\
你是实时字幕的校对与同声传译引擎。输入是一句{src_lang}流式语音识别结果，可能有同音字错误或残缺的英文单词。
规则：
- 只修正确定的识别错误（同音字、残缺英文词），修正后的词要与原词发音相近；不确定的保持原样；不改写语义、不增删内容、不替换人名
- 「修正」行必须仍是{src_lang}（修正后的完整原文，不是翻译，不能丢掉任何分句）；「翻译」行才是{dst_lang}译文，口语化、简洁
{glossary}
上文语境（仅供理解，不要输出）：
{context}

输出格式示例（严格两行，无其他内容）：
修正：预算下周才能确认，我们先把接口文档写完
翻译：The budget won't be confirmed until next week; let's finish the API docs first.

【原文】
{text}
""",
    # ``caption_fix_mt_strict`` is the zero-shot twin of ``caption_fix_mt``,
    # used when GBNF grammar is active (live_captions.qwen.grammar, default
    # on): the two-line contract is then enforced by the llama.cpp decoder,
    # so the format example can go away — and it should, because a 1.5B model
    # copies example wording into its own output (measured failure mode) and
    # few-shot prompting underperforms zero-shot at this size. Same
    # placeholders and same `_parse_fix_mt` output contract as caption_fix_mt,
    # which stays in use when the grammar can't be built (older
    # llama-cpp-python) — there the example is what keeps the format honest.
    "caption_fix_mt_strict": """\
你是实时字幕的校对与同声传译引擎。输入是一句{src_lang}流式语音识别结果，可能有同音字错误或残缺的英文单词。
规则：
- 只修正确定的识别错误（同音字、残缺英文词），修正后的词要与原词发音相近；不确定的保持原样；不改写语义、不增删内容、不替换人名
- 「修正」行必须仍是{src_lang}（修正后的完整原文，不是翻译，不能丢掉任何分句）；「翻译」行才是{dst_lang}译文，口语化、简洁
{glossary}
上文语境（仅供理解，不要输出）：
{context}

先输出「修正：」行，再输出「翻译：」行；两行之外不要输出任何内容。

【原文】
{text}
""",
    # ``caption_review`` is the periodic batch re-check of live captions
    # (live_captions.review): every N minutes the finalized lines of the
    # last N minutes go to the SAME provider that polishes transcripts, with
    # the whole window as context — which is why it catches what the
    # per-line passes cannot. Placeholders: {hotwords} (the current
    # stt.funasr.hotword list), {lines} (numbered `N ||| 原文 ||| 当前译文`).
    # Output contract parsed by `_parse_caption_review`: one line per entry,
    # `N ||| 修正后的原文 ||| 译文`.
    "caption_review": """\
你是会议实时字幕的校对助手。下面是最近一段会议的流式识别字幕，按顺序编号，每条包含识别原文和当前译文。流式识别常有同音字错误、残缺英文单词、错误断句，译文也会因此跟着错。

请结合**整段上下文**和术语表，逐条给出修正后的原文与译文：
1. 只修正确定的识别错误（同音字、残缺或拼错的英文词、明显错词）；不确定的保持原样
2. 不要改写语气或风格，不要合并或拆分条目，不要增删内容——条数必须与输入完全一致
3. 人名、产品名、技术术语优先采用术语表中的写法
4. 译文必须与修正后的原文对应：中文原文译成英文，英文原文译成中文；口语化、简洁
5. 无需修改的条目也要原样输出
6. 术语和人名的写法必须与【上文】保持一致——同一个词在前文已确定的写法，本批不要再换一种写法

【术语表】（转写中与其发音相近的错误词请优先修正为表内写法；表内没有的不要强行替换）
{hotwords}

【上文】（本场前面已经校对完成的字幕，只作理解上下文和统一术语之用；不要输出这部分，也不要修改它们）
{context}

【字幕】（只校对并输出这一部分）
{lines}

严格按下面格式输出，每行一条，除此之外不要输出任何内容：
编号 ||| 修正后的原文 ||| 译文
""",
    "meeting": {
        "notes_zh": """\
你是一位专业的会议纪要助手。请根据以下转写文本生成结构化会议纪要。

要求：
1. **会议概要** — 2~3 句话概括核心内容
2. **主要议题** — 逐条列出讨论的关键议题
3. **决策事项** — 明确达成的决定或共识，整节合并为一个 `> [!important]` callout 块，条目在块内用列表罗列
4. **行动项** — 格式：负责人 · 事项 · 截止时间（无明确信息则标"待确认"），整节合并为一个 `> [!tip]` callout 块，条目在块内用列表罗列
5. **关键洞察** — 值得记录的重要观点
6. **风险与提醒** — 会议中提到的风险、阻塞点或需特别注意的事项，整节合并为一个 `> [!warning]` callout 块，条目在块内用列表罗列；无则省略本节

callout 语法示例（每节只用一个 callout：标记行 + 列表条目，每行都以 > 开头；不要为每个条目单独开 callout）：
> [!important]
> - 确定采用方案 B，下季度上线。
> - 数据迁移窗口定在 9 月第一周。

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
3. **Decisions Made** — explicit decisions or consensus reached, merged into one `> [!important]` callout block with a bullet list inside
4. **Action Items** — format: Owner · Task · Due Date (mark "TBD" if unknown), merged into one `> [!tip]` callout block with a bullet list inside
5. **Key Insights** — notable observations worth recording
6. **Risks & Reminders** — risks, blockers, or points needing special attention, merged into one `> [!warning]` callout block with a bullet list inside; omit this section if none

Callout syntax example (one callout per section: marker line + list items, every line starts with >; do NOT open a separate callout per item):
> [!important]
> - Adopt option B; ship next quarter.
> - Data migration window: first week of September.

Output in English, Markdown format, concise and clear. If the content is short or incomplete, state so honestly.

---
[Meeting Transcript]
{transcript}
""",
    },
    "sharing": {
        "notes_zh": """\
你是一位专业的分享/讲座整理助手。以下转写文本来自一次知识分享 / 技术分享 / 最佳实践讲座，可能有一位或多位主讲人详细分享内容，过程中或结尾通常会有听众提问、主讲人答疑。请生成一份既忠实记录分享内容、又便于事后查阅的整理稿。

要求（请按以下结构输出，使用 Markdown）：

1. **分享概览** — 2~4 句，说明本次分享的主题、主讲人（如可识别）、整体定位（介绍 / 经验复盘 / 教程 / 案例剖析等）。
2. **分享正文** — **这是本输出最重要的部分**。请按主讲人的讲述顺序，分小节（## 二级标题）忠实重述分享内容：
   - 保留主讲人提到的具体案例、数字、命令、代码片段、链接、人名、产品名、方法论名称等关键事实，不要为了"简洁"而抹掉。
   - 保留逻辑层次：主讲人怎么展开论点的（背景 → 问题 → 方案 → 验证 → 总结），就按那个顺序写。
   - 用通顺的书面中文重写口语化的表达，但**不要做主观加工或评价**；这是"详细文字版本"，不是"摘要"。
   - 如果分享内容较长，可以拆为多个小节，每节都用 `##` 二级标题。
3. **核心要点** — 用 3~7 条要点列出分享中最重要的结论 / 主张，每条 1~2 句，整节合并为一个 `> [!important]` callout 块，条目在块内用列表罗列。
4. **最佳实践 / 可复用经验** — 列出主讲人明确推荐的做法、踩过的坑、避免的陷阱，整节合并为一个 `> [!tip]` callout 块，条目在块内用列表罗列。无则写"未明确提及"。
5. **关键洞察** — 主讲人独到的观点或反直觉的结论；与行业常见做法的差异。无则写"无"。
6. **适用边界 / 前提条件** — 这些经验在什么场景下成立？依赖哪些技术 / 团队规模 / 业务特征？主讲人有没有明确说"不适用于 X"？
7. **风险与权衡** — 主讲人提到的限制、副作用、tradeoff，整节合并为一个 `> [!warning]` callout 块，条目在块内用列表罗列。无则写"未明确提及"。
8. **问答（Q&A）** — 把听众提问和主讲人回答配对列出。格式：
   - **问 [提问者，如已知]**：……
   - **答 [主讲人，如已知]**：……
   - 如果某个问题主讲人没有正面回答或当场承认不知道，明确标注"（未直接回答 / 待跟进）"。
   - 若没有问答环节，写"本次分享未包含问答环节"。
9. **行动建议 / 后续动作** — 听众可立即落地的 2~5 条建议；如果分享中提到了延伸阅读 / 工具 / 文档链接，整理在此处。

callout 语法示例（每节只用一个 callout：标记行 + 列表条目，每行都以 > 开头；不要为每个条目单独开 callout）：
> [!important]
> - 灰度发布应以场景为单位，而不是按服务切分。
> - 可观测性建设要先于自动化排障。

发言者标注规则：
- 如果上一步校对结果里已经标注了角色或姓名（如「主讲人：」「提问者 A：」「张三：」），请在「分享正文」「问答」中**保留这些标注**。
- 如果转写没有可靠区分发言者，**不要凭空捏造身份**；用"主讲人"、"听众"这样的通用称呼即可。
- 多位主讲人时分别标注，不要混为一谈。

输出语言：中文。Markdown 格式，专业、客观。若转写较短、不完整或没有问答，如实在对应小节里说明。

---
【分享转写】
{transcript}
""",
        "notes_en": """\
You are a professional knowledge-sharing / tech-talk summarisation assistant. The transcript below is from a sharing session, tech talk, or best-practices walkthrough delivered by one or more presenters. Audience members may have asked questions during or at the end of the talk. Produce a faithful, structured rendering of the talk that readers can use as a substitute for having attended.

Output structure (Markdown):

1. **Session Overview** — 2–4 sentences naming the topic, presenter(s) where identifiable, and the talk's positioning (introduction / experience report / tutorial / case study).
2. **Detailed Walkthrough** — **the most important section.** Following the presenter's own narrative order, retell the content in numbered or `##`-headed subsections:
   - Preserve every concrete fact mentioned: examples, numbers, commands, code snippets, links, names of people / products / methodologies. Do **not** drop them for the sake of brevity.
   - Preserve the logical structure (background → problem → approach → validation → takeaway), in the same order the presenter used.
   - Rewrite spoken English into clean written English, but do **not** add your own interpretation or evaluation. This section is a *long-form retelling*, not a summary.
   - Break the walkthrough into multiple subsections (`##` headings) if the talk is long.
3. **Key Points** — 3–7 bullets capturing the most important claims / conclusions, 1–2 sentences each, merged into one `> [!important]` callout block with a bullet list inside.
4. **Best Practices / Reusable Lessons** — list practices the presenter explicitly recommended, plus pitfalls they called out, merged into one `> [!tip]` callout block with a bullet list inside. Write "Not explicitly mentioned" if absent.
5. **Insights** — non-obvious takeaways, contrarian observations, points of divergence from common industry practice. Write "None" if absent.
6. **Applicability & Preconditions** — the contexts in which these practices hold (technology stack, team size, business shape). Did the presenter call out anything as **not** applicable?
7. **Risks & Trade-offs** — limitations, side effects, and trade-offs the presenter acknowledged, merged into one `> [!warning]` callout block with a bullet list inside. Write "Not explicitly mentioned" if absent.
8. **Q&A** — pair audience questions with the presenter's answers. Format:
   - **Q [asker, if known]:** …
   - **A [presenter, if known]:** …
   - If the presenter deflected, deferred, or acknowledged not knowing, mark "(not directly answered / follow-up needed)".
   - If there was no Q&A segment, write "This session had no Q&A segment."
9. **Action Items / Next Steps** — 2–5 actionable suggestions for the audience; collect any extended-reading links / tools / docs the presenter mentioned.

Callout syntax example (one callout per section: marker line + list items, every line starts with >; do NOT open a separate callout per item):
> [!important]
> - Roll out by scenario, not by service.
> - Build observability before automating diagnosis.

Speaker-attribution rules:
- If the polish step already labelled segments (e.g. "Presenter:", "Audience member A:", a specific name), **preserve those labels** in both the Walkthrough and the Q&A.
- If the transcript does not reliably distinguish speakers, do **not** invent identities. Use generic terms like "presenter" and "audience member".
- For multi-presenter talks, attribute each segment to the correct presenter; do not blend their voices.

Output language: English. Markdown format, professional and objective. If the transcript is short, incomplete, or has no Q&A, say so honestly in the relevant section.

---
[Sharing Transcript]
{transcript}
""",
    },
    "interview": {
        "notes_zh": """\
你是一位专业的面试评估助手。请根据以下面试转写文本生成结构化的面试总结。

要求：
1. **候选人概况** — 姓名（如提及）、应聘岗位、整体印象（2~3 句）
2. **核心问答摘要** — 按主题归纳关键问题与候选人的回答要点
3. **技术 / 专业能力** — 具体技能掌握程度、深度、广度
4. **综合素质** — 沟通表达、逻辑思维、学习能力、团队意识等
5. **亮点** — 突出表现或印象深刻的回答，整节合并为一个 `> [!tip]` callout 块，条目在块内用列表罗列
6. **不足 / 待确认** — 回答模糊、经验欠缺或需进一步了解的方面，整节合并为一个 `> [!warning]` callout 块，条目在块内用列表罗列
7. **专业能力评估** — 从专业知识、方案设计、项目管理、数据分析等维度逐项评估
8. **价值观评估** — 从客户成功、极客精神、快速交付、简单直接、多元兼容等维度逐项评估
9. **综合评价与建议** — 是否推荐进入下一轮，及理由，整节放入一个 `> [!important]` callout 块

callout 语法示例（每节只用一个 callout：标记行 + 内容，每行都以 > 开头；不要为每个条目单独开 callout）：
> [!tip]
> - 对分布式事务的回答深入，且有真实生产案例佐证。
> - 主动补充压测数据支撑方案选型。

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
5. **Highlights** — standout moments or particularly impressive answers, merged into one `> [!tip]` callout block with a bullet list inside
6. **Gaps / To Verify** — vague answers, lacking experience, or areas needing follow-up, merged into one `> [!warning]` callout block with a bullet list inside
7. **Professional Competency Assessment** — evaluate across dimensions such as domain knowledge, solution design, project management, and data analysis
8. **Values Assessment** — evaluate across dimensions such as customer success, geek spirit, fast delivery, simple & direct, and diversity & inclusion
9. **Overall Assessment & Recommendation** — whether to advance to next round, with reasoning, wrapped in one `> [!important]` callout block

Callout syntax example (one callout per section: marker line + content, every line starts with >; do NOT open a separate callout per item):
> [!tip]
> - Deep answer on distributed transactions, backed by a real production case.
> - Proactively cited load-test numbers to support the design choice.

Output in English, Markdown format, objective and professional. If the content is short or incomplete, state so honestly.

---
[Interview Transcript]
{transcript}
""",
    },
}


DEFAULT_CONFIG = {
    # ── 并发控制 / 性能 ────────────────────────────────────────────────────
    # LLM 调用超时（秒），长会议建议调大
    "llm_timeout": 1200,
    # 校对时单块最大字符数，超出则分块处理
    "polish_chunk_size": 3000,
    # 校对并发数（同时调用 LLM 的块数），0 = 自动(max(4, cpu核数/2))
    "polish_max_workers": 4,
    # ── 运行模式 ──────────────────────────────────────────────────────────
    # meeting（会议纪要）| interview（面试总结）
    "mode": "meeting",
    # ── 各环节 provider ───────────────────────────────────────────────────
    "transcribe_provider": "funasr",
    "polish_provider": "claude",
    "meeting_notes_provider": "claude",
    # ── 实时双语字幕（GUI 录音页；全本地推理）──────────────────────────────
    "live_captions": {
        "enabled": True,           # GUI 字幕面板开关的默认值
        # 识别链路只有一条：sherpa-onnx 流式 zipformer（中英双语）+ opus-mt 翻译，
        # 定稿行再用离线 FunASR 大模型做 refine 二次校验。原来的「准确模式」
        # （FunASR 流式 2pass + NLLB）已移除——它的流式模型是纯中文的，英文会话
        # 上明显更差，而它的 NLLB 翻译仍可通过 mt_provider="nllb" 单独选用。
        "asr_model_dir": "",       # 留空 = 首次使用时自动下载流式 zipformer 中英双语模型
        "mt_zh_en": "Helsinki-NLP/opus-mt-zh-en",
        "mt_en_zh": "Helsinki-NLP/opus-mt-en-zh",
        "refine": True,            # 句子定稿后用离线 FunASR 大模型后台二次校验并替换
        # 二次校验的最大排队段数（每段约 1.0-1.4 秒）。语速快时超出的段
        # 直接跳过——迟到几十秒的修正没有价值，还会拖住后面的段。
        "refine_max_backlog": 3,
        "partial_interval_ms": 500,  # 进行中字幕的最小刷新间隔（调大 = 更稳定不闪跳）
        # 断句（端点检测）阈值，直接透传给 sherpa。这是「字幕碎不碎」的唯一旋钮：
        # 一个人连续讲话时断得越少，每行携带的上下文越多，后面的 refine 和批量
        # 复核就越准；代价是这一行要更晚才出现在屏幕上。
        #   rule2 是主要旋钮——已有文字后需要多长的静音才切句。
        #   rule1 是「还没出字时」的静音阈值，rule3 是单行最长时长的强制上限。
        # rule2=1.2 是实测选出来的（同一段 4 分钟单人连续讲话，只有一个说话人）：
        #   0.8（本项目原先的硬编码，比 sherpa 默认更激进）18 行 / 平均 75.8 字
        #   1.2（sherpa 原生默认）                          14 行 / 平均 97.6 字
        #   1.6                                             14 行 / 平均 95.4 字
        # 0.8→1.2 行数 −22%、行长 +29%，而总字数 1365 vs 1366——同样的内容换了
        # 个切法，没有丢内容；1.2→1.6 行数完全不变，已经到顶，再放大没有收益。
        # 行数少 22% 直接省复核开销：实测单次复核调用固定开销 41.6s、每行只
        # 3.4s，所以「少几行」比「每行短一点」值钱得多。
        # 改这里需要重启（识别器在构造时吃掉这些参数）。想自己复测：
        #   python3 meetingscribe.py captions <wav> --rule2 1.6
        "endpoint": {
            "rule1_min_trailing_silence": 2.4,
            "rule2_min_trailing_silence": 1.2,
            "rule3_min_utterance_length": 20.0,
        },
        # 显示层段落合并：相邻两句定稿间隔 ≤ 此毫秒数时合并为同一段显示
        # （只影响展示，不影响识别/翻译数据）。0 = 不合并。
        "merge_gap_ms": 2500,
        # 字幕面板回看时长（分钟）：滚轮往上翻能看到最近这么久的字幕。
        # 按时间保留，与语速无关；0 = 不按时间淘汰（只受 history_max 限制）。
        "history_minutes": 180,
        # 行数上限，仅作内存兜底（12000 行 ≈ 3 小时 × 66 行/分钟的极端语速）。
        # 实测每行重绘成本约 11.6 µs，重绘间隔会随行数自动放宽。
        "history_max": 12000,
        # 字幕面板重绘合并窗口（毫秒）：把突发的 partial/translation/refined
        # 事件合并成每窗口一次 setHtml()，消除滚动条闪烁。0 = 每事件立即重绘。
        "render_debounce_ms": 120,
        # 字幕行前标注声道来源（[我方] / [对方]）：按每段音频里哪一路
        # （麦克风 / 系统音）相对能量更大来归属。这只是「哪一侧」，不是身份，
        # 用作 speaker_id 出结果前的兜底；只有一路有声音时不标注。
        "speaker_labels": True,
        # 定期批量复核（独立线程）：每 interval_minutes 把这段时间的定稿字幕
        # 连同当前译文整批交给大模型（默认复用 polish_provider），带整段上下文
        # 重新校对并重译。这是唯一能利用「跨句上下文」的一层——单句 pass 不可能
        # 知道三句前的「拆特ops」和现在的「ChatOps」是同一个词。
        # 代价：每 interval_minutes 一次 LLM 调用（录音期间）。
        "review": {
            "enabled": True,
            "interval_minutes": 5,   # 复核周期（分钟），也就是每批覆盖的时长
            # 单批最多送多少行，超出的留到下一批。约束是「输出 token」而不是
            # 上下文窗口：模型要把每行重新吐一遍，实测 120 行 ≈ 入 5.6k / 出
            # 5.2k tok，刚好在常见的 8k 输出上限之下；而实测语速 6.1 行/分钟，
            # 一个 5 分钟周期只攒约 30 行，120 已是 4 倍余量。
            "max_lines": 120,
            # 待复核队列的总量上限（max_lines 只约束单批）。这是兜底而不是调优
            # 旋钮：provider 卡住时队列会无界增长，而迟到 40 分钟才修正一行
            # 早已滚出屏幕的字幕没有价值。超限丢最旧的，并在日志里写明丢了
            # 哪几段（seg=A..B），不做静默截断。实测摄入 5.9 行/分，240 行
            # 约等于容忍 40 分钟的完全停摆。不会低于 max_lines。
            "max_buffer": 240,
            # 送进 prompt 的「上文」行数：本场前面已复核定稿的最后 N 行，
            # 只读参考，不重新校对。跨批次术语一致性靠它——第 N 批把
            # 「拆特ops」定成「ChatOps」，第 N+1 批不该又换回去。
            # 只增加输入 token；实测单次调用固定开销 41.6s、每行边际 3.4s
            # （80 批线性回归），十来行参考文本的代价可忽略。0 = 关闭。
            "context_lines": 12,
            "max_tries": 2,          # 回复里漏掉的行最多重投几次
            "provider": "",          # 留空 = 用 polish_provider
        },
        # 声纹识别（独立线程）：定稿后用 cam++ 提取 192 维声纹（约 60ms），
        # 在线聚类。两个声道一视同仁——会议室里几个人共用一支麦克风也能分开，
        # 所有人统一编号为「说话人1/2/3」，声道只决定标签颜色（蓝=我方一侧）。
        "speaker_id": {
            "enabled": True,
            "model": "cam++",      # iic/speech_campplus_sv_zh-cn_16k-common（约 27MB，首次自动下载）
            "threshold": 0.5,      # 余弦相似度阈值：调低=更容易并成同一个人，调高=更容易分裂
            "max_speakers": 8,     # 上限；超出后新声音并入最近的已知说话人
            "min_secs": 1.0,       # 短于此长度的片段不做声纹（太短不可靠）
            # 一个声纹簇累计多少段才算「一个人」并分配编号。1 段不是证据：
            # 实测一场 34 分钟会议里 8 个簇有 4 个只有 1 段（咳嗽、敲键盘、
            # 一个字的插话），它们还占满了 max_speakers，导致后面真正的新
            # 声音被并进已有说话人。未达标的行先不打标签，等第 2 段到达时
            # 回溯补发——真人只是晚几秒拿到标签，不会拿不到。
            "min_segments": 2,
            "max_backlog": 3,      # 排队上限，超出的片段跳过识别
        },
        # 仅在 mt_provider="nllb" 时使用
        "nllb": {
            "mt_model": "facebook/nllb-200-distilled-600M",
            "mt_int8": True,       # 动态 int8 量化（CPU 提速 2-4 倍，质量损失极小）
        },
        # 字幕翻译引擎（方案 C）：qwen = 本地 Qwen 小模型（llama.cpp/GGUF），
        # 命中 stt.funasr.hotword 的术语会作为术语表注入翻译 prompt（译文保留原样）；
        # 不可用（未装 llama-cpp-python / 模型下载失败）时自动回退到 default。
        # marian（默认）/ nllb = 指定 MT 模型（实测两者在术语上互有胜负，
        # NLLB 慢 6-8 倍）。default 等同 marian。
        # 注意：英文源句上 1.5B 的 qwen 经常把「翻译」行吐回英文（实测 4 句中 3
        # 句），英文会议建议用 marian / nllb。
        "mt_provider": "qwen",
        "qwen": {
            "model_repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "model_file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            "model_path": "",      # 本地 GGUF 路径；留空 = 首次使用时自动下载 model_repo/model_file
            "n_ctx": 2048,
            "threads": 0,          # llama.cpp 线程数；0 = 自动
            # GPU offload 层数：-1 = 全部（macOS Metal，实测 CPU 占用降 ~170 倍
            # 且快 4 倍；无 GPU 时 llama.cpp 自动回退 CPU）；0 = 强制纯 CPU。
            "n_gpu_layers": -1,
            # 定稿行走"纠错+翻译"合并调用（prompts.caption_fix_mt）：
            # 用前几句上下文 + 热词表修正同音字/残缺英文词，修正结果替换
            # 字幕行。false = 只翻译不纠错。
            "correct": True,
            # GBNF 语法约束（llama.cpp 解码期强制输出格式）。开启时改用
            # prompts.caption_fix_mt_strict（零样本，无格式示例）——格式由语法
            # 保证，示例反而会被 1.5B 模型抄进译文。语法不可用（旧版
            # llama-cpp-python）时自动回退到带示例的 caption_fix_mt。
            "grammar": True,
            # 术语表注入上限：只注入本行命中（含发音近似）的热词，最多这么多条。
            # 整表灌进 prompt（热词可达 100 条）是模型胡乱套用术语的主因。
            "glossary_max": 12,
        },
    },
    # ── 热词自动维护（方案 A+B；写回本文件的 stt.funasr.hotword）────────────
    "hotwords": {
        "auto_update": True,       # 纪要/总结生成后自动提取热词并合并进 stt.funasr.hotword
        "rule_extract": True,      # A：规则法（英文缩写/驼峰/含数字词等，无 LLM 开销）
        "llm_extract": True,       # B：LLM 术语提取（复用 meeting_notes_provider，离线阶段执行）
        # 热词总量上限；超出时最早加入的先被淘汰。
        # 实测放大的代价（同一段音频，填充随机中文名到指定规模）：
        #   流式 sherpa：初始化不变(~1s)，解码 20x → 15.6x 实时（1000 时 +28%）
        #   离线 FunASR：30s 音频 11.6s → 14.2s(1000) → 15.2s(5000)
        # 性能不是瓶颈，但精度是：5000 个 2 字中文名会造成误替换
        # （「分眼都在那边」→「分眼都郑娜边」——日常读音被生僻人名抢走）。
        # 2 字纯中文热词最危险；含 ASCII 或 3 字以上的安全得多。
        # 1000 是实测下的稳妥上限，继续放大请先用 `captions` 回放做 A/B。
        "max_count": 1000,
    },
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
            "workers": 4,               # 并发实例数；0 = 自动（max(2, CPU核数/2)）
            "chunk_secs": 300,          # 超过此时长自动分块并发（秒），0 = 始终串行
            "model": "paraformer-zh",   # ASR 模型（首次运行自动下载）
            "vad_model": "fsmn-vad",    # VAD 分句模型，支持长音频
            "punc_model": "ct-punc",    # 标点恢复模型
            # 说话人区分（声纹聚类）。留空 = 关闭；"cam++" = 开启，转写行变成
            # [12.3s] [说话人1] …。注意：为保证说话人编号全程一致，开启后会
            # 放弃分块并发、单趟转写，长录音耗时明显变长（约 chunk 并发数倍）。
            "spk_model": "",
            "hotword": "",              # 热词（空格分隔），提升专有名词识别率
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
    # ── 音频高级选项（一般不需要修改）─────────────────────────────────────
    "audio": {
        # AudioDeviceMonitor 的兜底唤醒间隔（秒）。HAL listener 失效时的自愈
        # 上限——平时不会触发；联调可调小到 5。
        "monitor_safety_timeout_sec": 30.0,
        # 自动探测 Multi-Output Device 时匹配的设备名（简中 / 繁中 / 英）。
        # 若你在 Audio MIDI Setup 里把 Multi-Output 设备重命名了，
        # 把那个名字加进来即可恢复自动探测。
        "multi_output_device_names": [
            "Multi-Output Device", "多输出设备", "多重輸出裝置",
        ],
    },
    # ── 流水线 Prompt 模板 ───────────────────────────────────────────────
    # The pipeline reads each prompt via `_resolve_prompt(cfg, key, mode)`.
    # The defaults defined above as `_PROMPT_DEFAULTS` are deep-merged with
    # any user overrides from `config.jsonc`; partial overrides (e.g. just
    # `prompts.meeting.notes_zh`) keep the other keys at their defaults.
    #
    # `copy.deepcopy` is defensive — `_deep_merge` shallow-copies each
    # nesting level, so without this any in-place mutation of
    # `cfg["prompts"][...]` would silently corrupt `_PROMPT_DEFAULTS` /
    # the module-level `DEFAULT_CONFIG` shared across the process. The
    # current call sites never mutate, but the copy is cheap insurance.
    "prompts": copy.deepcopy(_PROMPT_DEFAULTS),
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


def _load_hotword_file() -> str:
    """Read `stt.funasr.hotword` out of hotword.jsonc, or "" if absent.

    Never raises: a hand-broken hotword file must not stop the tool from
    recording — it just means no contextual biasing this run.
    """
    if not HOTWORD_FILE.exists():
        return ""
    try:
        data = json.loads(
            _strip_jsonc_comments(HOTWORD_FILE.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError) as e:
        _log("WARN", f"hotword file unreadable, ignoring: "
                     f"{type(e).__name__}: {e}")
        return ""
    if isinstance(data, dict):
        hw = data.get("hotword")
        if isinstance(hw, list):          # tolerate a list of terms
            hw = " ".join(str(t) for t in hw)
        return (hw or "").strip()
    return ""


def _load_hotword_store() -> dict:
    """Read hotword.jsonc as a store: terms plus per-term bookkeeping.

    Backward compatible in both directions. A flat `{"hotword": "a b c"}`
    file — everything written before this existed, and anything a human
    hand-edits — loads as all-rolling with no history, and the flat key is
    still written on every save so the file stays greppable and older builds
    keep working.

    - `pinned`  terms that must never be evicted (workspace names from
                Notion, or anything the user adds by hand). They are
                authoritative; a one-off token mined from one meeting has no
                business pushing a colleague's name out of the list.
    - `hits` / `last_seen`  how often and how recently a term actually showed
                up in a transcript. Eviction reads these instead of insertion
                order, so a term added ten meetings ago but still in daily
                use outranks yesterday's garbage.
    """
    empty = {"terms": [], "pinned": set(), "hits": {}, "last_seen": {},
             "epoch": 0}
    if not HOTWORD_FILE.exists():
        return empty
    try:
        data = json.loads(
            _strip_jsonc_comments(HOTWORD_FILE.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError) as e:
        _log("WARN", f"hotword file unreadable, ignoring: "
                     f"{type(e).__name__}: {e}")
        return empty
    if not isinstance(data, dict):
        return empty
    hw = data.get("hotword")
    if isinstance(hw, list):
        hw = " ".join(str(t) for t in hw)
    terms = (hw or "").split()
    pinned_raw = data.get("pinned")
    if isinstance(pinned_raw, str):
        pinned_raw = pinned_raw.split()
    pinned = {str(t).lower() for t in (pinned_raw or [])}
    # Pinned terms missing from `hotword` are still part of the list: that is
    # how a hand-edited `pinned` entry takes effect without also being typed
    # into the flat string.
    known = {t.lower() for t in terms}
    for t in (pinned_raw or []):
        if str(t).lower() not in known:
            terms.append(str(t))
    def _num_map(key):
        got = data.get(key)
        if not isinstance(got, dict):
            return {}
        out = {}
        for k, v in got.items():
            try:
                out[str(k).lower()] = int(v)
            except (TypeError, ValueError):
                continue
        return out
    try:
        epoch = int(data.get("epoch") or 0)
    except (TypeError, ValueError):
        epoch = 0
    return {"terms": terms, "pinned": pinned, "hits": _num_map("hits"),
            "last_seen": _num_map("last_seen"), "epoch": epoch}


def _load_hotword_file() -> str:
    """Read `stt.funasr.hotword` out of hotword.jsonc, or "" if absent.

    Never raises: a hand-broken hotword file must not stop the tool from
    recording — it just means no contextual biasing this run.
    """
    return " ".join(_load_hotword_store()["terms"]).strip()


def _save_hotword_file(hotword: str, store: "dict | None" = None) -> None:
    """Write hotword.jsonc, keeping the header comment that explains why the
    file is separate (and gitignored).

    `hotword` stays the first key and the source of truth for the term list,
    so the file remains readable and hand-editable. The bookkeeping keys are
    written only when there is something to record.
    """
    payload: dict = {"hotword": hotword}
    if store:
        kept = {t.lower() for t in hotword.split()}
        pinned = sorted(t for t in hotword.split()
                        if t.lower() in (store.get("pinned") or set()))
        if pinned:
            payload["pinned"] = pinned
        hits = {k: v for k, v in (store.get("hits") or {}).items()
                if k in kept and v}
        if hits:
            payload["hits"] = dict(sorted(hits.items()))
        last = {k: v for k, v in (store.get("last_seen") or {}).items()
                if k in kept and v}
        if last:
            payload["last_seen"] = dict(sorted(last.items()))
        if store.get("epoch"):
            payload["epoch"] = int(store["epoch"])
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    header = "\n".join((
        "// ASR 热词（空格分隔），由每次纪要生成后自动维护。",
        "// 单独成文件、且不入版本库：热词会累积同事姓名、客户名、内部代号，",
        "// 属于本地数据，不该随项目发布。模板见 hotword.jsonc.example。",
        "//",
        "// hotword   词表本身，空格分隔；手工增删这一行即可。",
        "// pinned    永不淘汰的词（Notion 导入的成员/项目名，或你手工钉的）。",
        "//           手工往 pinned 里加词即可生效，不必同时写进 hotword。",
        "// hits      该词在历次转写中出现的次数。",
        "// last_seen 该词最近一次出现时的 epoch（下面的计数器）。",
        "// epoch     热词更新次数；淘汰时按 (last_seen, hits) 从小到大清理，",
        "//           所以「十次会议前加入但仍在天天用」的词优于「昨天的垃圾词」。",
    ))
    HOTWORD_FILE.write_text(f"{header}\n{body}\n", encoding="utf-8")


def load_config() -> dict:
    CONFIG_DIR.mkdir(exist_ok=True)
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, encoding="utf-8") as f:
            on_disk = json.loads(_strip_jsonc_comments(f.read()))
        cfg = _deep_merge(DEFAULT_CONFIG, on_disk)
    else:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
    # hotword.jsonc wins when it has content; an inline config.jsonc value
    # still works so an existing setup keeps its list until the next
    # extraction migrates it out (see _persist_hotwords).
    hw = _load_hotword_file()
    if hw:
        cfg.setdefault("stt", {}).setdefault("funasr", {})["hotword"] = hw
    _apply_audio_overrides(cfg)
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


def _resolve_prompt(cfg: dict, key: str, mode: str | None = None) -> str:
    """Resolve a pipeline prompt.

    Two layouts are supported:

      * Top-level (mode-agnostic) — ``cfg["prompts"][key]``. Used for
        prompts whose behaviour doesn't change between ``meeting`` and
        ``interview`` (e.g. ``polish``).
      * Mode-scoped — ``cfg["prompts"][mode][key]``. Used for prompts
        that legitimately differ per mode (``notes_zh`` / ``notes_en``).

    ``mode=None`` selects the top-level layout. ``mode="meeting"`` or
    ``"interview"`` selects the mode-scoped layout. The same shape is
    expected in ``DEFAULT_CONFIG["prompts"]`` as the fallback.

    Each value may be either:

      * ``str``       — used verbatim.
      * ``list[str]`` — joined with ``"\\n"``. JSON forbids literal newlines
                        inside a string, so users typically write long
                        multi-line prompts as an array of lines in
                        ``config.jsonc`` — this loader accepts that form.

    Raises ``TypeError`` if the resolved value is neither ``str`` nor
    ``list[str]``. Raises ``KeyError`` if the default itself is missing
    (i.e. an unknown ``mode`` / ``key`` combination — should not happen
    for the canonical prompts shipped in ``_PROMPT_DEFAULTS``).

    The returned string still contains the literal ``{transcript}`` token;
    callers substitute it with ``str.replace`` (not ``str.format``) so
    other braces in the user's prompt do not need to be escaped.
    """
    user_prompts = cfg.get("prompts") if isinstance(cfg, dict) else None

    def _lookup_user() -> "object | None":
        if not isinstance(user_prompts, dict):
            return None
        if mode is None:
            return user_prompts.get(key)
        mode_block = user_prompts.get(mode)
        if not isinstance(mode_block, dict):
            return None
        return mode_block.get(key)

    user_value = _lookup_user()
    if user_value is None:
        # KeyError here surfaces a programming bug (unknown mode/key) —
        # the user can't cause it, so let it propagate.
        default_block = DEFAULT_CONFIG["prompts"]
        value: object = default_block[key] if mode is None else default_block[mode][key]
    else:
        value = user_value
    if isinstance(value, list):
        return "\n".join(str(line) for line in value)
    if isinstance(value, str):
        return value
    path = f"prompts.{key}" if mode is None else f"prompts.{mode}.{key}"
    raise TypeError(
        f"{path} must be str or list[str], got {type(value).__name__}"
    )


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
        # Try to splice in **new top-level keys** (the common case of a
        # schema bump — e.g. adding a "prompts" section to an existing
        # user config) before falling back to a comment-destroying full
        # rewrite. Nested missing-key inserts are too brittle to do
        # line-locally, so we still bail on those.
        new_top_level: list[tuple[str, object]] = []
        other_unapplied: list[tuple[str, ...]] = []
        for path, value in pending.items():
            if len(path) == 1 and path[0] not in existing:
                new_top_level.append((path[0], value))
            else:
                other_unapplied.append(path)
        if other_unapplied:
            _log("CONFIG", f"in-place save: {len(other_unapplied)} non-top-level "
                           f"diff(s) unapplied (keys: {other_unapplied}); "
                           f"falling back to json.dump")
            return False
        if not _append_new_top_level_keys(lines, new_top_level):
            return False

    CONFIG_FILE.write_text("".join(lines), encoding="utf-8")
    return True


def _append_new_top_level_keys(
    lines: list[str], new_entries: list[tuple[str, object]]
) -> bool:
    """Splice ``new_entries`` into the in-memory line list ``lines`` so the
    result is the same JSONC document with each new top-level key inserted
    just before the root closing brace. Returns True on success (caller
    writes the lines back to disk), False on a structural surprise that
    can't be handled safely (caller falls back to ``json.dump``).

    The previous last top-level entry's line gets a trailing comma added
    if it didn't already have one. Inline ``// comments`` on that line are
    preserved.
    """
    if not new_entries:
        return True
    import re

    # Locate the root closing brace (last "}" on its own line, ignoring
    # inline comments). We can't be smarter than "last `}`" because we
    # don't track scope here — but this is sufficient since JSONC files
    # produced by this codebase always end with `}\n`.
    closing_idx = None
    for i in range(len(lines) - 1, -1, -1):
        commentless = re.sub(r'//.*$', '', lines[i])
        if commentless.strip() == "}":
            closing_idx = i
            break
    if closing_idx is None:
        _log("CONFIG", "in-place save: no root closing brace; "
                       "falling back to json.dump")
        return False

    # Ensure the previous last content line ends with a comma so the
    # inserted block parses cleanly. Walk backwards over blank /
    # comment-only lines.
    for j in range(closing_idx - 1, -1, -1):
        commentless = re.sub(r'//.*$', '', lines[j]).strip()
        if not commentless:
            continue
        cur = lines[j]
        cur_no_eol = cur.rstrip("\r\n")
        eol = cur[len(cur_no_eol):]
        # Split inline `// comment` off (respecting strings is overkill
        # here — JSONC values containing `//` are rare and we only need
        # to NOT touch the comment side).
        body = cur_no_eol
        comment = ""
        m_cmt = re.search(r'\s*//.*$', cur_no_eol)
        if m_cmt:
            body = cur_no_eol[:m_cmt.start()]
            comment = cur_no_eol[m_cmt.start():]
        body_rstripped = body.rstrip()
        if not body_rstripped.endswith(",") and not body_rstripped.endswith("{"):
            trailing_ws = body[len(body_rstripped):]
            lines[j] = body_rstripped + "," + trailing_ws + comment + eol
        break

    # Render each new entry as a 2-space-indented JSON block and splice
    # in. Lines from ``json.dumps({k: v}, indent=2)`` already have the
    # right indent for top-level placement; we strip the outer braces.
    insert_block: list[str] = []
    for idx, (key, value) in enumerate(new_entries):
        formatted = json.dumps({key: value}, ensure_ascii=False, indent=2)
        inner = formatted.split("\n")[1:-1]
        # If not the last new entry, the last line gets a trailing comma.
        if idx < len(new_entries) - 1:
            inner[-1] = inner[-1] + ","
        insert_block.extend(line + "\n" for line in inner)

    lines[closing_idx:closing_idx] = insert_block
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
# Mutable list (not tuple) so _apply_audio_overrides() can replace its contents in place
# from cfg["audio"]["multi_output_device_names"] without touching the many reference sites.
_MULTI_OUT_NAMES = ["Multi-Output Device", "多输出设备", "多重輸出裝置"]

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
# resyncs within half a minute. Overridable via cfg["audio"]["monitor_safety_timeout_sec"].
_AUDIO_MONITOR_SAFETY_TIMEOUT_SEC = 30.0


def _apply_audio_overrides(cfg: dict) -> None:
    """Push user-configurable audio knobs from cfg into module-level globals.

    Called from load_config() so AudioDeviceMonitor and the device-resolver
    helpers can keep reading simple globals without cfg threaded through every
    callsite. Idempotent: safe to call on every load. Silently ignores invalid
    values (keeps the built-in defaults).
    """
    global _AUDIO_MONITOR_SAFETY_TIMEOUT_SEC
    audio_cfg = cfg.get("audio") or {}

    timeout = audio_cfg.get("monitor_safety_timeout_sec")
    if isinstance(timeout, (int, float)) and timeout > 0:
        _AUDIO_MONITOR_SAFETY_TIMEOUT_SEC = float(timeout)

    names = audio_cfg.get("multi_output_device_names")
    if isinstance(names, list) and names:
        _MULTI_OUT_NAMES[:] = [str(n) for n in names if isinstance(n, str) and n]

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

# Devices the user has explicitly muted via the in-app slider (dragged to 0%).
# `_reconcile_recording_mutes` reads this to distinguish "user intent" from
# "stale leftover" when deciding whether to clear the mute on the active
# listening target. Protected by `_mutes_lock`. Not persisted on purpose —
# slider state is transient UI; a crash resets the user's mute intent and the
# device-level mute itself is recovered via `_recover_persisted_mutes`.
_slider_intent_muted: set[str] = set()


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
            # Honour explicit user intent: if the user has parked the in-app
            # slider at 0% for this device, they want it muted right now.
            # The "active listener mute" branch below would normally treat
            # this as a stale leftover and clear it; skip that here so the
            # slider's mute survives every periodic reconcile.
            if not desired and sub in _slider_intent_muted:
                _log(
                    "MUTE",
                    f"device={sub!r} mute kept (slider intent); reconcile skipped",
                )
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
            f"reconcile: active={active!r} multi_subs={physical_subs} "
            f"tracked={list(_active_mutes.keys())} "
            f"slider_intent={sorted(_slider_intent_muted)}",
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


# The resolved plan is re-derived on every monitor tick — once a second while
# recording, from the recorder's own monitor fallback. Logging it each time
# made it 37 % of a day's log (7 385 lines, only 3 distinct values, 6 191 of
# the gaps exactly 1 s). A CHANGE is what has diagnostic value, so a change
# prints immediately and identical repeats are throttled to a heartbeat. The
# heartbeat can be raised freely — nothing is lost, since changes bypass it.
_DEVICE_PLAN_LOG_INTERVAL_SEC = 3.0
_device_plan_log_state: dict = {"msg": None, "at": 0.0}


def _log_device_plan(msg: str) -> None:
    """Log the resolved audio plan, collapsing identical repeats.

    Racy by design across the monitor threads: the worst outcome of two
    threads passing the check together is one extra identical line, which is
    cheaper than serialising every device resolution behind a lock.
    """
    now = time.time()
    if (msg == _device_plan_log_state["msg"]
            and now - _device_plan_log_state["at"] < _DEVICE_PLAN_LOG_INTERVAL_SEC):
        return
    _device_plan_log_state["msg"] = msg
    _device_plan_log_state["at"] = now
    _log("DEVICE", msg)


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
    _log_device_plan(
        f"plan: mic={mic_name!r} sys={sys_source_name!r} "
        f"multi={multi_out!r} restore={restore!r} "
        f"external={is_external} warnings={warnings}"
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
        # Memoizes the last `plan.restore_output_name` we fired
        # `on_recording_plan_change` for. Tracks idle AND recording branches
        # so the callback fires exactly once per real change of the active
        # output device, regardless of which branch detects it first.
        self._prev_active_device: "str | None" = None
        # Optional callback fired (from this monitor thread) every time the
        # resolver's active output device changes — covers BOTH the idle
        # branch (user plugs headphones with no recording in progress) and
        # the recording branch (mid-recording hotplug). GUI code sets this
        # to keep widgets like the volume slider bound to the current
        # physical output. None outside the GUI (CLI). The callback is
        # responsible for its own thread-safety — it MUST NOT block; common
        # pattern is to schedule a Qt update via QTimer.singleShot(0, ...).
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
        # Fire the active-output-changed callback for hotplugs that happen
        # while not recording (e.g. user pairs AirPods after launching the
        # GUI but before starting a recording). Independent of `_prev_triple`
        # so a mic-only change doesn't also trigger a slider transfer.
        new_active = plan.restore_output_name
        if new_active and new_active != self._prev_active_device:
            self._prev_active_device = new_active
            cb = self.on_recording_plan_change
            if cb is not None:
                try:
                    cb(plan)
                except Exception as e:
                    _log(
                        "ERR",
                        f"idle-branch plan-change callback: "
                        f"{type(e).__name__}: {e}",
                    )

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
        self._prev_active_device = plan.restore_output_name
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
    on_audio_chunk: "callable | None" = None   # callback(role, frames, samplerate) — live-caption tap

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
        self.warnings: list[str] = []        # de-duped warning codes (logged via _emit_warning)
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
        """Map (device, 'not-opened'|'disappeared') to a stable warning code, or None
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
        try:
            role = self.role_labels[self.wanted.index(device)]
        except (ValueError, IndexError):
            role = device
        def _cb(indata, frames, time_info, status):
            if self.recording:
                data = indata.copy()
                self._frames[device].append(data)  # GIL 保护 list.append
                hook = self.on_audio_chunk
                if hook is not None:
                    # Same copy the recorder keeps — the tap never mutates it.
                    # hook must be non-blocking (PortAudio callback thread).
                    try:
                        hook(role, data, self.sample_rate)
                    except Exception:
                        pass  # never let the tap break capture
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
                # Downmix the device's own channels (see LiveCaptionEngine.feed):
                # channel 0 alone loses anything panned right.
                channels.append(
                    audio.mean(axis=1) if audio.ndim > 1 else audio)
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

        _log(
            "REC",
            f"save: path={path.name} channels={2} frames={mixed.shape[0]} "
            f"warnings={self.warnings}",
        )
        return True


# ── 实时双语字幕引擎 ─────────────────────────────────────────────────────────

_CAPTION_SAMPLE_RATE = 16000       # every caption ASR backend consumes 16 kHz mono
# Peak RMS above which a source counts as "heard" for speaker attribution.
# 0.01 was measured too high: a built-in mic at conversational distance sits
# around 0.003-0.008 per 250 ms window, so neither source ever qualified.
_CAPTION_ROLE_RMS_FLOOR = 0.002
_CAPTION_RING_SECONDS = 30.0       # per-source backlog cap while models are loading
_CAPTION_PACER_INTERVAL = 0.25     # seconds between drain/mix/feed iterations

# Endpoint (sentence-break) rules forwarded to sherpa. Bounds are sanity
# guards, not preferences: rule2 below ~0.3 s splits inside a normal pause and
# above ~3 s merges two speakers' turns into one line, and rule3 is the
# hard ceiling on how long a single caption line may keep growing.
_CAPTION_ENDPOINT_BOUNDS = {
    "rule1_min_trailing_silence": (0.3, 10.0),
    "rule2_min_trailing_silence": (0.3, 5.0),
    "rule3_min_utterance_length": (5.0, 120.0),
}


def _caption_endpoint_rules(lc_cfg: dict) -> dict:
    """Endpoint thresholds for the streaming recognizer, clamped to sane range.

    Longer thresholds mean fewer, longer caption lines: more context per line
    for the refine and review passes to work with, at the cost of the line
    appearing on screen later. A junk value in config must not silently
    disable sentence breaking, so each key falls back to its default and is
    clamped rather than trusted.
    """
    defaults = DEFAULT_CONFIG["live_captions"]["endpoint"]
    got = (lc_cfg or {}).get("endpoint") or {}
    out = {}
    for key, default in defaults.items():
        lo, hi = _CAPTION_ENDPOINT_BOUNDS[key]
        try:
            val = float(got.get(key, default))
        except (TypeError, ValueError):
            _log("CAPTION", f"endpoint {key}={got.get(key)!r} not a number, "
                            f"using {default}")
            val = float(default)
        clamped = min(max(val, lo), hi)
        if clamped != val:
            _log("CAPTION", f"endpoint {key}={val} out of [{lo}, {hi}], "
                            f"clamped to {clamped}")
        out[key] = clamped
    return out

_SHERPA_ZIPFORMER_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2"
)

# Heavy caption backends survive across recording sessions so a new recording
# doesn't re-load multi-GB models. Keyed ("asr"|"mt", mode); ASR entries get
# reset_session() before reuse.
_caption_backend_cache: dict = {}


def _caption_resample(x: "np.ndarray", sr_from: int, sr_to: int = _CAPTION_SAMPLE_RATE) -> "np.ndarray":
    """Linear-interpolation mono resample; fidelity is plenty for ASR feeds."""
    x = np.asarray(x, dtype=np.float32)
    if sr_from == sr_to or x.size == 0:
        return x
    n_out = int(round(x.size * sr_to / sr_from))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    xp = np.linspace(0.0, 1.0, num=x.size, endpoint=False)
    xq = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(xq, xp, x).astype(np.float32)


def _caption_is_english(text: str) -> bool:
    """Crude language sniff: mostly ASCII letters → treat as English source."""
    letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return letters >= 3 and letters >= 3 * max(cjk, 1)


# Pure filler syllables safe to drop anywhere. Grammatical sentence
# particles (呢 / 啊 / 吧 / 嘛 / 呀) intentionally stay — they carry tone.
# Multi-char interjections listed before their single-char substrings.
_CAPTION_FILLERS = ("哎呀", "哎哟", "哎呦", "嗯", "呃", "唉", "哎")

# 2-char stutter collapse whitelist: characters that never legitimately
# reduplicate as a word. Deliberately excludes verbs like 看/说/做/听/问
# ("看看" / "说说" are valid soft-imperatives) and words like 好/慢/常/明
# ("好好" / "慢慢" / "常常" / "明明" are valid). Applied AFTER the 3+ char
# collapse so cases like 「都都都 → 都」 still work at that rule.
_CAPTION_STUTTER_2CHAR = re.compile(
    r"(我|你|他|她|它|没|是|但|而|或|也|还|就|都|这|那|哪|谁|什|怎|如|因|所|"
    r"把|被|让|给|从|到|在)\1"
)


def _caption_tidy_zh(text: str) -> str:
    """Clean verbatim speech transcripts for subtitle display.

    1. Drops pure filler syllables (嗯 / 呃 / 哎呀 …) — see _CAPTION_FILLERS.
    2. Collapses adjacent stutter repeats: 「所以就像所以就像」→「所以就像」，
       「但是但是」→「但是」，「都都都」→「都」. Longest n-grams first so
       nested repeats collapse correctly. Two-char reduplications (谢谢 /
       刚刚 / 慢慢) are intentional Chinese and stay: single chars only
       collapse at 3+ repeats, n-gram collapsing starts at n=2.
    3. Sweeps up punctuation orphans the removals leave behind.

    Applied centrally before captions are displayed / translated, so the
    English line benefits too.
    """
    if not text or _caption_is_english(text):
        return text
    for f in _CAPTION_FILLERS:
        text = text.replace(f, "")
    text = re.sub(r"(.)\1{2,}", r"\1", text)          # 都都都 → 都
    for n in range(8, 1, -1):                          # 所以就像所以就像 → 所以就像
        text = re.sub(r"(.{%d})\1+" % n, r"\1", text)
    text = _CAPTION_STUTTER_2CHAR.sub(r"\1", text)     # 没没 / 就就 / 我我 → 单字
    text = re.sub(r"[，、]{2,}", "，", text)
    text = re.sub(r"[，、]+(?=[。？！，])", "", text)
    text = re.sub(r"^[，、。？！\s]+", "", text)
    return text


# Decoding guards shared by the Marian / NLLB caption backends. Greedy
# decoding (num_beams=1, chosen for latency) degenerates into loops when the
# ASR hands it noise; forbidding any repeated 6-token n-gram plus a mild
# penalty stops it at the source, before the display-layer collapse.
_MT_NO_REPEAT_NGRAM = 6
_MT_REPETITION_PENALTY = 1.15


def _collapse_mt_repeats(text: str) -> str:
    """Collapse degenerate repetition in MT output.

    Backstop for NMT's classic failure on garbage input: a caption line of
    「AND THEN OKAY WERE CON YEAH」 came back as 「…雅 的 内容 雅 的 内容 …」
    ×30 followed by 「Ya Ya Ya …」 ×100. `no_repeat_ngram_size` on the
    generate() call prevents most of it; this catches whatever slips through
    (e.g. a hand-configured backend, or repeats longer than the n-gram bound).

    Word-level rules run first and need 3+ occurrences, so English keeps its
    legitimate repeats ("very very good"). Character-level rules only apply to
    non-English output, where a repeated long n-gram is broken even at 2
    occurrences — running them on English would eat "very very" as the 5-char
    n-gram "very ". This is the translation-only tidy; source-side rules live
    in `_caption_tidy_zh`, which deliberately leaves English alone.
    """
    if not text:
        return text
    # Word-level, SHORTEST phrase first: "Ya Ya Ya Ya" → "Ya",
    # "the plan the plan the plan" → "the plan". Longest-first would leave
    # "Ya Ya" behind — 6 × "Ya" reads as 3 × the phrase "Ya Ya", which
    # collapses to a pair that no longer trips the 3+ threshold.
    for n in range(1, 7):
        text = re.sub(
            r"\b((?:[A-Za-z0-9']+ ){%d}[A-Za-z0-9']+)(?: \1\b){2,}" % (n - 1),
            r"\1", text)
    if not _caption_is_english(text):
        for n in range(8, 3, -1):       # long n-grams: 2+ repeats is broken
            text = re.sub(r"(.{%d})\1+" % n, r"\1", text)
        for n in range(3, 0, -1):       # short ones need 3+
            text = re.sub(r"(.{%d})\1{2,}" % n, r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _caption_fix_case(text: str, hotwords: "list[str] | None" = None) -> str:
    """Normalize SHOUTED English from the streaming ASR into sentence case.

    The fast-mode zipformer emits English in all caps ("AND THEN OKAY WERE
    CON YEAH"), which reads as shouting next to the mixed-case lines the
    refine pass produces for the same session. Only touches text that is
    genuinely all-caps: a line with any lowercase is left alone, so
    deliberate acronyms in normal text ("GKE 集群") never get downcased.

    Known hotwords keep their configured casing (`ChatOps`, `gVisor`), and
    the standalone pronoun "I" stays capitalized.
    """
    if not text:
        return text
    uppers = [c for c in text if c.isascii() and c.isalpha() and c.isupper()]
    if len(uppers) < 3 or any(
            c.isascii() and c.isalpha() and c.islower() for c in text):
        return text
    out = text.lower()
    for w in (hotwords or []):
        if len(w) < 2 or not any(c.isascii() and c.isalpha() for c in w):
            continue
        out = re.sub(r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])" % re.escape(w.lower()),
                     w, out)
    out = re.sub(r"(?<![A-Za-z0-9'])i(?![A-Za-z0-9'])", "I", out)
    out = re.sub(r"\bi'(m|ll|ve|d)\b", lambda m: "I'" + m.group(1), out)
    # Sentence case: first letter of the line and after . ! ?
    out = re.sub(r"(^|[.!?]\s+)([a-z])",
                 lambda m: m.group(1) + m.group(2).upper(), out)
    return out


# Interjections that carry no content even as a whole line. Distinct from
# _CAPTION_FILLERS (which are stripped INSIDE a line): 啊 / 吧 / 呢 are kept
# mid-sentence because they carry tone, but a line that is nothing but 啊 is
# noise. Short real answers (对 / 好 / 是 / 可以) are deliberately absent.
_CAPTION_NOISE_ONLY = set("啊呢嗯呃哦噢唉哎呀吧嘛唔额诶欸")


def _caption_is_noise(text: str) -> bool:
    """True for finalized lines with no content worth showing.

    The streaming model emits a stray letter or particle for a cough, a
    keyboard tap or an "mmm" — measured on a real 34-minute meeting, those
    junk lines (`M`, `N`, `A`, `啊。`) were 4 of the 8 speaker clusters the
    voice print created, each from a single segment, and they exhausted
    `max_speakers` so genuinely new speakers later got folded into existing
    ones. Dropping them at the source fixes the captions AND the clustering.
    """
    core = re.sub(r"[\s\W_]+", "", text or "", flags=re.UNICODE)
    if not core:
        return True
    if len(core) == 1 and core.isascii() and core.isalpha():
        return True     # "M" / "N" / "A" — a cough, not a word
    return all(ch in _CAPTION_NOISE_ONLY for ch in core)


def _dedup_caption_boundary(prev: str, curr: str, max_overlap: int = 12) -> str:
    """Strip curr's leading n-gram when it matches prev's trailing n-gram.

    Handles the paraformer streaming chunk-boundary duplicate pattern:
        prev = "…运维体系那一块"
        curr = "那一块。对，运维体系那块就"
        → curr becomes "。对，运维体系那块就"

    Trailing punctuation on prev is stripped for comparison (paraformer
    auto-adds 。 / ，). Minimum overlap = 2 chars to avoid single-char
    coincidences (e.g. two lines both starting with 我).
    """
    if not prev or not curr:
        return curr
    prev_norm = re.sub(r"[，。？！、,.?!;； \t]+$", "", prev)
    limit = min(max_overlap, len(prev_norm), len(curr))
    for n in range(limit, 1, -1):
        if prev_norm.endswith(curr[:n]):
            return curr[n:]
    return curr


def _append_hypothesis(clean: str, delta: str) -> str:
    """Append a streaming chunk's new text, stripping a boundary duplicate.

    Streaming decoders are append-only: once a token is emitted it cannot be
    revised, so a syllable split across a chunk boundary gets committed twice
    — measured on 18 s of real audio, the streaming pass produced 129 chars
    where the offline pass produced 101 (+28 %), all of it repeats like
    「今天今天天」/「十三十三」/「如果你如果」.

    Since the duplicate always sits at the JOIN, the fix is to check the new
    chunk's head against the text already accumulated — `_dedup_caption_boundary`
    with its 2-character minimum, so a single-character coincidence (a genuine
    「看看」 split across the boundary) is left alone.
    """
    if not delta:
        return clean
    return clean + _dedup_caption_boundary(clean, delta)


def _join_caption_texts(parts: "list[str]") -> str:
    """Join caption fragments for display: bare join between CJK chars,
    single space when either side of the boundary is ASCII alphanumeric
    (so merged English fragments don't run together)."""
    out = ""
    for p in parts:
        p = (p or "").strip()
        if not p:
            continue
        if out and (
            (out[-1].isascii() and out[-1].isalnum())
            or (p[0].isascii() and p[0].isalnum())
        ):
            out += " "
        out += p
    return out


def _prune_caption_rows(rows: "list[dict]", now: float,
                        max_secs: float, max_rows: int,
                        slack_secs: float = 60.0,
                        slack_rows: int = 200) -> int:
    """Drop caption rows that fell out of the retention window, in place.

    Retention is primarily by TIME: a row cap can't promise "the last 3
    hours" because the covered span depends on speech cadence (a fast
    meeting finalizes 40 lines/min, a slow one 15). `max_rows` stays as a
    memory backstop for pathological cadence. Rows are chronological, so
    both passes just trim from the front. Returns the number dropped.

    Eviction is BATCHED via the slack allowances, and that is not a
    micro-optimization: dropping one row per append shifts every paragraph
    in the pane, so the first display group changes and
    `_CaptionDocRenderer` can no longer patch a tail — the whole document
    gets re-laid-out on every single line once the window is full. Waiting
    until the overflow exceeds the slack turns that into one rebuild per
    minute (or per 200 rows) instead of one per line. The window is
    therefore a *minimum*: up to `slack` extra history may be retained.
    """
    before = len(rows)
    if max_secs > 0 and rows:
        if rows[0].get("t", 0.0) < now - max_secs - slack_secs:
            cutoff = now - max_secs
            keep = 0
            while keep < len(rows) and rows[keep].get("t", 0.0) < cutoff:
                keep += 1
            if keep:
                del rows[:keep]
    if max_rows > 0 and len(rows) > max_rows + slack_rows:
        del rows[:-max_rows]
    return before - len(rows)


def _group_caption_rows(rows: "list[dict]", gap_secs: float = 2.5,
                        max_chars: int = 120) -> "list[list[dict]]":
    """Display-layer paragraph merging (data layer untouched): a row joins
    the previous group when it arrived within `gap_secs` of the group's
    last row, the group is still below `max_chars` of source text, AND both
    rows carry the same speaker role — merging across a speaker change would
    attribute one side's words to the other.
    gap_secs <= 0 disables merging (one row per group)."""
    groups: "list[list[dict]]" = []
    for row in rows:
        if groups and gap_secs > 0:
            last = groups[-1]
            close = (row.get("t", 0.0) - last[-1].get("t", 0.0)) <= gap_secs
            small = sum(len(r.get("src", "")) for r in last) < max_chars
            same_speaker = (
                (row.get("speaker"), row.get("side"), row.get("role"))
                == (last[-1].get("speaker"), last[-1].get("side"),
                    last[-1].get("role")))
            if close and small and same_speaker:
                last.append(row)
                continue
        groups.append([row])
    return groups


_REVIEW_SEP = "|||"


def _format_caption_review_lines(batch: "list[dict]") -> str:
    """Render a review batch as `N ||| 原文 ||| 当前译文` lines."""
    out = []
    for i, row in enumerate(batch, start=1):
        src = (row.get("text") or "").replace("\n", " ").strip()
        dst = (row.get("dst") or "").replace("\n", " ").strip()
        out.append(f"{i} {_REVIEW_SEP} {src} {_REVIEW_SEP} {dst or '（未翻译）'}")
    return "\n".join(out)


def _format_caption_review_context(rows: "list[dict]", limit: int) -> str:
    """Render already-reviewed lines as read-only context for the next batch.

    Deliberately NOT in the numbered `N ||| … ||| …` shape the batch uses:
    numbering in the reply is positional into `batch`, so a model that echoed
    a context line in that format would land its text on an unrelated
    segment. An unnumbered bullet gives it no template to copy.
    """
    if limit <= 0 or not rows:
        return "（无，本批是本场第一批）"
    out = []
    for row in rows[-limit:]:
        src = (row.get("text") or "").replace("\n", " ").strip()
        dst = (row.get("dst") or "").replace("\n", " ").strip()
        if not src:
            continue
        out.append(f"- {src}" + (f"（{dst}）" if dst else ""))
    return "\n".join(out) or "（无，本批是本场第一批）"


def _review_log_text(s: str, limit: int = 500) -> str:
    """One-line, delimited rendering of caption text for the audit log.

    Newlines and runs of whitespace collapse so a change is exactly one log
    line (greppable), and the 「」 brackets keep the field boundaries readable
    even when the text itself contains arrows or colons.
    """
    flat = re.sub(r"\s+", " ", (s or "").strip())
    if len(flat) > limit:
        flat = flat[:limit] + f"…(+{len(flat) - limit})"
    return f"「{flat}」"


def _parse_caption_review(reply: str, batch: "list[dict]") -> "dict[int, tuple]":
    """Parse the caption_review reply into {segment_id: (text, dst)}.

    Defensive in the same spirit as `_parse_fix_mt`, because this pass
    rewrites lines the user has already read: an entry is dropped (leaving
    the line untouched) when its index is unknown, when either field is
    empty, or when the "correction" is not a minimal edit — length outside
    0.5x–2.0x of the original OR SequenceMatcher similarity < 0.45. A batch
    reply that omits entries simply leaves those lines alone, so a truncated
    or chatty answer degrades to "no change" rather than to lost captions.
    """
    import difflib
    out: "dict[int, tuple]" = {}
    if not reply:
        return out
    for raw in reply.splitlines():
        line = raw.strip()
        if not line or _REVIEW_SEP not in line:
            continue
        parts = [p.strip() for p in line.split(_REVIEW_SEP)]
        if len(parts) < 3:
            continue
        idx_txt = re.sub(r"[^\d]", "", parts[0])
        if not idx_txt:
            continue
        pos = int(idx_txt) - 1
        if not (0 <= pos < len(batch)):
            continue
        fixed, dst = parts[1], parts[2]
        row = batch[pos]
        original = (row.get("text") or "").strip()
        sid = row.get("id")
        if sid is None or not fixed or not dst:
            continue
        ratio = len(fixed) / max(len(original), 1)
        if (
            not (0.5 <= ratio <= 2.0)
            or difflib.SequenceMatcher(None, original, fixed).ratio() < 0.45
        ):
            # Rewrite, hallucination or dropped clause — keep the original
            # source but still take the translation, which is judged
            # separately (it legitimately looks nothing like the source).
            fixed = original
        out[sid] = (fixed, dst)
    return out


class _SpeakerClusterer:
    """Online cosine clustering of speaker embeddings.

    Live captions can't run offline diarization (which needs the whole
    recording to cluster globally), but each finalized segment already has
    its audio, and a 192-dim cam++ voice print costs ~60 ms. So speakers are
    identified incrementally: compare the segment's embedding against the
    centroids seen so far, join the nearest one above `threshold`, otherwise
    start a new speaker. Centroids are running means, so a speaker's print
    sharpens as they talk more.

    Channel role (mic vs system audio) can't do this job: everyone on the
    call arrives mixed into one system-audio stream. Role still decides which
    cluster is "me" — that's what the microphone channel authoritatively
    knows — via `role_votes`.
    """

    def __init__(self, threshold: float = 0.6, max_speakers: int = 8,
                 min_segments: int = 2):
        self.threshold = float(threshold)
        self.max_speakers = max(1, int(max_speakers))
        # Segments a cluster must accumulate before it is shown as a person.
        # One segment is not evidence of a speaker: on a real 34-minute
        # meeting, 4 of 8 clusters held exactly one segment each — a cough, a
        # keyboard tap, a one-word interjection — and they consumed
        # max_speakers, so genuinely new voices later got folded into
        # existing ones. `_caption_is_noise` catches the junk that decodes to
        # a stray letter or particle; this catches the rest.
        self.min_segments = max(1, int(min_segments))
        self._centroids: "list[np.ndarray]" = []
        self._counts: "list[int]" = []
        self.role_votes: "list[dict]" = []
        # Display numbers are handed out in QUALIFICATION order, not discovery
        # order: numbering by discovery would leave gaps (说话人1, then 说话人3
        # because cluster 1 never earned a number) which reads as a bug.
        self._numbers: "dict[int, int]" = {}
        self._next_number = 1

    @staticmethod
    def _unit(vec: "np.ndarray") -> "np.ndarray":
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def assign(self, embedding: "np.ndarray", role: "str | None" = None) -> int:
        """Cluster index for this embedding, creating a speaker if needed."""
        emb = self._unit(embedding)
        if emb.size == 0:
            return -1
        idx, created = -1, False
        if self._centroids:
            sims = [float(np.dot(c, emb)) for c in self._centroids]
            best = int(np.argmax(sims))
            if sims[best] >= self.threshold:
                idx = best
            elif len(self._centroids) >= self.max_speakers:
                # Cap reached: fold into the nearest speaker rather than
                # inventing 说话人9 for every new voice in a noisy room.
                idx = best
        if idx < 0:
            self._centroids.append(emb)
            self._counts.append(1)
            self.role_votes.append({})
            idx, created = len(self._centroids) - 1, True
        if not created:
            n = self._counts[idx]
            self._centroids[idx] = self._unit(
                (self._centroids[idx] * n + emb) / (n + 1))
            self._counts[idx] = n + 1
        if role:
            votes = self.role_votes[idx]
            votes[role] = votes.get(role, 0) + 1
        return idx

    def majority_side(self, idx: int) -> "str | None":
        """Which channel this speaker mostly arrives on ("mic" / "system").

        A SIDE, never an identity: a single microphone can carry several
        people (everyone in one meeting room), so "came in on the mic" does
        not mean "is the person running the app". Identity comes from the
        voice print alone; this only decides which colour the tag gets.
        """
        if not (0 <= idx < len(self.role_votes)):
            return None
        votes = self.role_votes[idx]
        if not votes:
            return None
        return max(votes.items(), key=lambda kv: kv[1])[0]

    def display_number(self, idx: int) -> "int | None":
        """1-based speaker number, or None while the cluster is unproven.

        Numbering is global across BOTH channels on purpose. Reserving 我 for
        the mic side broke as soon as two people shared one microphone: they
        both scored a mic majority and both got labelled 我.

        None means "not enough evidence yet" — the caller should hold the
        segment rather than tag it. A cluster earns its number on reaching
        `min_segments`, and keeps it for the rest of the session.
        """
        if not (0 <= idx < len(self._counts)):
            return None
        got = self._numbers.get(idx)
        if got is not None:
            return got
        if self._counts[idx] < self.min_segments:
            return None
        self._numbers[idx] = self._next_number
        self._next_number += 1
        return self._numbers[idx]

    def numbered_count(self) -> int:
        """How many speakers have actually earned a number."""
        return len(self._numbers)

    def segment_count(self, idx: int) -> int:
        """Segments assigned to this cluster so far."""
        return self._counts[idx] if 0 <= idx < len(self._counts) else 0


class _CaptionSpeakerId:
    """cam++ voice-print embeddings for live-caption speaker identification.

    `iic/speech_campplus_sv_zh-cn_16k-common` (~27 MB, auto-downloaded by
    modelscope on first use) returns a 192-dim embedding in ~60 ms per
    segment on M2 — cheap enough to run on every finalized line. It is a
    zh-CN trained model, but a voice print captures timbre rather than
    language, so it still separates speakers in English meetings.
    """

    def __init__(self, model_id: str = "cam++"):
        try:
            from funasr import AutoModel
        except ImportError as e:
            raise RuntimeError(
                "说话人识别需要 funasr：python3 -m pip install funasr modelscope torch"
            ) from e
        with _QuietCapture("CAPTION"):
            self._model = AutoModel(model=model_id, disable_update=True)
        _log("CAPTION", f"speaker id ready: {model_id}")

    def embed(self, audio: "np.ndarray") -> "np.ndarray | None":
        with _QuietCapture("CAPTION"):
            res = self._model.generate(input=audio)
        if not res:
            return None
        emb = res[0].get("spk_embedding") if isinstance(res[0], dict) else None
        if emb is None:
            return None
        arr = emb.detach().cpu().numpy() if hasattr(emb, "detach") else np.asarray(emb)
        return np.asarray(arr, dtype=np.float32).reshape(-1)


class _CaptionDocRenderer:
    """Incremental HTML renderer for the caption pane.

    `setHtml` re-lays-out the WHOLE document and is linear in its size
    (measured offscreen: 6.9 ms at 500 rows → 140 ms at 12 000), so with a
    3-hour scroll-back buffer a full rebuild per caption event cannot keep
    up. In steady state only the tail changes — a new paragraph is appended,
    or the last one gets its translation — so this keeps the HTML of every
    committed group and replaces just the tail from the first divergence.

    Bookkeeping is by BLOCK NUMBER rather than character position: each group
    renders to exactly `_BLOCKS_PER_GROUP` paragraphs, so group *i* starts at
    block *i × 2* and the invariant `blockCount == 2 × len(committed)` is
    checkable before every incremental edit. Any violation (or a divergence
    at index 0) falls back to a full `setHtml`, so a bookkeeping bug can
    only cost performance, never a corrupted pane.
    """

    _BLOCKS_PER_GROUP = 2   # source paragraph + translation paragraph

    def __init__(self, view):
        self._view = view
        self._committed: "list[str]" = []

    def reset(self) -> None:
        """Forget the document (call when the pane is cleared)."""
        self._committed = []

    @staticmethod
    def _first_divergence(old: "list[str]", new: "list[str]") -> int:
        for i in range(min(len(old), len(new))):
            if old[i] != new[i]:
                return i
        return min(len(old), len(new))

    def render(self, groups_html: "list[str]", placeholder_html: str = "") -> str:
        """Sync the view to `groups_html`. Returns the path taken, for tests
        and logging: "unchanged" / "tail" / "full"."""
        from PyQt6.QtGui import QTextCursor

        view = self._view
        doc = view.document()
        sb = view.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 8
        prev = sb.value()

        if not groups_html:
            if not self._committed and doc.toPlainText().strip():
                return "unchanged"    # placeholder already on screen
            self._set_full(placeholder_html)
            self._committed = []
            return "full"

        k = self._first_divergence(self._committed, groups_html)
        if k == len(self._committed) == len(groups_html):
            return "unchanged"

        # Rewrite from the LAST COMMITTED group even on a pure append. Block
        # 2k does not exist yet when appending (it is one past the end), and
        # inserting at the document end instead merges the new paragraph into
        # the previous block — Qt's insertHtml is inline when the cursor sits
        # in a non-empty block. Backing up one group keeps a single code path
        # whose target block always exists, at the cost of re-laying-out one
        # extra paragraph pair.
        k_eff = min(k, len(self._committed) - 1) if self._committed else -1
        expected_blocks = self._BLOCKS_PER_GROUP * len(self._committed)
        # k_eff must leave at least one untouched leading group, otherwise the
        # "patch" would rewrite the whole document anyway — setHtml is simpler
        # and at least as fast for that, and reporting it as "full" keeps the
        # diagnostic honest (a stream of them means the incremental path is
        # not actually engaging).
        block = (
            doc.findBlockByNumber(self._BLOCKS_PER_GROUP * k_eff)
            if k_eff >= 1 and doc.blockCount() == expected_blocks
            else None
        )
        if block is None or not block.isValid():
            self._set_full("".join(groups_html))
            self._committed = list(groups_html)
            self._restore_scroll(at_bottom, prev)
            return "full"

        was_append = k == len(self._committed)
        view.setUpdatesEnabled(False)
        try:
            cursor = QTextCursor(doc)
            cursor.setPosition(block.position())
            cursor.movePosition(QTextCursor.MoveOperation.End,
                                QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertHtml("".join(groups_html[k_eff:]))
            self._committed = list(groups_html)
        finally:
            view.setUpdatesEnabled(True)
        self._restore_scroll(at_bottom, prev)
        return "append" if was_append else "tail"

    def _set_full(self, doc_html: str) -> None:
        view = self._view
        view.setUpdatesEnabled(False)
        try:
            view.setHtml(doc_html)
        finally:
            view.setUpdatesEnabled(True)

    def _restore_scroll(self, at_bottom: bool, prev: int) -> None:
        # Follow the tail only when the user was already pinned at the
        # bottom; otherwise keep their place so scrolling back through a
        # 3-hour session isn't yanked away on every new line.
        sb = self._view.verticalScrollBar()
        sb.setValue(sb.maximum() if at_bottom else min(prev, sb.maximum()))


def _caption_download(url: str, dest: Path) -> None:
    """Stream a URL to disk with certifi's CA bundle.

    Plain urllib uses the interpreter's default SSL context, which on
    python.org macOS builds has no CA certificates wired in ("Install
    Certificates.command" never run) → CERTIFICATE_VERIFY_FAILED. certifi
    ships with transformers/modelscope, so prefer its bundle and keep the
    default context only as fallback. Never disables verification.
    """
    import ssl
    import urllib.request
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "meetingscribe"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp, \
            open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        next_pct = 0
        while True:
            block = resp.read(256 * 1024)
            if not block:
                break
            f.write(block)
            done += len(block)
            if total and done * 100 // total >= next_pct:
                _log("CAPTION", f"model download {done * 100 // total}% "
                                f"({done // (1024 * 1024)} MB)")
                next_pct += 10


def _ensure_sherpa_model(url: str = _SHERPA_ZIPFORMER_URL) -> Path:
    """Download + extract the fast-mode streaming zipformer once; returns model dir."""
    models_dir = CONFIG_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[-1]
    target = models_dir / name.replace(".tar.bz2", "")
    if (target / "tokens.txt").exists():
        return target
    import tarfile
    tmp = models_dir / (name + ".part")
    _log("CAPTION", f"downloading fast-mode ASR model ({url})")
    _caption_download(url, tmp)
    try:
        with tarfile.open(tmp, "r:bz2") as tf:
            try:
                tf.extractall(models_dir, filter="data")
            except TypeError:  # Python < 3.12: validate members manually
                base = models_dir.resolve()
                for m in tf.getmembers():
                    dest = (models_dir / m.name).resolve()
                    if not str(dest).startswith(str(base) + os.sep):
                        raise RuntimeError(f"unsafe path in model archive: {m.name}")
                tf.extractall(models_dir)
    finally:
        tmp.unlink(missing_ok=True)
    if not (target / "tokens.txt").exists():
        raise RuntimeError(f"model archive extracted but {target} is incomplete")
    _log("CAPTION", f"fast-mode ASR model ready at {target}")
    return target


def _ensure_qwen_gguf(repo: str, filename: str) -> Path:
    """Download the caption-MT Qwen GGUF once into CONFIG_DIR/models.

    Uses the HF `resolve` URL through `_caption_download` (certifi CA
    bundle, progress logging) instead of huggingface_hub so the download
    path is identical to the sherpa model's. Honours HF_ENDPOINT for
    mirror setups.
    """
    models_dir = CONFIG_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / filename
    if target.exists():
        return target
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/{repo}/resolve/main/{filename}"
    tmp = models_dir / (filename + ".part")
    _log("CAPTION", f"downloading qwen MT model ({url})")
    try:
        _caption_download(url, tmp)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    _log("CAPTION", f"qwen MT model ready at {target}")
    return target


def _pick_model_file(model_dir: Path, stem: str) -> str:
    """Prefer the int8-quantised onnx file when present (smaller + faster)."""
    for pattern in (f"{stem}*int8.onnx", f"{stem}*.onnx"):
        hits = sorted(p for p in model_dir.glob(pattern) if p.is_file())
        if hits:
            return str(hits[0])
    raise RuntimeError(f"no {stem}*.onnx found in {model_dir}")


class _SherpaCaptionASR:
    """Fast mode: sherpa-onnx streaming zipformer (bilingual zh-en, ~320 ms)."""

    def __init__(self, lc_cfg: dict, emit_partial, emit_final, hotword: str = ""):
        self._emit_partial = emit_partial
        self._emit_final = emit_final
        try:
            import sherpa_onnx
        except ImportError as e:
            raise RuntimeError(
                "快速模式需要 sherpa-onnx：python3 -m pip install sherpa-onnx"
            ) from e
        cfg_dir = (lc_cfg.get("asr_model_dir") or "")
        model_dir = Path(cfg_dir).expanduser() if cfg_dir else _ensure_sherpa_model()
        ep = _caption_endpoint_rules(lc_cfg)
        kwargs = dict(
            tokens=str(model_dir / "tokens.txt"),
            encoder=_pick_model_file(model_dir, "encoder"),
            decoder=_pick_model_file(model_dir, "decoder"),
            joiner=_pick_model_file(model_dir, "joiner"),
            num_threads=2,
            sample_rate=_CAPTION_SAMPLE_RATE,
            feature_dim=80,
            enable_endpoint_detection=True,
            decoding_method="greedy_search",
            **ep,
        )
        _log("CAPTION", "endpoint rules: " + " ".join(
            f"{k.split('_min_')[0]}={v}" for k, v in sorted(ep.items())))
        # Contextual biasing for domain terms (sandbox / API 网关 / …):
        # space-separated stt.funasr.hotword phrases, one per line for
        # sherpa. Needs modified_beam_search + the model's bpe.vocab.
        hotword = (hotword or "").strip()
        bpe_vocab = model_dir / "bpe.vocab"
        if hotword and bpe_vocab.exists():
            hw_file = CONFIG_DIR / "models" / "caption_hotwords.txt"
            hw_file.parent.mkdir(parents=True, exist_ok=True)
            hw_file.write_text(
                "\n".join(hotword.split()) + "\n", encoding="utf-8")
            kwargs.update(
                decoding_method="modified_beam_search",
                hotwords_file=str(hw_file),
                hotwords_score=1.5,
                modeling_unit="cjkchar+bpe",
                bpe_vocab=str(bpe_vocab),
            )
        try:
            self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)
            if "hotwords_file" in kwargs:
                _log("CAPTION", f"fast ASR hotwords active: {hotword}")
        except Exception as e:
            if "hotwords_file" not in kwargs:
                raise
            # Older sherpa-onnx builds lack hotword kwargs — degrade to
            # greedy decoding rather than losing captions entirely.
            _log("CAPTION", f"hotwords unsupported, falling back to greedy: "
                            f"{type(e).__name__}: {e}")
            for k in ("hotwords_file", "hotwords_score",
                      "modeling_unit", "bpe_vocab"):
                kwargs.pop(k, None)
            kwargs["decoding_method"] = "greedy_search"
            self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)
        # Kept for `_caption_fix_case`: this model emits English in ALL CAPS,
        # and the configured spelling of a hotword must survive the downcase.
        self._hotwords = [w for w in hotword.split() if len(w) >= 2]
        self._stream = self._rec.create_stream()
        self._seg_audio: list = []
        # `_raw_hyp` mirrors what the recognizer last returned (needed to
        # isolate each chunk's delta); `_clean_hyp` is our boundary-deduped
        # version, which is what gets displayed. They diverge on purpose —
        # the recognizer keeps its own append-only history either way.
        self._raw_hyp = ""
        self._clean_hyp = ""

    def reset_session(self, emit_partial, emit_final):
        """Rebind a cached instance to a new engine + fresh decoding stream."""
        self._emit_partial = emit_partial
        self._emit_final = emit_final
        self._stream = self._rec.create_stream()
        self._seg_audio: list = []
        self._raw_hyp = ""
        self._clean_hyp = ""

    def _take_seg_audio(self) -> "np.ndarray":
        audio = (
            np.concatenate(self._seg_audio)
            if self._seg_audio else np.zeros(0, dtype=np.float32)
        )
        self._seg_audio = []
        return audio

    def accept(self, samples: "np.ndarray"):
        if samples.size:
            self._stream.accept_waveform(_CAPTION_SAMPLE_RATE, samples)
            self._seg_audio.append(np.asarray(samples, dtype=np.float32))
            # Safety cap: sherpa's rule3 endpoint bounds utterances (~20 s),
            # but never let the refine buffer grow unboundedly regardless.
            excess = (
                sum(len(a) for a in self._seg_audio)
                - int(_CAPTION_RING_SECONDS * _CAPTION_SAMPLE_RATE)
            )
            while excess > 0 and len(self._seg_audio) > 1:
                excess -= len(self._seg_audio.pop(0))
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)
        text = _caption_fix_case(self._advance_hypothesis(), self._hotwords)
        if self._rec.is_endpoint(self._stream):
            audio = self._take_seg_audio()
            if text:
                self._emit_final(text, audio)
            self._rec.reset(self._stream)
            self._raw_hyp = self._clean_hyp = ""
        elif text:
            self._emit_partial(text)

    def _advance_hypothesis(self) -> str:
        """Pull the recognizer's current hypothesis and fold in only what is
        genuinely new, dropping a chunk-boundary duplicate at the join."""
        raw = self._rec.get_result(self._stream).strip()
        if raw != self._raw_hyp:
            if raw.startswith(self._raw_hyp):
                self._clean_hyp = _append_hypothesis(
                    self._clean_hyp, raw[len(self._raw_hyp):])
            else:
                # The recognizer rewrote earlier text (a rescoring build, not
                # the append-only behaviour we measured) — trust it wholesale
                # rather than splicing two disagreeing histories.
                self._clean_hyp = raw
            self._raw_hyp = raw
        return self._clean_hyp

    def flush(self):
        text = _caption_fix_case(self._advance_hypothesis(), self._hotwords)
        if text:
            self._emit_final(text, self._take_seg_audio())
        self._raw_hyp = self._clean_hyp = ""


class _MarianCaptionMT:
    """Fast mode MT: opus-mt Marian models, one per direction, lazily loaded."""

    def __init__(self, lc_cfg: dict):
        self._ids = {
            "zh-en": lc_cfg.get("mt_zh_en", "Helsinki-NLP/opus-mt-zh-en"),
            "en-zh": lc_cfg.get("mt_en_zh", "Helsinki-NLP/opus-mt-en-zh"),
        }
        self._models: dict = {}
        self._get("zh-en")  # load the primary direction eagerly → early failure

    def _get(self, direction: str):
        if direction not in self._models:
            try:
                from transformers import MarianMTModel, MarianTokenizer
            except ImportError as e:
                raise RuntimeError(
                    "字幕翻译需要 transformers：python3 -m pip install transformers"
                ) from e
            with _QuietCapture("CAPTION"):
                tok = MarianTokenizer.from_pretrained(self._ids[direction])
                mdl = MarianMTModel.from_pretrained(self._ids[direction])
                mdl.eval()
            self._models[direction] = (tok, mdl)
        return self._models[direction]

    def translate(self, text: str) -> str:
        import torch
        direction = "en-zh" if _caption_is_english(text) else "zh-en"
        tok, mdl = self._get(direction)
        with torch.no_grad():
            batch = tok([text], return_tensors="pt", truncation=True, max_length=512)
            # No max_new_tokens: Marian's own generation_config caps length,
            # and passing both triggers a warning print on every call.
            # no_repeat_ngram_size + repetition_penalty are the fix for the
            # degenerate loop greedy decoding falls into on garbage ASR input
            # (one caption came back as 「雅 的 内容」×30 then "Ya"×100).
            out = mdl.generate(
                **batch, num_beams=1,
                no_repeat_ngram_size=_MT_NO_REPEAT_NGRAM,
                repetition_penalty=_MT_REPETITION_PENALTY)
        return tok.decode(out[0], skip_special_tokens=True)


class _NLLBCaptionMT:
    """Optional MT backend: NLLB-200-distilled-600M, bidirectional zh↔en.

    Reachable via `live_captions.mt_provider = "nllb"`. Slower than opus-mt
    (760-1200 ms vs 120-150 ms per line, measured) but stronger on some
    domain terminology, so it stays selectable now that the accurate MODE
    that used to bundle it is gone.
    """

    def __init__(self, lc_cfg: dict):
        model_id = (lc_cfg.get("nllb") or {}).get(
            "mt_model", "facebook/nllb-200-distilled-600M"
        )
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "字幕翻译需要 transformers：python3 -m pip install transformers"
            ) from e
        with _QuietCapture("CAPTION"):
            self._tok = AutoTokenizer.from_pretrained(model_id)
            self._mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            self._mdl.eval()
        if bool((lc_cfg.get("nllb") or {}).get("mt_int8", True)):
            # Dynamic int8 quantization of the Linear layers: 2-4× faster
            # sentence latency on CPU with minor quality loss — the
            # difference between "…" backlogs and usable live translation.
            try:
                import torch
                self._mdl = torch.quantization.quantize_dynamic(
                    self._mdl, {torch.nn.Linear}, dtype=torch.qint8)
                _log("CAPTION", "NLLB dynamic int8 quantization applied")
            except Exception as e:
                _log("CAPTION", f"NLLB int8 quantization skipped: "
                                f"{type(e).__name__}: {e}")

    def translate(self, text: str) -> str:
        import torch
        src, tgt = (
            ("eng_Latn", "zho_Hans")
            if _caption_is_english(text) else ("zho_Hans", "eng_Latn")
        )
        self._tok.src_lang = src
        batch = self._tok([text], return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = self._mdl.generate(
                **batch,
                forced_bos_token_id=self._tok.convert_tokens_to_ids(tgt),
                num_beams=1,
                no_repeat_ngram_size=_MT_NO_REPEAT_NGRAM,
                repetition_penalty=_MT_REPETITION_PENALTY,
            )
        return self._tok.decode(out[0], skip_special_tokens=True)


_GLOSSARY_SPAN_RE = re.compile(r"[A-Za-z0-9]+")
_GLOSSARY_MIXED_RE = re.compile(
    r"[一-鿿]{0,2}[A-Za-z0-9]+[一-鿿]{0,2}")


def _glossary_candidates(text: str, hotwords: "list[str]",
                        limit: int = 12) -> "list[str]":
    """Pick the hotwords worth showing the caption model for THIS line.

    Dumping the whole hotword table (which auto-grows to
    `hotwords.max_count`, 100 by default) into the prompt is what makes a
    1.5B model invent semantic relations between unrelated terms. So only
    terms the line plausibly contains get injected:

    * literal containment (case-insensitive) — the term survived ASR intact;
    * fuzzy match against the line's alnum / CJK-flanked-alnum spans — the
      term got mangled, e.g. `拆特ops` still shares "ops" with `ChatOps`
      (SequenceMatcher 0.6), which is exactly the case correction exists for.

    Fuzzy matching needs ASCII to latch onto: a pure-CJK homophone
    (`李雷` → `里雷`) shares no characters and would need pinyin, so those
    terms are literal-match only. Literal hits rank first, then fuzzy hits by
    descending similarity; the list is capped at `limit`.
    """
    import difflib
    words = [w for w in (hotwords or []) if len(w) >= 2]
    if not words or not text:
        return []
    low = text.lower()
    spans = {
        s.lower() for s in
        _GLOSSARY_SPAN_RE.findall(text) + _GLOSSARY_MIXED_RE.findall(text)
        if len(s) >= 3
    }
    literal = [w for w in words if w.lower() in low]
    if _caption_is_english(text):
        # Fuzzy matching is for ASR-mangled CJK-mixed terms (拆特ops →
        # ChatOps). On an English line it is pure noise: English words share
        # 3-character runs with unrelated hotwords constantly — measured on
        # the real 100-term table, "Now we were told about this RFP around
        # four day" matched Acme (were/ere), and "…they've ked us to
        # commit…" pulled in eight terms. Literal hits only here.
        return literal[:max(1, limit)]
    # Fuzzy matching exists to recover terms the ASR mangled. A span that
    # already matched a term literally is spelled correctly, so it needs no
    # candidates — without this, `用 LangGraph` also dragged in LangSmith,
    # LangChain and LangFuse just for sharing the "lang" prefix.
    spans = {
        s for s in spans
        if not any(s in w.lower() or w.lower() in s for w in literal)
    }
    fuzzy = []
    for w in words:
        wl = w.lower()
        if wl in low:
            continue
        if not any(c.isascii() and c.isalnum() for c in w):
            continue  # pure CJK: no ASCII anchor for a fuzzy comparison
        best = 0.0
        for s in spans:
            m = difflib.SequenceMatcher(None, wl, s)
            # A shared run of ≥3 characters is what makes a near-miss
            # credible. Ratio alone is too loose on short spans: measured on
            # a real 100-term table, "dora" pulled in RMA (0.57) and
            # Portland (0.50) purely on scattered single-letter matches.
            if m.find_longest_match(0, len(wl), 0, len(s)).size < 3:
                continue
            best = max(best, m.ratio())
        if best >= 0.45:
            fuzzy.append((best, w))
    fuzzy.sort(key=lambda p: -p[0])
    return (literal + [w for _, w in fuzzy])[:max(1, limit)]


# GBNF grammar for the qwen fix+translate call. The decoder itself then
# guarantees the shape `_parse_fix_mt` expects, which structurally kills two
# of the observed failure modes: leaked prompt/example text before the 修正
# line, and infinite repetition (each line is length-bounded).
#
# Measured cost on M2 Metal (Qwen2.5-1.5B q4_K_M, min of 3 runs): 1.22 s →
# 1.80 s per finalized line. Prompt length is NOT the factor — few-shot vs
# zero-shot came out at 1.15 s vs 1.22 s — it's the per-token vocab filtering.
# Only the fix path pays it; plain `translate` (which also serves partials)
# stays unconstrained.
_FIX_MT_GRAMMAR = r"""
root ::= "修正：" line "\n" "翻译：" line
line ::= [^\n]{1,400}
"""


def _parse_fix_mt(out: str, original: str) -> "tuple[str, str]":
    """Parse the caption_fix_mt reply (two lines: 修正：… / 翻译：…).

    Defensive against a small model ignoring the contract: when no 翻译：
    line is found, the whole reply is treated as a plain translation and
    the source line is left untouched. The corrected line must be a
    *minimal edit* of the original — length within 0.66x–2.0x (catches
    dropped clauses / hallucinated additions, which pure similarity
    misses because a bare prefix still scores high) AND SequenceMatcher
    similarity ≥ 0.5 (catches a translation or rewrite masquerading as
    the correction). Anything suspicious → keep the raw ASR line.
    """
    import difflib
    fixed = ""
    trans = ""
    for line in (out or "").splitlines():
        s = line.strip()
        if s.startswith(("修正：", "修正:")):
            fixed = s[3:].strip()
        elif s.startswith(("翻译：", "翻译:")):
            trans = s[3:].strip()
    if not trans:
        return original, (out or "").strip()
    ratio = len(fixed) / max(len(original), 1)
    if (
        not fixed
        or not (0.66 <= ratio <= 2.0)
        or difflib.SequenceMatcher(None, original, fixed).ratio() < 0.5
    ):
        fixed = original
    return fixed, trans


class _QwenCaptionMT:
    """Scheme C MT: local Qwen instruct model (GGUF) via llama.cpp.

    Prompt-driven, unlike Marian/NLLB — so it supports a hotword
    glossary: every `stt.funasr.hotword` term that literally appears in
    the source line is injected as a keep-verbatim terminology list, and
    the rest of the line is translated normally by the same model.
    Finalized lines can additionally go through `correct_and_translate`
    (qwen.correct, default on): one combined call that first repairs
    obvious ASR errors (homophones, broken English words) using recent
    caption context + the hotwords this line plausibly contains
    (`_glossary_candidates`), then translates — the same mechanism that
    makes the offline polish step so much better than raw. Output shape is
    pinned by a GBNF grammar when llama.cpp supports it (qwen.grammar).
    """

    def __init__(self, cfg: dict, lc_cfg: dict, hotword: str = ""):
        qcfg = lc_cfg.get("qwen") or {}
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError(
                "字幕翻译（qwen）需要 llama-cpp-python："
                "python3 -m pip install llama-cpp-python"
            ) from e
        path = (qcfg.get("model_path") or "").strip()
        model_path = (
            Path(path).expanduser() if path
            else _ensure_qwen_gguf(
                qcfg.get("model_repo", "Qwen/Qwen2.5-1.5B-Instruct-GGUF"),
                qcfg.get("model_file", "qwen2.5-1.5b-instruct-q4_k_m.gguf"))
        )
        if not model_path.exists():
            raise RuntimeError(f"Qwen GGUF 模型不存在: {model_path}")
        self._prompt_tpl = _resolve_prompt(cfg, "caption_mt")
        self._correct = bool(qcfg.get("correct", True))
        self._glossary_max = max(1, int(qcfg.get("glossary_max", 12)))
        # Grammar first: it decides which fix prompt we use (strict/zero-shot
        # when the decoder enforces the format, few-shot when it can't).
        self._fix_grammar = None
        if bool(qcfg.get("grammar", True)):
            try:
                from llama_cpp import LlamaGrammar
                with _QuietCapture("CAPTION"):
                    self._fix_grammar = LlamaGrammar.from_string(
                        _FIX_MT_GRAMMAR, verbose=False)
            except Exception as e:
                # Bounded repetition `{m,n}` needs a recent llama.cpp; on older
                # builds we lose the structural guarantee and lean on the
                # few-shot example + `_parse_fix_mt`'s defensive checks instead.
                self._fix_grammar = None
                _log("CAPTION", f"GBNF grammar unavailable "
                                f"({type(e).__name__}: {e}); "
                                f"using few-shot caption_fix_mt")
        self._fix_tpl = _resolve_prompt(
            cfg,
            "caption_fix_mt_strict" if self._fix_grammar is not None
            else "caption_fix_mt")
        threads = int(qcfg.get("threads", 0) or 0)
        with _QuietCapture("CAPTION"):
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=int(qcfg.get("n_ctx", 2048)),
                n_threads=threads if threads > 0 else None,
                # Full GPU offload by default: on Apple Silicon this moves
                # inference to Metal (measured 6.8 → 0.04 CPU-seconds per
                # caption line, 2.4s → 0.54s wall). Falls back to CPU
                # automatically when no GPU is available.
                n_gpu_layers=int(qcfg.get("n_gpu_layers", -1)),
                verbose=False,
            )
        self.set_hotword(hotword)
        _log("CAPTION", f"qwen MT ready: {model_path.name}")

    def set_hotword(self, hotword: str):
        self._hotwords = [w for w in (hotword or "").split() if len(w) >= 2]

    def _glossary_hits(self, text: str) -> "list[str]":
        """Literal hits only — the translate path must not rewrite words the
        ASR got right, so near-misses are the fix path's business."""
        low = text.lower()
        return [w for w in self._hotwords if w.lower() in low][
            :self._glossary_max]

    def translate(self, text: str) -> str:
        to_zh = _caption_is_english(text)
        hits = self._glossary_hits(text)
        glossary = (
            "\n译文中出现以下术语时必须原样保留，不要翻译或改写：" + "、".join(hits)
            if hits else ""
        )
        prompt = (
            self._prompt_tpl
            .replace("{src_lang}", "英文" if to_zh else "中文")
            .replace("{dst_lang}", "中文" if to_zh else "英文")
            .replace("{glossary}", glossary)
            .replace("{text}", text)
        )
        # Deliberately NOT grammar-constrained: this path also translates
        # in-flight partials, and GBNF sampling measured +0.30 s (0.42 → 0.72 s
        # per line on M2 Metal) — enough to trip the partial-translation gate.
        # There is no structure to enforce either: a single free-form line,
        # already bounded by max_tokens.
        with _QuietCapture("CAPTION"):
            out = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
        return (out["choices"][0]["message"]["content"] or "").strip()

    def correct_and_translate(self, text: str,
                              context: "list[str]") -> "tuple[str, str]":
        """One combined call: fix obvious ASR errors, then translate.
        Returns (corrected_source, translation). Falls back to plain
        translation when qwen.correct is off."""
        if not self._correct:
            return text, self.translate(text)
        to_zh = _caption_is_english(text)
        hits = _glossary_candidates(text, self._hotwords, self._glossary_max)
        glossary = (
            "\n术语表（仅当原文中的词与下列术语发音相近时才修正为术语写法，发音不相近时忽略此表）："
            + " ".join(hits)
            if hits else ""
        )
        prompt = (
            self._fix_tpl
            .replace("{src_lang}", "英文" if to_zh else "中文")
            .replace("{dst_lang}", "中文" if to_zh else "英文")
            .replace("{glossary}", glossary)
            .replace("{context}", "\n".join(context[-3:]) or "（无）")
            .replace("{text}", text)
        )
        with _QuietCapture("CAPTION"):
            out = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
                **({"grammar": self._fix_grammar}
                   if self._fix_grammar is not None else {}),
            )
        content = (out["choices"][0]["message"]["content"] or "").strip()
        return _parse_fix_mt(content, text)


class LiveCaptionEngine:
    """Streaming ASR → MT pipeline behind the GUI's live bilingual captions.

    Lifecycle: construct → start() → feed() from PortAudio callbacks → stop().
    Events arrive on `on_event(dict)` from worker threads (never the Qt thread):
      {"type": "status", "state": "loading" | "ready" | "stopped"}
      {"type": "error", "message": str}
      {"type": "partial", "text": str}                  # current unfinished line
      {"type": "final", "id": int, "text": str}         # finished source line
      {"type": "translation", "id": int, "text": str}   # its translation
    """

    def __init__(self, cfg: dict, on_event):
        self.cfg = cfg or {}
        self.on_event = on_event
        self._lc = _deep_merge(
            DEFAULT_CONFIG["live_captions"], self.cfg.get("live_captions", {})
        )
        self._buffers: "dict[str, collections.deque]" = {}
        self._buf_lock = threading.Lock()
        self._dropped_secs = 0.0
        self._drop_log_at = 0.0
        self._running = False
        self._asr_thread: "threading.Thread | None" = None
        self._mt_thread: "threading.Thread | None" = None
        self._refine_thread: "threading.Thread | None" = None
        self._mt_queue: "queue.Queue" = queue.Queue()
        self._refine_queue: "queue.Queue" = queue.Queue()
        self._seg_id = 0
        # Last finalized (post-tidy) text, used by _dedup_caption_boundary to
        # strip paraformer chunk-boundary duplicates when the next segment
        # starts with the tail of the previous one.
        self._last_final_text = ""
        # Latest in-flight partial line, translated opportunistically when
        # the MT queue is idle so English appears before the line finalizes.
        self._partial_lock = threading.Lock()
        self._partial_pending = ""
        # Throttle partial caption updates so the line doesn't jitter with
        # every decoder step (user-tunable via live_captions.partial_interval_ms).
        self._partial_min_interval = (
            max(0, int(self._lc.get("partial_interval_ms", 500))) / 1000.0)
        self._last_partial_emit = 0.0
        # MT coalescing: segments queue their id; the worker always translates
        # the LATEST text for that id, so a refine that lands before the
        # original translation ran replaces it instead of doubling the work.
        self._mt_lock = threading.Lock()
        self._mt_texts: dict = {}
        self._mt_done: dict = {}
        self._mt_last_secs = 0.0
        # Cost of the last PARTIAL translation specifically — the gate for
        # whether opportunistic partial translation stays on.
        self._mt_partial_secs = 0.0
        # Speaker attribution for live captions. Streaming voice-print
        # clustering (cam++) is not viable here, but the recorder already
        # tells us which ROLE each audio chunk came from — mic = me, system
        # audio = whoever is on the call — so each finalized segment is
        # attributed to whichever source carried more energy while it was
        # being spoken. Caveat: several remote speakers share one role.
        self._role_energy: dict = {}
        self._role_peak: dict = {}
        self._role_seen: set = set()
        self._speaker_labels = bool(self._lc.get("speaker_labels", True))
        # Voice-print speaker identification (own worker thread, so a 60 ms
        # embedding never sits in front of ASR or MT). Splits the single
        # system-audio channel into the individual people on the call, which
        # channel role alone cannot do.
        _sid = self._lc.get("speaker_id") or {}
        self._spk_enabled = bool(_sid.get("enabled", True))
        self._spk_model_id = (_sid.get("model") or "cam++").strip()
        self._spk_min_samples = int(
            max(0.2, float(_sid.get("min_secs", 1.0))) * _CAPTION_SAMPLE_RATE)
        self._spk_max_backlog = max(1, int(_sid.get("max_backlog", 3)))
        self._spk_dropped = 0
        self._spk_clusterer = _SpeakerClusterer(
            threshold=float(_sid.get("threshold", 0.6)),
            max_speakers=int(_sid.get("max_speakers", 8)),
            min_segments=int(_sid.get("min_segments", 2)))
        # Segments whose cluster has not earned a number yet, per cluster.
        # They get tagged retroactively the moment it does.
        self._spk_pending: "dict[int, list[int]]" = {}
        self._spk_queue: "queue.Queue" = queue.Queue()
        self._spk_thread: "threading.Thread | None" = None
        # Periodic batch re-check: every `interval_minutes` the finalized
        # lines of that window go to the SAME provider that polishes
        # transcripts, with the whole window as context. That context is the
        # point — a per-line pass cannot know that 「拆特ops」 three lines ago
        # and 「ChatOps」 now are the same thing.
        _rev = self._lc.get("review") or {}
        self._review_enabled = bool(_rev.get("enabled", True))
        self._review_interval = max(
            30.0, float(_rev.get("interval_minutes", 5)) * 60.0)
        self._review_max_lines = max(1, int(_rev.get("max_lines", 120)))
        self._review_provider = (_rev.get("provider") or "").strip()
        # A line the model omitted (truncated reply) goes back for another
        # round, but only this many times — otherwise one stubborn line
        # would sit at the head of the buffer forever.
        self._review_max_tries = max(0, int(_rev.get("max_tries", 2)))
        # Total backlog cap. `max_lines` bounds one BATCH; without this the
        # buffer itself is unbounded, so a stalled provider both grows memory
        # without limit and turns the pass useless — a correction that lands
        # 40 minutes after the line scrolled away is not worth showing. Never
        # below max_lines: that would throw away rows one call could handle.
        self._review_max_buffer = max(
            self._review_max_lines, int(_rev.get("max_buffer", 240)))
        # Read-only context: the tail of lines this pass already corrected.
        # Cross-batch terminology consistency needs it — batch N settling on
        # 「ChatOps」 is useless if batch N+1 independently picks 「拆特ops」
        # again. Read-only on purpose: those lines were already shown as
        # corrected, and re-editing them would make the pane rewrite history
        # every interval. Costs input tokens only, and the measured per-call
        # cost is 41.6 s of fixed overhead against 3.4 s per line, so a dozen
        # extra reference lines is noise.
        self._review_context_lines = max(0, int(_rev.get("context_lines", 12)))
        self._review_context: "collections.deque" = collections.deque(
            maxlen=max(1, self._review_context_lines))
        self._review_lock = threading.Lock()
        self._review_buffer: "list[dict]" = []
        self._review_depth_warned = False
        self._review_thread: "threading.Thread | None" = None
        # Rolling discourse context for the qwen correct+translate call:
        # the last few finalized (corrected) source lines.
        self._mt_context: "collections.deque" = collections.deque(maxlen=3)
        # Fast mode re-decodes finalized segments with the offline FunASR
        # stack in the background ("对过去的断句二次校验").
        self._refine_enabled = bool(self._lc.get("refine", True))
        # A re-decode costs ~1.0–1.4 s per segment (measured on M2, offline
        # paraformer-large + vad + punc). When speech comes in faster than
        # that the queue grows without bound and refinements land minutes
        # after the line scrolled away — worse than not refining at all. Cap
        # the backlog and drop the excess: the streaming text stays on screen.
        self._refine_max_backlog = max(
            1, int(self._lc.get("refine_max_backlog", 3)))
        self._refine_dropped = 0

    # ── event plumbing ──
    def _emit(self, **ev):
        try:
            self.on_event(ev)
        except Exception as e:
            _log("ERR", f"caption on_event: {type(e).__name__}: {e}")

    def _emit_partial(self, text: str):
        # Tidy the IN-FLIGHT line too, not just finalized ones. A streaming
        # decoder oscillates between hypotheses while a sentence is still
        # open, and it shows: measured on one real meeting, the partial line
        # carried 4-19x the duplication of the offline transcript of the SAME
        # audio (single-char 85 vs 21 per 1000, triples 25 vs 1.3, 2-char word
        # repeats 55 vs 5.4) — 「大大大家大家」/「在在在在」/「训练练练习」.
        # Running the same collapse here removes the triples and word repeats
        # outright, and gives the partial translation a clean input as well.
        text = _caption_tidy_zh(text)
        with self._partial_lock:
            self._partial_pending = text
        now = time.time()
        if now - self._last_partial_emit < self._partial_min_interval:
            return  # pending text still reaches MT; only the UI update waits
        self._last_partial_emit = now
        self._emit(type="partial", text=text)

    def _queue_translation(self, sid: int, text: str):
        with self._mt_lock:
            self._mt_texts[sid] = text
        self._mt_queue.put(sid)

    def _review_record(self, sid: int, text: str):
        if not self._review_enabled:
            return
        with self._review_lock:
            for row in self._review_buffer:
                if row["id"] == sid:      # refine/correct rewrote the line
                    row["text"] = text
                    return
            self._review_buffer.append({"id": sid, "text": text, "dst": ""})
            dropped, depth = self._trim_review_buffer()
        # Logged outside the lock, and never silently: a bounded pass that
        # reports "batch=8 parsed=8" while having thrown rows away reads as
        # full coverage when it wasn't. Oldest go first — they scrolled off
        # screen long ago, and under sustained overflow keeping the recent
        # window is what preserves any value at all.
        if dropped:
            _log("CAPTION",
                 f"review buffer full ({self._review_max_buffer}), dropped "
                 f"{len(dropped)} unreviewed seg="
                 f"{dropped[0]['id']}..{dropped[-1]['id']}")
        elif depth > self._review_max_lines and not self._review_depth_warned:
            # One full batch behind: the early warning before anything is lost.
            self._review_depth_warned = True
            _log("CAPTION", f"review backlog {depth} rows > one batch "
                            f"({self._review_max_lines}) — provider behind")
        elif depth <= self._review_max_lines:
            self._review_depth_warned = False

    def _trim_review_buffer(self) -> "tuple[list[dict], int]":
        """Enforce the total cap. Caller must hold `_review_lock`."""
        depth = len(self._review_buffer)
        excess = depth - self._review_max_buffer
        if excess <= 0:
            return [], depth
        dropped = self._review_buffer[:excess]
        del self._review_buffer[:excess]
        return dropped, len(self._review_buffer)

    def _review_record_translation(self, sid: int, dst: str):
        if not self._review_enabled:
            return
        with self._review_lock:
            for row in self._review_buffer:
                if row["id"] == sid:
                    row["dst"] = dst
                    return

    def _take_segment_role(self) -> "str | None":
        """Role that carried this segment, or None when labels would be noise.

        Requires two roles to have been heard in the session: labelling every
        line 「我」 in a solo recording (or when system audio never opened)
        tells the user nothing.
        """
        tally, self._role_energy = self._role_energy, {}
        if not self._speaker_labels or len(self._role_seen) < 2 or not tally:
            return None
        return max(tally.items(), key=lambda kv: kv[1])[0]

    def _finalize_segment(self, text: str, audio: "np.ndarray | None" = None):
        with self._partial_lock:
            self._partial_pending = ""
        text = _caption_tidy_zh(text)
        text = _dedup_caption_boundary(self._last_final_text, text).strip()
        if not text:
            return  # entire segment was a duplicate of the previous tail
        if _caption_is_noise(text):
            # Never reaches the pane, MT, refine or the voice print: a junk
            # segment used to cost a caption line AND a phantom speaker.
            _log("CAPTION", f"noise segment dropped: {text!r}")
            return
        self._last_final_text = text
        self._seg_id += 1
        sid = self._seg_id
        role = self._take_segment_role()
        self._emit(type="final", id=sid, text=text, role=role)
        self._review_record(sid, text)
        self._queue_translation(sid, text)
        # The refine pass re-decodes with paraformer-zh — a CHINESE model. On
        # an English line it does not correct, it corrupts: a real session
        # (2026-08-26 15:15) logged changed=True on 53 of 56 English segments,
        # 1.7–2.5 s each, and the user saw the pass "do nothing useful". So
        # English-dominant segments keep the bilingual streaming model's own
        # text, which is the better source for them.
        if (
            self._refine_enabled
            and audio is not None
            and getattr(audio, "size", 0) >= int(0.5 * _CAPTION_SAMPLE_RATE)
            and not _caption_is_english(text)
        ):
            if self._refine_queue.qsize() >= self._refine_max_backlog:
                self._refine_dropped += 1
                _log("CAPTION",
                     f"refine skipped seg={sid} (backlog="
                     f"{self._refine_queue.qsize()} >= "
                     f"{self._refine_max_backlog}; dropped_total="
                     f"{self._refine_dropped})")
            else:
                self._refine_queue.put((sid, audio, text))
        if (
            self._spk_enabled
            and audio is not None
            and getattr(audio, "size", 0) >= self._spk_min_samples
        ):
            if self._spk_queue.qsize() >= self._spk_max_backlog:
                self._spk_dropped += 1
                _log("CAPTION",
                     f"speaker id skipped seg={sid} (backlog="
                     f"{self._spk_queue.qsize()}; dropped_total="
                     f"{self._spk_dropped})")
            else:
                self._spk_queue.put((sid, audio, role))

    # ── audio input; called on PortAudio callback threads — keep it light ──
    def feed(self, source: str, frames: "np.ndarray", samplerate: int):
        if not self._running:
            return
        try:
            # Downmix ALL of a device's channels. Taking only channel 0
            # silently discarded content: playing a role-separated stereo
            # recording back through BlackHole put the speech on channel 1,
            # so the clean digital copy read as pure silence (RMS 0.0000) and
            # the captions came from the microphone re-recording the speakers
            # through the air. Same loss would hit any hard-right-panned
            # participant in a live call.
            mono = (frames.mean(axis=1)
                    if getattr(frames, "ndim", 1) > 1 else frames)
            dropped = 0.0
            with self._buf_lock:
                buf = self._buffers.setdefault(source, collections.deque())
                buf.append((np.asarray(mono, dtype=np.float32), int(samplerate)))
                total = sum(len(c) / sr for c, sr in buf)
                while buf and total > _CAPTION_RING_SECONDS:
                    c, sr = buf.popleft()
                    total -= len(c) / sr
                    dropped += len(c) / sr
            if dropped:
                # Audio thrown away because the ASR fell this far behind. It
                # used to happen in complete silence, which made a whole class
                # of "why did the captions miss that" unanswerable — and the
                # `captions` replay tool hit it immediately by feeding faster
                # than the recognizer consumes.
                self._dropped_secs += dropped
                now = time.time()
                if now - self._drop_log_at > 5.0:
                    self._drop_log_at = now
                    _log("CAPTION",
                         f"ring buffer overflow: dropped {dropped:.1f}s from "
                         f"{source!r} (ASR behind by >{_CAPTION_RING_SECONDS:.0f}s; "
                         f"session total {self._dropped_secs:.1f}s)")
        except Exception as e:
            _log("ERR", f"caption feed: {type(e).__name__}: {e}")

    def backlog_secs(self) -> float:
        """Seconds of audio waiting on the slowest source's ring buffer.

        Exposed so a caller can pace itself: past `_CAPTION_RING_SECONDS`
        `feed()` starts dropping, which the `captions` replay tool would
        otherwise hit immediately by pushing samples faster than the
        recognizer consumes them.
        """
        with self._buf_lock:
            return max((sum(len(c) / sr for c, sr in buf)
                        for buf in self._buffers.values()), default=0.0)

    def _drain_mixed(self) -> "np.ndarray":
        """Drain all per-source backlogs, resample to 16 kHz, sum into mono."""
        with self._buf_lock:
            drained = {s: list(b) for s, b in self._buffers.items()}
            for b in self._buffers.values():
                b.clear()
        tracks = []
        for source, chunks in drained.items():
            if not chunks:
                continue
            parts = [_caption_resample(c, sr) for c, sr in chunks]
            track = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
            if track.size:
                tracks.append(track)
                # Energy per role, accumulated until the segment finalizes.
                # Scored RELATIVE to each source's own running peak: a mic at
                # -30 dBFS and a digital playback stream at -6 dBFS are not
                # comparable in absolute RMS, and comparing them raw made the
                # louder channel win nearly every segment (observed: almost
                # every line tagged 对方). Peak decays slowly so a one-off
                # loud burst doesn't desensitise a source for the session.
                rms = float(np.sqrt(np.mean(np.square(track))))
                peak = max(self._role_peak.get(source, 0.0) * 0.999, rms,
                           _CAPTION_ROLE_RMS_FLOOR)
                self._role_peak[source] = peak
                self._role_energy[source] = (
                    self._role_energy.get(source, 0.0)
                    + (rms / peak) * track.size)
                if peak > _CAPTION_ROLE_RMS_FLOOR:
                    # Peak over the session, not the current window: a
                    # built-in mic never reaches the old 0.01 instantaneous
                    # bar, so `side` came back None for all 208 segments of a
                    # real meeting and the tag never got its colour.
                    self._role_seen.add(source)
        if not tracks:
            return np.zeros(0, dtype=np.float32)
        n = max(len(t) for t in tracks)
        mixed = np.zeros(n, dtype=np.float32)
        for t in tracks:
            mixed[: len(t)] += t
        return np.clip(mixed, -1.0, 1.0)

    # ── lifecycle ──
    def start(self):
        if self._running:
            return
        self._running = True
        self._asr_thread = threading.Thread(
            target=self._asr_worker, daemon=True, name="caption-asr")
        self._mt_thread = threading.Thread(
            target=self._mt_worker, daemon=True, name="caption-mt")
        self._asr_thread.start()
        self._mt_thread.start()
        if self._refine_enabled:
            self._refine_thread = threading.Thread(
                target=self._refine_worker, daemon=True, name="caption-refine")
            self._refine_thread.start()
        if self._spk_enabled:
            self._spk_thread = threading.Thread(
                target=self._speaker_worker, daemon=True, name="caption-spk")
            self._spk_thread.start()
        if self._review_enabled:
            self._review_thread = threading.Thread(
                target=self._review_worker, daemon=True, name="caption-review")
            self._review_thread.start()
        _log("CAPTION", f"engine start refine={self._refine_enabled}")

    def stop(self):
        self._running = False
        self._mt_queue.put(None)
        self._refine_queue.put(None)
        self._spk_queue.put(None)
        for t in (self._asr_thread, self._mt_thread, self._refine_thread,
                  self._spk_thread, self._review_thread):
            if t:
                t.join(timeout=5.0)
        self._asr_thread = self._mt_thread = self._refine_thread = None
        self._emit(type="status", state="stopped")
        _log("CAPTION", "engine stopped")

    # ── backend selection (overridable in tests) ──
    def _stt_funasr_cfg(self) -> dict:
        return _deep_merge(
            DEFAULT_CONFIG["stt"], self.cfg.get("stt", {})).get("funasr", {})

    def _load_asr_backend(self):
        # Endpoint rules are baked into the recognizer at construction, so they
        # belong in the cache key — otherwise an A/B run in one process would
        # silently reuse the first set of thresholds.
        ep = _caption_endpoint_rules(self._lc)
        key = ("asr",) + tuple(sorted(ep.items()))
        backend = _caption_backend_cache.get(key)
        if backend is not None:
            backend.reset_session(self._emit_partial, self._finalize_segment)
            return backend
        backend = _SherpaCaptionASR(
            self._lc, self._emit_partial, self._finalize_segment,
            hotword=self._stt_funasr_cfg().get("hotword", ""))
        _caption_backend_cache[key] = backend
        return backend

    def _load_mt_backend(self):
        provider = (self._lc.get("mt_provider") or "default").strip().lower()
        hotword = self._stt_funasr_cfg().get("hotword", "")
        if provider == "qwen":
            backend = _caption_backend_cache.get(("mt", "qwen"))
            if backend is not None:
                backend.set_hotword(hotword)  # hotword may have grown since caching
                return backend
            try:
                backend = _QwenCaptionMT(self.cfg, self._lc, hotword=hotword)
                _caption_backend_cache[("mt", "qwen")] = backend
                return backend
            except Exception as e:
                _log("CAPTION",
                     f"qwen MT unavailable, falling back to opus-mt: "
                     f"{type(e).__name__}: {e}")
        # marian / nllb pin the MT model independently of the ASR mode. That
        # combination matters: fast mode's bilingual streaming ASR is the
        # better English recognizer, but on domain terminology the two MT
        # models trade wins — measured on real meeting lines, NLLB kept `PoC`
        # and `包裹` where opus-mt produced 「钻石交易市场」and 「土地」, while
        # opus-mt kept `RFP` where NLLB dropped a letter. NLLB also costs
        # 760–1200 ms per line against opus-mt's 120–150 ms. No default is
        # right for everyone, hence the knob.
        want = "nllb" if provider == "nllb" else "marian"
        key = ("mt", want)
        backend = _caption_backend_cache.get(key)
        if backend is None:
            backend = (
                _MarianCaptionMT(self._lc) if want == "marian"
                else _NLLBCaptionMT(self._lc)
            )
            _caption_backend_cache[key] = backend
        return backend

    # ── workers ──
    def _asr_worker(self):
        self._emit(type="status", state="loading")
        try:
            backend = self._load_asr_backend()
        except Exception as e:
            _log("ERR", f"caption asr load: {type(e).__name__}: {e}")
            self._emit(type="error", message=str(e))
            self._running = False
            return
        self._emit(type="status", state="ready")
        try:
            while self._running:
                time.sleep(_CAPTION_PACER_INTERVAL)
                chunk = self._drain_mixed()
                try:
                    backend.accept(chunk)
                except Exception as e:
                    _log("ERR", f"caption asr step: {type(e).__name__}: {e}")
        finally:
            try:
                backend.flush()
            except Exception as e:
                _log("CAPTION", f"asr flush: {type(e).__name__}: {e}")

    def _mt_worker(self):
        backend = None

        def _ensure_backend():
            nonlocal backend
            if backend is None:
                try:
                    backend = self._load_mt_backend()
                except Exception as e:
                    _log("ERR", f"caption mt load: {type(e).__name__}: {e}")
                    self._emit(type="error", message=str(e))
                    backend = False  # sentinel: don't retry every segment
            return backend

        # Preload so the first finalized line doesn't pay model-load latency.
        _ensure_backend()
        last_partial = ""
        while True:
            try:
                item = self._mt_queue.get(timeout=0.25)
            except queue.Empty:
                # Idle tick: opportunistically translate the in-flight
                # partial so English shows up before the line finalizes.
                # Skipped when the PARTIAL path itself is slow (e.g. NLLB on
                # CPU): a partial in flight delays the next finalized line by
                # its own duration, so that — not the fix path's cost — is the
                # right thing to gate on. Gating on the shared last-call time
                # would let the grammar-constrained fix call (~1.8 s) disable
                # partial translation outright.
                if not self._running or not _ensure_backend():
                    continue
                if self._mt_partial_secs > 0.8:
                    continue
                with self._partial_lock:
                    text = self._partial_pending
                if text and text != last_partial and len(text) >= 6:
                    last_partial = text
                    try:
                        t0 = time.time()
                        out = _collapse_mt_repeats(backend.translate(text))
                        self._mt_partial_secs = time.time() - t0
                        self._emit(type="partial_translation", text=out)
                    except Exception as e:
                        _log("ERR", f"caption partial translate: "
                                    f"{type(e).__name__}: {e}")
                continue
            if item is None:
                break
            if not _ensure_backend():
                continue
            with self._mt_lock:
                text = self._mt_texts.get(item, "")
            if not text or self._mt_done.get(item) == text:
                continue  # superseded duplicate or nothing to do
            try:
                t0 = time.time()
                fix = getattr(backend, "correct_and_translate", None)
                if fix is not None:
                    # Combined ASR-correction + translation (qwen backend).
                    # The corrected line replaces the caption via the same
                    # `refined` event the FunASR refine pass uses; if that
                    # pass later re-queues this id with new text, the
                    # coalescing above simply corrects it again.
                    fixed, out = fix(text, list(self._mt_context))
                    corrected = bool(fixed) and fixed != text
                    if corrected:
                        self._emit(type="refined", id=item, text=fixed)
                    self._mt_context.append(fixed or text)
                else:
                    corrected = False
                    out = backend.translate(text)
                self._mt_last_secs = time.time() - t0
                self._mt_done[item] = text
                out = _collapse_mt_repeats(out)
                self._emit(type="translation", id=item, text=out)
                self._review_record_translation(item, out)
                _log("CAPTION",
                     f"mt seg={item} took={self._mt_last_secs:.2f}s "
                     f"corrected={corrected} out_chars={len(out or '')}")
            except Exception as e:
                _log("ERR", f"caption translate: {type(e).__name__}: {e}")

    def _take_review_batch(self) -> "list[dict]":
        """Pop up to `max_lines` buffered rows; the rest wait for the next
        round so a long window can't build an unbounded prompt."""
        with self._review_lock:
            batch = self._review_buffer[:self._review_max_lines]
            self._review_buffer = self._review_buffer[len(batch):]
        return batch

    def _review_llm(self, prompt: str) -> str:
        """Overridable in tests. Uses the transcript-polish provider, i.e.
        the same big model whose offline output is visibly better than
        anything the realtime path can produce."""
        provider = self._review_provider or self.cfg.get(
            "polish_provider", "claude")
        return _llm_run(prompt, provider, self.cfg, "字幕复核")

    def _review_worker(self):
        """Every `interval_minutes`, re-check that window's finalized lines.

        Runs on its own thread and never touches the realtime path: a batch
        LLM call takes seconds to tens of seconds, which is fine for lines
        the user read minutes ago. Failure is swallowed — the captions simply
        stay as they were.
        """
        next_run = time.time() + self._review_interval
        while self._running:
            time.sleep(0.5)
            if time.time() < next_run:
                continue
            # next_run advances BEFORE the call on purpose: if a batch takes
            # longer than the interval (measured 53 s for 6 lines through the
            # claude CLI, most of it process startup), the next tick fires
            # immediately and the worker catches up instead of drifting.
            next_run = time.time() + self._review_interval
            batch = self._take_review_batch()
            if not batch:
                continue
            try:
                t0 = time.time()
                with self._review_lock:
                    context = list(self._review_context)
                prompt = (
                    _resolve_prompt(self.cfg, "caption_review")
                    .replace("{hotwords}",
                             _cfg_hotword(self.cfg) or "（无）")
                    .replace("{context}", _format_caption_review_context(
                        context, self._review_context_lines))
                    .replace("{lines}", _format_caption_review_lines(batch))
                )
                fixes = _parse_caption_review(self._review_llm(prompt), batch)
                changed, dst_changed, missed, settled = 0, 0, [], []
                for row in batch:
                    got = fixes.get(row["id"])
                    if not got:
                        # Popped from the buffer but absent from the reply —
                        # a truncated or partial answer. Without putting it
                        # back the line would never be reviewed again.
                        if row.get("tries", 0) < self._review_max_tries:
                            row["tries"] = row.get("tries", 0) + 1
                            missed.append(row)
                        continue
                    text, dst = got
                    # Every rewrite is logged with its before/after, one line
                    # each, so a session can be audited after the fact: this
                    # pass silently changes lines the user already read, and
                    # aggregate counts alone make that unverifiable.
                    if text != row["text"]:
                        self._emit(type="refined", id=row["id"], text=text)
                        changed += 1
                        _log("CAPTION",
                             f"review seg={row['id']} src: "
                             f"{_review_log_text(row['text'])} → "
                             f"{_review_log_text(text)}")
                    if dst and dst != row["dst"]:
                        self._emit(type="translation", id=row["id"], text=dst)
                        dst_changed += 1
                        _log("CAPTION",
                             f"review seg={row['id']} dst: "
                             f"{_review_log_text(row['dst'])} → "
                             f"{_review_log_text(dst)}")
                    # Settled state feeds the next batch's read-only context,
                    # so terminology decided here carries forward.
                    settled.append({"id": row["id"], "text": text,
                                    "dst": dst or row["dst"]})
                if settled:
                    with self._review_lock:
                        self._review_context.extend(settled)
                if missed:
                    # Front of the queue, oldest first: retrying is bounded by
                    # `tries`, so a model that keeps ignoring a line drops it
                    # rather than blocking the buffer forever. The cap is
                    # re-checked here too — a requeue during a busy window can
                    # push the buffer over on its own.
                    with self._review_lock:
                        self._review_buffer[:0] = missed
                        dropped, _ = self._trim_review_buffer()
                    if dropped:
                        _log("CAPTION",
                             f"review requeue over cap, dropped {len(dropped)} "
                             f"seg={dropped[0]['id']}..{dropped[-1]['id']}")
                _log("CAPTION",
                     f"review batch={len(batch)} parsed={len(fixes)} "
                     f"src_changed={changed} dst_changed={dst_changed} "
                     f"requeued={len(missed)} took={time.time() - t0:.1f}s")
            except Exception as e:
                _log("ERR", f"caption review: {type(e).__name__}: {e}")

    def _load_speaker_backend(self):
        return _CaptionSpeakerId(self._spk_model_id)

    def _speaker_worker(self):
        """Voice-print identification for finalized segments.

        Emits `speaker` events carrying the display number and whether this
        speaker is the local one, so the pane can relabel a line from the
        channel-based guess (`[对方]`) to the actual person (`[说话人2]`).
        Failure is non-fatal: the label just stays at whatever role said.
        """
        backend = None
        while True:
            item = self._spk_queue.get()
            if item is None:
                break
            if backend is None:
                try:
                    backend = self._load_speaker_backend()
                except Exception as e:
                    _log("ERR", f"caption speaker id load: "
                                f"{type(e).__name__}: {e}")
                    backend = False   # sentinel: off for this session
            if not backend:
                continue
            sid, audio, role = item
            try:
                t0 = time.time()
                emb = backend.embed(audio)
                if emb is None or emb.size == 0:
                    continue
                idx = self._spk_clusterer.assign(emb, role)
                if idx < 0:
                    continue
                took = time.time() - t0
                num = self._spk_clusterer.display_number(idx)
                if num is None:
                    # Unproven voice: hold the segment instead of minting a
                    # 说话人N for what may be a cough. If a second segment
                    # joins this cluster the held ids get tagged then, so a
                    # real speaker loses only the delay, not the label.
                    self._spk_pending.setdefault(idx, []).append(sid)
                    _log("CAPTION",
                         f"speaker seg={sid} cluster={idx} n=pending "
                         f"(seen={self._spk_clusterer.segment_count(idx)}/"
                         f"{self._spk_clusterer.min_segments}) role={role} "
                         f"took={took:.2f}s")
                    continue
                side = self._spk_clusterer.majority_side(idx)
                held = self._spk_pending.pop(idx, [])
                for old in held:
                    self._emit(type="speaker", id=old, speaker=num, side=side)
                self._emit(type="speaker", id=sid, speaker=num, side=side)
                _log("CAPTION",
                     f"speaker seg={sid} cluster={idx} n={num} side={side} "
                     f"role={role} took={took:.2f}s"
                     + (f" backfilled={held}" if held else ""))
            except Exception as e:
                _log("ERR", f"caption speaker id: {type(e).__name__}: {e}")

    def _load_refine_backend(self):
        stt = self._stt_funasr_cfg()
        key = (
            stt.get("model", "paraformer-zh"),
            stt.get("vad_model", "fsmn-vad"),
            stt.get("punc_model", "ct-punc"),
        )
        if key not in _funasr_model_cache:
            _funasr_model_cache[key] = _load_funasr_automodel(*key)
        return _funasr_model_cache[key]

    def _refine_worker(self):
        """Fast-mode second pass: re-decode finalized segment audio with the
        offline FunASR stack (same cached models as the batch pipeline) and
        replace the caption line + its translation when the text improves."""
        model = None
        while True:
            item = self._refine_queue.get()
            if item is None:
                break
            if model is None:
                try:
                    model = self._load_refine_backend()
                except Exception as e:
                    _log("ERR", f"caption refine load: {type(e).__name__}: {e}")
                    model = False  # sentinel: refinement off for this session
            if not model:
                continue
            sid, audio, before = item
            try:
                hw = (self._stt_funasr_cfg().get("hotword") or "").strip()
                kwargs = {"hotword": hw} if hw else {}
                t0 = time.time()
                with _QuietCapture("CAPTION"):
                    res = model.generate(input=audio, batch_size_s=60, **kwargs)
                text = (res[0].get("text") or "").strip() if res else ""
                text = _caption_tidy_zh(text)
                # One line per segment quantifies what the pass actually buys:
                # `changed=False` runs are pure cost, so a session's ratio tells
                # you whether refine is worth its ~1.2 s and RAM.
                _log("CAPTION",
                     f"refine seg={sid} took={time.time() - t0:.2f}s "
                     f"chars={len(before)}→{len(text)} "
                     f"changed={bool(text) and text != before}")
                if text:
                    self._emit(type="refined", id=sid, text=text)
                    self._review_record(sid, text)
                    # Re-queue under the same id: coalescing in _mt_worker
                    # drops the original translation if it hasn't run yet.
                    self._queue_translation(sid, text)
            except Exception as e:
                _log("ERR", f"caption refine: {type(e).__name__}: {e}")


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
    spk_model   = (pcfg.get("spk_model") or "").strip()
    hotword     = pcfg.get("hotword", "")
    chunk_secs       = int(pcfg.get("chunk_secs", 300))
    _workers_cfg     = int(pcfg.get("workers", 0))
    max_workers      = _workers_cfg if _workers_cfg > 0 else max(6, (os.cpu_count() or 4) // 2)
    _log(
        "STT",
        f"funasr concurrency: workers={max_workers} "
        f"(source={'config' if _workers_cfg > 0 else 'auto'}, "
        f"config_value={_workers_cfg}, cpu_count={os.cpu_count()}) "
        f"chunk_secs={chunk_secs}",
    )

    with wave.open(str(audio_path), "rb") as wf:
        total_frames = wf.getnframes()
        framerate    = wf.getframerate()
        n_channels   = wf.getnchannels()
        sampwidth    = wf.getsampwidth()
    total_secs = total_frames / framerate

    def _load_model():
        key = _funasr_cache_key(asr_model, vad_model, punc_model, spk_model)
        if key not in _funasr_model_cache:
            _funasr_model_cache[key] = _load_funasr_automodel(
                asr_model, vad_model, punc_model, spk_model,
            )
        return _funasr_model_cache[key]

    def _format(items, offset_s=0.0):
        """Speaker-labelled lines when diarization is on, plain otherwise."""
        if not spk_model:
            return _items_to_lines(items, offset_s)
        lines = _spk_lines(items, offset_s)
        if lines:
            return lines
        _log("WARN", f"spk_model='{spk_model}' produced no sentence_info; "
                     f"falling back to unlabelled lines")
        return _items_to_lines(items, offset_s)

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

    # Diarization forces the serial path: cam++ clusters voices per generate()
    # call, so chunk 2's 说话人1 is a different person from chunk 1's. Chunked
    # labels would be worse than none — the notes LLM would merge two people
    # into one. The cost is real (no 6-way parallelism), so it's logged and
    # printed, and the whole feature stays opt-in via stt.funasr.spk_model.
    diarize_serial = bool(spk_model) and chunk_secs > 0 and total_secs > chunk_secs
    if diarize_serial:
        _log("STT", f"diarization on (spk_model='{spk_model}'): forcing serial "
                    f"single-pass over {total_secs / 60:.1f} min "
                    f"(chunk_secs={chunk_secs} ignored) to keep speaker ids "
                    f"consistent across the recording")
        print(f"[转写] 已开启说话人区分（{spk_model}）：为保证说话人编号全程一致，"
              f"本次不分块并发，{total_secs / 60:.1f} 分钟录音将单趟转写，耗时明显变长")

    # ── 短录音 / 说话人区分：直接串行转写 ────────────────────────────────────
    if chunk_secs <= 0 or total_secs <= chunk_secs or diarize_serial:
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
        result_text = "\n".join(_format(results))
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
            return idx, _format(items, offset_s)
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


# Prompts moved to module-top `_PROMPT_DEFAULTS` / `DEFAULT_CONFIG["prompts"]`.
# Resolution happens via `_resolve_prompt(cfg, mode, key)` — see top of file.


# ── Pipeline cancellation primitives ─────────────────────────────────────────
#
# The GUI's "停止任务" button needs to truly interrupt the active pipeline (not
# just hide the front-end progress). Cancellation is cooperative:
#
#   * `_PIPELINE_CANCEL` — a module-wide ``threading.Event``. Stage boundaries
#     in `_PipelineWorker.run` plus every LLM helper call `_pipeline_check_
#     cancelled` before doing work; if the event is set they raise
#     `_PipelineCancelled`, which `_PipelineWorker.run` catches and turns into
#     a `cancelled` Qt signal.
#   * `_PIPELINE_POPENS` — every long-running ``subprocess.Popen`` the
#     pipeline spawns (currently just `claude -p …`) is registered here.
#     `_pipeline_kill_popens()` calls ``.kill()`` on each so the subprocess
#     dies immediately instead of waiting for its natural completion.
#
# Only one pipeline runs at a time (the UI disables the action buttons during
# a run), so module-level state is safe. The event is cleared in
# `_PipelineWorker.__init__` so a fresh worker always starts uncancelled.
#
# HTTP-backed providers (`_llm_openai`, `_llm_gemini`) and FunASR's
# `m.generate(...)` are uninterruptible mid-call — the in-flight call has to
# return naturally — but `_pipeline_check_cancelled()` at the next stage
# boundary still short-circuits the rest of the pipeline.

class _PipelineCancelled(RuntimeError):
    """Raised inside `_PipelineWorker.run` when the user clicks 停止任务."""


_PIPELINE_CANCEL = threading.Event()
_PIPELINE_POPENS_LOCK = threading.Lock()
_PIPELINE_POPENS: "list[subprocess.Popen]" = []


def _pipeline_register_popen(p: "subprocess.Popen") -> None:
    with _PIPELINE_POPENS_LOCK:
        _PIPELINE_POPENS.append(p)


def _pipeline_unregister_popen(p: "subprocess.Popen") -> None:
    with _PIPELINE_POPENS_LOCK:
        try:
            _PIPELINE_POPENS.remove(p)
        except ValueError:
            pass


def _pipeline_kill_popens() -> None:
    with _PIPELINE_POPENS_LOCK:
        victims = list(_PIPELINE_POPENS)
    _log("PIPELINE", f"kill_popens: terminating {len(victims)} subprocess(es)")
    for p in victims:
        pid = getattr(p, "pid", None)
        try:
            if sys.platform == "win32":
                # Best-effort: kill the immediate child. Killing the
                # process tree on Windows would need taskkill /T /F /PID,
                # which adds platform-specific complexity we'll defer
                # until a Windows user reports a leak.
                p.kill()
                _log("PIPELINE", f"kill_popens: killed pid={pid} (win)")
            else:
                # SIGKILL the whole process group — `claude` (Node.js) and
                # other CLI wrappers typically fork child processes that
                # do the actual API call; SIGKILL on the immediate child
                # leaves those children orphaned-but-alive (reparented to
                # init), so the "background task" appears to keep running.
                # Requires `start_new_session=True` on the Popen so the
                # child became a new process-group leader.
                try:
                    pgid = os.getpgid(p.pid)
                    os.killpg(pgid, signal.SIGKILL)
                    _log("PIPELINE", f"kill_popens: SIGKILL pgid={pgid} pid={pid}")
                except ProcessLookupError:
                    _log("PIPELINE", f"kill_popens: pid={pid} already exited")
                except PermissionError as e:
                    # killpg can fail if the child re-pgrouped itself
                    # (rare); fall back to a plain kill of the leader.
                    _log("PIPELINE", f"kill_popens: killpg pid={pid} denied ({e}); falling back to plain kill")
                    p.kill()
        except Exception as e:
            _log("PIPELINE", f"kill_popens: error killing pid={pid}: {type(e).__name__}: {e}")


def _pipeline_check_cancelled() -> None:
    if _PIPELINE_CANCEL.is_set():
        raise _PipelineCancelled()


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
    # Pre-check before spawning so a cancel that already happened doesn't
    # waste a subprocess startup.
    _pipeline_check_cancelled()
    cmd = ["claude", "-p", prompt]
    if model:
        cmd += ["--model", model]
    popen_kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }
    if sys.platform != "win32":
        # Put the child in its own process group so we can SIGKILL the
        # ENTIRE tree (claude wrapper + Node + any spawned helpers) on
        # cancel. Without this, `proc.kill()` only kills the immediate
        # child, leaving orphan grandchildren that keep talking to the
        # API and produce the "cancel didn't terminate" UX.
        popen_kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except FileNotFoundError:
        print("[错误] 找不到 claude 命令，请确认 Claude Code CLI 已安装且在 PATH 中")
        sys.exit(1)
    # Register so `_pipeline_kill_popens()` can SIGKILL this process when
    # the user clicks "停止任务" — that's what actually terminates the
    # background task instead of letting it run to completion.
    _pipeline_register_popen(proc)
    # Race guard: if cancel fired in the tiny window between Popen and
    # register, the previous kill_popens pass missed this proc. Kill it
    # now so `communicate()` returns immediately instead of blocking
    # forever on a live child.
    if _PIPELINE_CANCEL.is_set():
        _pipeline_kill_popens()
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            print(f"[错误] claude-cli 超时（{label}，{timeout}s）；可通过 config --set llm_timeout=900 调大")
            sys.exit(1)
    finally:
        _pipeline_unregister_popen(proc)
    # If we got here because cancel killed the subprocess, returncode will
    # be negative (signal); short-circuit before sys.exit so the worker
    # sees a clean `_PipelineCancelled` instead of a generic failure.
    if _PIPELINE_CANCEL.is_set():
        raise _PipelineCancelled()
    if proc.returncode != 0:
        out = (stdout or "").strip()
        err = (stderr or "").strip()
        parts = [f"[错误] claude-cli 非零退出（{label}，returncode={proc.returncode}）"]
        if err:
            parts.append(f"stderr: {err}")
        if out:
            parts.append(f"stdout: {out}")
        if not err and not out:
            parts.append(
                "stdout/stderr 均为空 — 可能是认证/限额/网络问题。"
                "请在终端直接执行 `claude -p 'hi'` 复现，并检查 `claude --version` 与登录态。"
            )
        print("\n".join(parts))
        sys.exit(1)
    return stdout.strip()


def _llm_openai(prompt: str, pcfg: dict, label: str, timeout: int) -> str:
    import json as _json
    import urllib.request, urllib.error

    # HTTP calls aren't killable mid-request, but checking here at least
    # prevents new calls from being made after the user has cancelled.
    _pipeline_check_cancelled()

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

    # HTTP calls aren't killable mid-request, but checking here at least
    # prevents new calls from being made after the user has cancelled.
    _pipeline_check_cancelled()

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
    effective = min(max_workers, total)
    _log(
        "POLISH",
        f"concurrency: workers={max_workers} effective={effective} "
        f"(source={'config' if _w > 0 else 'auto'}, config_value={_w}, "
        f"cpu_count={os.cpu_count()}) total_chunks={total} provider={provider} mode={mode}",
    )
    print(f"[校对] 并行调用 {provider}（{mode} 模式，共 {total} 块，并发 {effective}）...")

    def _run(i_chunk):
        i, chunk = i_chunk
        prompt = (_resolve_prompt(cfg, "polish")
                  .replace("{transcript}", chunk)
                  .replace("{hotwords}", _cfg_hotword(cfg) or "（无）"))
        result = _llm_run(prompt, provider, cfg, f"校对[{i}/{total}]")
        print(f"[校对] 第 {i}/{total} 块完成")
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_run, enumerate(chunks, 1)))

    return "\n\n".join(results)


_MODE_LABELS_ZH = {
    "meeting": "会议纪要",
    "interview": "面试总结",
    "sharing": "分享总结",
}


def generate_notes(transcript: str, provider: str, cfg: dict, mode: str = "meeting") -> str:
    import concurrent.futures
    label = _MODE_LABELS_ZH.get(mode, _MODE_LABELS_ZH["meeting"])
    print(f"[{label}] 并行生成中英文版本（{provider}）...")
    prompt_zh = _resolve_prompt(cfg, "notes_zh", mode=mode).replace("{transcript}", transcript)
    prompt_en = _resolve_prompt(cfg, "notes_en", mode=mode).replace("{transcript}", transcript)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fut_zh = ex.submit(_llm_run, prompt_zh, provider, cfg, f"{label}(中文)")
        fut_en = ex.submit(_llm_run, prompt_en, provider, cfg, f"{label}(English)")
        notes_zh = fut_zh.result()
        notes_en = fut_en.result()
    divider = "\n\n---\n\n"
    return notes_zh + divider + notes_en


# ── 热词自动维护（方案 A+B）───────────────────────────────────────────────────
# A（规则）+ B（LLM）从转写/纪要中提取专有名词，合并进 config.jsonc 的
# stt.funasr.hotword —— 本次会议提取的热词从下一次录音开始生效。
# 不引入任何额外的配置/状态文件：config.jsonc 是唯一存储。

# Not "meaningless" words — just ones so common in code-switched speech
# that boosting them as hotwords would hurt more than help.
_HOTWORD_STOPWORDS = frozenset("""
a an the and or but if then else for to of in on at by with from as is are was
were be been being am do does did done have has had having will would can could
should shall may might must not no yes it its this that these those there their
here what when where which who whom why how all any both each few more most
other some such own same so than too very just also about into over under again
once during before after above below up down out off further then we you they
he she me my your our his her them him us ok okay right well like get got make
made go going went gone come came take took taken see saw seen know knew known
think thought want wanted need needed use used using work works worked working
one two three four five ten new old good bad big small long short high low time
day week month year way thing things people team teams part parts lot bit end
start next last first second per via etc app apps item items case cases
""".split())


def _cfg_hotword(cfg: dict) -> str:
    """The current stt.funasr.hotword string ("" when unset)."""
    return (((cfg.get("stt") or {}).get("funasr") or {})
            .get("hotword") or "").strip()


def _extract_hotword_candidates(text: str) -> "list[str]":
    """方案 A：规则法热词候选（无 LLM 开销）。

    Mines English/alphanumeric tokens that look like terminology:
    acronyms (GKE), CamelCase / mixed-case (BigQuery), digit-bearing
    (N4D). In CJK-dominant text (the normal meetingscribe transcript),
    plain English words that recur are also kept — code-switched English
    inside Chinese speech is inherently likely to be terminology. In
    English-dominant text (e.g. the notes' English half) plain words are
    skipped, otherwise every sentence-initial capital would qualify.
    Chinese-only terms are scheme B's (LLM) job. Returns candidates
    ordered by frequency (desc), then first appearance.
    """
    import re
    if not text:
        return []
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    english_doc = ascii_letters > cjk
    counts: "dict[str, int]" = {}
    display: "dict[str, str]" = {}
    order: "list[str]" = []
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9+#._\-]*", text):
        tok = tok.strip("._-")
        if not 2 <= len(tok) <= 32:
            continue
        low = tok.lower()
        if low in _HOTWORD_STOPWORDS:
            continue
        if low not in counts:
            order.append(low)
            display[low] = tok
        elif tok != tok.lower() and display[low] == display[low].lower():
            display[low] = tok  # prefer a cased variant over all-lowercase
        counts[low] = counts.get(low, 0) + 1
    out = []
    for low in order:
        tok = display[low]
        acronym = tok.isupper() and len(tok) <= 10
        has_digit = any(c.isdigit() for c in tok)
        mixed_case = any(c.isupper() for c in tok[1:])
        if acronym or has_digit or mixed_case:
            out.append(low)
        elif not english_doc and (counts[low] >= 2 or tok[0].isupper()):
            # In CJK text a Capitalized English token is almost surely a
            # proper noun (FunASR capitalizes recognized names), so one
            # occurrence suffices; plain lowercase needs repetition.
            out.append(low)
    out.sort(key=lambda w: (-counts[w], order.index(w)))
    return [display[w] for w in out[:50]]


def _clean_hotword_token(tok: str) -> str:
    return tok.strip(".,;:!?，。、；：！？·\"'`()（）[]【】<>《》")


def _hotword_hits(text: str, terms: "list[str]") -> "set[str]":
    """Lower-cased terms that actually appear in `text`.

    This is the signal eviction runs on, so a false hit is not cosmetic: it
    protects exactly the junk the eviction exists to remove. Measured on one
    real 15.5k-character transcript, plain substring matching scored hits for
    `ch` (inside "check") and `ac` (inside "face") — short ASCII terms match
    constantly inside ordinary English, inflating 53 real hits to 58.

    So ASCII terms require word boundaries and CJK terms do not: Chinese is
    written without spaces, and 「李雷」 inside 「李雷说」 is a genuine hit.
    """
    if not text or not terms:
        return set()
    low = text.lower()
    out: "set[str]" = set()
    for term in terms:
        if not term:
            continue
        t = term.lower()
        if any("一" <= c <= "鿿" for c in t):
            if t in low:
                out.add(t)
        elif re.search(rf"(?<![0-9a-z]){re.escape(t)}(?![0-9a-z])", low):
            # Hand-rolled instead of \b: \b sits between a letter and a CJK
            # character too, so 「用GKE集群」 would fail a \bgke\b test.
            out.add(t)
    return out


def _merge_hotwords(existing: str, new_terms: "list[str]",
                    max_count: int = 100,
                    pinned: "set[str] | None" = None,
                    rank: "dict[str, tuple] | None" = None) -> str:
    """Merge new hotword candidates into the existing space-separated
    hotword string: case-insensitive dedup, first-seen casing wins,
    insertion order preserved.

    Over `max_count`, entries are evicted. Two things protect a term:

    - `pinned` (lower-cased) is never evicted. Workspace names imported from
      Notion live here, because a token mined once from one meeting has no
      business pushing a colleague's name out of the list.
    - `rank` maps a lower-cased term to a sort key; the LOWEST keys are
      evicted first. `_persist_hotwords` passes (last_seen, hits) so a term
      still in daily use outranks yesterday's garbage.

    With neither argument the behaviour is the original rolling FIFO — oldest
    out — which is what a plain string merge should do.
    """
    seen: "set[str]" = set()
    out: "list[str]" = []
    for tok in (existing or "").split() + list(new_terms):
        tok = _clean_hotword_token(tok)
        if not 2 <= len(tok) <= 32 or " " in tok or tok.isdigit():
            continue
        low = tok.lower()
        if low in seen or low in _HOTWORD_STOPWORDS:
            continue
        seen.add(low)
        out.append(tok)
    if max_count <= 0 or len(out) <= max_count:
        return " ".join(out)
    pinned = pinned or set()
    keep_pinned = [t for t in out if t.lower() in pinned]
    rolling = [t for t in out if t.lower() not in pinned]
    slots = max_count - len(keep_pinned)
    if slots <= 0:
        # Pinned alone fills the cap. Keeping them is the lesser evil:
        # silently dropping authoritative names is worse than a list that is
        # slightly too long, and the caller logs the overshoot.
        return " ".join(keep_pinned)
    if rank:
        # Stable: index breaks ties so equal-rank terms keep insertion order.
        order = sorted(range(len(rolling)),
                       key=lambda i: (rank.get(rolling[i].lower(), (0, 0)), i))
        drop = {order[i] for i in range(len(rolling) - slots)}
        rolling = [t for i, t in enumerate(rolling) if i not in drop]
    else:
        rolling = rolling[-slots:]
    # Rebuild in the original order so the file stays readable.
    survivors = set(keep_pinned) | set(rolling)
    return " ".join(t for t in out if t in survivors)


def _parse_llm_hotwords(out: str) -> "list[str]":
    """Parse the hotwords_extract LLM reply. Instructed to emit one
    space-separated line, but tolerate bullets / 、 separators / a stray
    preamble line ending in a colon."""
    import re
    toks: "list[str]" = []
    for line in (out or "").splitlines():
        line = line.strip().lstrip("-*•").strip()
        if not line or line.startswith("#") or line.endswith(("：", ":")):
            continue
        for tok in re.split(r"[\s,，、;；/|]+", line):
            tok = _clean_hotword_token(tok)
            if 2 <= len(tok) <= 32:
                toks.append(tok)
    return toks[:50]


def _extract_hotwords_llm(text: str, provider: str, cfg: dict) -> "list[str]":
    """方案 B：LLM 术语提取（离线阶段，不影响实时字幕/录音路径）。"""
    prompt = _resolve_prompt(cfg, "hotwords_extract").replace("{transcript}", text)
    return _parse_llm_hotwords(_llm_run(prompt, provider, cfg, "热词提取"))


def _persist_hotwords(new_terms: "list[str]", cfg: dict,
                      max_count: int = 100, pin: bool = False,
                      transcript: str = "") -> "list[str]":
    """Merge `new_terms` into stt.funasr.hotword, update the runtime cfg in
    place, and write hotword.jsonc. Returns the terms actually added.

    `pin=True` marks the new terms as never-evictable — used by the Notion
    import, whose terms are authoritative workspace names rather than guesses
    mined from one meeting's transcript.

    `transcript` is scanned for existing hotwords first, and every term found
    gets its hit count and `last_seen` epoch bumped. That is what turns
    eviction from "oldest out" into "least recently useful out": a term added
    ten meetings ago and still said every day now outranks a token mined
    yesterday and never heard again.

    Also clears any leftover inline `stt.funasr.hotword` in config.jsonc:
    that is how an existing setup migrates off the old single-file layout,
    and leaving a stale copy behind in a version-controlled file is exactly
    the footgun this split exists to remove.
    """
    store = _load_hotword_store()
    existing = ((cfg.get("stt") or {}).get("funasr") or {}).get("hotword", "") or ""
    # The runtime cfg wins as the term list (it is what this process has been
    # recognising with), but the bookkeeping only lives in the file.
    if existing.split():
        store["terms"] = existing.split()

    epoch = int(store.get("epoch") or 0) + 1
    store["epoch"] = epoch
    hit = _hotword_hits(transcript, store["terms"])
    for low in hit:
        store["hits"][low] = store["hits"].get(low, 0) + 1
        store["last_seen"][low] = epoch
    if pin:
        store["pinned"] |= {t.lower() for t in new_terms}
    # Seed anything with no history at the CURRENT epoch, not 0. Two reasons:
    # a new term was just mined from this transcript, so ranking it worst
    # makes no sense; and on the first run after an upgrade the whole existing
    # list has no history, so epoch 0 would let today's junk outrank a term
    # like CockroachDB that simply wasn't mentioned today. Unknown history
    # means "benefit of the doubt once" — real usage differentiates them from
    # the next meeting onwards, since only hit terms advance after this.
    for t in list(store["terms"]) + list(new_terms):
        store["last_seen"].setdefault(t.lower(), epoch)

    rank = {t.lower(): (store["last_seen"].get(t.lower(), 0),
                        store["hits"].get(t.lower(), 0))
            for t in store["terms"] + list(new_terms)}
    merged = _merge_hotwords(existing, new_terms, max_count,
                            pinned=store["pinned"], rank=rank)
    baseline = _merge_hotwords(existing, [], max_count,
                               pinned=store["pinned"], rank=rank)
    known = {t.lower() for t in existing.split()}
    added = [t for t in merged.split() if t.lower() not in known]
    dropped = [t for t in existing.split()
               if t.lower() not in {x.lower() for x in merged.split()}]
    if merged == baseline and not hit:
        return []          # nothing new and no hit history to record
    cfg.setdefault("stt", {}).setdefault("funasr", {})["hotword"] = merged
    try:
        _save_hotword_file(merged, store)
        pinned_kept = sum(1 for t in merged.split()
                          if t.lower() in store["pinned"])
        _log("STT", f"hotword updated: +{len(added)} "
                    f"({' '.join(added)}) -{len(dropped)}"
                    + (f" ({' '.join(dropped)})" if dropped else "")
                    + f" → {len(merged.split())} total "
                      f"(pinned={pinned_kept}, hit={len(hit)}, epoch={epoch})"
                      f" → {HOTWORD_FILE.name}")
        if pinned_kept > max_count:
            _log("WARN", f"pinned hotwords ({pinned_kept}) exceed "
                         f"hotwords.max_count ({max_count}); keeping them, "
                         f"but a longer list costs recognition accuracy")
        _migrate_inline_hotword()
    except (OSError, json.JSONDecodeError, ValueError) as e:
        _log("WARN", f"hotword persist failed: {type(e).__name__}: {e}")
    return added


def _migrate_inline_hotword() -> None:
    """Blank a legacy inline hotword list in config.jsonc, comment-preserving."""
    if not CONFIG_FILE.exists():
        return
    try:
        on_disk = json.loads(
            _strip_jsonc_comments(CONFIG_FILE.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError):
        return
    inline = ((on_disk.get("stt") or {}).get("funasr") or {}).get("hotword") or ""
    if not inline.strip():
        return
    on_disk.setdefault("stt", {}).setdefault("funasr", {})["hotword"] = ""
    save_config(on_disk)
    _log("STT", f"moved {len(inline.split())} inline hotword(s) out of "
                f"{CONFIG_FILE.name} into {HOTWORD_FILE.name}")


def _auto_update_hotwords(transcript: str, provider: str, cfg: dict) -> "list[str]":
    """方案 A+B 入口：纪要/总结生成后调用。绝不抛异常——热词维护失败
    不允许影响已经成功的转写/纪要流水线。Returns the newly added terms."""
    try:
        hw_cfg = _deep_merge(DEFAULT_CONFIG["hotwords"], cfg.get("hotwords") or {})
        if not hw_cfg.get("auto_update", True):
            return []
        candidates: "list[str]" = []
        if hw_cfg.get("rule_extract", True):
            candidates += _extract_hotword_candidates(transcript or "")
        if hw_cfg.get("llm_extract", True):
            try:
                candidates += _extract_hotwords_llm(transcript or "", provider, cfg)
            except (Exception, SystemExit) as e:
                # _llm_* error paths may sys.exit(); a failed extraction
                # must not kill a pipeline that already produced notes.
                _log("WARN", f"hotword llm extract failed: {type(e).__name__}: {e}")
        if not candidates:
            return []
        # The transcript doubles as the hit signal: every existing hotword
        # found in it gets its last_seen bumped, so eviction drops the terms
        # that stopped being useful rather than the ones added longest ago.
        return _persist_hotwords(candidates, cfg,
                                 int(hw_cfg.get("max_count", 100)),
                                 transcript=transcript or "")
    except Exception as e:
        _log("WARN", f"hotword auto-update failed: {type(e).__name__}: {e}")
        return []


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
    _log(
        "POLISH",
        f"transcribe_and_polish concurrency: polish_workers={max_polish_workers} "
        f"(source={'config' if _w > 0 else 'auto'}, config_value={_w}, "
        f"cpu_count={os.cpu_count()}) polish_provider={polish_provider}",
    )

    pending: list[tuple[int, concurrent.futures.Future]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_polish_workers + 1) as polish_ex:
        # Warm-up: send a tiny request concurrently with transcription so the first
        # real polish batch doesn't suffer the API cold-start penalty (~150s → ~60s).
        print(f"[预热] 并发预热 {polish_provider} API...")
        _warmup_fut = polish_ex.submit(_llm_run, "x", polish_provider, cfg, "预热")

        def on_chunk_done(text: str, idx: int):
            prompt = (_resolve_prompt(cfg, "polish")
                      .replace("{transcript}", text)
                      .replace("{hotwords}", _cfg_hotword(cfg) or "（无）"))
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

_NOTES_SUFFIX = {
    "meeting": ".meeting.md",
    "interview": ".interview.md",
    "sharing": ".sharing.md",
}
# Single source of truth for the supported pipeline modes. Drives argparse
# validation (`choices=MODES`), the mode→label lookup in `generate_notes`,
# and the `_list_recordings` artifact-detection logic. Adding a fourth
# mode is: extend `_NOTES_SUFFIX` + add a `<mode>` block to
# `_PROMPT_DEFAULTS` + wire UI labels — no other call sites to touch.
MODES = tuple(_NOTES_SUFFIX.keys())


def save_minutes(minutes: str, audio_path: Path, mode: str = "meeting") -> Path:
    """Persist generated notes / interview report next to the audio file.

    Output path:
      mode='meeting'   → <stem>.meeting.md   (会议纪要)
      mode='interview' → <stem>.interview.md (面试报告)

    Unknown modes fall back to .meeting.md to avoid silent data loss.
    """
    suffix = _NOTES_SUFFIX.get(mode, ".meeting.md")
    note_path = audio_path.with_name(audio_path.stem + suffix)
    note_path.write_text(minutes, encoding="utf-8")
    return note_path


# Files in the recordings dir follow this naming convention:
#     <timestamp>[.<custom_name>].<suffix>
# e.g.
#     20260518_201522.wav
#     20260518_201522.客户访谈.wav
#     20260518_201522.客户访谈.raw.txt
#     20260518_201522.客户访谈.meeting.md
#     20260518_201522.客户访谈.interview.md
#     20260518_201522.客户访谈.polish.txt
# `_split_meeting_stem` separates the timestamp prefix (which is always
# present, validated against the YYYYmmdd_HHMMSS format) from the optional
# custom label. `_rename_meeting_files` atomically renames every sibling
# file sharing the same stem when the user assigns a new custom name.
_MEETING_STEM_TIMESTAMP_RE = __import__("re").compile(r"^(\d{8}_\d{6})(?:\.(.+))?$")


def _split_meeting_stem(stem: str) -> tuple[str | None, str | None]:
    """Decompose a recording stem into ``(timestamp, custom_name)``.

    The timestamp portion (``YYYYmmdd_HHMMSS``) is always present in a
    well-formed stem. The optional ``custom_name`` is what the user set via
    the right-click rename action.

    Returns ``(None, None)`` when the stem doesn't match the expected
    pattern (e.g. an unrelated .wav file dropped into the recordings dir).
    """
    m = _MEETING_STEM_TIMESTAMP_RE.match(stem)
    if not m:
        return None, None
    ts, custom = m.group(1), m.group(2)
    return ts, custom or None


def _sanitize_meeting_custom_name(name: str) -> str:
    """Strip filesystem-hostile characters from a user-provided custom name.

    Reserved characters across macOS / Windows / Linux are mapped to ``_``.
    Leading / trailing dots and whitespace are stripped. Returns ``""`` when
    the input is empty after sanitisation — callers should treat that as
    "drop the custom-name segment entirely".
    """
    import re
    cleaned = re.sub(r'[\\/:\*\?"<>\|\x00-\x1f]', "_", name or "")
    return cleaned.strip().strip(".").strip()


def _rename_meeting_files(wav_path: Path, new_custom_name: str | None) -> Path | None:
    """Rename a meeting's ``.wav`` and every companion file sharing the
    same stem prefix to use a new custom-name segment.

    The new stem is ``<timestamp>`` (when ``new_custom_name`` is empty) or
    ``<timestamp>.<sanitised_custom_name>``. Refuses to proceed when:

      * ``wav_path`` is missing or doesn't match the timestamp pattern
      * any target filename already exists (collision)
      * sanitisation reduces the requested name to empty AND the existing
        stem already has no custom segment (no-op)

    Returns the new ``.wav`` ``Path`` on success, ``None`` otherwise.
    """
    if not wav_path.exists():
        return None
    old_stem = wav_path.stem
    ts, _old_custom = _split_meeting_stem(old_stem)
    if not ts:
        return None
    clean = _sanitize_meeting_custom_name(new_custom_name or "")
    new_stem = f"{ts}.{clean}" if clean else ts
    if new_stem == old_stem:
        return wav_path  # no change requested

    parent = wav_path.parent
    siblings: list[Path] = []
    prefix = old_stem + "."
    for f in parent.iterdir():
        if not f.is_file():
            continue
        if f.name == old_stem + ".wav" or f.name.startswith(prefix):
            siblings.append(f)
    if not siblings:
        return None

    moves: list[tuple[Path, Path]] = []
    for f in siblings:
        suffix_part = f.name[len(old_stem):]  # ".wav", ".raw.txt", ...
        target = f.with_name(new_stem + suffix_part)
        if target.exists() and target != f:
            _log("REC", f"rename collision: {target.name} already exists; aborting")
            return None
        moves.append((f, target))

    for src, dst in moves:
        src.rename(dst)
        _log("REC", f"rename: {src.name} → {dst.name}")
    return parent / (new_stem + ".wav")


def _delete_meeting_files(wav_path: Path) -> tuple[int, list[str]]:
    """Delete a meeting's ``.wav`` and every companion file sharing the
    same stem prefix (``.raw.txt``, ``.polish.txt``, ``.meeting.md``,
    ``.interview.md``, ``.sharing.md``, ``.meta.json``, legacy ``.md``,
    …).

    Returns ``(deleted_count, errors)`` where ``errors`` is a list of
    human-readable ``"<filename>: <reason>"`` strings for files that
    couldn't be removed. A missing ``wav_path`` is treated as a no-op
    (``(0, [])``).
    """
    if not wav_path.exists():
        return 0, []
    stem = wav_path.stem
    parent = wav_path.parent
    prefix = stem + "."
    targets: list[Path] = []
    for f in parent.iterdir():
        if not f.is_file():
            continue
        if f.name == stem + ".wav" or f.name.startswith(prefix):
            targets.append(f)
    deleted = 0
    errors: list[str] = []
    for f in targets:
        try:
            f.unlink()
            deleted += 1
            _log("REC", f"delete: {f.name}")
        except Exception as e:
            errors.append(f"{f.name}: {type(e).__name__}: {e}")
            _log("ERR", f"delete failed {f.name}: {type(e).__name__}: {e}")
    return deleted, errors


# ── 外部音频导入（iPhone .m4a / .mp3 / 等 → 16k mono PCM WAV）─────────────────

# Extensions we attempt to transcode via ffmpeg. The list is intentionally
# permissive — ffmpeg recognises far more, but these cover the common sources
# (iPhone voice memos, Zoom recordings, downloaded podcasts, etc.).
_IMPORTABLE_AUDIO_EXTS = frozenset({
    ".m4a", ".mp4", ".aac",
    ".mp3", ".mpga", ".mpeg",
    ".flac", ".ogg", ".opus",
    ".aif", ".aiff", ".aifc",
    ".wma", ".amr", ".caf",
    ".webm",
})


def _ffmpeg_path() -> "str | None":
    """Return the absolute path to `ffmpeg` if available, else None.

    Cached implicitly by `shutil.which` (it walks PATH; on macOS that's a few
    stat calls, fine to call ad-hoc). Returns None when ffmpeg isn't on PATH —
    callers surface the install hint.
    """
    import shutil
    return shutil.which("ffmpeg")


def _transcode_to_wav(src: Path, dst: Path) -> None:
    """Transcode `src` (any ffmpeg-recognised container) into a 16 kHz mono
    16-bit PCM WAV at `dst`. Format matches FunASR's expected input so the
    downstream pipeline doesn't need any special-case branch.

    Raises `RuntimeError` on ffmpeg failure with stderr included for
    debugging. Caller is responsible for choosing `dst` (we don't add
    suffixes — what's passed is what's written).
    """
    ffmpeg = _ffmpeg_path()
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install with `brew install ffmpeg` "
            "(macOS) or your distro's package manager."
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-ar", "16000",   # 16 kHz sample rate (FunASR canonical input)
        "-ac", "1",       # mono
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    _log("AUDIO", f"transcode: src={src.name!r} dst={dst.name!r}")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"ffmpeg invocation failed: {e}") from e
    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or "(no stderr)"
        _log("ERR", f"ffmpeg rc={proc.returncode}: {err[:500]}")
        raise RuntimeError(
            f"ffmpeg failed (exit code {proc.returncode}): {err[:500]}"
        )
    _log("AUDIO", f"transcode ok: {dst.name}")


def _import_audio_to_recordings(src: Path) -> Path:
    """Copy or transcode an external audio file into `RECORDINGS_DIR`,
    returning the canonical `.wav` Path it now lives at.

    Naming follows the existing `<timestamp>[.<custom>].wav` convention so
    every downstream helper (`_split_meeting_stem`, `_list_recordings`,
    `_rename_meeting_files`, `_delete_meeting_files`) keeps working without
    modification. The custom-name segment is the sanitised source stem, so
    `客户访谈.m4a` becomes `20260521_141500.客户访谈.wav`.

    Behavioural matrix:
      - .wav  → file is hard-copied (no transcode; format may already be
        FunASR-compatible PCM, and re-encoding would be wasteful).
      - other recognised audio extension → ffmpeg transcode to 16 kHz mono
        PCM WAV.
      - unrecognised extension → still attempted via ffmpeg (it might work);
        if ffmpeg fails the user gets the underlying error message.
    """
    import shutil
    if not src.exists():
        raise FileNotFoundError(f"audio file not found: {src}")
    if not src.is_file():
        raise ValueError(f"not a regular file: {src}")

    # `RECORDINGS_DIR` is a local in cmd_ui() / cmd_record(); this helper is
    # module-level so we recompute from CONFIG_DIR. Same path, same naming.
    recordings_dir = CONFIG_DIR / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    custom = _sanitize_meeting_custom_name(src.stem)
    stem = f"{ts}.{custom}" if custom else ts
    dst = recordings_dir / f"{stem}.wav"
    # Disambiguate same-second collisions (rapid imports) by appending a
    # numeric suffix. Almost never happens in practice but cheap to handle.
    n = 2
    while dst.exists():
        dst = recordings_dir / f"{stem}_{n}.wav"
        n += 1

    if src.suffix.lower() == ".wav":
        shutil.copy2(src, dst)
        _log("AUDIO", f"import (wav copy): src={src.name!r} → {dst.name!r}")
        return dst

    _transcode_to_wav(src, dst)
    return dst


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
    if mode not in MODES:
        print(f"[错误] 未知模式 '{mode}'，可选：{list(MODES)}")
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

    notes_label = _MODE_LABELS_ZH.get(mode, _MODE_LABELS_ZH["meeting"])
    notes = generate_notes(transcript_polished, notes_provider, cfg, mode)
    print(f"\n── {notes_label} " + "─" * (58 - len(notes_label)))
    print(notes)
    print("─" * 60)
    note_path = save_minutes(notes, audio_path, mode)
    print(f"\n✅ 完成！{notes_label}已保存: {note_path}")
    added_hw = _auto_update_hotwords(transcript_polished, notes_provider, cfg)
    if added_hw:
        print(f"[热词] 新增 {len(added_hw)} 个：{' '.join(added_hw)}"
              f"（已写入 config.jsonc，下次录音生效）")


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
    if mode not in MODES:
        print(f"[错误] 未知模式 '{mode}'，可选：{list(MODES)}")
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
        # Non-WAV input → transcode into RECORDINGS_DIR first so the
        # downstream pipeline (FunASR's `wave.open`, OpenAI/Gemini binary
        # upload) gets a clean PCM container. iPhone .m4a, Zoom .m4a,
        # downloaded .mp3, etc. all flow through here.
        if audio_path.suffix.lower() != ".wav":
            print(f"[导入] 检测到非 WAV 输入，转码到 16k mono PCM WAV…")
            try:
                audio_path = _import_audio_to_recordings(audio_path)
                print(f"[导入] 完成：{audio_path}")
            except Exception as e:
                print(f"[错误] 音频导入失败：{type(e).__name__}: {e}")
                sys.exit(1)
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

    notes_label = _MODE_LABELS_ZH.get(mode, _MODE_LABELS_ZH["meeting"])
    print(f"\n[校对] 已保存: {polish_path}")
    print("\n── 校对后转写 " + "─" * 46)
    print(transcript_polished)
    print("─" * 60)

    notes = generate_notes(transcript_polished, notes_provider, cfg, mode)
    print(f"\n── {notes_label} " + "─" * (58 - len(notes_label)))
    print(notes)
    print("─" * 60)
    note_path = save_minutes(notes, audio_path, mode)
    print(f"\n✅ 完成！{notes_label}已保存: {note_path}")
    added_hw = _auto_update_hotwords(transcript_polished, notes_provider, cfg)
    if added_hw:
        print(f"[热词] 新增 {len(added_hw)} 个：{' '.join(added_hw)}"
              f"（已写入 config.jsonc，下次录音生效）")


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


_NOTION_API = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"


def _notion_request(path: str, token: str, payload: "dict | None" = None) -> dict:
    """One Notion API call. GET when `payload` is None, else POST.

    The token is read from the environment by the caller and never logged,
    stored or echoed — not in errors either, since those reach the console.
    Uses certifi's CA bundle for the same reason `_caption_download` does.
    """
    import json as _json
    import ssl
    import urllib.error
    import urllib.request
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    data = _json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"{_NOTION_API}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Notion-Version": _NOTION_VERSION,
            "Content-Type": "application/json",
        },
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            return _json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = _json.loads(e.read().decode("utf-8")).get("message", "")
        except Exception:
            pass
        raise RuntimeError(f"Notion API {path} 返回 {e.code}: {detail}") from None


def _notion_paginate(path: str, token: str, payload: "dict | None" = None,
                     max_pages: int = 20) -> "list[dict]":
    """Follow Notion's cursor pagination, bounded so a huge workspace can't
    turn a one-off import into an unbounded crawl."""
    out: "list[dict]" = []
    cursor = None
    for _ in range(max_pages):
        body = dict(payload or {})
        if payload is None and cursor:
            page = _notion_request(f"{path}?start_cursor={cursor}", token)
        else:
            if cursor:
                body["start_cursor"] = cursor
            page = _notion_request(path, token, body if payload is not None else None)
        out.extend(page.get("results") or [])
        if not page.get("has_more"):
            break
        cursor = page.get("next_cursor")
        if not cursor:
            break
    return out


def _notion_member_names(token: str) -> "list[str]":
    """Workspace member names — the highest-value slice for ASR hotwords,
    since names are the error class the recognizer misses most."""
    names = []
    for u in _notion_paginate("/users", token):
        if u.get("type") == "bot":
            continue
        name = (u.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def _notion_titles(token: str) -> "list[str]":
    """Page + database titles — where project names and internal codenames live.

    Walks each result for any `title`-typed rich text rather than special-casing
    page vs database vs database-row shapes, which differ across API versions.
    """
    def _titles_in(node) -> "list[str]":
        found = []
        if isinstance(node, dict):
            if node.get("type") == "title" and isinstance(node.get("title"), list):
                found.append("".join(
                    part.get("plain_text", "") for part in node["title"]))
            elif isinstance(node.get("title"), list):
                found.append("".join(
                    part.get("plain_text", "") for part in node["title"]
                    if isinstance(part, dict)))
            for v in node.values():
                found.extend(_titles_in(v))
        elif isinstance(node, list):
            for v in node:
                found.extend(_titles_in(v))
        return found

    out = []
    for item in _notion_paginate("/search", token, {"page_size": 100}):
        out.extend(t.strip() for t in _titles_in(item) if t.strip())
    return out


def _notion_hotword_candidates(names: "list[str]", titles: "list[str]",
                               limit: int = 400) -> "list[str]":
    """Turn Notion names + titles into hotword candidates.

    Names become the form people SAY: a CJK name stays whole (`李雷`), while
    "Aaron Chen" splits into tokens because nobody says the full name mid
    sentence. Titles go through the existing rule-based miner
    (`_extract_hotword_candidates`) instead of being added verbatim — a title
    like 「2026 Q3 运维体系规划」 is a sentence, not a hotword.
    """
    out: "list[str]" = []
    for raw in names:
        name = re.sub(r"[（(\[].*?[)）\]]", " ", raw)       # drop "(Kevin)" etc.
        if "@" in name:                                     # an email, not a name
            name = name.split("@", 1)[0].replace(".", " ")
        if any("一" <= c <= "鿿" for c in name):
            out.extend(t for t in re.findall(r"[一-鿿]{2,6}", name))
        else:
            # Latin tokens must start capitalised. An email login splits into
            # lowercase fragments ("he.huang" → he / huang) and those are
            # common syllables that would bias real speech; a display name is
            # capitalised, so this keeps Aaron / Bella / Zhang and drops the
            # debris. ("he" is already a stopword; "huang" would not be.)
            out.extend(t for t in re.split(r"[\s_./-]+", name)
                       if len(t) >= 2 and t[:1].isupper())
    if titles:
        out.extend(_extract_hotword_candidates("。".join(titles)))
    seen, uniq = set(), []
    for t in out:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq[:max(1, limit)]


def _wav_channels_by_role(path: Path) -> "tuple[dict, int]":
    """Split a recording into {role: samples}, mirroring how it was captured.

    `MultiStreamRecorder.save` writes channel 0 = system audio, channel 1 =
    microphone (the `wanted` order). Feeding them back as SEPARATE sources is
    what makes a replay faithful: the engine mixes and attributes them exactly
    as it would live. A mono file is treated as microphone-only.
    """
    with wave.open(str(path), "rb") as wf:
        sr, ch, n = wf.getframerate(), wf.getnchannels(), wf.getnframes()
        raw = np.frombuffer(wf.readframes(n), dtype=np.int16)
    data = raw.astype(np.float32) / 32768.0
    if ch <= 1:
        return {"mic": data}, sr
    frames = data.reshape(-1, ch)
    return {"system": frames[:, 0], "mic": frames[:, 1]}, sr


def cmd_captions(args, cfg):
    """`captions` 子命令：把已有录音回放进实时字幕引擎。"""
    _, restore = _setup_log_file()
    try:
        _cmd_captions_body(args, cfg)
    finally:
        restore()


def _cmd_captions_body(args, cfg):
    """Replay a WAV through LiveCaptionEngine with no audio devices involved.

    Playing a file through the speakers and letting the microphone pick it up
    is NOT a way to test caption changes: it adds a whole speaker → room →
    mic round trip, and a role-separated recording puts its speech on the
    channel the capture device does not carry. This feeds the samples straight
    into the engine instead, so two runs over the same file are comparable and
    a caption change can be judged without a room.
    """
    path = Path(args.file).expanduser()
    if not path.exists():
        print(f"[错误] 文件不存在: {path}")
        sys.exit(1)

    tracks, sr = _wav_channels_by_role(path)
    total_secs = max(len(v) for v in tracks.values()) / sr
    start = max(0.0, float(getattr(args, "start", 0) or 0))
    span = float(getattr(args, "seconds", 0) or 0)
    if start or span:
        a = int(start * sr)
        b = int((start + span) * sr) if span else None
        tracks = {k: v[a:b] for k, v in tracks.items()}
        total_secs = max(len(v) for v in tracks.values()) / sr

    cfg = copy.deepcopy(cfg)
    lc = cfg.setdefault("live_captions", {})
    # The review pass is time-driven and costs an LLM call per interval, so a
    # replay opts in explicitly; --review then forces one pass at the end over
    # everything that accumulated, which is the comparable thing to look at.
    lc.setdefault("review", {})["enabled"] = False
    # Endpoint overrides: one A/B is one command, so the thresholds under test
    # come from the flags rather than an edit to config.jsonc between runs.
    ep_override = {
        key: float(getattr(args, flag))
        for flag, key in (("rule1", "rule1_min_trailing_silence"),
                          ("rule2", "rule2_min_trailing_silence"),
                          ("rule3", "rule3_min_utterance_length"))
        if getattr(args, flag, None) is not None
    }
    if ep_override:
        ep = dict(lc.get("endpoint") or {})
        ep.update(ep_override)
        lc["endpoint"] = ep

    rows: dict = {}
    order: list = []
    trace: list = []
    live = not getattr(args, "quiet", False)
    # Events arrive from the ASR / MT / refine / speaker threads, so printing
    # needs serialising or lines interleave mid-character.
    out_lock = threading.Lock()
    # Hold our OWN stream handle instead of using print(). The refine worker
    # wraps every FunASR call in `_QuietCapture`, which swaps the
    # process-global sys.stdout for an in-memory buffer — so a plain print()
    # from this thread during one of those windows (seconds at a time, many
    # times a run) is diverted into that buffer and forwarded to the
    # file-only logger. Measured: a 4-minute replay emitted its header and
    # then nothing at all, report included, which is fatal for a command
    # whose entire purpose is printing a comparable report.
    stdout = sys.stdout

    def _emit_line(line: str) -> None:
        with out_lock:
            stdout.write(line + "\n")
            stdout.flush()

    def _say(line: str) -> None:
        if live:
            _emit_line(line)

    def on_event(ev):
        t = ev.get("type")
        trace.append(ev)
        if t == "final":
            sid = ev["id"]
            rows[sid] = {"src": ev.get("text", ""), "dst": "",
                         "role": ev.get("role"), "speaker": None,
                         "revisions": 0}
            order.append(sid)
            _say(f"[{sid:3d}] {ev.get('text', '')}")
        elif t == "refined" and ev.get("id") in rows:
            rows[ev["id"]]["src"] = ev.get("text", "")
            rows[ev["id"]]["revisions"] += 1
            _say(f"[{ev['id']:3d}] ↻ {ev.get('text', '')}")
        elif t == "translation" and ev.get("id") in rows:
            rows[ev["id"]]["dst"] = ev.get("text", "")
            _say(f"[{ev['id']:3d}]   → {ev.get('text', '')}")
        elif t == "speaker" and ev.get("id") in rows:
            rows[ev["id"]]["speaker"] = ev.get("speaker")
            _say(f"[{ev['id']:3d}]   ⇢ 说话人{ev.get('speaker')}")
        elif t == "error":
            print(f"[字幕] 错误: {ev.get('message')}", file=sys.stderr)

    engine = LiveCaptionEngine(cfg, on_event)
    fast = bool(getattr(args, "fast", False))
    _emit_line(f"[字幕] 回放 {path.name} — {total_secs:.1f} 秒，"
               f"声源 {list(tracks)}，"
               f"{'超实时（分句会与实况不同）' if fast else '实时速度'}")
    if fast:
        # The ASR worker drains on a 250 ms WALL-CLOCK pacer, so feeding 10x
        # hands it 2.5 s per accept() — and `is_endpoint()` is checked once per
        # call, so sentence boundaries inside a batch collapse. Measured on the
        # same 104 s passage: 3 long run-on segments instead of 6. Fine for a
        # smoke test, useless for judging segmentation or comparing runs.
        _emit_line("[字幕] 提示：超实时模式下识别器每次收到的音频块变大，"
                   "断句会明显变少变长；对比字幕质量请用默认的实时速度。")
    engine.start()
    step = int(_CAPTION_PACER_INTERVAL * sr)
    n_samples = max(len(v) for v in tracks.values())
    next_note = _CAPTIONS_PROGRESS_EVERY_SEC
    try:
        for off in range(0, n_samples, step):
            for role, samples in tracks.items():
                chunk = samples[off:off + step]
                if chunk.size:
                    engine.feed(role, chunk, sr)
            fed = off / sr
            if fed >= next_note:
                next_note = fed + _CAPTIONS_PROGRESS_EVERY_SEC
                _emit_line(f"[字幕] … 已喂入 {fed:.0f}/{total_secs:.0f} 秒，"
                           f"定稿 {len(order)} 行")
            if fast:
                # Feeding faster than the ASR consumes would only pile audio
                # into the ring buffer (and past _CAPTION_RING_SECONDS it gets
                # dropped), so pace against the backlog rather than blindly.
                _captions_pace(engine)
            else:
                time.sleep(_CAPTION_PACER_INTERVAL)

        _emit_line(f"[字幕] 音频喂完，等待后台修正落地"
                   f"（上限 {args.wait:.0f} 秒）...")
        _captions_drain(engine, args.wait)
        if getattr(args, "review", False):
            _captions_force_review(engine)
    finally:
        engine.stop()

    _captions_report(rows, order, trace, verbose=getattr(args, "trace", False),
                     stream=stdout,
                     audio_secs=total_secs,
                     endpoint=_caption_endpoint_rules(lc),
                     review_cfg=_deep_merge(
                         DEFAULT_CONFIG["live_captions"]["review"],
                         cfg.get("live_captions", {}).get("review", {})))


_CAPTIONS_PROGRESS_EVERY_SEC = 30.0


def _captions_pace(engine) -> None:
    """Yield to the workers, backing off while the ASR is behind.

    The replay can push samples far faster than the recognizer consumes them.
    `feed()` caps each source's backlog at `_CAPTION_RING_SECONDS` and DROPS
    the excess, so an unpaced replay would silently transcribe a fraction of
    the file.
    """
    for _ in range(400):                    # ≈ 10 s ceiling per chunk
        if engine.backlog_secs() < _CAPTION_RING_SECONDS / 2:
            return
        time.sleep(_CAPTION_PACER_INTERVAL / 10)


def _captions_drain(engine, timeout: float) -> None:
    """Wait for the async workers to catch up before tearing the engine down."""
    deadline = time.time() + max(0.0, timeout)
    while time.time() < deadline:
        pending = (engine._mt_queue.qsize() + engine._refine_queue.qsize()
                   + engine._spk_queue.qsize())
        if pending == 0:
            time.sleep(0.5)          # let the in-flight item finish
            if (engine._mt_queue.qsize() + engine._refine_queue.qsize()
                    + engine._spk_queue.qsize()) == 0:
                return
        time.sleep(0.25)
    _log("CAPTION", "replay drain timed out; some corrections may be missing")


def _captions_force_review(engine) -> None:
    """Run the batch re-check once over everything, ignoring its timer."""
    engine._review_enabled = True
    batch = engine._take_review_batch()
    if not batch:
        print("[字幕] 复核：没有可复核的行")
        return
    print(f"[字幕] 复核 {len(batch)} 行（一次 LLM 调用）...")
    prompt = (
        _resolve_prompt(engine.cfg, "caption_review")
        .replace("{hotwords}", _cfg_hotword(engine.cfg) or "（无）")
        .replace("{lines}", _format_caption_review_lines(batch))
    )
    try:
        fixes = _parse_caption_review(engine._review_llm(prompt), batch)
    except Exception as e:
        print(f"[字幕] 复核失败: {type(e).__name__}: {e}", file=sys.stderr)
        return
    for row in batch:
        got = fixes.get(row["id"])
        if not got:
            continue
        text, dst = got
        if text != row["text"]:
            engine._emit(type="refined", id=row["id"], text=text)
        if dst and dst != row["dst"]:
            engine._emit(type="translation", id=row["id"], text=dst)


def _captions_report(rows: dict, order: list, trace: list, verbose: bool,
                     audio_secs: float = 0.0, endpoint: "dict | None" = None,
                     review_cfg: "dict | None" = None, stream=None) -> None:
    """Emit the settled state as ONE flushed write.

    Built as a single string on purpose: the per-line prints were lost when
    the process ended before stdout's block buffer was flushed, so a long
    replay could finish having printed nothing at all.

    `stream` is the caller's own stdout handle, captured before any worker
    thread could install a `_QuietCapture` over the process-global one. Using
    print() here loses the entire report whenever a refine call happens to be
    in flight — measured on a 4-minute replay, which printed its header and
    nothing else.
    """
    out: list = []
    if verbose:
        out.append("\n── 事件轨迹 ──")
        for ev in trace:
            if ev.get("type") in ("partial", "partial_translation"):
                continue          # too noisy to diff; several per second
            body = {k: v for k, v in ev.items() if k != "type"}
            out.append(f"  {ev.get('type'):12s} {body}")

    out.append(f"\n── 字幕 ({len(order)} 行) ──")
    for sid in order:
        r = rows[sid]
        tag = ""
        if r["speaker"] is not None:
            tag = f"[说话人{r['speaker']}] "
        elif r["role"]:
            tag = f"[{r['role']}] "
        mark = f" ×{r['revisions']}" if r["revisions"] else ""
        out.append(f"[{sid:3d}]{mark} {tag}{r['src']}")
        out.append(f"      → {r['dst'] or '（无译文）'}")

    src_chars = sum(len(r["src"]) for r in rows.values())
    revised = sum(1 for r in rows.values() if r["revisions"])
    translated = sum(1 for r in rows.values() if r["dst"])
    speakers = {r["speaker"] for r in rows.values() if r["speaker"] is not None}
    out.append("\n── 汇总 ──")
    out.append(f"  定稿行数    : {len(order)}")
    out.append(f"  原文总字数  : {src_chars}")
    out.append(f"  被修正的行  : {revised}")
    out.append(f"  有译文的行  : {translated}/{len(order)}")
    out.append(f"  识别出说话人: {len(speakers)}")

    # Segmentation block: the numbers a threshold A/B is decided on. Line
    # LENGTH is the payload (context per line for refine / review) and lines
    # per minute is the cost (each one is an LLM row to re-emit later), so
    # both are reported against the audio actually fed.
    if audio_secs > 0:
        lens = sorted(len(r["src"]) for r in rows.values())
        per_min = len(order) / (audio_secs / 60.0)
        median = lens[len(lens) // 2] if lens else 0
        out.append("\n── 断句 ──")
        if endpoint:
            out.append("  阈值        : " + " ".join(
                f"{k.split('_min_')[0]}={v:g}" for k, v in sorted(endpoint.items())))
        out.append(f"  音频时长    : {audio_secs / 60:.1f} 分钟")
        out.append(f"  行/分钟     : {per_min:.1f}")
        out.append(f"  平均行长    : {src_chars / max(1, len(order)):.1f} 字"
                   f"（中位 {median}，最长 {lens[-1] if lens else 0}）")
        # A review batch is one LLM call. At the measured 66 s median per
        # call, batches-per-hour is what says whether the pass keeps up.
        rc = review_cfg or {}
        interval = float(rc.get("interval_minutes") or 5) or 5
        max_lines = int(rc.get("max_lines") or 120)
        batches = max(1, math.ceil(per_min * interval / max(1, max_lines)))
        out.append(f"  复核批次    : 每 {interval:g} 分钟 {per_min * interval:.0f} 行 "
                   f"→ {batches} 批/周期（单批上限 {max_lines}）")
    out_stream = stream or sys.stdout
    out_stream.write("\n".join(out) + "\n")
    out_stream.flush()


def cmd_hotwords(args, cfg):
    _, restore = _setup_log_file()
    try:
        _cmd_hotwords_body(args, cfg)
    finally:
        restore()


def _hotwords_from_notion(args, cfg, hw_cfg: dict) -> None:
    """`hotwords --notion`: one-off import of member names, project names and
    terminology from a Notion workspace.

    Deliberately on demand rather than a background sync — the workspace does
    not change per meeting, and a scheduled crawl would be a standing network
    dependency for a tool that otherwise only talks to localhost and model
    downloads.

    The token comes from NOTION_TOKEN / NOTION_API_KEY in the environment and
    is never written to config, logged, or echoed in an error.
    """
    token = (os.environ.get("NOTION_TOKEN")
             or os.environ.get("NOTION_API_KEY") or "").strip()
    if not token:
        print("[热词] 需要 Notion token。请先创建一个 internal integration，"
              "把要读取的页面共享给它，然后：")
        print("    export NOTION_TOKEN=ntn_xxx   # 只从环境变量读取，不写入配置")
        print("    python3 meetingscribe.py hotwords --notion")
        sys.exit(1)

    limit = int(getattr(args, "notion_limit", 0) or 400)
    try:
        names = _notion_member_names(token)
        print(f"[热词] Notion 成员 {len(names)} 人")
        titles = _notion_titles(token)
        print(f"[热词] Notion 页面/数据库标题 {len(titles)} 条")
    except (RuntimeError, OSError) as e:
        print(f"[热词] Notion 拉取失败：{e}")
        sys.exit(1)

    cands = _notion_hotword_candidates(names, titles, limit=limit)
    print(f"[热词] 候选 {len(cands)} 个（上限 {limit}）")
    if not cands:
        print("[热词] 没有可用候选，未改动")
        return

    # Short pure-CJK terms are the measured danger: at scale they steal
    # ordinary speech (「分眼都在那边」→「分眼都郑娜边」). Report the count so
    # the effect is visible rather than silent, and let --min-cjk drop them.
    min_cjk = int(getattr(args, "min_cjk", 0) or 0)
    if min_cjk:
        before = len(cands)
        cands = [t for t in cands
                 if not all("一" <= c <= "鿿" for c in t) or len(t) >= min_cjk]
        print(f"[热词] 过滤掉 {before - len(cands)} 个短于 {min_cjk} 字的纯中文词")
    risky = [t for t in cands if all("一" <= c <= "鿿" for c in t) and len(t) <= 2]
    if risky:
        print(f"[热词] 注意：其中 {len(risky)} 个是 2 字纯中文词，"
              f"放大后容易抢走日常读音；如需剔除用 --min-cjk 3")

    if getattr(args, "dry_run", False):
        print("[热词] --dry-run，仅预览不写入：")
        print("  " + " ".join(cands))
        return

    # Pinned: these are authoritative workspace names, not guesses mined from
    # one meeting, so a rolling auto-extracted token must not evict them.
    added = _persist_hotwords(cands, cfg, hw_cfg.get("max_count", 1000),
                              pin=True)
    print(f"[热词] 新增 {len(added)} 个 → {HOTWORD_FILE.name}"
          f"（共 {len(((cfg.get('stt') or {}).get('funasr') or {}).get('hotword', '').split())} 个）")
    if added:
        print("  " + " ".join(added))
    print("[热词] 快速模式字幕需重启 app 才会用上新热词（识别器有缓存）")


def _hotwords_set_pins(args, cfg) -> None:
    """`hotwords --pin/--unpin`: mark terms as never-evictable, or release them.

    A pinned term that is not yet in the list is ADDED — pinning a name you
    know will come up is a reasonable thing to want to do in one command.
    """
    store = _load_hotword_store()
    current = ((cfg.get("stt") or {}).get("funasr") or {}).get("hotword", "") or ""
    if current.split():
        store["terms"] = current.split()
    known = {t.lower(): t for t in store["terms"]}

    added, pinned, unpinned, missing = [], [], [], []
    for raw in (getattr(args, "pin", None) or []):
        term = _clean_hotword_token(raw)
        if not 2 <= len(term) <= 32:
            missing.append(raw)
            continue
        low = term.lower()
        if low not in known:
            store["terms"].append(term)
            known[low] = term
            added.append(term)
        store["pinned"].add(low)
        pinned.append(known[low])
    for raw in (getattr(args, "unpin", None) or []):
        low = _clean_hotword_token(raw).lower()
        if low in store["pinned"]:
            store["pinned"].discard(low)
            unpinned.append(known.get(low, raw))
        else:
            missing.append(raw)

    merged = " ".join(store["terms"])
    cfg.setdefault("stt", {}).setdefault("funasr", {})["hotword"] = merged
    try:
        _save_hotword_file(merged, store)
    except OSError as e:
        print(f"[热词] 写入失败：{type(e).__name__}: {e}")
        return
    if pinned:
        print(f"✅ 已钉住 {len(pinned)} 个词（永不淘汰）：{' '.join(pinned)}")
    if added:
        print(f"   其中 {len(added)} 个是新加入词表的：{' '.join(added)}")
    if unpinned:
        print(f"✅ 已取消钉住 {len(unpinned)} 个：{' '.join(unpinned)}")
    if missing:
        print(f"⚠️  忽略 {len(missing)} 个（长度不合法或本来就没钉住）："
              f"{' '.join(str(m) for m in missing)}")
    print(f"[热词] 共 {len(merged.split())} 个，钉住 {len(store['pinned'])} 个"
          f" → {HOTWORD_FILE.name}")


def _cmd_hotwords_body(args, cfg):
    """`hotwords` 子命令：存量回填。

    扫描 recordings 下已有的 .polish.txt（方案 A 规则提取）与
    .meeting.md / .interview.md / .sharing.md（方案 B LLM 提取，纪要文本
    紧凑、分块并发调用），合并写回 config.jsonc 的 stt.funasr.hotword。
    日常增量由纪要流水线在每次生成后自动完成，无需手动跑本命令。
    """
    current = ((cfg.get("stt") or {}).get("funasr") or {}).get("hotword", "") or ""
    if getattr(args, "show", False):
        store = _load_hotword_store()
        terms = current.split() or store["terms"]
        print(f"当前热词（{len(terms)} 个，其中钉住 "
              f"{sum(1 for t in terms if t.lower() in store['pinned'])} 个）：")
        print(" ".join(terms) or "（空）")
        if store["pinned"]:
            print(f"\n钉住（永不淘汰）："
                  f"{' '.join(t for t in terms if t.lower() in store['pinned'])}")
        # Least recently used first: this is the eviction order, so it is the
        # useful view — these are the terms that go when the cap is reached.
        ranked = sorted(
            (t for t in terms if t.lower() not in store["pinned"]),
            key=lambda t: (store["last_seen"].get(t.lower(), 0),
                           store["hits"].get(t.lower(), 0)))
        if any(store["hits"].values()):
            print(f"\n最可能先被淘汰的 15 个（epoch={store['epoch']}）：")
            for t in ranked[:15]:
                print(f"  {t:24s} last_seen={store['last_seen'].get(t.lower(), 0)}"
                      f" hits={store['hits'].get(t.lower(), 0)}")
        else:
            print("\n（还没有命中历史；下一次生成纪要后开始记录）")
        return

    if getattr(args, "pin", None) or getattr(args, "unpin", None):
        _hotwords_set_pins(args, cfg)
        return

    hw_cfg = _deep_merge(DEFAULT_CONFIG["hotwords"], cfg.get("hotwords") or {})

    if getattr(args, "notion", False):
        _hotwords_from_notion(args, cfg, hw_cfg)
        return

    recordings_dir = CONFIG_DIR / "recordings"
    if not recordings_dir.exists():
        print(f"[热词] 录音目录不存在：{recordings_dir}")
        return

    candidates: "list[str]" = []
    # Text scanned for hit counting. Built from the rule pass, which always
    # runs — the LLM branch is optional (--no-llm), so a blob defined only
    # there would be undefined on that path.
    seen_text: "list[str]" = []

    polish_files = sorted(recordings_dir.glob("*.polish.txt"))
    print(f"[热词] 规则扫描 {len(polish_files)} 个 .polish.txt ...")
    for p in polish_files:
        try:
            body = p.read_text(encoding="utf-8")
        except OSError as e:
            print(f"[热词] 跳过 {p.name}：{e}")
            continue
        candidates += _extract_hotword_candidates(body)
        seen_text.append(body)

    note_files = sorted(
        f for suffix in _NOTES_SUFFIX.values()
        for f in recordings_dir.glob(f"*{suffix}"))
    if (not getattr(args, "no_llm", False)
            and hw_cfg.get("llm_extract", True) and note_files):
        import concurrent.futures
        provider = cfg.get("meeting_notes_provider", "claude")
        texts = []
        for f in note_files:
            try:
                texts.append(f.read_text(encoding="utf-8"))
            except OSError as e:
                print(f"[热词] 跳过 {f.name}：{e}")
        blob = "\n\n".join(texts)
        seen_text.append(blob)
        chunks = [blob[i:i + 12000] for i in range(0, len(blob), 12000)]
        print(f"[热词] LLM 提取（{provider}）：{len(note_files)} 份纪要 / "
              f"{len(chunks)} 块并发 ...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(_extract_hotwords_llm, c, provider, cfg)
                    for c in chunks]
            for fut in futs:
                try:
                    candidates += fut.result()
                except (Exception, SystemExit) as e:
                    print(f"[热词] LLM 提取块失败：{type(e).__name__}: {e}")

    if not candidates:
        print("[热词] 未提取到候选热词")
        return
    # Backfill also carries the hit signal: these are the transcripts the
    # terms came from, so existing hotwords found in them count as in use.
    added = _persist_hotwords(candidates, cfg, int(hw_cfg.get("max_count", 100)),
                              transcript="\n\n".join(seen_text))
    total = len(((cfg.get("stt") or {}).get("funasr") or {})
                .get("hotword", "").split())
    if added:
        print(f"✅ 新增 {len(added)} 个热词（共 {total} 个），已写入 {HOTWORD_FILE.name}：")
        print("   " + " ".join(added))
    else:
        print(f"[热词] 无新增（共 {total} 个）")


# ── 桌面 UI (PyQt6 + PyQt6-Fluent-Widgets) ──────────────────────────────────
#
# The only desktop GUI for this project. Reached via ``python3
# meetingscribe.py ui``. Imports are lazy inside the function so that the
# headless / CLI subcommands (record / transcribe / devices / config) keep
# working when PyQt6 isn't installed; the GUI itself prints a helpful pip
# command and exits if the import fails.
#
# Features:
#   - Sidebar nav with THREE items (录音 / 历史 / 配置) inside a plain
#     QMainWindow + NavigationInterface — gives macOS native traffic
#     lights on the LEFT with standard ×/−/+ hover icons.
#   - Recording: no mic-selector row (always system audio + resolver-chosen
#     mic). Subtitle clarifies the two captured streams.
#   - History sidebar: live substring search; right-click → rename / delete
#     (cascades across every sibling file via _rename_meeting_files /
#     _delete_meeting_files).
#   - History view: four filter tabs (全部 / 已总结 / 已录音转文字 / 待处理).
#     Detail pane renders body differently per tab.
#   - Config view: SpinBox to bump the two concurrency limits at once
#     plus a raw editor for config.jsonc (JSONC-aware save via save_config).
#   - Pipeline prompts are user-editable via ``cfg["prompts"]`` —
#     overrideable from the JSONC editor.
#   - Live 中文 / EN toggle in the top bar (single button — flips every
#     translatable widget across all views).
#
# Future ideas: structured meta data (participants, todos, key points),
# waveform meter, account / settings panes.

def cmd_ui(args, cfg):
    """PyQt6 + PyQt6-Fluent-Widgets desktop GUI — the project's only UI.

    Run: ``python3 meetingscribe.py ui``
    Requires: ``python3 -m pip install PyQt6 PyQt6-Fluent-Widgets``

    Note: do NOT install ``PyQt-Fluent-Widgets`` (without the ``6``) — that one
    pulls in PyQt5 as a dependency and collides with PyQt6 in the same process
    (duplicate ObjC class registrations → QApplication construction abort on
    macOS). Always use the ``PyQt6-Fluent-Widgets`` package name.

    Wraps the body with _setup_log_file() (matching cmd_ui / cmd_record /
    cmd_transcribe) so every print() and `logging` WARNING+ record from the
    pipeline thread is timestamped and persisted to the daily log file.
    """
    _, restore = _setup_log_file()
    try:
        _cmd_ui_body(args, cfg)
    finally:
        restore()


def _cmd_ui_body(args, cfg):
    try:
        from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QPoint, QSize
        from PyQt6.QtGui import (
            QFont, QAction, QColor, QSyntaxHighlighter, QTextCharFormat,
            QTextCursor, QTextFormat, QGuiApplication,
        )
        from PyQt6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame,
            QStackedWidget, QListWidgetItem, QFileDialog, QMenu, QInputDialog,
            QMessageBox, QPlainTextEdit, QPushButton,
        )
        from qfluentwidgets import (
            NavigationInterface, NavigationItemPosition,
            FluentIcon, setTheme, Theme,
            PrimaryPushButton, PushButton, Slider, ListWidget,
            TextEdit, ProgressBar, SegmentedWidget, BodyLabel,
            StrongBodyLabel, TitleLabel, SubtitleLabel, CaptionLabel,
            SearchLineEdit, TextBrowser, ScrollArea, SpinBox,
            InfoBar, InfoBarPosition, SwitchButton,
        )
    except ImportError as e:
        print(
            "[错误] ui 需要 PyQt6 和 PyQt6-Fluent-Widgets。\n"
            "请安装：\n"
            "    python3 -m pip install PyQt6 PyQt6-Fluent-Widgets\n"
            "注意：包名是 PyQt6-Fluent-Widgets，不是 PyQt-Fluent-Widgets。\n"
            "后者会拖入 PyQt5，与 PyQt6 在同一进程中冲突（ObjC class 重复注册）。\n"
            f"详细错误: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    RECORDINGS_DIR = CONFIG_DIR / "recordings"

    # ── Helpers ────────────────────────────────────────────────────────────

    def _list_recordings() -> list[dict]:
        """Scan recordings dir for .wav files with their companion artifacts."""
        if not RECORDINGS_DIR.exists():
            return []
        out: list[dict] = []
        for p in sorted(RECORDINGS_DIR.glob("*.wav"),
                        key=lambda x: x.stat().st_mtime, reverse=True):
            stem = p.stem
            # New format: <stem>.meeting.md / <stem>.interview.md /
            # <stem>.sharing.md. Legacy format: <stem>.md (still detected
            # for backward compat).
            meeting_md = p.with_name(stem + ".meeting.md")
            interview_md = p.with_name(stem + ".interview.md")
            sharing_md = p.with_name(stem + ".sharing.md")
            legacy_md = p.with_name(stem + ".md")
            # Pick the first existing artifact; preference order is meeting
            # > interview > sharing > legacy. `md_path` / `md_mode` exist
            # mainly for backward compat with code paths that haven't been
            # taught about the mode-specific path fields yet.
            if meeting_md.exists():
                md_path, md_mode = meeting_md, "meeting"
            elif interview_md.exists():
                md_path, md_mode = interview_md, "interview"
            elif sharing_md.exists():
                md_path, md_mode = sharing_md, "sharing"
            elif legacy_md.exists():
                md_path, md_mode = legacy_md, "meeting"  # legacy assumed meeting
            else:
                md_path, md_mode = None, None
            raw = p.with_name(stem + ".raw.txt")
            polish = p.with_name(stem + ".polish.txt")
            out.append({
                "wav_path": p,
                "stem": stem,
                "mtime": p.stat().st_mtime,
                "has_md": md_path is not None,
                "has_polish": polish.exists(),
                "has_raw": raw.exists(),
                "md_path": md_path,
                "md_mode": md_mode,
                "meeting_md_path": meeting_md if meeting_md.exists() else None,
                "interview_md_path": interview_md if interview_md.exists() else None,
                "sharing_md_path": sharing_md if sharing_md.exists() else None,
                "polish_path": polish if polish.exists() else None,
                "raw_path": raw if raw.exists() else None,
            })
        return out

    def _format_timestamp_label(stem: str) -> str:
        """``20260518_174926`` → ``2026/05/18 17:49``;
        ``20260518_174926.客户访谈`` → ``2026/05/18 17:49 · 客户访谈``."""
        ts, custom = _split_meeting_stem(stem)
        if not ts:
            return stem
        try:
            pretty = datetime.strptime(ts, "%Y%m%d_%H%M%S").strftime("%Y/%m/%d %H:%M")
        except ValueError:
            pretty = ts
        return f"{pretty} · {custom}" if custom else pretty

    def _audio_duration_secs(wav_path: Path) -> int | None:
        try:
            with wave.open(str(wav_path), "rb") as w:
                return int(w.getnframes() / max(w.getframerate(), 1))
        except Exception:
            return None

    def _format_duration(secs: int | None) -> str:
        if not secs:
            return _t("dur.unknown")
        if secs >= 3600:
            return _t("dur.hour_minute", h=secs // 3600, m=(secs % 3600) // 60)
        if secs >= 60:
            return _t("dur.minute", m=secs // 60)
        return _t("dur.second", s=secs)

    # Shared visual styling so RecordingInterface / HistoryInterface columns
    # share the same Fluent-light "card" look — a soft rounded background
    # with consistent internal padding. We use objectName-qualified QSS so
    # nested widgets (lists, buttons, etc.) are NOT inadvertently restyled.
    _CARD_BG = "#f5f7fa"

    def _confirm_dialog(parent, title: str, msg: str) -> bool:
        """Modal Yes/No dialog whose button labels are routed through
        ``_t("ctx.confirm_yes")`` / ``_t("ctx.confirm_no")`` so they flip
        between Chinese and English with the rest of the UI.

        Qt's built-in ``QMessageBox.question(...)`` labels its buttons
        from the system locale via QTranslator — independent of our
        ``_LANG["current"]`` — which means the EN toggle would otherwise
        leave the buttons in Chinese (or vice-versa) depending on the
        user's macOS locale. Using ``addButton`` lets us control the
        labels directly.
        """
        box = QMessageBox(parent)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(title)
        box.setText(msg)
        yes_btn = box.addButton(_t("ctx.confirm_yes"), QMessageBox.ButtonRole.YesRole)
        box.addButton(_t("ctx.confirm_no"), QMessageBox.ButtonRole.NoRole)
        box.exec()
        return box.clickedButton() is yes_btn

    def _apply_open_btn_style(btn, is_open: bool) -> None:
        """Paint a pipeline action button. Both states share the exact
        same rounded geometry (border-radius, padding, border width) so
        the button doesn't visually "jump" when an artifact appears /
        disappears on disk — only the fill colour changes:

          * ``is_open=True``  → accent blue (matches the idle mic-button
            colour), reads as a primary "open the result" affordance.
          * ``is_open=False`` → light gray, neutral "generate" affordance.

        Defined at the cmd_ui scope so RecordingInterface and
        HistoryInterface share the exact same visual treatment.
        """
        if is_open:
            btn.setStyleSheet(
                "PushButton {"
                "  background-color: #0a84ff;"
                "  color: white;"
                "  border: 1px solid #0a84ff;"
                "  border-radius: 6px;"
                "  padding: 6px 16px;"
                "}"
                "PushButton:hover { background-color: #0066d6; }"
                "PushButton:pressed { background-color: #0050a8; }"
                "PushButton:disabled {"
                "  background-color: #9bbfe6;"
                "  border-color: #9bbfe6;"
                "  color: #f0f0f0;"
                "}"
            )
        else:
            btn.setStyleSheet(
                "PushButton {"
                "  background-color: #e8eaed;"
                "  color: #1f2328;"
                "  border: 1px solid #d0d7de;"
                "  border-radius: 6px;"
                "  padding: 6px 16px;"
                "}"
                "PushButton:hover { background-color: #dadce0; }"
                "PushButton:pressed { background-color: #c8ccd1; }"
                "PushButton:disabled {"
                "  background-color: #f0f1f3;"
                "  border-color: #e0e2e5;"
                "  color: #9aa0a6;"
                "}"
            )

    def _style_as_card(w: "QWidget", padding: int = 18, name: str = "columnCard") -> None:
        w.setObjectName(name)
        # WA_StyledBackground makes a plain QWidget honour the QSS
        # background-color + border-radius (QFrame does this by default,
        # but the history-view detail pane is a bare QWidget so without
        # this attribute its rounded corners wouldn't render).
        w.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        w.setStyleSheet(
            f"#{name} {{ background-color: {_CARD_BG}; border-radius: 14px; }}"
        )
        lyt = w.layout()
        if lyt is not None:
            lyt.setContentsMargins(padding, padding, padding, padding)

    def _open_path(path) -> None:
        path = str(path)
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", path])
            else:
                subprocess.run(["xdg-open", path])
        except Exception as e:
            _log("ERR", f"qt open path={path!r}: {type(e).__name__}: {e}")

    # ── Markdown callouts (GitHub / Obsidian `> [!type]` blockquotes) ──────
    import re

    # type -> (icon, title, foreground, background); GitHub-flavoured colors.
    _CALLOUT_STYLES = {
        "note":      ("📝", "Note",      "#0969da", "#ddf4ff"),
        "tip":       ("💡", "Tip",       "#1a7f37", "#dcffe4"),
        "todo":      ("✅", "To-do",     "#1a7f37", "#dcffe4"),
        "important": ("⭐", "Important", "#8250df", "#fbefff"),
        "warning":   ("⚠️", "Warning",   "#9a6700", "#fff8c5"),
        "caution":   ("🚫", "Caution",   "#cf222e", "#ffebe9"),
    }
    _CALLOUT_MARKER_RE = re.compile(r"^\[!([A-Za-z]+)\]\s*")

    def _apply_callout_styles(doc) -> None:
        """Restyle callout blockquotes after ``setMarkdown``.

        Qt's Markdown importer renders ``> [!important]`` as a plain
        blockquote with the literal ``[!important]`` text; this pass finds
        those blocks (via the ``BlockQuoteLevel`` block property), paints a
        colored card background across the quote run, and swaps the marker
        for a bold colored icon+title.
        """
        def _quote_level(b) -> int:
            return b.blockFormat().intProperty(QTextFormat.Property.BlockQuoteLevel)

        block = doc.begin()
        while block.isValid():
            m = _CALLOUT_MARKER_RE.match(block.text()) if _quote_level(block) > 0 else None
            if not m:
                block = block.next()
                continue
            icon, title, fg, bg = _CALLOUT_STYLES.get(
                m.group(1).lower(), ("💬", m.group(1).capitalize(), "#57606a", "#eef1f4")
            )
            # Body = this block + following quote blocks, stopping at the
            # next non-quote block or the next callout marker.
            run = [block]
            nxt = block.next()
            while (
                nxt.isValid()
                and _quote_level(nxt) > 0
                and not _CALLOUT_MARKER_RE.match(nxt.text())
            ):
                run.append(nxt)
                nxt = nxt.next()
            for rb in run:
                bf = rb.blockFormat()
                bf.setBackground(QColor(bg))
                cur = QTextCursor(rb)
                cur.setBlockFormat(bf)
            # Soft line breaks merge "> [!type]\n> body" into one paragraph,
            # so replace just the marker and re-split with U+2028 when body
            # text follows in the same block.
            has_body = m.end() < len(block.text())
            cur = QTextCursor(doc)
            cur.setPosition(block.position())
            cur.setPosition(block.position() + m.end(), QTextCursor.MoveMode.KeepAnchor)
            title_fmt = QTextCharFormat()
            title_fmt.setFontWeight(QFont.Weight.Bold)
            title_fmt.setForeground(QColor(fg))
            cur.insertText(f"{icon} {title}" + ("\u2028" if has_body else ""), title_fmt)
            block = nxt

    # ── Internationalisation ─────────────────────────────────────────────
    #
    # Single-process language state. The 中文 / EN toggle in MainWindow
    # flips `_LANG["current"]` and calls each interface's `apply_language()`
    # — every translatable widget either registered a callback in
    # `_lang_callbacks` (for static labels) or gets re-rendered via the
    # interface's own refresh path (for dynamic strings like the history
    # count label or recording status text).
    _LANG: dict[str, str] = {"current": "zh"}
    _LABELS: dict[str, dict[str, str]] = {
        "zh": {
            # ── Top bar / window
            "app.title": "MeetingScribe",
            "topbar.lang_zh": "中文/EN",
            "topbar.lang_en": "中文/EN",
            # ── Nav
            "nav.recording": "录音",
            "nav.history": "历史",
            "nav.config": "配置",
            # ── Recording view
            "rec.title.idle": "开始录音",
            "rec.title.recording": "正在录音…",
            "rec.subtitle.idle": "默认录制（内外）扬声器和（内外）麦克风",
            "rec.subtitle.recording": "再次点击中间按钮停止录音",
            "rec.volume": "输出音量：",
            "rec.volume_hint": "💡 录音期间用此滑块调音量（F11/F12 在录音中会失效）",
            "rec.btn.transcribe": "语音转文字",
            "rec.btn.notes": "生成会议纪要",
            "rec.btn.interview": "生成面试报告",
            "rec.btn.sharing": "生成分享总结",
            "rec.btn.open_transcribe": "打开语音转文字结果",
            "rec.btn.open_notes": "打开会议纪要",
            "rec.btn.open_interview": "打开面试报告",
            "rec.btn.open_sharing": "打开分享总结",
            "rec.btn.cancel": "停止任务",
            "rec.btn.import": "导入外部音频…",
            "rec.import.dialog_title": "选择要导入的音频文件",
            "rec.import.filter": "音频文件 (*.wav *.m4a *.mp3 *.aac *.flac *.ogg *.opus *.aif *.aiff *.aifc *.wma *.amr *.caf *.webm);;所有文件 (*)",
            "rec.import.progress": "正在导入并转码…",
            "rec.import.done_fmt": "已导入：{name}",
            "rec.import.failed_title": "导入失败",
            "rec.import.failed_msg_fmt": "{err}",
            "rec.import.no_ffmpeg_msg": "未在 PATH 中找到 ffmpeg。\n在 macOS 上可执行：brew install ffmpeg",
            "rec.no_file": "未选择录音文件",
            "rec.selected_prefix": "已选择：",
            "rec.captions.title": "实时双语字幕",
            "rec.captions.idle_hint": "开始录音后，此处实时显示中英双语字幕",
            "rec.captions.disabled_hint": "字幕已关闭",
            "rec.captions.status.loading": "字幕模型加载中…（首次使用需下载模型）",
            "rec.captions.status.ready": "字幕运行中",
            "rec.captions.status.stopped": "字幕已停止",
            "rec.captions.error_prefix": "字幕出错：",
            "rec.captions.role_mic": "我方",
            "rec.captions.role_system": "对方",
            "rec.captions.speaker_n": "说话人{n}",
            # ── History view
            "hist.search_placeholder": "按文件名搜索…（子串匹配）",
            "hist.tab.all": "全部",
            "hist.tab.done": "已总结",
            "hist.tab.stt_only": "已录音转文字",
            "hist.tab.pending": "待处理",
            "hist.default_title": "选择左侧会议查看详情",
            "hist.body_title": "会议内容",
            "hist.participants": "参与者",
            "hist.todos": "待办事项",
            "hist.no_data": "未获取相关信息",
            "hist.count_fmt": "共 {n} 条会议记录",
            "hist.btn.transcribe": "语音转文字",
            "hist.btn.notes": "生成会议纪要",
            "hist.btn.interview": "生成面试报告",
            "hist.btn.sharing": "生成分享总结",
            "hist.btn.open_transcribe": "打开语音转文字结果",
            "hist.btn.open_notes": "打开会议纪要",
            "hist.btn.open_interview": "打开面试报告",
            "hist.btn.open_sharing": "打开分享总结",
            "hist.btn.cancel": "停止任务",
            # ── Right-click context menus + delete confirmation
            "ctx.rename": "重命名…",
            "ctx.reveal": "在 Finder/资源管理器中显示",
            "ctx.delete": "删除本次会议所有记录…",
            "ctx.delete_title": "确认删除",
            "ctx.delete_confirm": "确认删除 “{name}” 及其所有相关文件（.wav / .raw.txt / .polish.txt / .meeting.md / .interview.md / .sharing.md 等）？\n此操作不可恢复。",
            "ctx.delete_failed_title": "删除失败",
            "ctx.delete_failed_msg": "{err}",
            "ctx.confirm_yes": "确认",
            "ctx.confirm_no": "取消",
            "ctx.delete_blocked_title": "无法删除",
            "ctx.delete_blocked_msg": "当前会议正在处理流水线任务，请等任务结束后再删除。",
            "ctx.rename_failed_title": "重命名失败",
            "ctx.rename_failed_format": "文件名不是 <时间戳>[.<名字>] 格式，无法重命名。",
            "ctx.rename_failed_collision": "目标文件名已存在或无法重命名，请换一个名字。",
            "ctx.rename_dialog_title": "重命名会议",
            "ctx.rename_dialog_prompt": "为 {ts} 设置一个可读的名字（留空则去掉自定义名字）：",
            # ── Pipeline log lines + warnings (shown in the in-UI log_view)
            "pipe.log.done": "✓ 完成 → {path}",
            "pipe.info.done_title": "已完成",
            "pipe.log.failed": "✗ 失败：{err}",
            "pipe.log.cancelling": "[提示] 正在终止后台任务…",
            "pipe.log.cancelled": "[已停止] 后台任务已终止",
            "pipe.warn.no_wav": "请先录音或选择 .wav 文件",
            "pipe.warn.prefix": "[警告] {msg}",
            "pipe.warn.file_missing_title": "文件不存在",
            "pipe.warn.file_missing_msg": "找不到 {name}。",
            # ── History list badges + body titles
            "hist.badge.done": "✓ 已总结",
            "hist.badge.transcribed": "📝 已转文字",
            "hist.badge.pending_summary": "待生成纪要",
            "hist.badge.audio_only": "仅录音",
            "hist.dur_prefix": "时长",
            "dur.unknown": "未知时长",
            "dur.hour_minute": "{h} 小时 {m} 分钟",
            "dur.minute": "{m} 分钟",
            "dur.second": "{s} 秒",
            "hist.body.notes_meeting": "会议纪要",
            "hist.body.notes_interview": "面试报告",
            "hist.body.notes_sharing": "分享总结",
            "hist.body.notes_both": "会议纪要 + 面试报告",
            "hist.body.notes_meeting_md": "会议纪要 (.meeting.md)",
            "hist.body.notes_interview_md": "面试报告 (.interview.md)",
            "hist.body.notes_sharing_md": "分享总结 (.sharing.md)",
            "hist.body.notes_legacy_md": "会议纪要 (.md 旧格式)",
            "hist.body.polish_only": "语音转文字（已校对，.polish.txt）",
            "hist.body.raw_only": "原始转写 (.raw.txt)",
            "hist.body.no_notes": "（无总结文件）",
            "hist.body.no_polish": "（无 .polish.txt 文件）",
            "hist.body.raw_pending": "原始转写 (.raw.txt) — 等待后续校对 / 总结",
            "hist.body.pending_placeholder": "尚未生成任何转写 / 纪要文件。\n在「录音」页选择此 .wav 然后点「语音转文字」或「生成会议纪要 / 面试报告 / 分享总结」启动流水线。",
            "hist.body.none": "（尚未生成转写 / 纪要文件）",
            "hist.body.default": "会议内容",
            "hist.body.read_error": "（无法读取 {name}: {err}）",
            # ── ConfigInterface InfoBars + errors
            "cfg.info.applied_title": "并发已应用",
            "cfg.info.applied_body": "已修改 LLM 校对 / FunASR 转写并发任务数 = {n}，已写回 config.jsonc",
            "cfg.info.save_failed_title": "保存失败",
            "cfg.info.saved_title": "配置已保存",
            "cfg.info.saved_body": "config.jsonc 已写回（注释和顺序按你写的样子保留）",
            "cfg.info.write_failed_title": "写入失败",
            "cfg.info.json_error_title": "JSON 语法错误，未保存",
            # ── Config view
            "cfg.title.concurrency": "后台任务并发数",
            "cfg.desc.concurrency": ("一次性同步两个并发：LLM 校对、FunASR 转写"
                                     "并发任务数。0 = 自动（由 CPU 核数推导）；"
                                     "> 0 时直接使用该值，不再二次计算。"),
            "cfg.label.workers": "并发任务数：",
            "cfg.btn.apply": "应用",
            "cfg.title.editor": "配置文件编辑器",
            "cfg.desc.editor": ("直接编辑 config.jsonc 原文（支持 // 注释）。"
                                "保存前会做一次 JSON 语法校验；校验失败不会覆盖磁盘上的文件。"),
            "cfg.btn.reload": "重新加载",
            "cfg.btn.save": "保存配置文件",
        },
        "en": {
            "app.title": "MeetingScribe",
            "topbar.lang_zh": "中文/EN",
            "topbar.lang_en": "中文/EN",
            "nav.recording": "Recording",
            "nav.history": "History",
            "nav.config": "Settings",
            "rec.title.idle": "Start Recording",
            "rec.title.recording": "Recording…",
            "rec.subtitle.idle": "Captures both system audio (speakers) and microphone by default.",
            "rec.subtitle.recording": "Click the centre button again to stop.",
            "rec.volume": "Output volume:",
            "rec.volume_hint": "💡 Use this slider during recording — F11/F12 are disabled while recording.",
            "rec.btn.transcribe": "Transcribe",
            "rec.btn.notes": "Generate meeting notes",
            "rec.btn.interview": "Generate interview report",
            "rec.btn.sharing": "Generate sharing summary",
            "rec.btn.open_transcribe": "Open transcript",
            "rec.btn.open_notes": "Open meeting notes",
            "rec.btn.open_interview": "Open interview report",
            "rec.btn.open_sharing": "Open sharing summary",
            "rec.btn.cancel": "Stop task",
            "rec.btn.import": "Import audio file…",
            "rec.import.dialog_title": "Select an audio file to import",
            "rec.import.filter": "Audio files (*.wav *.m4a *.mp3 *.aac *.flac *.ogg *.opus *.aif *.aiff *.aifc *.wma *.amr *.caf *.webm);;All files (*)",
            "rec.import.progress": "Importing & transcoding…",
            "rec.import.done_fmt": "Imported: {name}",
            "rec.import.failed_title": "Import failed",
            "rec.import.failed_msg_fmt": "{err}",
            "rec.import.no_ffmpeg_msg": "ffmpeg not found on PATH.\nOn macOS: brew install ffmpeg",
            "rec.no_file": "No recording selected",
            "rec.selected_prefix": "Selected: ",
            "rec.captions.title": "Live bilingual captions",
            "rec.captions.idle_hint": "Bilingual captions appear here once recording starts",
            "rec.captions.disabled_hint": "Captions are off",
            "rec.captions.status.loading": "Loading caption models… (first use downloads them)",
            "rec.captions.status.ready": "Captions live",
            "rec.captions.status.stopped": "Captions stopped",
            "rec.captions.error_prefix": "Caption error: ",
            "rec.captions.role_mic": "Us",
            "rec.captions.role_system": "Them",
            "rec.captions.speaker_n": "Speaker {n}",
            "hist.search_placeholder": "Search by filename… (substring)",
            "hist.tab.all": "All",
            "hist.tab.done": "Summarized",
            "hist.tab.stt_only": "Transcribed",
            "hist.tab.pending": "Pending",
            "hist.default_title": "Select a meeting on the left to view details",
            "hist.body_title": "Content",
            "hist.participants": "Participants",
            "hist.todos": "Todos",
            "hist.no_data": "No data available",
            "hist.count_fmt": "{n} meeting(s)",
            "hist.btn.transcribe": "Transcribe",
            "hist.btn.notes": "Generate meeting notes",
            "hist.btn.interview": "Generate interview report",
            "hist.btn.sharing": "Generate sharing summary",
            "hist.btn.open_transcribe": "Open transcript",
            "hist.btn.open_notes": "Open meeting notes",
            "hist.btn.open_interview": "Open interview report",
            "hist.btn.open_sharing": "Open sharing summary",
            "hist.btn.cancel": "Stop task",
            "ctx.rename": "Rename…",
            "ctx.reveal": "Show in Finder / Explorer",
            "ctx.delete": "Delete this meeting…",
            "ctx.delete_title": "Confirm delete",
            "ctx.delete_confirm": "Delete \"{name}\" and all related files (.wav / .raw.txt / .polish.txt / .meeting.md / .interview.md / .sharing.md etc.)?\nThis cannot be undone.",
            "ctx.delete_failed_title": "Delete failed",
            "ctx.delete_failed_msg": "{err}",
            "ctx.confirm_yes": "Confirm",
            "ctx.confirm_no": "Cancel",
            "ctx.delete_blocked_title": "Can't delete",
            "ctx.delete_blocked_msg": "A pipeline is running on this meeting. Wait for it to finish, then try again.",
            "ctx.rename_failed_title": "Rename failed",
            "ctx.rename_failed_format": "Filename does not match the <timestamp>[.<name>] pattern; can't rename.",
            "ctx.rename_failed_collision": "Target filename already exists or can't be renamed. Try a different name.",
            "ctx.rename_dialog_title": "Rename meeting",
            "ctx.rename_dialog_prompt": "Pick a readable name for {ts} (leave blank to drop the custom name):",
            "pipe.log.done": "✓ Done → {path}",
            "pipe.info.done_title": "Done",
            "pipe.log.failed": "✗ Failed: {err}",
            "pipe.log.cancelling": "[info] Cancelling background task…",
            "pipe.log.cancelled": "[stopped] Background task cancelled",
            "pipe.warn.no_wav": "Record first or pick a .wav file.",
            "pipe.warn.prefix": "[warn] {msg}",
            "pipe.warn.file_missing_title": "File not found",
            "pipe.warn.file_missing_msg": "Could not find {name}.",
            "hist.badge.done": "✓ Summarized",
            "hist.badge.transcribed": "📝 Transcribed",
            "hist.badge.pending_summary": "Awaiting summary",
            "hist.badge.audio_only": "Audio only",
            "hist.dur_prefix": "Duration",
            "dur.unknown": "Unknown duration",
            "dur.hour_minute": "{h}h {m}m",
            "dur.minute": "{m} min",
            "dur.second": "{s} sec",
            "hist.body.notes_meeting": "Meeting Notes",
            "hist.body.notes_interview": "Interview Report",
            "hist.body.notes_sharing": "Sharing Summary",
            "hist.body.notes_both": "Meeting Notes + Interview Report",
            "hist.body.notes_meeting_md": "Meeting Notes (.meeting.md)",
            "hist.body.notes_interview_md": "Interview Report (.interview.md)",
            "hist.body.notes_sharing_md": "Sharing Summary (.sharing.md)",
            "hist.body.notes_legacy_md": "Meeting Notes (.md, legacy)",
            "hist.body.polish_only": "Transcript (polished, .polish.txt)",
            "hist.body.raw_only": "Raw transcript (.raw.txt)",
            "hist.body.no_notes": "(no summary file)",
            "hist.body.no_polish": "(no .polish.txt file)",
            "hist.body.raw_pending": "Raw transcript (.raw.txt) — awaiting polish / notes",
            "hist.body.pending_placeholder": "No transcript / notes file yet.\nGo to Recording, pick this .wav, then click Transcribe / Generate notes / Generate interview report / Generate sharing summary.",
            "hist.body.none": "(no transcript / notes file yet)",
            "hist.body.default": "Content",
            "hist.body.read_error": "(failed to read {name}: {err})",
            "cfg.info.applied_title": "Concurrency applied",
            "cfg.info.applied_body": "Updated LLM polish / FunASR workers = {n}, saved to config.jsonc.",
            "cfg.info.save_failed_title": "Save failed",
            "cfg.info.saved_title": "Config saved",
            "cfg.info.saved_body": "config.jsonc written (comments and ordering preserved).",
            "cfg.info.write_failed_title": "Write failed",
            "cfg.info.json_error_title": "JSON syntax error — not saved",
            "cfg.title.concurrency": "Background task concurrency",
            "cfg.desc.concurrency": ("Sets two concurrency limits at once: LLM polish and FunASR workers. "
                                     "0 = auto (derived from CPU count); "
                                     "> 0 uses the value as-is, no fallback computation."),
            "cfg.label.workers": "Workers:",
            "cfg.btn.apply": "Apply",
            "cfg.title.editor": "config.jsonc editor",
            "cfg.desc.editor": ("Edit config.jsonc as raw text (// comments supported). "
                                "The text is JSON-validated before saving; an invalid edit "
                                "will NOT overwrite the file on disk."),
            "cfg.btn.reload": "Reload",
            "cfg.btn.save": "Save config",
        },
    }

    def _t(key: str, **fmt) -> str:
        s = _LABELS.get(_LANG["current"], {}).get(key, key)
        return s.format(**fmt) if fmt else s

    # Dynamic placeholder helper — the participants / todos panes call this
    # each render so they re-translate on language switch.
    def _placeholder_no_data() -> str:
        return _t("hist.no_data")

    # ── Pipeline worker (runs on QThread) ─────────────────────────────────

    class _JSONCHighlighter(QSyntaxHighlighter):
        """Light-theme JSONC syntax highlighter used by the config editor.

        Tokens (rendered left-to-right; later passes override earlier ones
        for comments so a "//" inside a string stays string-coloured):

          keys      → blue   ``"polish_max_workers":``
          strings   → green  ``"meeting"``
          numbers   → red    ``48000``, ``-0.5``, ``1e6``
          keywords  → purple ``true`` / ``false`` / ``null``
          comments  → grey italic, ``// …`` to end of line

        Multi-line block comments are NOT recognised — the project's
        JSONC files only use single-line ``//`` comments.
        """

        def __init__(self, document):
            super().__init__(document)
            import re as _re
            self._re = _re

            def _fmt(rgb, italic=False):
                f = QTextCharFormat()
                f.setForeground(QColor(rgb))
                if italic:
                    f.setFontItalic(True)
                return f

            self._fmt_string = _fmt("#0a7c00")
            self._fmt_key = _fmt("#0451a5")
            self._fmt_number = _fmt("#a31515")
            self._fmt_keyword = _fmt("#7d4eb1")
            self._fmt_comment = _fmt("#6a737d", italic=True)

        def highlightBlock(self, text: str) -> None:
            re = self._re
            # 1. Strings first (so we can detect what's inside one).
            string_spans = []
            for m in re.finditer(r'"(?:\\.|[^"\\])*"', text):
                self.setFormat(m.start(), m.end() - m.start(), self._fmt_string)
                string_spans.append((m.start(), m.end()))

            def _in_string(pos: int) -> bool:
                return any(s <= pos < e for s, e in string_spans)

            # 2. Re-colour string-keys (a string immediately followed by `:`)
            #    so they stand out from value strings.
            for m in re.finditer(r'"(?:\\.|[^"\\])*"(?=\s*:)', text):
                self.setFormat(m.start(), m.end() - m.start(), self._fmt_key)

            # 3. Numbers (outside strings).
            for m in re.finditer(r'-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b', text):
                if not _in_string(m.start()):
                    self.setFormat(m.start(), m.end() - m.start(), self._fmt_number)

            # 4. Keywords.
            for m in re.finditer(r'\b(?:true|false|null)\b', text):
                if not _in_string(m.start()):
                    self.setFormat(m.start(), m.end() - m.start(), self._fmt_keyword)

            # 5. Line comments LAST so they override any earlier colouring
            #    for characters after the "//" (but a "//" inside a string
            #    is suppressed by the in_string check).
            for m in re.finditer(r'//.*$', text):
                if not _in_string(m.start()):
                    self.setFormat(m.start(), m.end() - m.start(), self._fmt_comment)


    class _PipelineWorker(QObject):
        """Runs transcribe → polish → notes on a background QThread, emitting
        Qt signals (``progress`` / ``log`` / ``done`` / ``failed`` /
        ``cancelled``) for the UI to consume.

        ``cancel()`` is the public hook the UI calls when the user clicks
        "停止任务"; it sets the module-level ``_PIPELINE_CANCEL`` event AND
        SIGKILLs every registered subprocess (currently `claude -p …`), so
        the background task actually terminates instead of running to
        completion. Stage boundaries inside `run()` call
        ``_pipeline_check_cancelled()`` to translate the event into a clean
        ``_PipelineCancelled`` exception, which we then surface as the
        ``cancelled`` signal."""
        progress = pyqtSignal(int)
        log = pyqtSignal(str)
        done = pyqtSignal(str)   # result md path
        failed = pyqtSignal(str)  # error message
        cancelled = pyqtSignal()  # user-initiated stop

        def __init__(self, audio_path: Path, mode: str, cfg_: dict,
                     transcribe_only: bool = False, parent=None):
            super().__init__(parent)
            self.audio_path = audio_path
            self.mode = mode
            self.cfg = cfg_
            self.transcribe_only = transcribe_only
            # Always start uncancelled — clear leftover state from a
            # previous (cancelled / completed) run.
            _PIPELINE_CANCEL.clear()

        def cancel(self):
            """Request termination from any thread. Safe to call multiple
            times. Does not block — the actual termination is observed
            inside ``run()`` and reported via the ``cancelled`` signal."""
            _PIPELINE_CANCEL.set()
            _pipeline_kill_popens()

        def run(self):
            tp = self.cfg.get("transcribe_provider", "funasr")
            pp = self.cfg.get("polish_provider", "claude")
            np_ = self.cfg.get("meeting_notes_provider", "claude")
            try:
                _pipeline_check_cancelled()
                audio_path = self.audio_path
                raw_txt = audio_path.with_name(audio_path.stem + ".raw.txt")
                polish_path = audio_path.with_name(audio_path.stem + ".polish.txt")

                self.progress.emit(5)
                need_transcribe = not raw_txt.exists()
                need_polish = not polish_path.exists()

                def _on_pct(pct: int):
                    # Reserve 0-80 for transcribe stage so the bar feels even
                    self.progress.emit(min(80, max(5, int(pct * 0.8))))

                if self.transcribe_only:
                    # "仅语音转文字" = transcribe + polish, no notes generation.
                    # Outputs the polished text (.polish.txt) — raw FunASR
                    # output has no punctuation/segmentation, so the polish
                    # pass is what makes it actually readable.
                    if need_transcribe and need_polish:
                        raw, polished = transcribe_and_polish(
                            audio_path, tp, pp, self.cfg, self.mode, on_progress=_on_pct)
                        _pipeline_check_cancelled()
                        raw_txt.write_text(raw, encoding="utf-8")
                        polish_path.write_text(polished, encoding="utf-8")
                    elif need_transcribe:
                        self.log.emit(f"[校对] 检测到 {polish_path.name}，跳过校对")
                        raw = transcribe(audio_path, tp, self.cfg, on_progress=_on_pct)
                        _pipeline_check_cancelled()
                        raw_txt.write_text(raw, encoding="utf-8")
                    elif need_polish:
                        self.log.emit(f"[转写] 检测到 {raw_txt.name}，跳过转写")
                        raw = raw_txt.read_text(encoding="utf-8")
                        polished = polish_transcript(raw, pp, self.cfg, self.mode)
                        _pipeline_check_cancelled()
                        polish_path.write_text(polished, encoding="utf-8")
                    else:
                        self.log.emit(f"[转写] 检测到 {raw_txt.name}，跳过转写")
                        self.log.emit(f"[校对] 检测到 {polish_path.name}，跳过校对")
                    self.progress.emit(100)
                    self.done.emit(str(polish_path))
                    return

                if need_transcribe and need_polish:
                    raw, polished = transcribe_and_polish(
                        audio_path, tp, pp, self.cfg, self.mode, on_progress=_on_pct)
                    _pipeline_check_cancelled()
                    raw_txt.write_text(raw, encoding="utf-8")
                    polish_path.write_text(polished, encoding="utf-8")
                elif need_transcribe:
                    self.log.emit(f"[校对] 检测到 {polish_path.name}，跳过校对")
                    raw = transcribe(audio_path, tp, self.cfg, on_progress=_on_pct)
                    _pipeline_check_cancelled()
                    raw_txt.write_text(raw, encoding="utf-8")
                    polished = polish_path.read_text(encoding="utf-8")
                elif need_polish:
                    self.log.emit(f"[转写] 检测到 {raw_txt.name}，跳过转写")
                    raw = raw_txt.read_text(encoding="utf-8")
                    polished = polish_transcript(raw, pp, self.cfg, self.mode)
                    _pipeline_check_cancelled()
                    polish_path.write_text(polished, encoding="utf-8")
                else:
                    self.log.emit(f"[转写] 检测到 {raw_txt.name}，跳过转写")
                    self.log.emit(f"[校对] 检测到 {polish_path.name}，跳过校对")
                    polished = polish_path.read_text(encoding="utf-8")

                _pipeline_check_cancelled()
                self.progress.emit(85)
                self.log.emit("[纪要] 生成中...")
                notes = generate_notes(polished, np_, self.cfg, self.mode)
                _pipeline_check_cancelled()
                note_path = save_minutes(notes, audio_path, self.mode)
                added_hw = _auto_update_hotwords(polished, np_, self.cfg)
                if added_hw:
                    self.log.emit(f"[热词] 新增 {len(added_hw)} 个："
                                  f"{' '.join(added_hw)}（下次录音生效）")
                self.progress.emit(100)
                self.done.emit(str(note_path))
            except _PipelineCancelled:
                _log("PIPELINE", "cancelled by user")
                self.cancelled.emit()
            except SystemExit as e:
                # Downstream library code (some provider error paths) may
                # call sys.exit(); without this guard the QThread dies
                # without `failed.emit`, leaving the progress bar +
                # disabled buttons frozen forever. Mirror cmd_record's
                # behaviour and route the exit code into the failure
                # signal so the UI resets cleanly.
                if _PIPELINE_CANCEL.is_set():
                    # A SystemExit raised by a provider that was killed
                    # mid-call (e.g. Claude CLI receiving SIGKILL → non-zero
                    # returncode → sys.exit) is really a cancellation, not
                    # a failure. Report it as such.
                    _log("PIPELINE", f"cancelled mid-call (SystemExit {e.code})")
                    self.cancelled.emit()
                else:
                    _log("ERR", f"Qt pipeline SystemExit({e.code})")
                    self.failed.emit(f"SystemExit: {e.code}")
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                _log("ERR", f"Qt pipeline: {type(e).__name__}: {e}\n{tb}")
                self.failed.emit(f"{type(e).__name__}: {e}")

    # ── Recording lifecycle wrapper (QObject for signals) ─────────────────

    class _RecorderState(QObject):
        """Wraps MultiStreamRecorder + dOut / mute lifecycle in Qt signals."""
        status_changed = pyqtSignal(str)   # 'idle' | 'recording' | 'processing'
        elapsed_changed = pyqtSignal(int)  # seconds
        warning = pyqtSignal(str)
        # LiveCaptionEngine events, re-emitted onto the Qt thread (queued
        # connection — the engine calls .emit from its worker threads).
        caption_event = pyqtSignal(dict)

        def __init__(self, parent=None):
            super().__init__(parent)
            self._recorder: "MultiStreamRecorder | None" = None
            self._audio_path: "Path | None" = None
            self._plan: "AudioPlan | None" = None
            self._start_time = 0.0
            self._status = "idle"
            # Whether the NEXT session runs captions; set by
            # RecordingInterface before start_recording(). There is only one
            # recognition path, so this is a plain on/off.
            self.captions_enabled = False
            self._caption_engine: "LiveCaptionEngine | None" = None
            self._tick = QTimer(self)
            self._tick.setInterval(1000)
            self._tick.timeout.connect(self._on_tick)

        @property
        def status(self) -> str:
            return self._status

        @property
        def audio_path(self) -> "Path | None":
            return self._audio_path

        @property
        def plan(self) -> "AudioPlan | None":
            return self._plan

        def set_status(self, s: str) -> None:
            if s != self._status:
                self._status = s
                self.status_changed.emit(s)

        def _on_tick(self):
            if self._start_time:
                self.elapsed_changed.emit(int(time.time() - self._start_time))

        def start_recording(self, plan: "AudioPlan", audio_path: Path) -> bool:
            wanted = [n for n in (plan.sys_source_name, plan.mic_name) if n]
            role_labels: list[str] = []
            if plan.sys_source_name:
                role_labels.append("system")
            if plan.mic_name:
                role_labels.append("mic")
            if not wanted:
                self.warning.emit("没有可用的录音设备")
                return False
            recorder = MultiStreamRecorder(wanted, cfg["sample_rate"], role_labels=role_labels)
            recorder.on_warning = lambda code: self.warning.emit(code)
            engine = None
            if self.captions_enabled:
                engine = LiveCaptionEngine(cfg, self.caption_event.emit)
                engine.start()
                recorder.on_audio_chunk = engine.feed
            try:
                recorder.start()
            except Exception as e:
                _log("ERR", f"Qt recorder.start: {type(e).__name__}: {e}")
                self.warning.emit(f"录音设备启动失败: {e}")
                if engine:
                    engine.stop()
                return False
            self._caption_engine = engine
            self._recorder = recorder
            self._audio_path = audio_path
            self._plan = plan
            self._start_time = time.time()
            self._tick.start()
            self.set_status("recording")
            return True

        def stop_recording(self) -> "Path | None":
            if not self._recorder:
                return None
            try:
                self._recorder.stop()
            except Exception as e:
                _log("ERR", f"Qt recorder.stop: {type(e).__name__}: {e}")
            if self._caption_engine:
                try:
                    self._caption_engine.stop()
                except Exception as e:
                    _log("ERR", f"Qt caption stop: {type(e).__name__}: {e}")
                self._caption_engine = None
            self._tick.stop()
            saved_path = None
            if self._audio_path and self._recorder.save(self._audio_path):
                saved_path = self._audio_path
            self._recorder = None
            self._audio_path = None
            self._start_time = 0.0
            self.set_status("idle")
            return saved_path

    # ── Recording view ────────────────────────────────────────────────────

    class RecordingInterface(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("recordingInterface")
            self.state = _RecorderState(self)
            self._last_recorded: "Path | None" = None
            self._result_path: "Path | None" = None
            self._vol_device: "str | None" = None
            # True iff the slider-driven device-level mute is currently engaged
            # (i.e. user parked the slider at 0%). Used to log only on boundary
            # crossings, not on every dragged value-change event.
            self._slider_mute_active: bool = False
            # One-shot guard for the startup-mute default: set to True after the
            # first successful application so the 300 ms retry (and any future
            # accidental re-entry) becomes a no-op once the user is in control.
            self._startup_mute_applied: bool = False
            self._pipeline_thread: "QThread | None" = None
            self._pipeline_worker: "_PipelineWorker | None" = None
            # i18n: each callback re-applies its widget's text on lang switch.
            self._lang_callbacks: list = []
            self._current_status = "idle"
            # Live-caption view model: finalized rows + the in-flight partial
            # (and its opportunistic translation).
            self._caption_rows: list = []
            self._caption_partial = ""
            self._caption_partial_en = ""
            _lc_cfg = cfg.get("live_captions") or {}
            self._caption_merge_gap = max(0, int(
                _lc_cfg.get("merge_gap_ms", 2500)
            )) / 1000.0
            # Scroll-back retention for the caption panel: primarily a TIME
            # window (history_minutes), with history_max as a memory backstop
            # for pathological cadence. Older rows drop from the panel;
            # on-disk transcripts are untouched.
            self._caption_history_secs = max(
                0, int(_lc_cfg.get("history_minutes", 180))) * 60
            self._caption_row_cap = max(80, int(_lc_cfg.get("history_max", 12000)))
            # Debounce window for document re-renders. Bursts of translation /
            # refined events collapse into one render per window. A fixed
            # window is enough now that _CaptionDocRenderer patches only the
            # changed tail — cost no longer scales with retained history.
            self._caption_render_debounce_ms = max(
                0, int(_lc_cfg.get("render_debounce_ms", 120)))
            self._caption_render_pending = False
            self._caption_status_key: "str | None" = None
            self._build_ui()
            # Needs caption_view, so it is created after _build_ui().
            self._caption_renderer = _CaptionDocRenderer(self.caption_view)
            self._wire()
            self._sync_caption_enabled_to_state()
            self._render_captions()
            self._refresh_action_buttons()
            # Startup default: mute the active output device and park the
            # slider at 0%. Forces the user to deliberately drag the slider
            # up before any system audio plays through them — a conservative
            # default for a recording app. One-shot, gated by
            # `_startup_mute_applied`; the 300 ms retry covers the case
            # where the audio monitor hasn't resolved `_vol_device` yet.
            try:
                self._apply_startup_mute()
            except Exception as e:
                _log("ERR", f"Qt startup mute: {type(e).__name__}: {e}")
            QTimer.singleShot(300, self._apply_startup_mute)

        def _i18n(self, widget, key, attr="setText"):
            """Track a translatable label so `apply_language()` can re-set it."""
            def update():
                getattr(widget, attr)(_t(key))
            self._lang_callbacks.append(update)
            update()

        def apply_language(self):
            for cb in self._lang_callbacks:
                cb()
            # Re-render dynamic strings whose text depends on state.
            self._on_status_changed(self._current_status)
            if self._caption_status_key:
                self.caption_status.setText(_t(self._caption_status_key))
            self._render_captions()
            if self._last_recorded:
                self.chosen_label.setText(
                    f"{_t('rec.selected_prefix')}{self._last_recorded.name}")
            else:
                self.chosen_label.setText(_t("rec.no_file"))
            # Action button labels depend on (current language) × (artifact
            # exists on disk) — refresh after the static callbacks so the
            # "open X" override wins over the default "generate X" text.
            self._refresh_action_buttons()

        # ── UI build
        def _build_ui(self):
            root = QHBoxLayout(self)
            root.setContentsMargins(28, 28, 28, 28)
            root.setSpacing(24)

            # Left column — recording controls.
            #
            # Layout shape (same as the history view's detail pane):
            #   left (QFrame, "recordingCardLeft")   ← rounded background
            #     └── left_scroll (ScrollArea, transparent)
            #           └── left_inner (QWidget, transparent)  ← `lv`
            #
            # The scroll area is load-bearing, not cosmetic. This column's
            # content is mostly fixed-size (132 px mic disc, a TitleLabel
            # timer, five buttons) and grows by ~200 px once 停止任务 +
            # progress + log_view appear. When the total exceeded the card
            # height Qt resolved the deficit by shrinking children BELOW
            # their minimums — `mic_block`'s setFixedHeight(180) included —
            # so the timer painted outside its block and over the volume
            # row, and the hint landed on the mic disc. Verified offscreen
            # at 1100x560: overlapping before, clean after. Budgeting
            # pixels in comments had been tried three times; scrolling ends
            # the class of bug.
            left = QFrame(self)
            left_outer = QVBoxLayout(left)
            left_outer.setContentsMargins(0, 0, 0, 0)
            left_scroll = ScrollArea(left)
            left_scroll.setWidgetResizable(True)
            left_scroll.setFrameShape(QFrame.Shape.NoFrame)
            # Horizontal scrolling stays ENABLED on purpose: with it off, a
            # window too narrow for the three notes buttons clipped their
            # labels silently ("生成分享总结" → "生成分享"). A scrollbar that
            # only shows up in genuinely cramped windows beats truncated text.
            left_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            left_scroll.setStyleSheet(
                "QScrollArea { background: transparent; border: none; }"
                "QScrollArea > QWidget > QWidget { background: transparent; }"
            )
            left_scroll.viewport().setAutoFillBackground(False)
            left_inner = QWidget()
            left_inner.setObjectName("recordingLeftInner")
            left_inner.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            left_inner.setStyleSheet(
                "#recordingLeftInner { background: transparent; }")
            left_outer.addWidget(left_scroll)
            left_scroll.setWidget(left_inner)
            lv = QVBoxLayout(left_inner)
            lv.setContentsMargins(0, 0, 0, 0)
            lv.setSpacing(16)

            # Title + subtitle are state-dependent (idle / recording); they're
            # set by `_on_status_changed` and re-applied by `apply_language`.
            self.title_label = TitleLabel(_t("rec.title.idle"), self)
            self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            self.subtitle_label = BodyLabel(_t("rec.subtitle.idle"), self)
            self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Big circular mic button.  Idle = blue; recording = red.
            # Visual state is toggled by `_apply_rec_btn_style(status)` from
            # `_on_status_changed`. We use a plain QPushButton (not a Fluent
            # PushButton) so we can override the entire visual via QSS —
            # PyQt-Fluent-Widgets' own styling doesn't expose border-radius
            # / state-coloured backgrounds in a portable way.
            self.rec_btn = QPushButton(self)
            self.rec_btn.setFixedSize(132, 132)
            self.rec_btn.setIcon(FluentIcon.MICROPHONE.icon())
            # Icon deliberately smaller than the 132 px disc: at 56 px it
            # crowded the circle's edge.
            self.rec_btn.setIconSize(QSize(40, 40))
            self.rec_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._apply_rec_btn_style("idle")

            self.timer_label = TitleLabel("00:00:00", self)
            self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Mic + timer block — packed into a single fixed-height
            # container so they're always stacked correctly. Putting the
            # timer directly into `lv` was fragile: when 停止任务 /
            # progress bar / log view become visible, `lv`'s total
            # content overflows the column height; Qt then proportionally
            # violates every child's min/fixed height, including the mic
            # button's `setFixedSize(132, 132)`, and the timer ends up
            # painted on top of the mic disc. Wrapping them here means
            # any squish from `lv` is applied to the BLOCK as a whole,
            # while the inner QVBoxLayout preserves the mic-above-timer
            # ordering with explicit spacing. Heights: mic 132 + 8
            # spacing + timer ~36 → 176; rounded up to 180 for breathing
            # room around the TitleLabel.
            mic_block = QWidget(self)
            # Minimum, not fixed: the scroll area guarantees the space now, so
            # the block only needs to refuse to shrink below its contents.
            mic_block.setMinimumHeight(180)
            mic_block_v = QVBoxLayout(mic_block)
            mic_block_v.setContentsMargins(0, 0, 0, 0)
            mic_block_v.setSpacing(8)
            mic_row = QHBoxLayout()
            mic_row.setContentsMargins(0, 0, 0, 0)
            mic_row.addStretch(1)
            mic_row.addWidget(self.rec_btn)
            mic_row.addStretch(1)
            mic_block_v.addLayout(mic_row)
            mic_block_v.addWidget(
                self.timer_label,
                alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            )

            # No mic-selector row — the app always captures system audio
            # (BlackHole) + the resolver-chosen mic (external > built-in).
            # The active devices appear in the post-recording log instead.

            # Plain-text hint just above the volume row. Reason: during
            # recording dOut is the Multi-Output Device, which is an
            # aggregate without master volume — so macOS disables F11/F12
            # / Touch Bar / menu-bar slider with the 🚫 cursor. The user
            # CAN still adjust playback level via this in-app slider,
            # which writes directly to the physical sub-device. We
            # surface the affordance here so users don't think the app
            # is broken when their hardware keys stop responding.
            self._vol_hint = BodyLabel("", self)
            self._i18n(self._vol_hint, "rec.volume_hint")
            self._vol_hint.setWordWrap(True)
            self._vol_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # Slightly eye-catching but consistent with the app's accent
            # palette: the same Fluent blue used by the idle mic button
            # (#0a84ff hover variant #0066d6, which reads better as text
            # on a white background) + a touch of weight to lift it off
            # the surrounding plain body text without screaming.
            self._vol_hint.setStyleSheet(
                "BodyLabel { color: #0066d6; font-weight: 500; }"
            )

            # Volume row
            vol_row = QHBoxLayout()
            vol_row.addStretch(1)
            self._vol_label = BodyLabel("", self)
            self._i18n(self._vol_label, "rec.volume")
            vol_row.addWidget(self._vol_label)
            self.vol_slider = Slider(Qt.Orientation.Horizontal, self)
            self.vol_slider.setRange(0, 100)
            self.vol_slider.setValue(50)
            self.vol_slider.setMinimumWidth(220)
            vol_row.addWidget(self.vol_slider)
            # Initialise the percentage label from the slider's value so the
            # handle position and the number always agree at first paint —
            # even before `_sync_vol_slider` has had a chance to query the
            # real device volume. (Previously hardcoded "--%", which made the
            # handle at 50 sit beside a "--%" label on startup.)
            self.vol_pct = CaptionLabel(f"{self.vol_slider.value()}%", self)
            self.vol_pct.setMinimumWidth(40)
            vol_row.addWidget(self.vol_pct)
            vol_row.addStretch(1)

            # Action buttons — each toggles between "generate" and "open" based
            # on whether the corresponding output artifact already exists on disk.
            # When the artifact exists, the button switches to the accent-blue
            # style (mirroring the record button's idle colour) so it reads as
            # a primary "open the result" affordance. See _refresh_action_buttons()
            # for the text + style toggle and _on_*_clicked() for the wiring.
            # Text/style are managed entirely by _refresh_action_buttons() — no
            # _i18n callback registration, so a language flip simply calls
            # _refresh_action_buttons() instead of fighting with a static label.
            self.transcribe_btn = PushButton("", self)
            self.notes_btn = PushButton("", self)
            self.interview_btn = PushButton("", self)
            self.sharing_btn = PushButton("", self)

            # Import button: lets the user bring in an external audio file
            # (iPhone .m4a, .mp3 download, Zoom .m4a, etc.). Opens a file
            # dialog, transcodes via ffmpeg to 16k mono PCM WAV in
            # RECORDINGS_DIR, then selects the imported recording so the
            # user can immediately run transcribe / notes on it.
            self.import_btn = PushButton("", self)
            self._i18n(self.import_btn, "rec.btn.import")
            # Match the visual treatment of the four "generate" buttons.
            # `_refresh_action_buttons` only restyles the transcribe / notes
            # / interview / sharing buttons, so the import button gets a
            # one-shot light-gray fill here at construction time and keeps
            # it for the widget's lifetime.
            _apply_open_btn_style(self.import_btn, is_open=False)

            # Row 1: import + transcribe (entry-point actions)
            actions_row1 = QHBoxLayout()
            actions_row1.addStretch(1)
            actions_row1.addWidget(self.import_btn)
            actions_row1.addWidget(self.transcribe_btn)
            actions_row1.addStretch(1)

            # Row 2: per-mode notes generation (meeting / interview / sharing)
            actions_row2 = QHBoxLayout()
            actions_row2.addStretch(1)
            actions_row2.addWidget(self.notes_btn)
            actions_row2.addWidget(self.interview_btn)
            actions_row2.addWidget(self.sharing_btn)
            actions_row2.addStretch(1)

            # Chosen-recording indicator. The standalone "选择已有音频文件"
            # button was removed — meetings are picked via the right-side
            # history list (click or right-click) or just-recorded inline.
            self.chosen_label = CaptionLabel(_t("rec.no_file"), self)
            choose_row = QHBoxLayout()
            choose_row.addStretch(1)
            choose_row.addWidget(self.chosen_label)
            choose_row.addStretch(1)

            # Progress + log + cancel
            self.progress_bar = ProgressBar(self)
            self.progress_bar.setVisible(False)
            self.cancel_btn = PushButton("", self)
            self._i18n(self.cancel_btn, "rec.btn.cancel")
            self.cancel_btn.setVisible(False)
            self.log_view = TextEdit(self)
            self.log_view.setReadOnly(True)
            self.log_view.setMaximumHeight(160)
            # Let log_view shrink all the way to 0 when vertical space
            # is tight (e.g. on smaller windows where mic_block + the
            # action rows + progress + cancel already fill the column).
            # Without this, log_view's default minimumHeight forces Qt
            # to squish the mic_block above, and the timer ends up
            # overlapping the mic disc.
            self.log_view.setMinimumHeight(0)
            self.log_view.setVisible(False)

            lv.addWidget(self.title_label)
            lv.addWidget(self.subtitle_label)
            # No leading spacer before mic_block — under squish pressure
            # (cancel_btn + progress + log_view all visible) every pixel
            # of explicit addSpacing competes with the mic_block's
            # fixed 180 px. Letting the mic ride right up under the
            # subtitle moves the circle visibly higher in the card.
            lv.addWidget(mic_block)
            # Bigger gap below mic_block so the volume hint sits clearly
            # apart from the timer (which lives at the bottom of
            # mic_block). Without this the hint visually crowds the
            # "00:00:00" digits as soon as the log_view expands.
            lv.addSpacing(20)
            lv.addWidget(self._vol_hint)
            lv.addSpacing(6)
            lv.addLayout(vol_row)
            lv.addSpacing(8)
            lv.addLayout(actions_row1)
            lv.addLayout(actions_row2)
            lv.addLayout(choose_row)
            lv.addWidget(self.progress_bar)
            lv.addWidget(self.cancel_btn)
            lv.addWidget(self.log_view)
            lv.addStretch(1)

            # Right column — live bilingual captions, full height.
            right_col = QWidget(self)
            right_col.setMinimumWidth(420)
            rc = QVBoxLayout(right_col)
            rc.setContentsMargins(0, 0, 0, 0)
            rc.setSpacing(16)

            # ── Caption card
            captions = QFrame(self)
            cv = QVBoxLayout(captions)
            cv.setContentsMargins(0, 0, 0, 0)
            cv.setSpacing(10)

            cap_header = QHBoxLayout()
            self._captions_title = SubtitleLabel("", self)
            self._i18n(self._captions_title, "rec.captions.title")
            cap_header.addWidget(self._captions_title)
            cap_header.addStretch(1)
            self.caption_switch = SwitchButton(self)
            lc_cfg = _deep_merge(
                DEFAULT_CONFIG["live_captions"], cfg.get("live_captions", {}))
            self.caption_switch.setChecked(bool(lc_cfg.get("enabled", True)))
            cap_header.addWidget(self.caption_switch)
            cv.addLayout(cap_header)


            self.caption_status = CaptionLabel("", self)
            self.caption_status.setWordWrap(True)
            cv.addWidget(self.caption_status)

            self.caption_view = TextBrowser(self)
            self.caption_view.setMinimumHeight(140)
            cv.addWidget(self.caption_view, stretch=1)

            # In-flight ("still being spoken") line: its own widget, NOT part
            # of the caption document — see _render_caption_partial.
            self.caption_partial_label = CaptionLabel("", self)
            self.caption_partial_label.setWordWrap(True)
            self.caption_partial_label.setTextFormat(Qt.TextFormat.RichText)
            self.caption_partial_label.setVisible(False)
            cv.addWidget(self.caption_partial_label)


            # Wrap each region in a soft rounded card so the three panels
            # read as visually separate surfaces.
            _style_as_card(left, padding=20, name="recordingCardLeft")
            _style_as_card(captions, padding=16, name="recordingCardCaptions")

            # Left column is the record-controls card alone; the caption
            # panel owns the whole right column. The meeting-history card
            # that used to sit below the controls is gone — the left nav
            # already has a 历史 view with the same list and the same
            # pipeline buttons, and duplicating it here was what squeezed
            # this column until the timer painted over the mic disc.
            left_col = QWidget(self)
            lcol = QVBoxLayout(left_col)
            lcol.setContentsMargins(0, 0, 0, 0)
            lcol.setSpacing(16)
            lcol.addWidget(left, stretch=1)

            rc.addWidget(captions, stretch=1)

            root.addWidget(left_col, stretch=2)
            root.addWidget(right_col, stretch=3)

        # ── Wiring
        def _wire(self):
            self.rec_btn.clicked.connect(self._on_record_clicked)
            self.import_btn.clicked.connect(self._on_import_clicked)
            self.transcribe_btn.clicked.connect(self._on_transcribe_clicked)
            self.notes_btn.clicked.connect(self._on_meeting_clicked)
            self.interview_btn.clicked.connect(self._on_interview_clicked)
            self.sharing_btn.clicked.connect(self._on_sharing_clicked)
            self.cancel_btn.clicked.connect(self._on_cancel_pipeline)
            self.vol_slider.valueChanged.connect(self._on_vol_changed)

            self.state.status_changed.connect(self._on_status_changed)
            self.state.elapsed_changed.connect(self._on_elapsed_changed)
            self.state.warning.connect(self._on_warning)
            self.state.caption_event.connect(self._on_caption_event)
            self.caption_switch.checkedChanged.connect(self._on_caption_prefs_changed)

            _get_audio_monitor().on_recording_plan_change = self._on_plan_change

        # ── Live captions
        def _sync_caption_enabled_to_state(self):
            """Push the panel's switch into _RecorderState for the NEXT session."""
            self.state.captions_enabled = self.caption_switch.isChecked()

        def _on_caption_prefs_changed(self, *_a):
            self._sync_caption_enabled_to_state()
            self._render_captions()
            lc = cfg.setdefault("live_captions", {})
            lc["enabled"] = self.caption_switch.isChecked()
            try:
                # In-place JSONC patch; silently skipped if the block is
                # absent from config.jsonc (prefs then live for this run only).
                _save_config_preserving_comments(cfg)
            except Exception as e:
                _log("ERR", f"persist caption prefs: {type(e).__name__}: {e}")

        def _on_caption_event(self, ev: dict):
            t = ev.get("type")
            if t == "partial":
                self._caption_partial = ev.get("text", "")
                self._render_caption_partial()
                return          # document untouched: no re-render needed
            elif t == "partial_translation":
                self._caption_partial_en = ev.get("text", "")
                self._render_caption_partial()
                return
            elif t == "final":
                self._caption_partial = ""
                self._caption_partial_en = ""
                self._render_caption_partial()
                self._caption_rows.append(
                    {"id": ev.get("id"), "src": ev.get("text", ""),
                     "dst": "", "t": time.time(), "role": ev.get("role"),
                     "speaker": None, "side": None})
                _prune_caption_rows(
                    self._caption_rows, time.time(),
                    self._caption_history_secs, self._caption_row_cap)
            elif t == "refined":
                for row in reversed(self._caption_rows):
                    if row["id"] == ev.get("id"):
                        row["src"] = ev.get("text", row["src"])
                        # Old translation no longer matches the refined
                        # source — show "…" until the re-translation lands.
                        row["dst"] = ""
                        break
            elif t == "translation":
                for row in reversed(self._caption_rows):
                    if row["id"] == ev.get("id"):
                        row["dst"] = ev.get("text", "")
                        break
            elif t == "speaker":
                # Voice print beats the channel-energy guess: it arrives
                # ~60 ms later and can tell two remote people apart, which
                # the single system-audio stream cannot.
                for row in reversed(self._caption_rows):
                    if row["id"] == ev.get("id"):
                        row["speaker"] = ev.get("speaker")
                        row["side"] = ev.get("side")
                        break
            elif t == "status":
                self._caption_status_key = (
                    f"rec.captions.status.{ev.get('state', 'stopped')}")
                self.caption_status.setText(_t(self._caption_status_key))
                return
            elif t == "error":
                self._caption_status_key = None
                self.caption_status.setText(
                    _t("rec.captions.error_prefix") + str(ev.get("message", "")))
                return
            else:
                return
            self._schedule_caption_render()

        def _schedule_caption_render(self):
            """Debounce setHtml() so caption event bursts don't flicker the
            scrollbar. All event-driven renders funnel through here; direct
            renders (init / language toggle) still call _render_captions()."""
            delay = self._caption_render_debounce_ms
            if delay <= 0:
                self._render_captions()
                return
            if self._caption_render_pending:
                return
            self._caption_render_pending = True
            QTimer.singleShot(delay, self._do_caption_render)

        def _do_caption_render(self):
            self._caption_render_pending = False
            self._render_captions()

        def _render_captions(self):
            groups_html = []
            # Display-layer paragraph merging: rapid-fire short finals show
            # as one paragraph (data rows stay 1:1 with engine segments).
            # One entry per group, each exactly two paragraphs — the unit
            # `_CaptionDocRenderer` diffs and patches, so a new line touches
            # only the tail no matter how long the retained history is.
            for group in _group_caption_rows(
                    self._caption_rows, self._caption_merge_gap):
                src = _join_caption_texts([r["src"] for r in group])
                dsts = [r["dst"] for r in group]
                dst = _join_caption_texts([d for d in dsts if d])
                if dst and any(not d for d in dsts):
                    dst += " …"  # part of the paragraph still translating
                # Speaker tag: mic = me, system audio = the other side. None
                # when the engine can't attribute (single active source).
                head = group[0]
                role, speaker = head.get("role"), head.get("speaker")
                label, colour = "", "#e0782c"
                if speaker is not None:
                    # Voice print = identity. Every speaker gets a number,
                    # including several people sharing one microphone; the
                    # channel only tints the tag (blue = our side).
                    label = _t("rec.captions.speaker_n").format(n=speaker)
                    if head.get("side") == "mic":
                        colour = "#0a84ff"
                elif role:
                    # Still only the channel-energy guess (or speaker id off).
                    key = f"rec.captions.role_{role}"
                    # _t echoes unknown keys; a raw device name is a better
                    # tag than "rec.captions.role_BlackHole 2ch".
                    label = _t(key)
                    label = role if label == key else label
                    colour = "#0a84ff" if role == "mic" else "#e0782c"
                tag = ""
                if label:
                    tag = (f'<span style="color:{colour}; font-weight:600;">'
                           f'[{html.escape(label)}]</span> ')
                groups_html.append(
                    f'<p style="margin:6px 0 0 0; color:#1f1f1f;">'
                    f'{tag}{html.escape(src)}</p>'
                    f'<p style="margin:1px 0 0 0; color:#0066d6;">'
                    f'{html.escape(dst) or "…"}</p>'
                )
            hint = (
                _t("rec.captions.idle_hint")
                if self.caption_switch.isChecked()
                else _t("rec.captions.disabled_hint"))
            mode = self._caption_renderer.render(
                groups_html, f'<p style="color:#8a8a8a;">{hint}</p>')
            # A "full" path means the incremental bookkeeping bailed out — fine
            # occasionally (first paint, language toggle, history eviction),
            # but a steady stream of them means every caption event is paying
            # a whole-document re-layout again.
            if mode == "full" and len(groups_html) > 200:
                _log("CAPTION", f"caption pane full re-render "
                                f"({len(groups_html)} groups)")

        def _render_caption_partial(self):
            """The in-flight line lives OUTSIDE the document.

            It refreshes ~2×/s (partial_interval_ms) and used to drag the
            whole pane through a re-render each time — ~70 % of all caption
            events. As its own widget it costs one setText and the document
            stays untouched between finalized lines."""
            if not self._caption_partial:
                self.caption_partial_label.setText("")
                self.caption_partial_label.setVisible(False)
                return
            body = (f'<span style="color:#8a8a8a; font-style:italic;">'
                    f'{html.escape(self._caption_partial)}</span>')
            if self._caption_partial_en:
                body += (f'<br/><span style="color:#7fa8d9; font-style:italic;">'
                         f'{html.escape(self._caption_partial_en)}</span>')
            self.caption_partial_label.setText(body)
            self.caption_partial_label.setVisible(True)



        # ── Volume slider
        def _on_vol_changed(self, val: int):
            self.vol_pct.setText(f"{val}%")
            if self._vol_device:
                try:
                    set_device_volume(self._vol_device, val / 100.0)
                except Exception as e:
                    _log("ERR", f"Qt set_volume: {type(e).__name__}: {e}")
                # Mirror macOS F11/F12 behaviour: vmvc=0.0 on Apple built-in
                # audio is "minimum", not silence. Toggle the device-level mute
                # in lockstep with the slider so 0% is truly silent and any
                # non-zero value clears the mute. Bypasses _active_mutes on
                # purpose — this is an explicit user action, not part of the
                # recording-lifecycle reconcile state machine. We log only on
                # boundary crossings (0 ↔ non-0) so dragging through the range
                # doesn't flood the log.
                desired = (val == 0)
                if desired != self._slider_mute_active:
                    try:
                        # Record the user's intent BEFORE writing mute so any
                        # concurrent reconcile (AudioDeviceMonitor recording
                        # branch wakes on hotplug / safety timeout) already
                        # sees the slider intent and won't undo our mute.
                        with _mutes_lock:
                            if desired:
                                _slider_intent_muted.add(self._vol_device)
                            else:
                                _slider_intent_muted.discard(self._vol_device)
                        ok = _ca_set_device_mute(self._vol_device, desired)
                        _log(
                            "MUTE",
                            f"slider {'muted' if desired else 'unmuted'} "
                            f"device={self._vol_device!r} val={val}% ok={ok}",
                        )
                        self._slider_mute_active = desired
                    except Exception as e:
                        _log("ERR", f"Qt slider mute: {type(e).__name__}: {e}")

        def _sync_vol_slider(self):
            if not self._vol_device:
                try:
                    self._vol_device = _get_current_output_device()
                except Exception:
                    return
            if not self._vol_device:
                return
            v = get_device_volume(self._vol_device)
            if v is None:
                return
            # Defensive clamp — some macOS drivers report > 1.0 when software
            # boost is engaged. Without this, the slider clamps to 100 but
            # the label would still read "150%" etc.
            pct = max(0, min(100, int(v * 100)))
            self.vol_slider.blockSignals(True)
            try:
                self.vol_slider.setValue(pct)
            finally:
                self.vol_slider.blockSignals(False)
            self.vol_pct.setText(f"{pct}%")

        def _apply_startup_mute(self):
            """Force the slider to 0% and engage device-level mute on the
            active output device. One-shot, gated by `_startup_mute_applied`
            so the 300 ms retry / any future re-entry is a no-op once the
            user has taken control.

            Writes the same property pair the regular slider-→-0 path writes
            (vmvc=0 + kAudioDevicePropertyMute=True) and registers the
            device in `_slider_intent_muted` so the recording-lifecycle
            reconcile honours the mute from the very first tick.
            """
            if self._startup_mute_applied:
                return
            if not self._vol_device:
                try:
                    self._vol_device = _get_current_output_device()
                except Exception:
                    return
            if not self._vol_device:
                return
            try:
                set_device_volume(self._vol_device, 0.0)
            except Exception as e:
                _log("ERR", f"Qt startup mute (vol): {type(e).__name__}: {e}")
            try:
                with _mutes_lock:
                    _slider_intent_muted.add(self._vol_device)
                ok = _ca_set_device_mute(self._vol_device, True)
                _log("MUTE", f"startup mute device={self._vol_device!r} ok={ok}")
            except Exception as e:
                _log("ERR", f"Qt startup mute (mute): {type(e).__name__}: {e}")
            self._slider_mute_active = True
            self.vol_slider.blockSignals(True)
            try:
                self.vol_slider.setValue(0)
            finally:
                self.vol_slider.blockSignals(False)
            self.vol_pct.setText("0%")
            self._startup_mute_applied = True

        # ── Recording lifecycle
        def _on_record_clicked(self):
            if self.state.status == "idle":
                self._start_recording()
            elif self.state.status == "recording":
                self._stop_recording()

        def _start_recording(self):
            try:
                plan = resolve_audio_devices(query_fresh=True)
            except Exception as e:
                self._on_warning(f"无法枚举设备: {e}")
                return

            # Mic / system-audio source are always picked by the resolver
            # (external > built-in, BlackHole for system audio). No UI
            # override — the recording is system audio + the resolver-chosen
            # microphone.

            _recording_did_switch.clear()
            _log("AUDIO", "session gate cleared (did_switch=False at start-of-lifecycle)")
            _log_device_raw_dump(reason="recording-start:qt-ui")

            prev_dout = _get_current_output_device()
            if plan.multi_output_name and prev_dout != plan.multi_output_name:
                switch_output(plan.multi_output_name)
                _recording_did_switch.set()
                _log("AUDIO", f"start switch: from={prev_dout!r} to={plan.multi_output_name!r} performed=True")
            else:
                _log("AUDIO", f"start switch: from={prev_dout!r} to={plan.multi_output_name!r} performed=False")

            _reconcile_recording_mutes(plan)

            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_path = RECORDINGS_DIR / f"{ts}.wav"

            self._vol_device = plan.restore_output_name
            # Fresh caption session: clear the transcript pane and hand the
            # panel's current switch+mode to the recorder state.
            self._caption_rows = []
            self._caption_partial = ""
            self._caption_partial_en = ""
            # The renderer's committed-HTML bookkeeping must forget the old
            # session, or the next render would diff against a document that
            # is about to be replaced.
            self._caption_renderer.reset()
            self._render_caption_partial()
            self._sync_caption_enabled_to_state()
            self._render_captions()
            ok = self.state.start_recording(plan, audio_path)
            if ok:
                QTimer.singleShot(300, self._sync_vol_slider)
            else:
                # Recording failed to start — undo the dOut switch we just did.
                try:
                    _restore_all_recording_mutes()
                except Exception:
                    pass

        def _stop_recording(self):
            try:
                _restore_all_recording_mutes()
            except Exception as e:
                _log("ERR", f"Qt stop restore_mutes: {type(e).__name__}: {e}")
            # If the user parked the slider at 0% during recording we will
            # have flipped the device-level mute on; clear it on stop so
            # F11/F12 / menu-bar slider work normally once recording ends.
            # Also drop the module-level slider-intent record so the next
            # recording's reconcile starts from a clean slate.
            if self._vol_device:
                try:
                    with _mutes_lock:
                        _slider_intent_muted.discard(self._vol_device)
                    ok = _ca_set_device_mute(self._vol_device, False)
                    _log(
                        "MUTE",
                        f"slider unmute at stop device={self._vol_device!r} "
                        f"was_muted={self._slider_mute_active} ok={ok}",
                    )
                except Exception as e:
                    _log("ERR", f"Qt stop slider unmute: {type(e).__name__}: {e}")
                self._slider_mute_active = False
            try:
                _restore_output_if_needed(
                    resolve_audio_devices(query_fresh=True),
                    reason="post-recording",
                )
            except Exception as e:
                _log("ERR", f"Qt stop output restore: {type(e).__name__}: {e}")

            saved = self.state.stop_recording()
            if saved:
                self._last_recorded = saved
                self.chosen_label.setText(f"{_t('rec.selected_prefix')}{saved.name}")
                self._result_path = None
            self._refresh_action_buttons()
            QTimer.singleShot(200, self._sync_vol_slider)

        def _on_status_changed(self, s: str):
            self._current_status = s
            # Caption prefs are session-scoped: lock the switch while a
            # recording (or pipeline) is active.
            caption_idle = s == "idle"
            self.caption_switch.setEnabled(caption_idle)
            if s == "idle":
                self._apply_rec_btn_style("idle")
                self.title_label.setText(_t("rec.title.idle"))
                self.subtitle_label.setText(_t("rec.subtitle.idle"))
                self.timer_label.setText("00:00:00")
                for b in (self.transcribe_btn, self.notes_btn,
                          self.interview_btn):
                    b.setEnabled(True)
            elif s == "recording":
                self._apply_rec_btn_style("recording")
                self.title_label.setText(_t("rec.title.recording"))
                self.subtitle_label.setText(_t("rec.subtitle.recording"))
                for b in (self.transcribe_btn, self.notes_btn,
                          self.interview_btn):
                    b.setEnabled(False)
            elif s == "processing":
                pass  # buttons handled by pipeline-start

        def _apply_rec_btn_style(self, status: str) -> None:
            """Repaint the circular mic button for the current lifecycle
            state. idle = blue; recording = red. Pressed/hover variants are
            slightly darker. The icon (white mic) stays the same — only the
            disk colour swaps."""
            if status == "recording":
                bg, hover, pressed = "#ef4444", "#dc2626", "#b91c1c"
            else:
                bg, hover, pressed = "#0a84ff", "#0066d6", "#0050a8"
            self.rec_btn.setStyleSheet(
                "QPushButton {"
                "  border: none;"
                "  border-radius: 66px;"
                f"  background-color: {bg};"
                "}"
                "QPushButton:hover {"
                f"  background-color: {hover};"
                "}"
                "QPushButton:pressed {"
                f"  background-color: {pressed};"
                "}"
            )

        def _on_elapsed_changed(self, secs: int):
            h, rem = divmod(secs, 3600)
            m, s = divmod(rem, 60)
            self.timer_label.setText(f"{h:02d}:{m:02d}:{s:02d}")

        def _on_warning(self, msg: str):
            self.log_view.setVisible(True)
            self._ui_log(_t("pipe.warn.prefix", msg=msg))

        # AudioDeviceMonitor → vol_device rebind on mid-recording hotplug.
        # Runs on the monitor thread; schedule UI work via QTimer.singleShot.
        def _on_plan_change(self, plan: "AudioPlan"):
            """Called from the AudioDeviceMonitor thread when the resolver's
            active output device changes (idle OR mid-recording hotplug).
            Marshals to the GUI thread for the actual slider transfer.
            """
            new_dev = plan.restore_output_name
            if not new_dev or new_dev == self._vol_device:
                return
            QTimer.singleShot(
                0, lambda nd=new_dev: self._transfer_slider_to_device(nd))

        def _transfer_slider_to_device(self, new_device: str):
            """Re-target the slider's volume + mute intent at `new_device`.

            Runs on the GUI thread. Splits cleanup of the old device based
            on lifecycle:

              • Idle: unmute the old device so the user isn't trapped in
                silence when they switch back (no recording-state
                reconcile to fix it for them).

              • Recording: leave the old device muted (reconcile wants
                inactive Multi-Output subs muted anyway), but register it
                in `_active_mutes` with original=False so
                `_restore_all_recording_mutes` brings it back at stop.
                Reconcile's `current == desired` skip would otherwise leave
                it untracked and permanently muted post-stop.

            On the new device: write the current slider value's volume +
            mute state, and add to `_slider_intent_muted` if at 0% so any
            subsequent reconcile honours the user's intent.
            """
            if not new_device or new_device == self._vol_device:
                return
            old_device = self._vol_device
            val = self.vol_slider.value()
            desired_mute = (val == 0)
            is_recording = _recording_active.is_set()

            with _mutes_lock:
                if old_device:
                    _slider_intent_muted.discard(old_device)
                    if is_recording and old_device not in _active_mutes:
                        _active_mutes[old_device] = False
                        _persist_mutes()
                self._vol_device = new_device
                if desired_mute:
                    _slider_intent_muted.add(new_device)
                else:
                    _slider_intent_muted.discard(new_device)
            self._slider_mute_active = desired_mute

            try:
                set_device_volume(new_device, val / 100.0)
            except Exception as e:
                _log("ERR", f"transfer set_vol new: {type(e).__name__}: {e}")
            try:
                ok = _ca_set_device_mute(new_device, desired_mute)
                _log(
                    "MUTE",
                    f"slider transfer from={old_device!r} to={new_device!r} "
                    f"val={val}% mute={desired_mute} ok={ok} "
                    f"recording={is_recording}",
                )
            except Exception as e:
                _log("ERR", f"transfer mute new: {type(e).__name__}: {e}")

            if old_device and not is_recording:
                try:
                    ok = _ca_set_device_mute(old_device, False)
                    _log("MUTE",
                         f"transfer: idle unmute old={old_device!r} ok={ok}")
                except Exception as e:
                    _log("ERR",
                         f"transfer idle unmute: {type(e).__name__}: {e}")

        # ── Pipeline
        def _start_pipeline(self, transcribe_only: bool = False,
                            mode: "str | None" = None):
            target = self._last_recorded
            if not target or not Path(target).exists():
                self._on_warning(_t("pipe.warn.no_wav"))
                return
            chosen_mode = mode or getattr(args, "mode", None) or cfg.get("mode", "meeting")

            self.log_view.setVisible(True)
            self.log_view.clear()
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.cancel_btn.setVisible(True)
            for b in (self.rec_btn, self.transcribe_btn, self.notes_btn,
                      self.interview_btn):
                b.setEnabled(False)
            self.state.set_status("processing")

            thread = QThread(self)
            worker = _PipelineWorker(target, chosen_mode, cfg,
                                     transcribe_only=transcribe_only)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.progress.connect(self.progress_bar.setValue)
            worker.log.connect(self._ui_log)
            worker.done.connect(self._on_pipeline_done)
            worker.failed.connect(self._on_pipeline_failed)
            worker.cancelled.connect(self._on_pipeline_cancelled)
            worker.done.connect(thread.quit)
            worker.failed.connect(thread.quit)
            worker.cancelled.connect(thread.quit)
            thread.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            self._pipeline_worker = worker
            self._pipeline_thread = thread
            thread.start()

        def _on_pipeline_done(self, path_str: str):
            self._reset_after_pipeline()
            self._result_path = Path(path_str)
            self._refresh_action_buttons()
            # The run log is transient: every line is already in the daily
            # log file, and the action button flipping to "Open …" is the
            # confirmation that matters. Leaving the pane parked on screen
            # after a successful run is just clutter.
            self._hide_pipeline_log()
            InfoBar.success(
                title=_t("pipe.info.done_title"),
                content=Path(path_str).name,
                isClosable=True, position=InfoBarPosition.TOP,
                duration=4000, parent=self,
            )

        def _on_pipeline_failed(self, msg: str):
            self._reset_after_pipeline()
            # Failures KEEP the log: it is the only place the traceback and
            # the step it died on are visible without opening the log file.
            self._ui_log(_t("pipe.log.failed", err=msg))

        def _on_pipeline_cancelled(self):
            self._reset_after_pipeline()
            self._hide_pipeline_log()

        def _hide_pipeline_log(self):
            self.log_view.clear()
            self.log_view.setVisible(False)

        def _reset_after_pipeline(self):
            self.progress_bar.setVisible(False)
            self.cancel_btn.setVisible(False)
            # Re-arm the cancel button in case `_on_cancel_pipeline` had
            # disabled it during the wrap-up window.
            self.cancel_btn.setEnabled(True)
            for b in (self.rec_btn, self.transcribe_btn, self.notes_btn,
                      self.interview_btn, self.sharing_btn):
                b.setEnabled(True)
            self.state.set_status("idle")

        def _on_cancel_pipeline(self):
            # Truly terminate the background pipeline: set the cancel event
            # and SIGKILL any registered subprocess (claude -p …) so the
            # worker thread escapes its blocking call and emits the
            # `cancelled` signal. We keep the cancel button visible but
            # disabled during the wrap-up window so the user sees that
            # termination is in progress; full UI reset happens in
            # `_on_pipeline_cancelled` when the worker confirms.
            worker = self._pipeline_worker
            if worker is None:
                # Worker may have already finished between the click and
                # this handler — fall back to a plain UI reset.
                self._reset_after_pipeline()
                return
            self._ui_log(_t("pipe.log.cancelling"))
            self.cancel_btn.setEnabled(False)
            worker.cancel()

        def _on_import_clicked(self):
            """Pick an external audio file (iPhone .m4a, .mp3, etc.) and
            transcode it into RECORDINGS_DIR. On success the imported file
            becomes the currently-selected recording so the user can run
            transcribe / notes / interview on it immediately.

            ffmpeg is invoked synchronously — typical iPhone Voice Memos
            transcode in <2 s on Apple Silicon, so blocking the UI thread
            with a busy cursor is acceptable. A pipeline already running
            disables the button via `_refresh_action_buttons` semantics,
            but we double-check here to be safe.
            """
            if self._pipeline_thread is not None:
                self._on_warning(_t("ctx.delete_blocked_msg"))
                return
            path_str, _filter = QFileDialog.getOpenFileName(
                self,
                _t("rec.import.dialog_title"),
                "",
                _t("rec.import.filter"),
            )
            if not path_str:
                return
            src = Path(path_str)
            # ffmpeg sanity check up-front so we show the install hint
            # before the user waits on a long transcode that's going to
            # fail anyway.
            if src.suffix.lower() != ".wav" and _ffmpeg_path() is None:
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Icon.Warning)
                box.setWindowTitle(_t("rec.import.failed_title"))
                box.setText(_t("rec.import.no_ffmpeg_msg"))
                box.addButton(_t("ctx.confirm_yes"),
                              QMessageBox.ButtonRole.AcceptRole)
                box.exec()
                return

            self.chosen_label.setText(_t("rec.import.progress"))
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                dst = _import_audio_to_recordings(src)
            except Exception as e:
                QApplication.restoreOverrideCursor()
                _log("ERR", f"import: {type(e).__name__}: {e}")
                self.chosen_label.setText(_t("rec.no_file"))
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Icon.Critical)
                box.setWindowTitle(_t("rec.import.failed_title"))
                box.setText(_t("rec.import.failed_msg_fmt").format(
                    err=f"{type(e).__name__}: {e}"))
                box.addButton(_t("ctx.confirm_yes"),
                              QMessageBox.ButtonRole.AcceptRole)
                box.exec()
                return
            QApplication.restoreOverrideCursor()

            self._last_recorded = dst
            self._result_path = None
            self.chosen_label.setText(
                _t("rec.import.done_fmt").format(name=dst.name))
            self._refresh_action_buttons()

        # ── Action buttons — toggle between "generate" and "open" based on
        # whether the corresponding artifact already exists for _last_recorded.
        # When an artifact exists, the button also gets the accent-blue style
        # (#0a84ff — matches the idle mic button) so it reads as a primary
        # "open" affordance instead of a neutral "generate" button.
        def _refresh_action_buttons(self):
            audio = self._last_recorded
            if not audio:
                self.transcribe_btn.setText(_t("rec.btn.transcribe"))
                self.notes_btn.setText(_t("rec.btn.notes"))
                self.interview_btn.setText(_t("rec.btn.interview"))
                self.sharing_btn.setText(_t("rec.btn.sharing"))
                for b in (self.transcribe_btn, self.notes_btn,
                          self.interview_btn, self.sharing_btn):
                    _apply_open_btn_style(b, is_open=False)
                return
            polish = audio.with_name(audio.stem + ".polish.txt")
            meeting_md = audio.with_name(audio.stem + ".meeting.md")
            interview_md = audio.with_name(audio.stem + ".interview.md")
            sharing_md = audio.with_name(audio.stem + ".sharing.md")
            # Legacy single-`.md` artefact (pre-`.meeting.md`/`.interview.md`
            # split). When it exists and the new-format artefacts don't, the
            # notes button still flips to "open" mode so the user can read
            # the existing summary instead of generating a duplicate file.
            legacy_md = audio.with_name(audio.stem + ".md")
            for btn, exists, gen_key, open_key in (
                (self.transcribe_btn, polish.exists(),
                 "rec.btn.transcribe", "rec.btn.open_transcribe"),
                (self.notes_btn, meeting_md.exists() or legacy_md.exists(),
                 "rec.btn.notes", "rec.btn.open_notes"),
                (self.interview_btn, interview_md.exists(),
                 "rec.btn.interview", "rec.btn.open_interview"),
                (self.sharing_btn, sharing_md.exists(),
                 "rec.btn.sharing", "rec.btn.open_sharing"),
            ):
                btn.setText(_t(open_key if exists else gen_key))
                _apply_open_btn_style(btn, is_open=exists)

        def _on_transcribe_clicked(self):
            audio = self._last_recorded
            if not audio:
                self._on_warning(_t("pipe.warn.no_wav"))
                return
            polish = audio.with_name(audio.stem + ".polish.txt")
            if polish.exists():
                _open_path(polish)
            else:
                self._start_pipeline(transcribe_only=True)

        def _on_meeting_clicked(self):
            audio = self._last_recorded
            if not audio:
                self._on_warning(_t("pipe.warn.no_wav"))
                return
            meeting_md = audio.with_name(audio.stem + ".meeting.md")
            legacy_md = audio.with_name(audio.stem + ".md")
            if meeting_md.exists():
                _open_path(meeting_md)
            elif legacy_md.exists():
                _open_path(legacy_md)
            else:
                self._start_pipeline(mode="meeting")

        def _on_interview_clicked(self):
            audio = self._last_recorded
            if not audio:
                self._on_warning(_t("pipe.warn.no_wav"))
                return
            interview_md = audio.with_name(audio.stem + ".interview.md")
            if interview_md.exists():
                _open_path(interview_md)
            else:
                self._start_pipeline(mode="interview")

        def _on_sharing_clicked(self):
            audio = self._last_recorded
            if not audio:
                self._on_warning(_t("pipe.warn.no_wav"))
                return
            sharing_md = audio.with_name(audio.stem + ".sharing.md")
            if sharing_md.exists():
                _open_path(sharing_md)
            else:
                self._start_pipeline(mode="sharing")

        # ── Log helper — prefix every UI log line with a wall-clock timestamp
        # so the log_view and the file log share the same time reference.
        def _ui_log(self, msg: str):
            ts = datetime.now().strftime("%H:%M:%S")
            self.log_view.append(f"[{ts}] {msg}")

    # ── History view ──────────────────────────────────────────────────────

    class HistoryInterface(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("historyInterface")
            self._current: "dict | None" = None
            # Pipeline state — mirrors RecordingInterface so that the user
            # can re-run transcribe / polish / notes from the history view
            # against a previously-recorded meeting.
            self._pipeline_thread: "QThread | None" = None
            self._pipeline_worker: "_PipelineWorker | None" = None
            # i18n: per-widget setText callbacks + per-tab button refs
            # (SegmentedWidget doesn't expose `setItemText` everywhere, so
            # we capture the underlying button from `addItem`'s return).
            self._lang_callbacks: list = []
            self._tab_buttons: dict = {}
            self._build_ui()
            self._wire()
            self.refresh()

        def _i18n(self, widget, key, attr="setText"):
            def update():
                getattr(widget, attr)(_t(key))
            self._lang_callbacks.append(update)
            update()

        def apply_language(self):
            for cb in self._lang_callbacks:
                cb()
            for route_key, btn in self._tab_buttons.items():
                if btn is not None and hasattr(btn, "setText"):
                    btn.setText(_t(f"hist.tab.{route_key}"))
            self.refresh()
            if self._current is not None:
                self._show_detail(self._current)
            else:
                self.detail_title.setText(_t("hist.default_title"))
            # Action button labels depend on (current language) × (artifact
            # exists on disk) — refresh after _show_detail so the "open X"
            # override wins if a meeting is selected.
            self._refresh_h_action_buttons()

        def _build_ui(self):
            root = QHBoxLayout(self)
            root.setContentsMargins(24, 24, 24, 24)
            root.setSpacing(20)

            # Left: search + filter + list
            left = QFrame(self)
            left.setMinimumWidth(340)
            left.setMaximumWidth(440)
            lv = QVBoxLayout(left)
            lv.setSpacing(12)

            self.search = SearchLineEdit(self)
            self._i18n(self.search, "hist.search_placeholder", attr="setPlaceholderText")

            # Four filter tabs:
            #   all      — every recording (parity with the recording-view sidebar)
            #   done     — has .meeting.md or .interview.md → right pane shows
            #              both summaries concatenated
            #   stt_only — has .polish.txt (but not yet summarised) → right
            #              pane shows the polished transcript
            #   pending  — neither .polish.txt nor any .md → just-recorded
            #              audio waiting for transcribe/polish/notes pipeline
            #
            # We track the active mode in `self._filter_mode` and let each tab
            # update it via its own onClick callback. `SegmentedWidget`'s
            # `currentItemChanged` signal isn't reliably emitted across
            # qfluentwidgets versions, so we don't depend on it here.
            self._filter_mode = "all"
            self.filter = SegmentedWidget(self)

            def _make_tab_handler(key):
                def _handler():
                    self._filter_mode = key
                    self.refresh()
                return _handler

            # Capture each tab's button so apply_language can re-set the
            # label later. `addItem` returns the SegmentedItem button in
            # current qfluentwidgets versions; if it returns None we just
            # leave the tab text fixed in Chinese (graceful fallback).
            self._tab_buttons["all"] = self.filter.addItem(
                "all", _t("hist.tab.all"), _make_tab_handler("all"))
            self._tab_buttons["done"] = self.filter.addItem(
                "done", _t("hist.tab.done"), _make_tab_handler("done"))
            self._tab_buttons["stt_only"] = self.filter.addItem(
                "stt_only", _t("hist.tab.stt_only"), _make_tab_handler("stt_only"))
            self._tab_buttons["pending"] = self.filter.addItem(
                "pending", _t("hist.tab.pending"), _make_tab_handler("pending"))
            self.filter.setCurrentItem("all")

            self.list_w = ListWidget(self)
            self.list_w.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.count_label = CaptionLabel("", self)
            self.count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            lv.addWidget(self.search)
            lv.addWidget(self.filter)
            lv.addWidget(self.list_w, stretch=1)
            lv.addWidget(self.count_label)

            # Wrap the list column in a soft card (paired with the detail
            # card below). Padding deliberately a bit smaller than the
            # recording-view cards because this column is narrower.
            _style_as_card(left, padding=14, name="historyCardLeft")

            # Right: detail (scrollable, wrapped in a rounded card).
            #
            # Layout shape:
            #   right_card (QFrame, "historyCardRight")  ← rounded background
            #     └── scroll (ScrollArea, transparent)
            #           └── detail (QWidget, transparent)
            #
            # Earlier we styled `detail` itself as the card, but a
            # `QScrollArea` ships an opaque viewport that clipped the
            # rounded corners. Now the card lives on the outer QFrame and
            # both the scroll area and its viewport are explicitly painted
            # transparent so the wrapper's rounded background shows
            # through on every side.
            scroll = ScrollArea(self)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setStyleSheet(
                "QScrollArea { background: transparent; border: none; }"
                "QScrollArea > QWidget > QWidget { background: transparent; }"
            )
            # Defensive: explicitly clear the viewport's auto-fill so a
            # system style can't slip an opaque background back in.
            scroll.viewport().setAutoFillBackground(False)

            detail = QWidget()
            detail.setObjectName("historyDetailInner")
            detail.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            detail.setStyleSheet("#historyDetailInner { background: transparent; }")
            dv = QVBoxLayout(detail)
            dv.setContentsMargins(16, 16, 16, 16)
            dv.setSpacing(14)

            # ── Action row (top-right) + progress + log view ───────────
            # Mirrors RecordingInterface's pipeline-launch UX so a user can
            # re-run transcribe / polish / notes on any previously-recorded
            # meeting from the history view directly.
            # Same treatment as the recording view's action row: plain
            # `PushButton`s (neutral surface colour) plus `addStretch` on
            # BOTH sides so the row centres in the detail pane instead of
            # right-aligning. Text + style are managed entirely by
            # `_refresh_h_action_buttons()` — mirrors RecordingInterface so
            # a meeting whose artifact already exists on disk renders the
            # corresponding button as a blue "open the result" affordance.
            self.h_transcribe_btn = PushButton("", self)
            self.h_notes_btn = PushButton("", self)
            self.h_interview_btn = PushButton("", self)
            self.h_sharing_btn = PushButton("", self)
            for b in (self.h_transcribe_btn, self.h_notes_btn,
                      self.h_interview_btn, self.h_sharing_btn):
                b.setEnabled(False)  # no meeting selected at startup
            self._refresh_h_action_buttons()
            actions_row = QHBoxLayout()
            actions_row.addStretch(1)
            actions_row.addWidget(self.h_transcribe_btn)
            actions_row.addWidget(self.h_notes_btn)
            actions_row.addWidget(self.h_interview_btn)
            actions_row.addWidget(self.h_sharing_btn)
            actions_row.addStretch(1)
            dv.addLayout(actions_row)

            self.h_progress = ProgressBar(self)
            self.h_progress.setVisible(False)
            dv.addWidget(self.h_progress)

            self.h_cancel_btn = PushButton("", self)
            self._i18n(self.h_cancel_btn, "hist.btn.cancel")
            self.h_cancel_btn.setVisible(False)
            dv.addWidget(self.h_cancel_btn)

            self.h_log_view = TextEdit(self)
            self.h_log_view.setReadOnly(True)
            self.h_log_view.setMaximumHeight(160)
            self.h_log_view.setVisible(False)
            dv.addWidget(self.h_log_view)

            dv.addWidget(self._sep())

            # ── Detail content (title / meta / body / participants / todos)
            self.detail_title = TitleLabel(_t("hist.default_title"), self)
            dv.addWidget(self.detail_title)

            meta_row = QHBoxLayout()
            self.meta_date = CaptionLabel("", self)
            self.meta_dur = CaptionLabel("", self)
            self.meta_participants = CaptionLabel("", self)
            meta_row.addWidget(self.meta_date)
            meta_row.addSpacing(12)
            meta_row.addWidget(self.meta_dur)
            meta_row.addSpacing(12)
            meta_row.addWidget(self.meta_participants)
            meta_row.addStretch(1)
            dv.addLayout(meta_row)
            dv.addWidget(self._sep())

            self.body_title = StrongBodyLabel("", self)
            self._i18n(self.body_title, "hist.body_title")
            self.body_browser = TextBrowser(self)
            self.body_browser.setMinimumHeight(320)
            dv.addWidget(self.body_title)
            dv.addWidget(self.body_browser)
            dv.addWidget(self._sep())

            _participants_title = StrongBodyLabel("", self)
            self._i18n(_participants_title, "hist.participants")
            dv.addWidget(_participants_title)
            self.participants_label = BodyLabel(_placeholder_no_data(), self)
            dv.addWidget(self.participants_label)
            dv.addWidget(self._sep())

            _todos_title = StrongBodyLabel("", self)
            self._i18n(_todos_title, "hist.todos")
            dv.addWidget(_todos_title)
            self.todos_label = BodyLabel(_placeholder_no_data(), self)
            dv.addWidget(self.todos_label)

            dv.addStretch(1)
            scroll.setWidget(detail)

            # Outer rounded-card wrapper around the scroll area. This is
            # what actually paints the soft-gray background and the 14-px
            # corner radius; `detail` itself stays transparent so we don't
            # double-paint the surface.
            right_card = QFrame(self)
            right_card.setObjectName("historyCardRight")
            right_card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            right_card.setStyleSheet(
                f"#historyCardRight {{ background-color: {_CARD_BG}; "
                f"border-radius: 14px; }}"
            )
            rcv = QVBoxLayout(right_card)
            rcv.setContentsMargins(0, 0, 0, 0)
            rcv.setSpacing(0)
            rcv.addWidget(scroll)

            root.addWidget(left)
            root.addWidget(right_card, stretch=1)

        def _sep(self) -> QFrame:
            s = QFrame(self)
            s.setFrameShape(QFrame.Shape.HLine)
            s.setFrameShadow(QFrame.Shadow.Sunken)
            return s

        def _wire(self):
            self.list_w.itemClicked.connect(self._on_pick)
            self.search.textChanged.connect(lambda _t: self.refresh())
            # Filter tabs already wire themselves via the onClick callbacks
            # registered in addItem(...) — no `currentItemChanged` signal
            # connection needed (and it isn't reliably emitted anyway).
            self.list_w.customContextMenuRequested.connect(self._on_context_menu)

            self.h_transcribe_btn.clicked.connect(self._on_h_transcribe_clicked)
            self.h_notes_btn.clicked.connect(self._on_h_meeting_clicked)
            self.h_interview_btn.clicked.connect(self._on_h_interview_clicked)
            self.h_sharing_btn.clicked.connect(self._on_h_sharing_clicked)
            self.h_cancel_btn.clicked.connect(self._on_cancel_pipeline)

        # ── Action buttons — mirror RecordingInterface: each toggles
        # between "generate" / "open" based on whether the corresponding
        # output file already exists for the currently-selected meeting,
        # and switches to the accent-blue style when it's an "open"
        # affordance (matches the idle mic button colour).
        def _refresh_h_action_buttons(self):
            m = self._current
            if not m:
                self.h_transcribe_btn.setText(_t("hist.btn.transcribe"))
                self.h_notes_btn.setText(_t("hist.btn.notes"))
                self.h_interview_btn.setText(_t("hist.btn.interview"))
                self.h_sharing_btn.setText(_t("hist.btn.sharing"))
                for b in (self.h_transcribe_btn, self.h_notes_btn,
                          self.h_interview_btn, self.h_sharing_btn):
                    _apply_open_btn_style(b, is_open=False)
                return
            polish_p = m.get("polish_path")
            meeting_p = m.get("meeting_md_path")
            interview_p = m.get("interview_md_path")
            sharing_p = m.get("sharing_md_path")
            # Legacy single-`.md` artefact (`md_path` is set + `md_mode` is
            # None means a pre-split `.md` file). Counts as "meeting notes
            # exist" for the open-toggle so the user can read the existing
            # summary without re-running the pipeline.
            legacy_p = (
                m.get("md_path") if (meeting_p is None and interview_p is None
                                     and sharing_p is None
                                     and m.get("has_md")) else None
            )
            for btn, exists, gen_key, open_key in (
                (self.h_transcribe_btn, polish_p is not None,
                 "hist.btn.transcribe", "hist.btn.open_transcribe"),
                (self.h_notes_btn, meeting_p is not None or legacy_p is not None,
                 "hist.btn.notes", "hist.btn.open_notes"),
                (self.h_interview_btn, interview_p is not None,
                 "hist.btn.interview", "hist.btn.open_interview"),
                (self.h_sharing_btn, sharing_p is not None,
                 "hist.btn.sharing", "hist.btn.open_sharing"),
            ):
                btn.setText(_t(open_key if exists else gen_key))
                _apply_open_btn_style(btn, is_open=exists)

        # Open-or-generate click handlers. If the corresponding artifact
        # already exists for `self._current`, just open it; otherwise kick
        # off the pipeline (transcribe-only / meeting / interview).
        def _on_h_transcribe_clicked(self):
            m = self._current
            if not m:
                return
            polish_p = m.get("polish_path")
            if polish_p:
                _open_path(polish_p)
            else:
                self._start_pipeline(transcribe_only=True)

        def _on_h_meeting_clicked(self):
            m = self._current
            if not m:
                return
            meeting_p = m.get("meeting_md_path")
            interview_p = m.get("interview_md_path")
            legacy_p = (
                m.get("md_path") if (meeting_p is None and interview_p is None
                                     and m.get("has_md")) else None
            )
            if meeting_p:
                _open_path(meeting_p)
            elif legacy_p:
                _open_path(legacy_p)
            else:
                self._start_pipeline(mode="meeting")

        def _on_h_interview_clicked(self):
            m = self._current
            if not m:
                return
            interview_p = m.get("interview_md_path")
            if interview_p:
                _open_path(interview_p)
            else:
                self._start_pipeline(mode="interview")

        def _on_h_sharing_clicked(self):
            m = self._current
            if not m:
                return
            sharing_p = m.get("sharing_md_path")
            if sharing_p:
                _open_path(sharing_p)
            else:
                self._start_pipeline(mode="sharing")

        def refresh(self):
            self.list_w.clear()
            q = (self.search.text() or "").strip().lower()
            mode = self._filter_mode  # 'all' | 'done' | 'stt_only' | 'pending'
            shown = 0
            for m in _list_recordings():
                # Filter by tab
                if mode == "done":
                    if not (m["has_md"]):
                        continue
                elif mode == "stt_only":
                    if not m["has_polish"]:
                        continue
                elif mode == "pending":
                    # Neither a polished transcript nor any summary yet —
                    # raw audio waiting on the pipeline.
                    if m["has_md"] or m["has_polish"]:
                        continue
                # Substring filter (case-insensitive) on the full stem
                if q and q not in m["stem"].lower():
                    continue
                label = _format_timestamp_label(m["stem"])
                dur = _audio_duration_secs(m["wav_path"])
                if m["has_md"]:
                    badge = _t("hist.badge.done")
                elif m["has_polish"]:
                    badge = _t("hist.badge.transcribed")
                else:
                    badge = _t("hist.badge.audio_only")
                item = QListWidgetItem(
                    f"{label}\n{_t('hist.dur_prefix')} {_format_duration(dur)}  ·  {badge}"
                )
                item.setData(Qt.ItemDataRole.UserRole, m)
                self.list_w.addItem(item)
                shown += 1
            self.count_label.setText(_t("hist.count_fmt", n=shown))

        def _on_pick(self, item: QListWidgetItem):
            data = item.data(Qt.ItemDataRole.UserRole)
            if not data:
                return
            self._show_detail(data)

        def _show_detail(self, m: dict):
            self._current = m
            label = _format_timestamp_label(m["stem"])
            self.detail_title.setText(label)
            self.meta_date.setText(f"📅 {label}")
            dur = _audio_duration_secs(m["wav_path"])
            self.meta_dur.setText(f"⏱ {_format_duration(dur)}")
            self.meta_participants.setText(f"👥 {_placeholder_no_data()}")

            # Body content is tab-dependent:
            #   全部       → meeting.md > interview.md > polish.txt > raw.txt
            #   已总结     → meeting.md + interview.md concatenated (if both)
            #   已录音转文字 → polish.txt only
            #   待处理     → raw.txt or placeholder
            self._render_body_for_mode(m, self._filter_mode)

            # Until a `.meta.json` sidecar is introduced (Phase 2), we have
            # no source for participants / todos — show the placeholder
            # both fields user said to show.
            self.participants_label.setText(_placeholder_no_data())
            self.todos_label.setText(_placeholder_no_data())

            # Now that a meeting is selected, the pipeline buttons are
            # actionable (unless one is already running).
            running = self._pipeline_thread is not None
            for b in (self.h_transcribe_btn, self.h_notes_btn,
                      self.h_interview_btn, self.h_sharing_btn):
                b.setEnabled(not running)
            # Refresh "generate / open" label + style for the new selection.
            self._refresh_h_action_buttons()

        def _set_body_markdown(self, md: str):
            self.body_browser.setMarkdown(md)
            _apply_callout_styles(self.body_browser.document())

        def _render_body_for_mode(self, m: dict, mode: str):
            meeting_p = m.get("meeting_md_path")
            interview_p = m.get("interview_md_path")
            sharing_p = m.get("sharing_md_path")
            polish_p = m.get("polish_path")
            raw_p = m.get("raw_path")
            # Legacy pre-split `.md` only fires when none of the new-format
            # artefacts exist; without this guard, a sharing-only recording
            # would render the legacy fallback alongside its real sharing.
            legacy_p = (
                m.get("md_path") if (m.get("md_mode") is None and m.get("has_md"))
                else None
            )

            def _read(p):
                try:
                    return Path(p).read_text(encoding="utf-8")
                except Exception as e:
                    return _t("hist.body.read_error", name=Path(p).name, err=str(e))

            if mode == "done":
                parts: list[str] = []
                if meeting_p:
                    parts.append(f"# 📝 {_t('hist.body.notes_meeting')}\n\n{_read(meeting_p)}")
                if interview_p:
                    parts.append(f"# 🎤 {_t('hist.body.notes_interview')}\n\n{_read(interview_p)}")
                if sharing_p:
                    parts.append(f"# 🎓 {_t('hist.body.notes_sharing')}\n\n{_read(sharing_p)}")
                if legacy_p and not (meeting_p or interview_p or sharing_p):
                    parts.append(_read(legacy_p))
                if parts:
                    self._set_body_markdown("\n\n---\n\n".join(parts))
                    # Title: name the single artifact if there's only one, else
                    # "Multiple summaries". (Don't try to enumerate every
                    # combination — the body itself shows the contents.)
                    present = [p for p in (meeting_p, interview_p, sharing_p) if p]
                    if len(present) == 1:
                        if meeting_p:
                            self.body_title.setText(_t("hist.body.notes_meeting"))
                        elif interview_p:
                            self.body_title.setText(_t("hist.body.notes_interview"))
                        else:
                            self.body_title.setText(_t("hist.body.notes_sharing"))
                    else:
                        # Pre-existing "notes_both" label still reads sensibly
                        # for any multi-artifact combination ("Meeting Notes +
                        # Interview Report"-style listing); the body content
                        # is unambiguous regardless.
                        self.body_title.setText(_t("hist.body.notes_both"))
                else:
                    self.body_browser.setPlainText(_t("hist.body.no_notes"))
                    self.body_title.setText(_t("hist.body.default"))
                return

            if mode == "stt_only":
                if polish_p:
                    self.body_browser.setPlainText(_read(polish_p))
                    self.body_title.setText(_t("hist.body.polish_only"))
                else:
                    self.body_browser.setPlainText(_t("hist.body.no_polish"))
                    self.body_title.setText(_t("hist.body.default"))
                return

            if mode == "pending":
                # 待处理 tab: no polish + no summary yet. Show the raw FunASR
                # transcript if it exists (e.g. transcribe finished but polish
                # didn't), otherwise a friendly placeholder.
                if raw_p:
                    self.body_browser.setPlainText(_read(raw_p))
                    self.body_title.setText(_t("hist.body.raw_pending"))
                else:
                    self.body_browser.setPlainText(_t("hist.body.pending_placeholder"))
                    self.body_title.setText(_t("hist.body.default"))
                return

            # mode == "all" (default): show the richest available artifact.
            # If multiple summary artifacts exist on the same recording,
            # concatenate them so the user doesn't have to flip tabs to see
            # "I have both a meeting summary AND a sharing summary on this
            # recording".
            summary_parts: list[str] = []
            if meeting_p:
                summary_parts.append(
                    f"# 📝 {_t('hist.body.notes_meeting')}\n\n{_read(meeting_p)}"
                )
            if interview_p:
                summary_parts.append(
                    f"# 🎤 {_t('hist.body.notes_interview')}\n\n{_read(interview_p)}"
                )
            if sharing_p:
                summary_parts.append(
                    f"# 🎓 {_t('hist.body.notes_sharing')}\n\n{_read(sharing_p)}"
                )
            if len(summary_parts) >= 2:
                self._set_body_markdown("\n\n---\n\n".join(summary_parts))
                self.body_title.setText(_t("hist.body.notes_both"))
            elif meeting_p:
                self._set_body_markdown(_read(meeting_p))
                self.body_title.setText(_t("hist.body.notes_meeting_md"))
            elif interview_p:
                self._set_body_markdown(_read(interview_p))
                self.body_title.setText(_t("hist.body.notes_interview_md"))
            elif sharing_p:
                self._set_body_markdown(_read(sharing_p))
                self.body_title.setText(_t("hist.body.notes_sharing_md"))
            elif legacy_p:
                self._set_body_markdown(_read(legacy_p))
                self.body_title.setText(_t("hist.body.notes_legacy_md"))
            elif polish_p:
                self.body_browser.setPlainText(_read(polish_p))
                self.body_title.setText(_t("hist.body.polish_only"))
            elif raw_p:
                self.body_browser.setPlainText(_read(raw_p))
                self.body_title.setText(_t("hist.body.raw_only"))
            else:
                self.body_browser.setPlainText(_t("hist.body.none"))
                self.body_title.setText(_t("hist.body.default"))

        # Right-click → rename / open enclosing folder (mirrors the
        # recording-view sidebar). Renames cascade across every sibling
        # file via _rename_meeting_files.
        def _on_context_menu(self, pos: "QPoint"):
            item = self.list_w.itemAt(pos)
            if item is None:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            if not data:
                return
            menu = QMenu(self.list_w)
            act_rename = QAction(_t("ctx.rename"), menu)
            act_delete = QAction(_t("ctx.delete"), menu)
            act_reveal = QAction(_t("ctx.reveal"), menu)
            menu.addAction(act_rename)
            menu.addAction(act_delete)
            menu.addAction(act_reveal)

            def _do_rename():
                ts, current_custom = _split_meeting_stem(data["stem"])
                if not ts:
                    QMessageBox.warning(self, _t("ctx.rename_failed_title"),
                                        _t("ctx.rename_failed_format"))
                    return
                new_name, ok = QInputDialog.getText(
                    self, _t("ctx.rename_dialog_title"),
                    _t("ctx.rename_dialog_prompt", ts=ts),
                    text=current_custom or "",
                )
                if not ok:
                    return
                clean = _sanitize_meeting_custom_name(new_name)
                if clean == (current_custom or ""):
                    return
                new_wav = _rename_meeting_files(data["wav_path"], clean)
                if new_wav is None:
                    QMessageBox.warning(self, _t("ctx.rename_failed_title"),
                                        _t("ctx.rename_failed_collision"))
                    return
                # If the renamed meeting is the one currently shown in the
                # right pane, follow it — without this `self._current` keeps
                # the OLD `wav_path` / `polish_path` / ... and the action
                # buttons act on the now-nonexistent old paths.
                was_current = (
                    self._current is not None
                    and self._current.get("wav_path") == data["wav_path"]
                )
                self.refresh()
                if was_current:
                    fresh = next(
                        (m for m in _list_recordings() if m["wav_path"] == new_wav),
                        None,
                    )
                    if fresh is not None:
                        self._show_detail(fresh)

            def _do_reveal():
                _open_path(data["wav_path"].parent)

            def _do_delete():
                wav = data["wav_path"]
                # Guard: a running pipeline owns the files. Same logic as
                # RecordingInterface — block delete and tell the user why.
                if self._pipeline_thread is not None:
                    QMessageBox.warning(
                        self, _t("ctx.delete_blocked_title"),
                        _t("ctx.delete_blocked_msg"),
                    )
                    return
                if not _confirm_dialog(
                    self, _t("ctx.delete_title"),
                    _t("ctx.delete_confirm", name=wav.name),
                ):
                    return
                _deleted, errors = _delete_meeting_files(wav)
                if errors:
                    QMessageBox.warning(
                        self, _t("ctx.delete_failed_title"),
                        _t("ctx.delete_failed_msg", err="\n".join(errors)),
                    )
                # If the deleted file was the currently-displayed meeting,
                # clear the right-pane detail.
                if self._current is not None and \
                        self._current.get("wav_path") == wav:
                    self._current = None
                    self.detail_title.setText(_t("hist.default_title"))
                    self.body_browser.setPlainText("")
                    for b in (self.h_transcribe_btn, self.h_notes_btn,
                              self.h_interview_btn, self.h_sharing_btn):
                        b.setEnabled(False)
                self.refresh()

            # Same async-popup pattern as RecordingInterface's history menu
            # (see the comment there): no nested exec() loop, handlers
            # deferred past the menu teardown, screen pinned explicitly.
            menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

            def _defer(fn):
                return lambda: QTimer.singleShot(0, fn)

            act_rename.triggered.connect(_defer(_do_rename))
            act_reveal.triggered.connect(_defer(_do_reveal))
            act_delete.triggered.connect(_defer(_do_delete))
            global_pos = self.list_w.mapToGlobal(pos)
            screen = QGuiApplication.screenAt(global_pos)
            if screen is not None:
                menu.setScreen(screen)
            self._ctx_menu = menu  # keep the wrapper alive while shown
            menu.popup(global_pos)

        # ── Pipeline (transcribe / meeting / interview) ────────────────
        # Re-runs the transcribe/polish/notes pipeline on a previously-
        # recorded meeting. Functionally identical to RecordingInterface's
        # pipeline flow, just driven from the currently-selected history
        # item instead of the freshly-recorded file.

        def _start_pipeline(self, transcribe_only: bool = False,
                            mode: "str | None" = None):
            if not self._current:
                return
            target: Path = self._current["wav_path"]
            if not target.exists():
                QMessageBox.warning(
                    self, _t("pipe.warn.file_missing_title"),
                    _t("pipe.warn.file_missing_msg", name=target.name),
                )
                return
            chosen_mode = mode or getattr(args, "mode", None) or cfg.get("mode", "meeting")

            self.h_log_view.setVisible(True)
            self.h_log_view.clear()
            self.h_progress.setVisible(True)
            self.h_progress.setValue(0)
            self.h_cancel_btn.setVisible(True)
            for b in (self.h_transcribe_btn, self.h_notes_btn,
                      self.h_interview_btn, self.h_sharing_btn):
                b.setEnabled(False)

            thread = QThread(self)
            worker = _PipelineWorker(target, chosen_mode, cfg,
                                     transcribe_only=transcribe_only)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.progress.connect(self.h_progress.setValue)
            worker.log.connect(self.h_log_view.append)
            worker.done.connect(self._on_pipeline_done)
            worker.failed.connect(self._on_pipeline_failed)
            worker.cancelled.connect(self._on_pipeline_cancelled)
            worker.done.connect(thread.quit)
            worker.failed.connect(thread.quit)
            worker.cancelled.connect(thread.quit)
            thread.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            self._pipeline_worker = worker
            self._pipeline_thread = thread
            thread.start()

        def _on_pipeline_done(self, path_str: str):
            self._reset_after_pipeline()
            # Same as the recording view: the run log is transient on success
            # (it all lives in the daily log file), a toast reports the result.
            self._hide_pipeline_log()
            InfoBar.success(
                title=_t("pipe.info.done_title"),
                content=Path(path_str).name,
                isClosable=True, position=InfoBarPosition.TOP,
                duration=4000, parent=self,
            )
            # Re-render: the .md / .polish.txt artifact list just changed.
            self.refresh()
            if self._current is not None:
                # Re-load the current meeting's detail so newly-produced
                # artifacts (.meeting.md / .interview.md / .polish.txt)
                # show up in the body pane immediately.
                fresh = next(
                    (m for m in _list_recordings()
                     if m["wav_path"] == self._current["wav_path"]),
                    None,
                )
                if fresh is not None:
                    self._show_detail(fresh)

        def _on_pipeline_failed(self, msg: str):
            self._reset_after_pipeline()
            self.h_log_view.append(_t("pipe.log.failed", err=msg))

        def _on_pipeline_cancelled(self):
            self._reset_after_pipeline()
            self._hide_pipeline_log()

        def _hide_pipeline_log(self):
            self.h_log_view.clear()
            self.h_log_view.setVisible(False)

        def _reset_after_pipeline(self):
            self.h_progress.setVisible(False)
            self.h_cancel_btn.setVisible(False)
            # Re-arm the cancel button in case `_on_cancel_pipeline` had
            # disabled it during the wrap-up window.
            self.h_cancel_btn.setEnabled(True)
            self._pipeline_worker = None
            self._pipeline_thread = None
            # Re-enable action buttons iff a meeting is currently selected.
            has_selection = self._current is not None
            for b in (self.h_transcribe_btn, self.h_notes_btn,
                      self.h_interview_btn, self.h_sharing_btn):
                b.setEnabled(has_selection)
            # A just-completed pipeline may have produced new artifacts —
            # re-render the "generate / open" toggle.
            self._refresh_h_action_buttons()

        def _on_cancel_pipeline(self):
            # Truly terminate the background pipeline: set the cancel event
            # and SIGKILL any registered subprocess (claude -p …) so the
            # worker thread escapes its blocking call and emits the
            # `cancelled` signal. We keep the cancel button visible but
            # disabled during the wrap-up window so the user sees that
            # termination is in progress; full UI reset happens in
            # `_on_pipeline_cancelled` when the worker confirms.
            worker = self._pipeline_worker
            if worker is None:
                self._reset_after_pipeline()
                return
            self.h_log_view.append(_t("pipe.log.cancelling"))
            self.h_cancel_btn.setEnabled(False)
            worker.cancel()

    # ── Config view ───────────────────────────────────────────────────────

    class ConfigInterface(QWidget):
        """Two stacked cards:

          1. **Background-task concurrency** — single SpinBox + Apply button.
             Bumps two keys at once (``polish_max_workers`` and
             ``stt.funasr.workers``) and persists via ``save_config`` so
             JSONC comments survive.

          2. **Raw editor for ``config.jsonc``** — the file's actual text
             (with ``//`` comments) in a PlainTextEdit. Save button writes
             back after a JSONC-strip → ``json.loads`` validation; an
             invalid edit shows an InfoBar.error and does NOT touch disk.
        """

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("configInterface")
            self._lang_callbacks: list = []
            self._build_ui()
            self._wire()
            self._load_into_widgets()

        def _i18n(self, widget, key, attr="setText"):
            def update():
                getattr(widget, attr)(_t(key))
            self._lang_callbacks.append(update)
            update()

        def apply_language(self):
            for cb in self._lang_callbacks:
                cb()

        def _build_ui(self):
            root = QVBoxLayout(self)
            root.setContentsMargins(28, 28, 28, 28)
            root.setSpacing(20)

            # Card 1: concurrency
            _t_concur_title = TitleLabel("", self)
            self._i18n(_t_concur_title, "cfg.title.concurrency")
            root.addWidget(_t_concur_title)
            _t_concur_desc = BodyLabel("", self)
            _t_concur_desc.setWordWrap(True)
            self._i18n(_t_concur_desc, "cfg.desc.concurrency")
            root.addWidget(_t_concur_desc)
            concurrency_row = QHBoxLayout()
            _t_workers_label = BodyLabel("", self)
            self._i18n(_t_workers_label, "cfg.label.workers")
            concurrency_row.addWidget(_t_workers_label)
            self.concurrency_spin = SpinBox(self)
            self.concurrency_spin.setRange(0, 64)
            self.concurrency_spin.setMinimumWidth(120)
            concurrency_row.addWidget(self.concurrency_spin)
            self.concurrency_apply = PrimaryPushButton("", self)
            self._i18n(self.concurrency_apply, "cfg.btn.apply")
            concurrency_row.addWidget(self.concurrency_apply)
            concurrency_row.addStretch(1)
            root.addLayout(concurrency_row)

            sep = QFrame(self)
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setFrameShadow(QFrame.Shadow.Sunken)
            root.addWidget(sep)

            # Card 2: raw editor
            _t_editor_title = TitleLabel("", self)
            self._i18n(_t_editor_title, "cfg.title.editor")
            root.addWidget(_t_editor_title)
            _t_editor_desc = BodyLabel("", self)
            _t_editor_desc.setWordWrap(True)
            self._i18n(_t_editor_desc, "cfg.desc.editor")
            root.addWidget(_t_editor_desc)
            self.editor = QPlainTextEdit(self)
            mono = QFont("Menlo")
            if not mono.exactMatch():
                mono = QFont("Monaco")
            if not mono.exactMatch():
                mono = QFont("Consolas")
            if not mono.exactMatch():
                mono = QFont()
            mono.setPointSize(12)
            self.editor.setFont(mono)
            self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            self.editor.setMinimumHeight(360)
            # Rounded card-style border + soft padding so the editor blends
            # with the rest of the Fluent surface instead of looking like
            # a raw native textarea.
            self.editor.setStyleSheet(
                "QPlainTextEdit {"
                "  background-color: #ffffff;"
                "  border: 1px solid #d0d7de;"
                "  border-radius: 10px;"
                "  padding: 10px;"
                "  selection-background-color: #cfe1ff;"
                "}"
            )
            # JSONC syntax highlighting (light theme):
            #   keys  → blue   strings → green   numbers → red
            #   keywords (true/false/null) → purple   comments → grey-italic
            self._highlighter = _JSONCHighlighter(self.editor.document())
            root.addWidget(self.editor, stretch=1)

            # Below the editor: 「重新加载」 + 「保存配置文件」 on the same
            # row. Reload re-fills the editor from disk (discarding any
            # in-flight edits). Save validates the editor text (JSONC strip
            # → json.loads) before writing it back; an invalid edit shows
            # an InfoBar.error and does NOT touch the file on disk.
            edit_row = QHBoxLayout()
            self.reload_btn = PushButton(FluentIcon.SYNC, "", self)
            self.save_btn = PrimaryPushButton(FluentIcon.SAVE, "", self)
            self._i18n(self.reload_btn, "cfg.btn.reload")
            self._i18n(self.save_btn, "cfg.btn.save")
            edit_row.addStretch(1)
            edit_row.addWidget(self.reload_btn)
            edit_row.addWidget(self.save_btn)
            root.addLayout(edit_row)

        def _wire(self):
            self.concurrency_apply.clicked.connect(self._on_apply_concurrency)
            self.reload_btn.clicked.connect(self._load_into_widgets)
            self.save_btn.clicked.connect(self._on_save_editor)

        def _load_into_widgets(self):
            """Refresh both widgets from disk. Called at construct time and
            via the 「重新加载」 button so an external edit (or a recent
            ``--set`` from the CLI) is picked up without restarting."""
            current = cfg.get("polish_max_workers", 0) or 0
            try:
                self.concurrency_spin.setValue(int(current))
            except (TypeError, ValueError):
                self.concurrency_spin.setValue(0)
            if CONFIG_FILE.exists():
                try:
                    self.editor.setPlainText(CONFIG_FILE.read_text(encoding="utf-8"))
                except Exception as e:
                    self.editor.setPlainText(
                        f"// 无法读取 config.jsonc: {type(e).__name__}: {e}\n")
            else:
                self.editor.setPlainText("{\n}\n")

        def _on_apply_concurrency(self):
            """Apply the SpinBox value to the two concurrency keys and
            persist via ``save_config`` (JSONC-aware in-place writer that
            preserves comments). Editor edits are saved by the separate
            「保存配置文件」 button below the editor."""
            n = int(self.concurrency_spin.value())
            cfg["polish_max_workers"] = n
            cfg.setdefault("stt", {}).setdefault("funasr", {})["workers"] = n
            try:
                save_config(cfg)
            except Exception as e:
                _log("ERR", f"Qt apply concurrency: {type(e).__name__}: {e}")
                InfoBar.error(
                    title=_t("cfg.info.save_failed_title"),
                    content=f"{type(e).__name__}: {e}",
                    isClosable=True, position=InfoBarPosition.TOP,
                    duration=4000, parent=self,
                )
                return
            self._load_into_widgets()
            InfoBar.success(
                title=_t("cfg.info.applied_title"),
                content=_t("cfg.info.applied_body", n=n),
                isClosable=True, position=InfoBarPosition.TOP,
                duration=4000, parent=self,
            )

        def _on_save_editor(self):
            """Validate the editor text as JSONC and persist it verbatim to
            disk (preserving every // comment exactly as typed). Reload the
            in-memory cfg so the running process sees the change."""
            text = self.editor.toPlainText()
            try:
                json.loads(_strip_jsonc_comments(text))
            except (json.JSONDecodeError, ValueError) as e:
                InfoBar.error(
                    title=_t("cfg.info.json_error_title"),
                    content=str(e),
                    isClosable=True, position=InfoBarPosition.TOP,
                    duration=6000, parent=self,
                )
                return
            try:
                CONFIG_FILE.write_text(text, encoding="utf-8")
            except Exception as e:
                InfoBar.error(
                    title=_t("cfg.info.write_failed_title"),
                    content=f"{type(e).__name__}: {e}",
                    isClosable=True, position=InfoBarPosition.TOP,
                    duration=4000, parent=self,
                )
                return
            try:
                reloaded = load_config()
                cfg.clear()
                cfg.update(reloaded)
            except Exception as e:
                _log("ERR", f"Qt editor reload cfg: {type(e).__name__}: {e}")
            self._load_into_widgets()
            InfoBar.success(
                title=_t("cfg.info.saved_title"),
                content=_t("cfg.info.saved_body"),
                isClosable=True, position=InfoBarPosition.TOP,
                duration=4000, parent=self,
            )


    # ── Main window (QMainWindow + NavigationInterface) ──────────────────
    #
    # Replaces the earlier FluentWindow base class so that:
    #   * macOS provides native traffic lights on the LEFT with the
    #     standard ×/−/+ hover icons (no custom code needed).
    #   * The Fluent-styled left navigation panel is still shown, but as
    #     an embedded NavigationInterface widget rather than as part of
    #     the window frame itself.
    #   * Windows / Linux keep their respective native chrome.

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("MeetingScribe")
            self.resize(1280, 800)
            self.setMinimumSize(1000, 640)

            self.nav = NavigationInterface(
                self, showMenuButton=False, showReturnButton=False)
            # Sidebar width: tuned so the longest English label
            # ("Recording" / "Settings") fits without truncation on macOS,
            # while leaving the recording / history / config panes plenty
            # of room to breathe. (Previously 96 px — too narrow for English.)
            self.nav.setFixedWidth(154)
            try:
                self.nav.setExpandWidth(154)
            except (AttributeError, TypeError):
                # Older qfluentwidgets versions don't expose setExpandWidth;
                # setFixedWidth alone is sufficient there.
                pass
            self.stack = QStackedWidget(self)

            # Track nav items by route key so apply_language() can re-set
            # their text. Each value is whatever NavigationInterface.addItem
            # returned — a NavigationTreeWidget exposing setText().
            self._nav_items: dict[str, object] = {}

            self.recording_view = RecordingInterface()
            self.history_view = HistoryInterface()
            self.config_view = ConfigInterface()

            self._register_view(self.recording_view, FluentIcon.MICROPHONE,
                                "nav.recording")
            self._register_view(self.history_view, FluentIcon.HISTORY,
                                "nav.history")
            self._register_view(self.config_view, FluentIcon.SETTING,
                                "nav.config")
            # 中文/EN toggle lives at the BOTTOM of the left nav (under 配置).
            # It's an action, not a view — hence selectable=False. Registered
            # in _nav_items so apply-language relabels it like the others.
            self._nav_items["topbar.lang_zh"] = self.nav.addItem(
                routeKey="langToggle",
                icon=FluentIcon.LANGUAGE,
                text=_t("topbar.lang_zh"),
                onClick=lambda _checked=False: self._toggle_language(),
                selectable=False,
                position=NavigationItemPosition.BOTTOM,
            )
            # Start on recording view.
            self.stack.setCurrentWidget(self.recording_view)
            self.nav.setCurrentItem(self.recording_view.objectName())
            self.stack.currentChanged.connect(self._on_view_changed)

            right_col = QWidget(self)
            rc = QVBoxLayout(right_col)
            rc.setContentsMargins(0, 0, 0, 0)
            rc.setSpacing(0)
            rc.addWidget(self.stack, stretch=1)

            # Thin light-gray vertical rule between the nav rail and the
            # right column — separates the navigation surface from the
            # active view's surface so they read as two distinct panels.
            nav_separator = QFrame(self)
            nav_separator.setObjectName("navSeparator")
            nav_separator.setFrameShape(QFrame.Shape.VLine)
            nav_separator.setFrameShadow(QFrame.Shadow.Plain)
            nav_separator.setFixedWidth(1)
            nav_separator.setStyleSheet(
                "#navSeparator { background-color: #e0e2e5; border: none; }"
            )

            central = QWidget(self)
            h = QHBoxLayout(central)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(0)
            h.addWidget(self.nav)
            h.addWidget(nav_separator)
            h.addWidget(right_col, stretch=1)
            self.setCentralWidget(central)

        def _register_view(self, widget: QWidget, icon, text_key: str):
            """`text_key` is a `_LABELS` key (e.g. ``nav.recording``) so the
            label flips on language switch via `apply_language()`."""
            key = widget.objectName() or text_key
            widget.setObjectName(key)
            self.stack.addWidget(widget)
            item = self.nav.addItem(
                routeKey=key,
                icon=icon,
                text=_t(text_key),
                onClick=lambda _checked=False, w=widget: self.stack.setCurrentWidget(w),
                position=NavigationItemPosition.TOP,
            )
            self._nav_items[text_key] = item

        def _toggle_language(self):
            """Flip between zh ↔ en on every click of the nav's bottom
            中文/EN item."""
            self._set_language("en" if _LANG["current"] == "zh" else "zh")

        def _set_language(self, lang: str):
            """Flip the process-wide language and re-render every label."""
            if lang not in ("zh", "en") or lang == _LANG["current"]:
                return
            _LANG["current"] = lang
            # Update nav labels.
            for text_key, item in self._nav_items.items():
                if item is not None and hasattr(item, "setText"):
                    try:
                        item.setText(_t(text_key))
                    except Exception as e:
                        _log("ERR", f"Qt nav setText: {type(e).__name__}: {e}")
            # Notify each interface to re-render its own translatable
            # widgets. apply_language is best-effort: a view without one
            # just keeps its current labels.
            for view in (self.recording_view, self.history_view,
                         self.config_view):
                fn = getattr(view, "apply_language", None)
                if fn is None:
                    continue
                try:
                    fn()
                except Exception as e:
                    _log("ERR", f"Qt apply_language {view.objectName()}: "
                                f"{type(e).__name__}: {e}")

        def _on_view_changed(self, _idx):
            w = self.stack.currentWidget()
            if w is self.history_view:
                self.history_view.refresh()
            elif w is self.config_view:
                self.config_view._load_into_widgets()
            # The recording view has nothing to refresh on entry any more —
            # its meeting-history card moved out to the history view.
            # Keep the nav highlight in sync with programmatic stack changes.
            if w is not None:
                self.nav.setCurrentItem(w.objectName())

        def closeEvent(self, ev):
            # Best-effort cleanup so a quit during a
            # recording doesn't leave dOut/mutes in a bad state.
            try:
                rec = self.recording_view.state._recorder
                if rec is not None:
                    rec.stop()
            except Exception as e:
                _log("ERR", f"Qt closeEvent stop recorder: {type(e).__name__}: {e}")
            try:
                _get_audio_monitor().stop()
            except Exception as e:
                _log("ERR", f"Qt closeEvent monitor.stop: {type(e).__name__}: {e}")
            try:
                _restore_all_recording_mutes()
            except Exception as e:
                _log("ERR", f"Qt closeEvent restore_mutes: {type(e).__name__}: {e}")
            try:
                _restore_output_if_needed(
                    resolve_audio_devices(query_fresh=False),
                    reason="post-recording",
                )
            except Exception as e:
                _log("ERR", f"Qt closeEvent restore output: {type(e).__name__}: {e}")
            try:
                _remove_device_listeners()
            except Exception as e:
                _log("ERR", f"Qt closeEvent remove_listeners: {type(e).__name__}: {e}")
            super().closeEvent(ev)

    # ── Run ───────────────────────────────────────────────────────────────
    _install_device_listeners()
    _get_audio_monitor().start()

    app = QApplication.instance() or QApplication(sys.argv)
    setTheme(Theme.LIGHT)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="一行命令：录音 + 转写 + 纪要",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", metavar="<命令>")

    stt_help = "可选：funasr（默认）/ openai / gemini，或 stt 配置中的任意 key"
    llm_help = "可选：claude / openai / gemini，或 llm 配置中的任意 key"

    mode_help = "运行模式：meeting（会议纪要，默认）| interview（面试总结）| sharing（分享总结）"

    # --mode / --debug 可放在子命令之前（全局）或之后（子命令级），两种写法均有效
    parser.add_argument("--mode", metavar="MODE", choices=MODES, help=mode_help)
    parser.add_argument("--debug", action="store_true", help="启用调试日志（每秒打印各路音频电平到 stderr）")

    p_rec = sub.add_parser("record", help="开始录音（Ctrl+C 停止）")
    p_rec.add_argument("--mode", metavar="MODE", choices=MODES, help=mode_help, default=argparse.SUPPRESS)
    p_rec.add_argument("--transcribe-provider", metavar="PROVIDER", help=f"语音转文字模型，{stt_help}")
    p_rec.add_argument("--polish-provider", metavar="PROVIDER", help=f"转写校对模型，{llm_help}")
    p_rec.add_argument("--meeting-notes-provider", metavar="PROVIDER", help=f"纪要/总结模型，{llm_help}")

    p_tr = sub.add_parser("transcribe", help="转写已有音频文件")
    p_tr.add_argument("file", help="音频文件路径")
    p_tr.add_argument("--mode", metavar="MODE", choices=MODES, help=mode_help, default=argparse.SUPPRESS)
    p_tr.add_argument("--transcribe-provider", metavar="PROVIDER", help=f"语音转文字模型，{stt_help}")
    p_tr.add_argument("--polish-provider", metavar="PROVIDER", help=f"转写校对模型，{llm_help}")
    p_tr.add_argument("--meeting-notes-provider", metavar="PROVIDER", help=f"纪要/总结模型，{llm_help}")

    sub.add_parser(
        "ui",
        help="打开桌面图形界面（PyQt6 + Fluent；需要 python3 -m pip install PyQt6 PyQt6-Fluent-Widgets）",
    )

    p_cap = sub.add_parser(
        "captions",
        help="把已有录音回放进实时字幕引擎（不经声卡，用于对比字幕改动）")
    p_cap.add_argument("file", help="录音文件路径（.wav）")
    p_cap.add_argument("--start", type=float, default=0,
                       metavar="SEC", help="从第几秒开始（默认 0）")
    p_cap.add_argument("--seconds", type=float, default=0,
                       metavar="SEC", help="只回放这么多秒（默认整段）")
    p_cap.add_argument("--fast", action="store_true",
                       help="超实时喂音频（快但断句会与实况不同，仅适合冒烟测试）")
    p_cap.add_argument("--review", action="store_true",
                       help="结束后强制跑一次大模型批量复核（会调用 LLM）")
    p_cap.add_argument("--trace", action="store_true",
                       help="额外打印事件轨迹")
    p_cap.add_argument("--quiet", action="store_true",
                       help="过程中不打字幕，只在结束时输出汇总（便于 diff）")
    p_cap.add_argument("--wait", type=float, default=60, metavar="SEC",
                       help="等待后台修正落地的上限（默认 60 秒）")
    # Endpoint overrides exist so one A/B is one command: the whole point of
    # this subcommand is judging a threshold change on identical audio.
    p_cap.add_argument("--rule1", type=float, metavar="SEC",
                       help="覆盖断句阈值 rule1（出字前的静音时长）")
    p_cap.add_argument("--rule2", type=float, metavar="SEC",
                       help="覆盖断句阈值 rule2（已出字后的静音时长，主要旋钮）")
    p_cap.add_argument("--rule3", type=float, metavar="SEC",
                       help="覆盖断句阈值 rule3（单行最长时长上限）")

    p_dev = sub.add_parser("devices", help="列出可用音频设备")
    p_dev.add_argument(
        "--raw",
        action="store_true",
        help="打印每个设备的 ClassID / transport / sub-device UID 等原始诊断信息，并写入日志",
    )

    p_cfg = sub.add_parser("config", help="查看或修改配置")
    p_cfg.add_argument("--set", metavar="key=value", help="设置配置项")

    p_hw = sub.add_parser(
        "hotwords",
        help="扫描已有转写/纪要提取 ASR 热词，写入 config.jsonc 的 stt.funasr.hotword",
    )
    p_hw.add_argument("--no-llm", action="store_true", help="只用规则提取，不调用 LLM")
    p_hw.add_argument("--show", action="store_true", help="仅显示当前热词列表")
    p_hw.add_argument("--notion", action="store_true",
                      help="一次性从 Notion 导入成员名/项目名/术语"
                           "（需环境变量 NOTION_TOKEN）")
    p_hw.add_argument("--notion-limit", type=int, default=400, metavar="N",
                      help="Notion 候选词上限（默认 400）")
    p_hw.add_argument("--min-cjk", type=int, default=0, metavar="N",
                      help="剔除短于 N 字的纯中文候选（推荐 3，减少误替换）")
    p_hw.add_argument("--dry-run", action="store_true",
                      help="只预览将导入的词，不写入")
    p_hw.add_argument("--pin", nargs="+", metavar="TERM",
                      help="把这些词标记为永不淘汰（写入 hotword.jsonc 的 pinned）")
    p_hw.add_argument("--unpin", nargs="+", metavar="TERM",
                      help="取消这些词的永不淘汰标记")

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
        "hotwords": cmd_hotwords,
        "captions": cmd_captions,
    }

    if args.cmd in dispatch:
        dispatch[args.cmd](args, cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
