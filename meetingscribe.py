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
  --transcribe-provider    funasr（默认，本地）| openai | gemini
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


class _QuietCapture:
    """Context manager: redirect stdout+stderr into in-memory buffers, then
    forward the captured lines to _log(category, ...) on exit. Used around
    third-party libraries (FunASR) whose tqdm progress bars
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

只输出整理后的正文，不要解释修改内容。

---
【原始转写】
{transcript}
""",
    "meeting": {
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


def load_config() -> dict:
    CONFIG_DIR.mkdir(exist_ok=True)
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, encoding="utf-8") as f:
            on_disk = json.loads(_strip_jsonc_comments(f.read()))
        cfg = _deep_merge(DEFAULT_CONFIG, on_disk)
    else:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
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
        prompt = _resolve_prompt(cfg, "polish").replace("{transcript}", chunk)
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
    prompt_zh = _resolve_prompt(cfg, "notes_zh", mode=mode).replace("{transcript}", transcript)
    prompt_en = _resolve_prompt(cfg, "notes_en", mode=mode).replace("{transcript}", transcript)
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
            prompt = _resolve_prompt(cfg, "polish").replace("{transcript}", text)
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

_NOTES_SUFFIX = {"meeting": ".meeting.md", "interview": ".interview.md"}


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
    ``.interview.md``, ``.warnings.txt``, ``.meta.json``, legacy ``.md``,
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
    note_path = save_minutes(notes, audio_path, mode)
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
    note_path = save_minutes(notes, audio_path, mode)
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
            InfoBar, InfoBarPosition,
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
            # New format: <stem>.meeting.md / <stem>.interview.md
            # Legacy format: <stem>.md (still detected for backward compat)
            meeting_md = p.with_name(stem + ".meeting.md")
            interview_md = p.with_name(stem + ".interview.md")
            legacy_md = p.with_name(stem + ".md")
            # Pick the first existing artifact; meeting > interview > legacy
            if meeting_md.exists():
                md_path, md_mode = meeting_md, "meeting"
            elif interview_md.exists():
                md_path, md_mode = interview_md, "interview"
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
            "topbar.lang_zh": "中文",
            "topbar.lang_en": "EN",
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
            "rec.btn.transcribe": "语音转文字",
            "rec.btn.notes": "生成会议纪要",
            "rec.btn.interview": "生成面试报告",
            "rec.btn.open_transcribe": "打开语音转文字结果",
            "rec.btn.open_notes": "打开会议纪要",
            "rec.btn.open_interview": "打开面试报告",
            "rec.btn.cancel": "停止任务",
            "rec.no_file": "未选择录音文件",
            "rec.selected_prefix": "已选择：",
            "rec.history_title": "会议历史",
            "rec.search_placeholder": "按文件名搜索…（子串匹配）",
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
            "hist.btn.open_transcribe": "打开语音转文字结果",
            "hist.btn.open_notes": "打开会议纪要",
            "hist.btn.open_interview": "打开面试报告",
            "hist.btn.cancel": "停止任务",
            # ── Right-click context menus + delete confirmation
            "ctx.rename": "重命名…",
            "ctx.reveal": "在 Finder/资源管理器中显示",
            "ctx.delete": "删除本次会议所有记录…",
            "ctx.delete_title": "确认删除",
            "ctx.delete_confirm": "确认删除 “{name}” 及其所有相关文件（.wav / .raw.txt / .polish.txt / .meeting.md / .interview.md 等）？\n此操作不可恢复。",
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
            "pipe.log.failed": "✗ 失败：{err}",
            "pipe.log.cancel_hint": "[提示] 已停止前台显示；后台任务仍会跑完",
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
            "hist.body.notes_both": "会议纪要 + 面试报告",
            "hist.body.notes_meeting_md": "会议纪要 (.meeting.md)",
            "hist.body.notes_interview_md": "面试报告 (.interview.md)",
            "hist.body.notes_legacy_md": "会议纪要 (.md 旧格式)",
            "hist.body.polish_only": "语音转文字（已校对，.polish.txt）",
            "hist.body.raw_only": "原始转写 (.raw.txt)",
            "hist.body.no_notes": "（无总结文件）",
            "hist.body.no_polish": "（无 .polish.txt 文件）",
            "hist.body.raw_pending": "原始转写 (.raw.txt) — 等待后续校对 / 总结",
            "hist.body.pending_placeholder": "尚未生成任何转写 / 纪要文件。\n在「录音」页选择此 .wav 然后点「语音转文字」或「生成会议纪要 / 面试报告」启动流水线。",
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
            "topbar.lang_zh": "中文",
            "topbar.lang_en": "EN",
            "nav.recording": "Recording",
            "nav.history": "History",
            "nav.config": "Settings",
            "rec.title.idle": "Start Recording",
            "rec.title.recording": "Recording…",
            "rec.subtitle.idle": "Captures both system audio (speakers) and microphone by default.",
            "rec.subtitle.recording": "Click the centre button again to stop.",
            "rec.volume": "Output volume:",
            "rec.btn.transcribe": "Transcribe",
            "rec.btn.notes": "Generate meeting notes",
            "rec.btn.interview": "Generate interview report",
            "rec.btn.open_transcribe": "Open transcript",
            "rec.btn.open_notes": "Open meeting notes",
            "rec.btn.open_interview": "Open interview report",
            "rec.btn.cancel": "Stop task",
            "rec.no_file": "No recording selected",
            "rec.selected_prefix": "Selected: ",
            "rec.history_title": "Meeting history",
            "rec.search_placeholder": "Search by filename… (substring)",
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
            "hist.btn.open_transcribe": "Open transcript",
            "hist.btn.open_notes": "Open meeting notes",
            "hist.btn.open_interview": "Open interview report",
            "hist.btn.cancel": "Stop task",
            "ctx.rename": "Rename…",
            "ctx.reveal": "Show in Finder / Explorer",
            "ctx.delete": "Delete this meeting…",
            "ctx.delete_title": "Confirm delete",
            "ctx.delete_confirm": "Delete \"{name}\" and all related files (.wav / .raw.txt / .polish.txt / .meeting.md / .interview.md etc.)?\nThis cannot be undone.",
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
            "pipe.log.failed": "✗ Failed: {err}",
            "pipe.log.cancel_hint": "[info] UI detached; the background task still runs to completion.",
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
            "hist.body.notes_both": "Meeting Notes + Interview Report",
            "hist.body.notes_meeting_md": "Meeting Notes (.meeting.md)",
            "hist.body.notes_interview_md": "Interview Report (.interview.md)",
            "hist.body.notes_legacy_md": "Meeting Notes (.md, legacy)",
            "hist.body.polish_only": "Transcript (polished, .polish.txt)",
            "hist.body.raw_only": "Raw transcript (.raw.txt)",
            "hist.body.no_notes": "(no summary file)",
            "hist.body.no_polish": "(no .polish.txt file)",
            "hist.body.raw_pending": "Raw transcript (.raw.txt) — awaiting polish / notes",
            "hist.body.pending_placeholder": "No transcript / notes file yet.\nGo to Recording, pick this .wav, then click Transcribe / Generate notes / Generate interview report.",
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
        Qt signals (``progress`` / ``log`` / ``done`` / ``failed``) for the
        UI to consume."""
        progress = pyqtSignal(int)
        log = pyqtSignal(str)
        done = pyqtSignal(str)   # result md path
        failed = pyqtSignal(str)  # error message

        def __init__(self, audio_path: Path, mode: str, cfg_: dict,
                     transcribe_only: bool = False, parent=None):
            super().__init__(parent)
            self.audio_path = audio_path
            self.mode = mode
            self.cfg = cfg_
            self.transcribe_only = transcribe_only

        def run(self):
            tp = self.cfg.get("transcribe_provider", "funasr")
            pp = self.cfg.get("polish_provider", "claude")
            np_ = self.cfg.get("meeting_notes_provider", "claude")
            try:
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
                        raw_txt.write_text(raw, encoding="utf-8")
                        polish_path.write_text(polished, encoding="utf-8")
                    elif need_transcribe:
                        self.log.emit(f"[校对] 检测到 {polish_path.name}，跳过校对")
                        raw = transcribe(audio_path, tp, self.cfg, on_progress=_on_pct)
                        raw_txt.write_text(raw, encoding="utf-8")
                    elif need_polish:
                        self.log.emit(f"[转写] 检测到 {raw_txt.name}，跳过转写")
                        raw = raw_txt.read_text(encoding="utf-8")
                        polished = polish_transcript(raw, pp, self.cfg, self.mode)
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
                    raw_txt.write_text(raw, encoding="utf-8")
                    polish_path.write_text(polished, encoding="utf-8")
                elif need_transcribe:
                    self.log.emit(f"[校对] 检测到 {polish_path.name}，跳过校对")
                    raw = transcribe(audio_path, tp, self.cfg, on_progress=_on_pct)
                    raw_txt.write_text(raw, encoding="utf-8")
                    polished = polish_path.read_text(encoding="utf-8")
                elif need_polish:
                    self.log.emit(f"[转写] 检测到 {raw_txt.name}，跳过转写")
                    raw = raw_txt.read_text(encoding="utf-8")
                    polished = polish_transcript(raw, pp, self.cfg, self.mode)
                    polish_path.write_text(polished, encoding="utf-8")
                else:
                    self.log.emit(f"[转写] 检测到 {raw_txt.name}，跳过转写")
                    self.log.emit(f"[校对] 检测到 {polish_path.name}，跳过校对")
                    polished = polish_path.read_text(encoding="utf-8")

                self.progress.emit(85)
                self.log.emit("[纪要] 生成中...")
                notes = generate_notes(polished, np_, self.cfg, self.mode)
                note_path = save_minutes(notes, audio_path, self.mode)
                self.progress.emit(100)
                self.done.emit(str(note_path))
            except SystemExit as e:
                # Downstream library code (some provider error paths) may
                # call sys.exit(); without this guard the QThread dies
                # without `failed.emit`, leaving the progress bar +
                # disabled buttons frozen forever. Mirror cmd_record's
                # behaviour and route the exit code into the failure
                # signal so the UI resets cleanly.
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

        def __init__(self, parent=None):
            super().__init__(parent)
            self._recorder: "MultiStreamRecorder | None" = None
            self._audio_path: "Path | None" = None
            self._plan: "AudioPlan | None" = None
            self._start_time = 0.0
            self._status = "idle"
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
            try:
                recorder.start()
            except Exception as e:
                _log("ERR", f"Qt recorder.start: {type(e).__name__}: {e}")
                self.warning.emit(f"录音设备启动失败: {e}")
                return False
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
            self._pipeline_thread: "QThread | None" = None
            self._pipeline_worker: "_PipelineWorker | None" = None
            # i18n: each callback re-applies its widget's text on lang switch.
            self._lang_callbacks: list = []
            self._current_status = "idle"
            self._build_ui()
            self._wire()
            self._refresh_history()
            self._refresh_action_buttons()
            # Try to align the slider + label with the real device volume
            # immediately so the first paint is already in sync. If the audio
            # monitor hasn't resolved the device yet, this is a silent no-op
            # and the 300 ms retry picks it up.
            try:
                self._sync_vol_slider()
            except Exception as e:
                _log("ERR", f"Qt initial vol sync: {type(e).__name__}: {e}")
            QTimer.singleShot(300, self._sync_vol_slider)

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
            if self._last_recorded:
                self.chosen_label.setText(
                    f"{_t('rec.selected_prefix')}{self._last_recorded.name}")
            else:
                self.chosen_label.setText(_t("rec.no_file"))
            self._refresh_history()
            # Action button labels depend on (current language) × (artifact
            # exists on disk) — refresh after the static callbacks so the
            # "open X" override wins over the default "generate X" text.
            self._refresh_action_buttons()

        # ── UI build
        def _build_ui(self):
            root = QHBoxLayout(self)
            root.setContentsMargins(28, 28, 28, 28)
            root.setSpacing(24)

            # Left column — recording controls
            left = QFrame(self)
            lv = QVBoxLayout(left)
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
            self.rec_btn.setIconSize(QSize(56, 56))
            self.rec_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._apply_rec_btn_style("idle")

            mic_row = QHBoxLayout()
            mic_row.addStretch(1)
            mic_row.addWidget(self.rec_btn)
            mic_row.addStretch(1)

            self.timer_label = TitleLabel("00:00:00", self)
            self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # No mic-selector row — the app always captures system audio
            # (BlackHole) + the resolver-chosen mic (external > built-in).
            # The active devices appear in the post-recording log instead.

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

            actions = QHBoxLayout()
            actions.addStretch(1)
            actions.addWidget(self.transcribe_btn)
            actions.addWidget(self.notes_btn)
            actions.addWidget(self.interview_btn)
            actions.addStretch(1)

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
            self.log_view.setVisible(False)

            lv.addWidget(self.title_label)
            lv.addWidget(self.subtitle_label)
            lv.addSpacing(12)
            lv.addLayout(mic_row)
            lv.addWidget(self.timer_label)
            lv.addSpacing(8)
            lv.addLayout(vol_row)
            lv.addSpacing(8)
            lv.addLayout(actions)
            lv.addLayout(choose_row)
            lv.addWidget(self.progress_bar)
            lv.addWidget(self.cancel_btn)
            lv.addWidget(self.log_view)
            lv.addStretch(1)

            # Right column — history sidebar
            right = QFrame(self)
            right.setMinimumWidth(320)
            right.setMaximumWidth(420)
            rv = QVBoxLayout(right)
            rv.setContentsMargins(0, 0, 0, 0)
            rv.setSpacing(12)

            self._history_title = SubtitleLabel("", self)
            self._i18n(self._history_title, "rec.history_title")
            rv.addWidget(self._history_title)
            self.search_sidebar = SearchLineEdit(self)
            self._i18n(self.search_sidebar, "rec.search_placeholder",
                       attr="setPlaceholderText")
            rv.addWidget(self.search_sidebar)

            self.history_list = ListWidget(self)
            self.history_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            rv.addWidget(self.history_list, stretch=1)

            # Wrap each column in a soft rounded card so the two regions
            # read as visually separate panels.
            _style_as_card(left, padding=20, name="recordingCardLeft")
            _style_as_card(right, padding=16, name="recordingCardRight")

            root.addWidget(left, stretch=2)
            root.addWidget(right, stretch=1)

        # ── Wiring
        def _wire(self):
            self.rec_btn.clicked.connect(self._on_record_clicked)
            self.transcribe_btn.clicked.connect(self._on_transcribe_clicked)
            self.notes_btn.clicked.connect(self._on_meeting_clicked)
            self.interview_btn.clicked.connect(self._on_interview_clicked)
            self.cancel_btn.clicked.connect(self._on_cancel_pipeline)
            self.vol_slider.valueChanged.connect(self._on_vol_changed)

            self.state.status_changed.connect(self._on_status_changed)
            self.state.elapsed_changed.connect(self._on_elapsed_changed)
            self.state.warning.connect(self._on_warning)

            self.search_sidebar.textChanged.connect(lambda _t: self._refresh_history())
            self.history_list.itemClicked.connect(self._on_history_pick)
            self.history_list.customContextMenuRequested.connect(
                self._on_history_context_menu)

            monitor = _get_audio_monitor()
            monitor.on_recording_plan_change = self._on_plan_change

        def _refresh_history(self):
            self.history_list.clear()
            q = (self.search_sidebar.text() or "").strip().lower()
            for m in _list_recordings():
                # Case-insensitive substring match against the full stem
                # (timestamp + optional custom name) so the user can type
                # either "0518" or "客户访谈" to filter.
                if q and q not in m["stem"].lower():
                    continue
                label = _format_timestamp_label(m["stem"])
                if m["has_md"]:
                    badge = _t("hist.badge.done")
                elif m["has_raw"] or m["has_polish"]:
                    badge = _t("hist.badge.pending_summary")
                else:
                    badge = _t("hist.badge.audio_only")
                item = QListWidgetItem(f"{label}\n{badge}")
                item.setData(Qt.ItemDataRole.UserRole, m)
                self.history_list.addItem(item)

        # Right-click on a history list item → rename / open enclosing folder
        # / refresh. Renames cascade across every sibling file sharing the
        # same stem, via the module-level _rename_meeting_files helper.
        def _on_history_context_menu(self, pos: "QPoint"):
            item = self.history_list.itemAt(pos)
            if item is None:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            if not data:
                return
            menu = QMenu(self.history_list)
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
                    return  # no change
                new_wav = _rename_meeting_files(data["wav_path"], clean)
                if new_wav is None:
                    QMessageBox.warning(self, _t("ctx.rename_failed_title"),
                                        _t("ctx.rename_failed_collision"))
                    return
                # If the renamed file is currently the chosen target for
                # pipeline / open-result actions, follow it.
                if self._last_recorded == data["wav_path"]:
                    self._last_recorded = new_wav
                    self.chosen_label.setText(f"{_t('rec.selected_prefix')}{new_wav.name}")
                self._refresh_history()
                self._refresh_action_buttons()

            def _do_reveal():
                _open_path(data["wav_path"].parent)

            def _do_delete():
                wav = data["wav_path"]
                # Guard: if a pipeline is currently writing files for this
                # meeting (or any meeting from this view's worker), deletion
                # would yank files from under the worker → crash.
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
                # If the deleted file was the chosen target, clear selection.
                if self._last_recorded == wav:
                    self._last_recorded = None
                    self._result_path = None
                    self.chosen_label.setText(_t("rec.no_file"))
                self._refresh_history()
                self._refresh_action_buttons()

            act_rename.triggered.connect(_do_rename)
            act_reveal.triggered.connect(_do_reveal)
            act_delete.triggered.connect(_do_delete)
            menu.exec(self.history_list.mapToGlobal(pos))

        # ── Volume slider
        def _on_vol_changed(self, val: int):
            self.vol_pct.setText(f"{val}%")
            if self._vol_device:
                try:
                    set_device_volume(self._vol_device, val / 100.0)
                except Exception as e:
                    _log("ERR", f"Qt set_volume: {type(e).__name__}: {e}")

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
            self._refresh_history()
            QTimer.singleShot(200, self._sync_vol_slider)

        def _on_status_changed(self, s: str):
            self._current_status = s
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
            new_dev = plan.restore_output_name
            if not new_dev or new_dev == self._vol_device:
                return
            self._vol_device = new_dev
            QTimer.singleShot(0, self._sync_vol_slider)

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
            worker.done.connect(thread.quit)
            worker.failed.connect(thread.quit)
            thread.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            self._pipeline_worker = worker
            self._pipeline_thread = thread
            thread.start()

        def _on_pipeline_done(self, path_str: str):
            self._reset_after_pipeline()
            self._ui_log(_t("pipe.log.done", path=path_str))
            self._result_path = Path(path_str)
            self._refresh_action_buttons()
            self._refresh_history()

        def _on_pipeline_failed(self, msg: str):
            self._reset_after_pipeline()
            self._ui_log(_t("pipe.log.failed", err=msg))

        def _reset_after_pipeline(self):
            self.progress_bar.setVisible(False)
            self.cancel_btn.setVisible(False)
            for b in (self.rec_btn, self.transcribe_btn, self.notes_btn,
                      self.interview_btn):
                b.setEnabled(True)
            self.state.set_status("idle")

        def _on_cancel_pipeline(self):
            # transcribe/polish/generate don't have built-in cancel hooks;
            # we simply detach the UI. The thread keeps running until natural
            # completion (and any partial writes are kept on disk so a re-run
            # resumes from the checkpoint).
            self._ui_log(_t("pipe.log.cancel_hint"))
            self._reset_after_pipeline()

        def _on_history_pick(self, item: QListWidgetItem):
            data = item.data(Qt.ItemDataRole.UserRole)
            if not data:
                return
            self._last_recorded = data["wav_path"]
            self.chosen_label.setText(f"{_t('rec.selected_prefix')}" + data['wav_path'].name)
            self._result_path = data.get("md_path")
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
                for b in (self.transcribe_btn, self.notes_btn,
                          self.interview_btn):
                    _apply_open_btn_style(b, is_open=False)
                return
            polish = audio.with_name(audio.stem + ".polish.txt")
            meeting_md = audio.with_name(audio.stem + ".meeting.md")
            interview_md = audio.with_name(audio.stem + ".interview.md")
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
            for b in (self.h_transcribe_btn, self.h_notes_btn, self.h_interview_btn):
                b.setEnabled(False)  # no meeting selected at startup
            self._refresh_h_action_buttons()
            actions_row = QHBoxLayout()
            actions_row.addStretch(1)
            actions_row.addWidget(self.h_transcribe_btn)
            actions_row.addWidget(self.h_notes_btn)
            actions_row.addWidget(self.h_interview_btn)
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
                for b in (self.h_transcribe_btn, self.h_notes_btn,
                          self.h_interview_btn):
                    _apply_open_btn_style(b, is_open=False)
                return
            polish_p = m.get("polish_path")
            meeting_p = m.get("meeting_md_path")
            interview_p = m.get("interview_md_path")
            # Legacy single-`.md` artefact (`md_path` is set + `md_mode` is
            # None means a pre-split `.md` file). Counts as "meeting notes
            # exist" for the open-toggle so the user can read the existing
            # summary without re-running the pipeline.
            legacy_p = (
                m.get("md_path") if (meeting_p is None and interview_p is None
                                     and m.get("has_md")) else None
            )
            for btn, exists, gen_key, open_key in (
                (self.h_transcribe_btn, polish_p is not None,
                 "hist.btn.transcribe", "hist.btn.open_transcribe"),
                (self.h_notes_btn, meeting_p is not None or legacy_p is not None,
                 "hist.btn.notes", "hist.btn.open_notes"),
                (self.h_interview_btn, interview_p is not None,
                 "hist.btn.interview", "hist.btn.open_interview"),
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
                      self.h_interview_btn):
                b.setEnabled(not running)
            # Refresh "generate / open" label + style for the new selection.
            self._refresh_h_action_buttons()

        def _render_body_for_mode(self, m: dict, mode: str):
            meeting_p = m.get("meeting_md_path")
            interview_p = m.get("interview_md_path")
            polish_p = m.get("polish_path")
            raw_p = m.get("raw_path")
            legacy_p = m.get("md_path") if (m.get("md_mode") is None and m.get("has_md")) else None

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
                if legacy_p and not (meeting_p or interview_p):
                    parts.append(_read(legacy_p))
                if parts:
                    self.body_browser.setMarkdown("\n\n---\n\n".join(parts))
                    self.body_title.setText(
                        _t("hist.body.notes_meeting") if not interview_p else
                        _t("hist.body.notes_interview") if not meeting_p else
                        _t("hist.body.notes_both")
                    )
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

            # mode == "all" (default): show the richest available artifact
            if meeting_p and interview_p:
                self.body_browser.setMarkdown(
                    f"# 📝 {_t('hist.body.notes_meeting')}\n\n{_read(meeting_p)}\n\n---\n\n"
                    f"# 🎤 {_t('hist.body.notes_interview')}\n\n{_read(interview_p)}"
                )
                self.body_title.setText(_t("hist.body.notes_both"))
            elif meeting_p:
                self.body_browser.setMarkdown(_read(meeting_p))
                self.body_title.setText(_t("hist.body.notes_meeting_md"))
            elif interview_p:
                self.body_browser.setMarkdown(_read(interview_p))
                self.body_title.setText(_t("hist.body.notes_interview_md"))
            elif legacy_p:
                self.body_browser.setMarkdown(_read(legacy_p))
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
                              self.h_interview_btn):
                        b.setEnabled(False)
                self.refresh()

            act_rename.triggered.connect(_do_rename)
            act_reveal.triggered.connect(_do_reveal)
            act_delete.triggered.connect(_do_delete)
            menu.exec(self.list_w.mapToGlobal(pos))

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
                      self.h_interview_btn):
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
            worker.done.connect(thread.quit)
            worker.failed.connect(thread.quit)
            thread.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            self._pipeline_worker = worker
            self._pipeline_thread = thread
            thread.start()

        def _on_pipeline_done(self, path_str: str):
            self._reset_after_pipeline()
            self.h_log_view.append(_t("pipe.log.done", path=path_str))
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

        def _reset_after_pipeline(self):
            self.h_progress.setVisible(False)
            self.h_cancel_btn.setVisible(False)
            self._pipeline_worker = None
            self._pipeline_thread = None
            # Re-enable action buttons iff a meeting is currently selected.
            has_selection = self._current is not None
            for b in (self.h_transcribe_btn, self.h_notes_btn,
                      self.h_interview_btn):
                b.setEnabled(has_selection)
            # A just-completed pipeline may have produced new artifacts —
            # re-render the "generate / open" toggle.
            self._refresh_h_action_buttons()

        def _on_cancel_pipeline(self):
            # transcribe/polish/generate don't have built-in cancel hooks;
            # we detach the UI. The thread keeps running until natural
            # completion (any partial writes stay on disk so a re-run
            # resumes from the checkpoint).
            self.h_log_view.append(_t("pipe.log.cancel_hint"))
            self._reset_after_pipeline()

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
            # Start on recording view.
            self.stack.setCurrentWidget(self.recording_view)
            self.nav.setCurrentItem(self.recording_view.objectName())
            self.stack.currentChanged.connect(self._on_view_changed)

            # ── Top language-switch bar ──────────────────────────────────
            # Right-aligned single 中文 / EN toggle, sits above the stacked
            # views. The button label shows the CURRENT language ("中文" in
            # zh mode, "EN" in en mode) rendered in the accent-blue style.
            # Clicking it flips `_LANG["current"]` to the other language
            # and calls `apply_language()` on every view + the nav.
            self.topbar = QWidget(self)
            self.topbar.setObjectName("langTopbar")
            tb = QHBoxLayout(self.topbar)
            tb.setContentsMargins(16, 8, 16, 8)
            tb.setSpacing(6)
            tb.addStretch(1)
            self.lang_toggle_btn = PushButton("", self.topbar)
            self.lang_toggle_btn.setMinimumWidth(72)
            self.lang_toggle_btn.clicked.connect(self._toggle_language)
            tb.addWidget(self.lang_toggle_btn)

            right_col = QWidget(self)
            rc = QVBoxLayout(right_col)
            rc.setContentsMargins(0, 0, 0, 0)
            rc.setSpacing(0)
            rc.addWidget(self.topbar)
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

            self._apply_lang_button_style()

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
            """Flip between zh ↔ en on every click of the single toggle
            button. Called from the topbar button's `clicked` signal."""
            self._set_language("en" if _LANG["current"] == "zh" else "zh")

        def _set_language(self, lang: str):
            """Flip the process-wide language and re-render every label."""
            if lang not in ("zh", "en") or lang == _LANG["current"]:
                self._apply_lang_button_style()
                return
            _LANG["current"] = lang
            self._apply_lang_button_style()
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

        def _apply_lang_button_style(self):
            """Render the single toggle button: label = the CURRENT language
            ("中文" in zh mode, "EN" in en mode) with the accent-blue fill,
            so the user can see at a glance which mode they're in. Clicking
            flips to the other language via `_toggle_language`."""
            key = "topbar.lang_zh" if _LANG["current"] == "zh" else "topbar.lang_en"
            self.lang_toggle_btn.setText(_t(key))
            self.lang_toggle_btn.setStyleSheet(
                "PushButton {"
                "  background-color: #0066d2;"
                "  color: white;"
                "  border: 1px solid #0066d2;"
                "  border-radius: 6px;"
                "  padding: 4px 14px;"
                "}"
                "PushButton:hover { background-color: #1577e0; }"
                "PushButton:pressed { background-color: #004fa5; }"
            )

        def _on_view_changed(self, _idx):
            w = self.stack.currentWidget()
            if w is self.history_view:
                self.history_view.refresh()
            elif w is self.recording_view:
                self.recording_view._refresh_history()
            elif w is self.config_view:
                self.config_view._load_into_widgets()
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

    sub.add_parser(
        "ui",
        help="打开桌面图形界面（PyQt6 + Fluent；需要 python3 -m pip install PyQt6 PyQt6-Fluent-Widgets）",
    )

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
