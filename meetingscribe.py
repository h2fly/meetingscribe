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
  # 传 .wav：若同目录已有 .raw.txt 则自动跳过 Whisper，直接校对 + 纪要
  python3 meetingscribe.py transcribe audio.wav
  python3 meetingscribe.py transcribe audio.wav --mode interview
  python3 meetingscribe.py transcribe audio.wav --transcribe-provider openai
  python3 meetingscribe.py transcribe audio.wav --polish-provider gemini
  python3 meetingscribe.py transcribe audio.wav --meeting-notes-provider openai
  python3 meetingscribe.py transcribe audio.wav --transcribe-provider openai --polish-provider gemini --meeting-notes-provider claude
  python3 meetingscribe.py transcribe audio.wav --mode interview --meeting-notes-provider gemini

  # 传 .raw.txt：直接跳过 Whisper，从校对步骤开始
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
  python3 meetingscribe.py config --set polish_max_workers=5   # 校对并发数，0=不限

━━━ 各环节 provider 可选值 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  --transcribe-provider    whisper（默认，本地）| openai | gemini
  --polish-provider        claude（默认）| openai | gemini
  --meeting-notes-provider claude（默认）| openai | gemini

━━━ 输出文件（与录音 / 输入文件同目录）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  <stem>.wav            录音文件
  <stem>.raw.txt        Whisper 原始转写
  <stem>.proofread.txt  校对后转写（会议纪要/面试总结的输入）
  <stem>.md             会议纪要 / 面试总结
"""

import argparse
import json
import math
import os
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import subprocess

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ── 配置 ──────────────────────────────────────────────────────────────────────

CONFIG_DIR = Path.home() / "Documents" / "meetingscribe"
CONFIG_FILE = Path(__file__).parent / "config.jsonc"

DEFAULT_CONFIG = {
    "sample_rate": 48000,
    "channels": 3,
    "output_record": None,
    "output_restore": None,
    "device_system_audio": "BlackHole 2ch",
    "device_mic": "MacBook Air Microphone",
    # 模式：meeting（会议纪要）| interview（面试总结）
    "mode": "meeting",
    # LLM 调用超时（秒），长会议建议调大
    "llm_timeout": 600,
    # 校对时单块最大字符数，超出则分块处理
    "polish_chunk_size": 3000,
    # 校对并发数（同时调用 LLM 的块数），0 = 不限
    "polish_max_workers": 8,
    # 转写 / 校对 / 纪要各自使用的 provider
    "transcribe_provider": "whisper",
    "polish_provider": "claude",
    "meeting_notes_provider": "claude",
    # 语音转文字 provider 配置
    "stt": {
        "whisper": {
            "model": "base",        # tiny / base / small / medium / large-v3
            "chunk_secs": 300,      # 超过此时长自动分块并行（秒），0 = 始终串行
            "workers": 2,           # 并行实例数；内存占用 = workers × 模型大小
            "cpu_threads": 0,       # 每个实例的内部线程数；0 = 自动（CPU 核数 / 2）
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
        cfg = DEFAULT_CONFIG.copy()
    return cfg


def save_config(cfg: dict):
    CONFIG_DIR.mkdir(exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# ── 音频输出切换 ──────────────────────────────────────────────────────────────

def switch_output(device_name: str):
    """Switch default audio output device. macOS only; no-op on other platforms."""
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

    # get all device IDs
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
                    kSystem, ctypes.byref(_Addr(_fcc("dOut"), kGlobal, 0)),
                    0, None, ctypes.c_uint32(4), ctypes.byref(val),
                )
                return
    print(f"[警告] 找不到输出设备: {device_name}", file=sys.stderr)


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

        # Drop silent channels; always write stereo (Whisper handles stereo fine)
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


# ── 双路录音（静音模式）─────────────────────────────────────────────────────────

class DualStreamRecorder:
    """同时从 BlackHole（系统音频）和麦克风两路录制，扬声器保持静音也能正常捕获。"""

    def __init__(self, sys_device: str, mic_device: str, sample_rate: int):
        self.sys_device = sys_device
        self.mic_device = mic_device
        self.sample_rate = sample_rate
        self._sys_frames: list[np.ndarray] = []
        self._mic_frames: list[np.ndarray] = []
        self._sys_stream = None
        self._mic_stream = None
        self.recording = False

    def _sys_cb(self, indata, frames, time_info, status):
        if self.recording:
            self._sys_frames.append(indata.copy())

    def _mic_cb(self, indata, frames, time_info, status):
        if self.recording:
            self._mic_frames.append(indata.copy())

    def start(self):
        self.recording = True
        self._sys_frames = []
        self._mic_frames = []
        block = int(self.sample_rate * 0.1)
        self._sys_stream = sd.InputStream(
            device=self.sys_device, samplerate=self.sample_rate,
            channels=2, dtype="float32", callback=self._sys_cb, blocksize=block,
        )
        self._mic_stream = sd.InputStream(
            device=self.mic_device, samplerate=self.sample_rate,
            channels=1, dtype="float32", callback=self._mic_cb, blocksize=block,
        )
        try:
            self._sys_stream.start()
            self._mic_stream.start()
        except Exception:
            self.stop()
            raise

    def stop(self):
        self.recording = False
        for s in (self._sys_stream, self._mic_stream):
            if s:
                try:
                    s.stop()
                finally:
                    s.close()

    def save(self, path: Path) -> bool:
        if not self._sys_frames and not self._mic_frames:
            return False

        # 两路帧数对齐（丢弃尾部多余帧）
        sys_audio = np.concatenate(self._sys_frames) if self._sys_frames else None
        mic_audio = np.concatenate(self._mic_frames) if self._mic_frames else None

        if sys_audio is not None and mic_audio is not None:
            n = min(len(sys_audio), len(mic_audio))
            sys_ch = sys_audio[:n, 0]   # BlackHole L（系统音频，取单声道）
            mic_ch = mic_audio[:n, 0]   # 麦克风
            # stereo: L=系统音频, R=麦克风
            mixed = np.column_stack([sys_ch, mic_ch])
        elif sys_audio is not None:
            ch = sys_audio[:, 0]
            mixed = np.column_stack([ch, ch])
        else:
            ch = mic_audio[:, 0]
            mixed = np.column_stack([ch, ch])

        audio_int16 = (np.clip(mixed, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return True


# ── 转写 ──────────────────────────────────────────────────────────────────────

def transcribe(audio_path: Path, provider: str, cfg: dict, on_progress=None) -> str:
    stt_cfgs = {**DEFAULT_CONFIG["stt"], **cfg.get("stt", {})}
    pcfg = stt_cfgs.get(provider)
    if pcfg is None:
        print(f"[错误] 未知转写 provider '{provider}'，请在 config stt 中配置")
        sys.exit(1)

    print(f"[转写] 使用 {provider} 转写 {audio_path.name} ...")
    if provider == "whisper":
        return _transcribe_whisper(audio_path, pcfg, on_progress=on_progress)
    elif provider == "openai":
        return _transcribe_openai(audio_path, pcfg)
    elif provider == "gemini":
        return _transcribe_gemini(audio_path, pcfg)
    else:
        print(f"[错误] 不支持的转写 provider: {provider}")
        sys.exit(1)


def _transcribe_whisper(audio_path: Path, pcfg: dict, on_progress=None) -> str:
    import concurrent.futures, tempfile

    model_size  = pcfg.get("model", "base")
    chunk_secs  = int(pcfg.get("chunk_secs", 300))
    max_workers     = max(1, int(pcfg.get("workers", 2)))
    _cpu_threads_cfg = int(pcfg.get("cpu_threads", 0))
    cpu_threads      = _cpu_threads_cfg if _cpu_threads_cfg > 0 else max(1, (os.cpu_count() or 4) // 2)

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
    print(
        f"[转写] 录音时长 {total_secs / 60:.1f} 分钟，分 {n_chunks} 块并发转写"
        f"（每块 {chunk_secs // 60} 分钟，并发 {actual_workers}）"
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
    except urllib.error.HTTPError as e:
        print(f"[错误] Gemini STT HTTP {e.code}: {e.read().decode()}")
        sys.exit(1)
    except urllib.error.URLError as e:
        if isinstance(e.reason, (TimeoutError, OSError)) and "timed out" in str(e.reason).lower():
            print("[错误] Gemini STT 请求超时，录音文件可能过大")
        else:
            print(f"[错误] Gemini STT 网络错误: {e.reason}")
        sys.exit(1)
    except (ValueError, KeyError) as e:
        print(f"[错误] Gemini STT 返回格式异常: {e}")
        sys.exit(1)
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


# ── Prompts ───────────────────────────────────────────────────────────────────

_POLISH_BASE = """\
以下是语音识别自动转写的文本，带有时间戳，可能存在错别字、同音字混淆、断句不当等问题。

请在**不改变原意**的前提下：
1. 去掉所有时间戳（如 [00.0s]）
2. 将所有片段合并为连贯的自然段落，按语义分段
3. 纠正明显的错别字和同音字错误
4. 修复错误的断句和标点
5. 无法确定的内容用【？】标注

只输出整理后的正文，不要解释修改内容。

---
【原始转写】
{transcript}
"""

PROMPTS = {
    "meeting": {
        "polish": "你是一位专业的文字校对助手，正在处理一段会议录音的转写文本。\n\n" + _POLISH_BASE,
        "notes": """\
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
    },
    "interview": {
        "polish": "你是一位专业的文字校对助手，正在处理一段面试录音的转写文本。如能区分面试官与候选人，请在段落前标注「面试官：」或「候选人：」。\n\n" + _POLISH_BASE,
        "notes": """\
你是一位专业的面试评估助手。请根据以下面试转写文本生成结构化的面试总结。

要求：
1. **候选人概况** — 姓名（如提及）、应聘岗位、整体印象（2~3 句）
2. **核心问答摘要** — 按主题归纳关键问题与候选人的回答要点
3. **技术 / 专业能力** — 具体技能掌握程度、深度、广度
4. **综合素质** — 沟通表达、逻辑思维、学习能力、团队意识等
5. **亮点** — 突出表现或印象深刻的回答
6. **不足 / 待确认** — 回答模糊、经验欠缺或需进一步了解的方面
7. **综合评价与建议** — 是否推荐进入下一轮，及理由

用中文输出，格式为 Markdown，客观专业。若内容较短或不完整，如实说明。

---
【面试转写】
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
        return _llm_claude_cli(prompt, label, timeout)
    elif ptype == "openai":
        return _llm_openai(prompt, pcfg, label, timeout)
    elif ptype == "gemini":
        return _llm_gemini(prompt, pcfg, label, timeout)
    else:
        print(f"[错误] 不支持的 provider type: {ptype}")
        sys.exit(1)


def _llm_claude_cli(prompt: str, label: str, timeout: int) -> str:
    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
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
    except urllib.error.HTTPError as e:
        print(f"[错误] OpenAI HTTP {e.code}（{label}）: {e.read().decode()}")
        sys.exit(1)
    except urllib.error.URLError as e:
        if isinstance(e.reason, (TimeoutError, OSError)) and "timed out" in str(e.reason).lower():
            print(f"[错误] OpenAI 超时（{label}，{timeout}s）；可通过 config --set llm_timeout=900 调大")
        else:
            print(f"[错误] OpenAI 网络错误（{label}）: {e.reason}")
        sys.exit(1)
    except (ValueError, KeyError) as e:
        print(f"[错误] OpenAI 返回格式异常（{label}）: {e}")
        sys.exit(1)
    return data["choices"][0]["message"]["content"].strip()


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
    except urllib.error.HTTPError as e:
        print(f"[错误] Gemini HTTP {e.code}（{label}）: {e.read().decode()}")
        sys.exit(1)
    except urllib.error.URLError as e:
        if isinstance(e.reason, (TimeoutError, OSError)) and "timed out" in str(e.reason).lower():
            print(f"[错误] Gemini 超时（{label}，{timeout}s）；可通过 config --set llm_timeout=900 调大")
        else:
            print(f"[错误] Gemini 网络错误（{label}）: {e.reason}")
        sys.exit(1)
    except (ValueError, KeyError) as e:
        print(f"[错误] Gemini 返回格式异常（{label}）: {e}")
        sys.exit(1)
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


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
    max_workers = cfg.get("polish_max_workers", DEFAULT_CONFIG["polish_max_workers"]) or total
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
    label = "面试总结" if mode == "interview" else "会议纪要"
    print(f"[{label}] 调用 {provider} 生成{label}...")
    prompt = PROMPTS[mode]["notes"].format(transcript=transcript)
    return _llm_run(prompt, provider, cfg, label)


# ── 保存纪要 ──────────────────────────────────────────────────────────────────

def save_minutes(minutes: str, audio_path: Path) -> Path:
    note_path = audio_path.with_suffix(".md")
    note_path.write_text(minutes, encoding="utf-8")
    return note_path


# ── 子命令 ────────────────────────────────────────────────────────────────────

def cmd_devices(args, cfg):
    sys_dev = cfg.get("device_system_audio", "")
    mic_dev = cfg.get("device_mic", "")
    print("\n可用音频输入设备：\n")
    for dev in sd.query_devices():
        if dev["max_input_channels"] < 1:
            continue
        name = dev["name"]
        hint = ""
        if "aggregate" in name.lower() or "聚合" in name or "omi" in name.lower():
            hint = "  ◀ 推荐（聚合设备）"
        elif "blackhole" in name.lower():
            hint = "  ◀ BlackHole"
        tags = []
        if name == sys_dev:
            tags.append("系统音频")
        if name == mic_dev:
            tags.append("麦克风")
        active = f" [当前: {'/'.join(tags)}]" if tags else ""
        print(f"  {name}{hint}{active}")
    print()


def cmd_record(args, cfg):
    recordings_dir = CONFIG_DIR / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_path = recordings_dir / f"{ts}.wav"

    out_record = cfg.get("output_record")
    out_restore = cfg.get("output_restore")

    recorder = DualStreamRecorder(
        sys_device=cfg.get("device_system_audio", "BlackHole 2ch"),
        mic_device=cfg.get("device_mic", "MacBook Air Microphone"),
        sample_rate=cfg["sample_rate"],
    )
    print(f"\n[录音] {cfg.get('device_system_audio')}（系统音频）+ {cfg.get('device_mic')}（麦克风）")

    if out_record:
        switch_output(out_record)
        print(f"[音频] 输出已切换至: {out_record}")
        print("[音频] 等待 1 秒让播放器重新路由...")
        time.sleep(1)

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

    if out_restore:
        switch_output(out_restore)
        print(f"[音频] 输出已还原至: {out_restore}")

    if not recorder.save(audio_path):
        print("[错误] 未录到任何音频")
        sys.exit(1)

    print(f"[录音] 完成，时长 {duration:.0f}s → {audio_path}\n")

    mode = getattr(args, "mode", None) or cfg.get("mode", "meeting")
    if mode not in PROMPTS:
        print(f"[错误] 未知模式 '{mode}'，可选：{list(PROMPTS)}")
        sys.exit(1)

    transcribe_provider  = getattr(args, "transcribe_provider", None)  or cfg.get("transcribe_provider", "whisper")
    polish_provider      = getattr(args, "polish_provider", None)       or cfg.get("polish_provider", "claude")
    notes_provider       = getattr(args, "meeting_notes_provider", None) or cfg.get("meeting_notes_provider", "claude")

    raw_txt_path = audio_path.with_name(audio_path.stem + ".raw.txt")
    proofread_path = audio_path.with_name(audio_path.stem + ".proofread.txt")

    if raw_txt_path.exists():
        print(f"[转写] 检测到已有转写文件 {raw_txt_path.name}，跳过 Whisper")
        transcript_raw = raw_txt_path.read_text(encoding="utf-8")
    else:
        transcript_raw = transcribe(audio_path, transcribe_provider, cfg)
        raw_txt_path.write_text(transcript_raw, encoding="utf-8")

    if proofread_path.exists():
        print(f"[校对] 检测到已有校对文件 {proofread_path.name}，跳过校对")
        transcript_polished = proofread_path.read_text(encoding="utf-8")
    else:
        transcript_polished = polish_transcript(transcript_raw, polish_provider, cfg, mode)
        proofread_path.write_text(transcript_polished, encoding="utf-8")

    print(f"\n[校对] 已保存: {proofread_path}")
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
    input_path = Path(args.file)
    if not input_path.exists():
        print(f"[错误] 文件不存在: {input_path}")
        sys.exit(1)

    mode = getattr(args, "mode", None) or cfg.get("mode", "meeting")
    if mode not in PROMPTS:
        print(f"[错误] 未知模式 '{mode}'，可选：{list(PROMPTS)}")
        sys.exit(1)

    transcribe_provider  = getattr(args, "transcribe_provider", None)   or cfg.get("transcribe_provider", "whisper")
    polish_provider      = getattr(args, "polish_provider", None)        or cfg.get("polish_provider", "claude")
    notes_provider       = getattr(args, "meeting_notes_provider", None) or cfg.get("meeting_notes_provider", "claude")

    # 支持直接传入 .raw.txt 或 .proofread.txt，或传 .wav 自动检测跳过已完成步骤
    if input_path.suffix == ".txt" and input_path.stem.endswith(".proofread"):
        # 直接传入 .proofread.txt，跳过转写和校对
        audio_path = input_path.with_name(input_path.stem[:-10] + ".wav")
        proofread_path = input_path
        print(f"[校对] 使用已有校对文件: {proofread_path.name}（跳过转写和校对）")
        transcript_polished = proofread_path.read_text(encoding="utf-8")
        transcript_raw = None
    elif input_path.suffix == ".txt" and input_path.stem.endswith(".raw"):
        # 直接传入 .raw.txt，跳过 Whisper
        audio_path = input_path.with_name(input_path.stem[:-4] + ".wav")
        raw_txt_path = input_path
        proofread_path = audio_path.with_name(audio_path.stem + ".proofread.txt")
        print(f"[转写] 使用已有转写文件: {raw_txt_path.name}（跳过 Whisper）")
        transcript_raw = raw_txt_path.read_text(encoding="utf-8")
        transcript_polished = None
    else:
        audio_path = input_path
        raw_txt_path = audio_path.with_name(audio_path.stem + ".raw.txt")
        proofread_path = audio_path.with_name(audio_path.stem + ".proofread.txt")
        if raw_txt_path.exists():
            print(f"[转写] 检测到已有转写文件 {raw_txt_path.name}，跳过 Whisper")
            transcript_raw = raw_txt_path.read_text(encoding="utf-8")
        else:
            transcript_raw = transcribe(audio_path, transcribe_provider, cfg)
            raw_txt_path.write_text(transcript_raw, encoding="utf-8")
        transcript_polished = None

    if transcript_polished is None:
        if proofread_path.exists():
            print(f"[校对] 检测到已有校对文件 {proofread_path.name}，跳过校对")
            transcript_polished = proofread_path.read_text(encoding="utf-8")
        else:
            transcript_polished = polish_transcript(transcript_raw, polish_provider, cfg, mode)
            proofread_path.write_text(transcript_polished, encoding="utf-8")

    notes_label = "面试总结" if mode == "interview" else "会议纪要"
    print(f"\n[校对] 已保存: {proofread_path}")
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

def cmd_ui(args, cfg):  # noqa: C901
    import queue as _q
    import threading as _t
    try:
        import tkinter as tk
        from tkinter import ttk, scrolledtext, filedialog
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
        ),
    }

    def t(key):
        return TR[st["lang"]][key]

    # ── 根窗口 ─────────────────────────────────────────────────────────────────
    root = tk.Tk()
    root.title("MeetingScribe")
    root.geometry("580x720")
    root.resizable(False, False)
    root.configure(bg=BG)

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TProgressbar",
                    troughcolor=BORDER, background=ACCENT,
                    darkcolor=ACCENT, lightcolor=ACCENT, thickness=4)

    def sep(parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=20, pady=6)

    def card(parent, pady=(0, 0)):
        outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
        outer.pack(fill="x", padx=20, pady=pady)
        inner = tk.Frame(outer, bg=CARD, padx=18, pady=14)
        inner.pack(fill="x")
        return inner

    # ── 辅助函数 ───────────────────────────────────────────────────────────────

    def add_log(msg):
        log_box.configure(state="normal")
        log_box.insert("end", msg + "\n")
        log_box.see("end")
        log_box.configure(state="disabled")

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
        btn_zh.configure(bg=LANG_ON if lang == "zh" else BTN,
                         fg="#1c2b3a" if lang == "zh" else MUTED)
        btn_en.configure(bg=LANG_ON if lang == "en" else BTN,
                         fg="#1c2b3a" if lang == "en" else MUTED)
        is_rec = st["status"] == "recording"
        rec_btn.configure(text=t("stop") if is_rec else t("start"))
        choose_btn.configure(text=t("choose"))
        btn_meeting.configure(text=t("action_meeting"))
        btn_interview.configure(text=t("action_interview"))
        stop_btn.configure(text=t("stop_task"))
        open_btn.configure(text=t("open_result"))

    def toggle_record():
        if st["status"] == "idle":
            _start_recording()
        elif st["status"] == "recording":
            _stop_recording()

    def _tick():
        timer_secs[0] += 1
        s = timer_secs[0]
        timer_var.set(f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}")
        timer_job[0] = root.after(1000, _tick)

    def _start_recording():
        out_record = cfg.get("output_record")
        if out_record:
            switch_output(out_record)
        recorder = DualStreamRecorder(
            sys_device=cfg.get("device_system_audio", "BlackHole 2ch"),
            mic_device=cfg.get("device_mic", "MacBook Air Microphone"),
            sample_rate=cfg["sample_rate"],
        )
        recorder.start()
        recordings_dir = CONFIG_DIR / "recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = recordings_dir / f"{ts}.wav"
        st.update(status="recording", recorder=recorder, audio_path=audio_path)
        rec_btn.configure(text=t("stop"), fg=DANGER, activeforeground=DANGER)
        timer_secs[0] = 0
        timer_var.set("00:00:00")
        timer_lbl.configure(fg=DANGER)
        timer_job[0] = root.after(1000, _tick)
        _set_action_btns(False)
        _draw_progress(0)
        add_log(f"[REC] {audio_path.name}")

    def _stop_recording():
        if timer_job[0]:
            root.after_cancel(timer_job[0])
        recorder = st["recorder"]
        recorder.stop()
        out_restore = cfg.get("output_restore")
        if out_restore:
            switch_output(out_restore)
        audio_path = st["audio_path"]
        if not recorder.save(audio_path):
            add_log("[ERR] 未录到任何音频")
            st["status"] = "idle"
            rec_btn.configure(text=t("start"), fg=ACCENT, activeforeground=ACCENT)
            timer_lbl.configure(fg=TEXT)
            return
        add_log(f"[REC] 完成 → {audio_path.name}")
        st.update(status="idle", chosen_path=audio_path)
        chosen_var.set(t("chosen_prefix") + audio_path.name)
        rec_btn.configure(text=t("start"), fg=ACCENT, activeforeground=ACCENT)
        timer_lbl.configure(fg=TEXT)
        _set_action_btns(True)

    def choose_file():
        path = filedialog.askopenfilename(
            title="选择录音文件",
            filetypes=[("WAV 录音", "*.wav")],
        )
        if path:
            st["chosen_path"] = Path(path)
            chosen_var.set(t("chosen_prefix") + Path(path).name)
            _set_action_btns(True)

    def stop_pipeline():
        cancel_flag[0] = True
        stop_btn.pack_forget()
        action_row.pack(fill="x")
        _draw_progress(0)
        st["status"] = "idle"
        rec_btn.configure(state="normal")
        _set_action_btns(bool(st.get("chosen_path")))
        add_log("[STOP] 已停止任务")

    def _start_pipeline(mode: str):
        path = st.get("chosen_path")
        if not path or pipeline_running[0]:
            return
        cancel_flag[0] = False
        pipeline_running[0] = True
        rec_btn.configure(state="disabled")
        _set_action_btns(False)
        st["status"] = "processing"
        _draw_progress(0)
        result_frame.pack_forget()
        action_row.pack_forget()
        stop_btn.pack(fill="x")
        _t.Thread(target=_run_pipeline, args=(path, mode), daemon=True).start()

    def _run_pipeline(input_path: Path, mode: str):
        class _Tee:
            def __init__(self, orig):
                # Unwrap any existing _Tee so nested pipelines never double-log
                real = orig
                while hasattr(real, '_orig'):
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
            tp  = cfg.get("transcribe_provider", "whisper")
            pp  = cfg.get("polish_provider", "claude")
            np_ = cfg.get("meeting_notes_provider", "claude")

            audio_path = input_path
            raw_txt   = audio_path.with_name(audio_path.stem + ".raw.txt")
            proofread = audio_path.with_name(audio_path.stem + ".proofread.txt")

            log_q.put(("progress", 5))
            if raw_txt.exists():
                print(f"[转写] 检测到 {raw_txt.name}，跳过 Whisper")
                transcript_raw = raw_txt.read_text(encoding="utf-8")
            else:
                def _transcribe_progress(pct):
                    log_q.put(("progress", pct))
                transcript_raw = transcribe(audio_path, tp, cfg, on_progress=_transcribe_progress)
                raw_txt.write_text(transcript_raw, encoding="utf-8")

            log_q.put(("progress", 40))
            if proofread.exists():
                print(f"[校对] 检测到 {proofread.name}，跳过校对")
                transcript_polished = proofread.read_text(encoding="utf-8")
            else:
                transcript_polished = polish_transcript(transcript_raw, pp, cfg, mode)
                proofread.write_text(transcript_polished, encoding="utf-8")

            log_q.put(("progress", 85))
            notes = generate_notes(transcript_polished, np_, cfg, mode)
            note_path = save_minutes(notes, audio_path)
            print(f"✓ 完成 → {note_path}")
            log_q.put(("done", str(note_path)))
        except SystemExit as e:
            log_q.put(("error", f"Pipeline exited ({e.code})"))
        except Exception as e:
            log_q.put(("error", str(e)))
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
                    add_log(msg)
                elif kind == "progress":
                    _draw_progress(msg)
                elif kind == "done":
                    stop_btn.pack_forget()
                    action_row.pack(fill="x")
                    if not cancel_flag[0]:
                        _draw_progress(100)
                        st.update(status="idle", result_path=msg)
                        rec_btn.configure(state="normal")
                        _set_action_btns(bool(st.get("chosen_path")))
                        result_frame.pack(fill="x", padx=20, pady=(8, 4))
                elif kind == "error":
                    stop_btn.pack_forget()
                    action_row.pack(fill="x")
                    if not cancel_flag[0]:
                        add_log(f"[ERR] {msg}")
                        _draw_progress(0)
                        st["status"] = "idle"
                        rec_btn.configure(state="normal")
                        _set_action_btns(bool(st.get("chosen_path")))
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
    btn_zh = tk.Button(lang_row, text="中文", relief="flat", bd=0, padx=8, pady=3,
                       font=("Menlo", 11), cursor="hand2", bg=LANG_ON, fg="#1c2b3a",
                       activebackground=LANG_ON, activeforeground="#1c2b3a",
                       command=lambda: set_lang("zh"))
    btn_zh.pack(side="left", padx=(0, 3))
    btn_en = tk.Button(lang_row, text="EN", relief="flat", bd=0, padx=8, pady=3,
                       font=("Menlo", 11), cursor="hand2", bg=BTN, fg=MUTED,
                       activebackground=LANG_ON, activeforeground="#1c2b3a",
                       command=lambda: set_lang("en"))
    btn_en.pack(side="left")

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
                            activeforeground=ACCENT, disabledforeground=ACCENT,
                            relief="flat", bd=0, cursor="hand2",
                            padx=12, pady=11, state="disabled",
                            command=lambda: _start_pipeline("meeting"))
    btn_meeting.pack(side="left", expand=True, fill="x", padx=(0, 4))
    btn_interview = tk.Button(action_row, text="", font=("Menlo", 12, "bold"),
                              bg=BTN, fg=ACCENT, activebackground=BTN,
                              activeforeground=ACCENT, disabledforeground=ACCENT,
                              relief="flat", bd=0, cursor="hand2",
                              padx=12, pady=11, state="disabled",
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

    # ⑥ 日志
    lc = card(root, pady=(0, 0))
    log_title_lbl = tk.Label(lc, text="", bg=CARD, fg=MUTED, font=("Menlo", 10))  # kept for set_lang ref
    log_box = scrolledtext.ScrolledText(
        lc, height=11, font=("Menlo", 11),
        bg="#111d2a", fg=MUTED, insertbackground=ACCENT,
        relief="flat", bd=0, state="disabled", wrap="word",
        selectbackground=BORDER,
    )
    log_box.pack(fill="x")
    log_box.configure(state="normal")
    log_box.insert("end", "LOG\n")
    log_box.configure(state="disabled")

    # ⑦ 结果（完成后显示）
    result_frame = tk.Frame(root, bg=BG)
    open_btn = tk.Button(result_frame, text="",
                         font=("Menlo", 12, "bold"),
                         bg=BTN, fg=ACCENT,
                         activebackground=BTN, activeforeground=ACCENT,
                         relief="flat", bd=0, cursor="hand2",
                         padx=28, pady=11, command=open_result)
    open_btn.pack()

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

    stt_help = "可选：whisper / openai / gemini，或 stt 配置中的任意 key"
    llm_help = "可选：claude / openai / gemini，或 llm 配置中的任意 key"

    mode_help = "运行模式：meeting（会议纪要，默认）| interview（面试总结）"

    # --mode 可放在子命令之前（全局）或之后（子命令级），两种写法均有效
    parser.add_argument("--mode", metavar="MODE", help=mode_help)

    p_rec = sub.add_parser("record", help="开始录音（Ctrl+C 停止）")
    p_rec.add_argument("--mode", metavar="MODE", help=mode_help, default=argparse.SUPPRESS)
    p_rec.add_argument("--title", metavar="标题", help="标题")
    p_rec.add_argument("--device", metavar="N", help="音频设备编号（见 devices 命令）")
    p_rec.add_argument("--transcribe-provider", metavar="PROVIDER", help=f"语音转文字模型，{stt_help}")
    p_rec.add_argument("--polish-provider", metavar="PROVIDER", help=f"转写校对模型，{llm_help}")
    p_rec.add_argument("--meeting-notes-provider", metavar="PROVIDER", help=f"纪要/总结模型，{llm_help}")

    p_tr = sub.add_parser("transcribe", help="转写已有音频文件")
    p_tr.add_argument("file", help="音频文件路径")
    p_tr.add_argument("--mode", metavar="MODE", help=mode_help, default=argparse.SUPPRESS)
    p_tr.add_argument("--title", metavar="标题", help="标题")
    p_tr.add_argument("--transcribe-provider", metavar="PROVIDER", help=f"语音转文字模型，{stt_help}")
    p_tr.add_argument("--polish-provider", metavar="PROVIDER", help=f"转写校对模型，{llm_help}")
    p_tr.add_argument("--meeting-notes-provider", metavar="PROVIDER", help=f"纪要/总结模型，{llm_help}")

    sub.add_parser("ui", help="打开桌面图形界面")

    sub.add_parser("devices", help="列出可用音频设备")

    p_cfg = sub.add_parser("config", help="查看或修改配置")
    p_cfg.add_argument("--set", metavar="key=value", help="设置配置项")

    args = parser.parse_args()
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
