# MeetingScribe

macOS meeting scribe — captures both speaker and microphone audio, auto-transcribes, and produces AI-generated meeting notes or interview summaries.

---

## Requirements

| Item | Detail |
|------|--------|
| macOS 12+ | Uses CoreAudio API, macOS only |
| Python 3.9+ | Verify with `python3 --version` |
| Node.js 18+ | Required for Claude Code CLI |
| Homebrew | Required to install BlackHole |

---

## Installation

### Step 1: Install Homebrew (skip if already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install BlackHole virtual audio driver

BlackHole is a free virtual audio device that captures system audio (e.g. audio from meeting apps).

```bash
brew install --cask blackhole-2ch
```

### Step 3: Configure macOS audio routing

This step routes system audio to both your speakers and BlackHole simultaneously.

1. Open **Audio MIDI Setup** — search in Spotlight, or run:
   ```bash
   open "/System/Applications/Utilities/Audio MIDI Setup.app"
   ```

2. Click **"+"** at the bottom left → **"Create Multi-Output Device"**

3. Check both:
   - ✅ Your speakers (e.g. `MacBook Air Speakers` / external headphones)
   - ✅ `BlackHole 2ch`

4. Check the **"Drift Correction"** checkbox on the `BlackHole 2ch` row

5. Right-click the new Multi-Output Device → **"Use This Device For Sound Output"**

6. Note the device name (default is `Multi-Output Device`; you can rename it)

> After this setup, system audio will flow into BlackHole and MeetingScribe can capture it.

### Step 4: Install Claude Code CLI

Claude Code CLI is used for AI polishing and generating meeting notes.

```bash
npm install -g @anthropic-ai/claude-code
claude login        # Follow the prompts to authorize
claude --version    # Confirm installation
```

### Step 5: Install Python dependencies

```bash
cd meetingscribe
pip3 install -r requirements.txt
```

### Step 6: Update config.jsonc

Open `config.jsonc` and update the following fields to match your device names:

```jsonc
"output_record":  "Multi-Output Device",   // Name of the Multi-Output Device from Step 3
"output_restore": "MacBook Air Speakers",  // Your speaker name (restored after recording)
"device_mic":     "MacBook Air Microphone" // Your microphone name
```

**How to find device names:**

```bash
python3 meetingscribe.py devices
```

Speaker names can be found in **System Settings → Sound → Output**.

---

## Quick Start

### GUI (recommended)

```bash
python3 meetingscribe.py ui
```

1. Click **▶ Start Recording** to begin
2. Click **◼ Stop Recording** when done
3. Click **Generate Meeting Notes** or **Generate Interview Summary**
4. When complete, click **Open Result** to view the Markdown output

### Command Line

```bash
# Record and auto-generate meeting notes (Ctrl+C to stop)
python3 meetingscribe.py record

# Interview mode
python3 meetingscribe.py record --mode interview

# Process an existing recording
python3 meetingscribe.py transcribe /path/to/audio.wav
python3 meetingscribe.py transcribe /path/to/audio.wav --mode interview
```

---

## Output Files

Recordings are saved to `~/Documents/meetingscribe/recordings/`. All output files are placed alongside the recording:

| Extension | Content |
|-----------|---------|
| `.wav` | Recording (system audio + microphone, dual stream) |
| `.raw.txt` | Raw Whisper transcript (with timestamps) |
| `.proofread.txt` | AI-polished transcript |
| `.md` | Meeting notes / interview summary (Markdown) |

> Re-running automatically detects completed steps and skips them — no need to re-transcribe.

---

## Configuration

Edit `config.jsonc` in the project directory (supports `//` comments). Common options:

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `meeting` | `meeting` or `interview` |
| `transcribe_provider` | `whisper` | `whisper` (local) / `openai` / `gemini` |
| `polish_provider` | `claude` | `claude` / `openai` / `gemini` |
| `meeting_notes_provider` | `claude` | `claude` / `openai` / `gemini` |
| `stt.whisper.model` | `base` | Whisper model size (see table below) |

**Whisper model comparison:**

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `tiny` | ~75 MB | Fastest | Fair |
| `base` | ~150 MB | Fast | Good (**default**) |
| `small` | ~480 MB | Medium | Better |
| `medium` | ~1.5 GB | Slow | Very good |
| `large-v3` | ~3 GB | Slowest | Best |

---

## Troubleshooting

**Q: No system audio captured (can't hear meeting app audio)**
- Check Audio MIDI Setup — confirm BlackHole 2ch is checked in the Multi-Output Device
- Check System Settings → Sound → Output — confirm Multi-Output Device is selected
- Check that `output_record` in `config.jsonc` exactly matches the device name (spaces matter)

**Q: `claude` command not found**
```bash
npm install -g @anthropic-ai/claude-code
claude login
```
Confirm Node.js is installed: `node --version` (requires 18+)

**Q: First Whisper run is slow**
First run downloads the model (~150 MB for `base`). It is cached locally and not re-downloaded. If your network is slow, switch to `tiny` in `config.jsonc`: `"model": "tiny"`.

**Q: How to confirm device names**
```bash
python3 meetingscribe.py devices
```
The listed names are exactly what you should put in `config.jsonc`. Names are case-sensitive.

**Q: macOS requests microphone permission**
Click **Allow** when prompted. If you previously denied it, re-enable it in **System Settings → Privacy & Security → Microphone**.

---
---

# MeetingScribe

macOS 会议记录助手 — 同时录制扬声器与麦克风声音，自动转写，最终由 AI 生成会议纪要或面试总结。

---

## 系统要求

| 条件 | 说明 |
|------|------|
| macOS 12+ | 使用 CoreAudio API，仅支持 macOS |
| Python 3.9+ | `python3 --version` 确认 |
| Node.js 18+ | 用于安装 Claude Code CLI |
| Homebrew | 用于安装 BlackHole |

---

## 安装步骤

### 第一步：安装 Homebrew（已安装可跳过）

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 第二步：安装 BlackHole 虚拟音频驱动

BlackHole 是一个免费的虚拟声卡，用于捕获系统播放的音频（会议软件声音）。

```bash
brew install --cask blackhole-2ch
```

### 第三步：配置 macOS 音频路由

这一步让系统在播放声音给扬声器的同时，也把声音送入 BlackHole，从而实现录制。

1. 打开「音频 MIDI 设置」：Spotlight 搜索，或运行
   ```bash
   open "/System/Applications/Utilities/Audio MIDI Setup.app"
   ```

2. 左下角点击「**+**」→「**创建多输出设备**」

3. 在右侧列表中勾选：
   - ✅ 你的扬声器（如 `MacBook Air 扬声器` / 外接耳机）
   - ✅ `BlackHole 2ch`

4. 勾选 `BlackHole 2ch` 那行的「**漂移校正**」复选框

5. 右键新建的「多输出设备」→「**将此设备用作系统声音输出**」

6. 记下该设备的名称（默认是「多输出设备」，可双击重命名）

> 完成后，系统播放声音时会同时进入 BlackHole，MeetingScribe 即可捕获。

### 第四步：安装 Claude Code CLI

Claude Code CLI 用于 AI 校对和生成会议纪要。

```bash
npm install -g @anthropic-ai/claude-code
claude login        # 按提示完成授权登录
claude --version    # 确认安装成功
```

### 第五步：安装 Python 依赖

```bash
cd meetingscribe
pip3 install -r requirements.txt
```

### 第六步：修改配置文件

打开项目目录下的 `config.jsonc`，根据你自己的设备名称修改以下三项：

```jsonc
"output_record":  "多输出设备",             // 第三步创建的多输出设备名称
"output_restore": "MacBook Air Speakers",  // 录音结束后恢复的扬声器名称
"device_mic":     "MacBook Air Microphone" // 麦克风名称
```

**如何查看设备名称：**

```bash
python3 meetingscribe.py devices
```

扬声器名称可在系统「声音」设置 → 输出 中查看。

---

## 快速开始

### 图形界面（推荐）

```bash
python3 meetingscribe.py ui
```

使用流程：

1. 点击「▶ 开始录音」→ 开始会议
2. 会议结束后点「◼ 停止录音」
3. 点击「开始整理会议纪要」或「开始整理面试记录」
4. 等待处理完成，点击「打开结果文件」查看 Markdown 纪要

### 命令行

```bash
# 录音 + 自动生成会议纪要（Ctrl+C 停止录音）
python3 meetingscribe.py record

# 面试模式
python3 meetingscribe.py record --mode interview

# 处理已有录音文件
python3 meetingscribe.py transcribe /path/to/audio.wav
python3 meetingscribe.py transcribe /path/to/audio.wav --mode interview
```

---

## 输出文件说明

录音文件默认保存在 `~/Documents/meetingscribe/recordings/`，处理结果与录音同目录：

| 文件后缀 | 内容 |
|----------|------|
| `.wav` | 录音文件（系统音频 + 麦克风双路） |
| `.raw.txt` | Whisper 原始转写文本（含时间戳） |
| `.proofread.txt` | AI 校对后文本 |
| `.md` | 会议纪要 / 面试总结（Markdown 格式） |

> 重复运行时会自动检测已完成的步骤并跳过，无需重新转写。

---

## 配置说明

配置文件为项目目录下的 `config.jsonc`，支持 `//` 注释。常用配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `mode` | `meeting` | 默认模式：`meeting`（会议）/ `interview`（面试） |
| `transcribe_provider` | `whisper` | 转写引擎：`whisper`（本地）/ `openai` / `gemini` |
| `polish_provider` | `claude` | 校对模型：`claude` / `openai` / `gemini` |
| `meeting_notes_provider` | `claude` | 纪要模型：`claude` / `openai` / `gemini` |
| `stt.whisper.model` | `base` | Whisper 模型大小（见下表） |

**Whisper 模型对比：**

| 模型 | 文件大小 | 速度 | 准确度 |
|------|----------|------|--------|
| `tiny` | ~75 MB | 最快 | 一般 |
| `base` | ~150 MB | 快 | 较好（**默认**） |
| `small` | ~480 MB | 中等 | 好 |
| `medium` | ~1.5 GB | 慢 | 很好 |
| `large-v3` | ~3 GB | 最慢 | 最佳 |

---

## 常见问题

**Q：录不到会议软件的声音**
- 检查「音频 MIDI 设置」→ 多输出设备是否勾选了 BlackHole 2ch
- 检查系统「声音」→「输出」是否选择了多输出设备
- 检查 `config.jsonc` 中 `output_record` 的名称是否与多输出设备完全一致（包括空格）

**Q：`claude` 命令找不到**
```bash
npm install -g @anthropic-ai/claude-code
claude login
```
确认 Node.js 已安装：`node --version`（需要 18+）

**Q：首次运行 Whisper 很慢**
首次运行会自动下载模型（base 约 150 MB），下载完成后本地缓存，后续无需重新下载。
如果网络较慢，可以先改用更小的模型：在 `config.jsonc` 中将 `"model": "base"` 改为 `"model": "tiny"`。

**Q：如何确认设备名称是否正确**
```bash
python3 meetingscribe.py devices
```
列出的名称即为可填入 `config.jsonc` 的值。注意名称区分大小写。

**Q：macOS 要求麦克风权限**
首次运行时系统会弹出权限请求，点击「允许」即可。
如已拒绝，可在「系统设置」→「隐私与安全性」→「麦克风」中重新开启 Terminal / Python 的权限。
