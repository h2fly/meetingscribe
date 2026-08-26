# MeetingScribe

Meeting scribe for macOS and Windows — captures both speaker and microphone audio, auto-transcribes, and produces AI-generated meeting notes or interview summaries.

---

## Requirements

### macOS

| Item | Detail |
|------|--------|
| macOS 12+ | Uses CoreAudio API |
| Python 3.9+ | Verify with `python3 --version` |
| Node.js 18+ | Required for Claude Code CLI |
| Homebrew | Required to install BlackHole |

### Windows

| Item | Detail |
|------|--------|
| Windows 10/11 | 64-bit |
| Python 3.9+ | Download from [python.org](https://www.python.org/downloads/) — check **"Add to PATH"** during install |
| Node.js 18+ | Required for Claude Code CLI |

---

## Installation — macOS

### Step 1: Install Homebrew (skip if already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install system dependencies

```bash
brew install --cask blackhole-2ch   # virtual audio driver (captures system audio)
brew install ffmpeg                  # required by FunASR for audio decoding
```

> ⚠️ **Restart your Mac after installing BlackHole.** The virtual audio driver is loaded at boot — without a restart, `BlackHole 2ch` will not appear in Audio MIDI Setup in Step 3.

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

# Install PyTorch first (CPU version; visit https://pytorch.org for GPU options)
pip3 install torch torchaudio

# Install remaining dependencies
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

## Installation — Windows

### Step 1: Install VB-Audio Virtual Cable

Download and install **VB-Audio Virtual Cable** (free):
[https://vb-audio.com/Cable/](https://vb-audio.com/Cable/)

This is the Windows equivalent of BlackHole — it creates a virtual audio device to capture system audio.

### Step 2: Install VoiceMeeter Banana

Download and install **VoiceMeeter Banana** (free):
[https://vb-audio.com/Voicemeeter/banana.htm](https://vb-audio.com/Voicemeeter/banana.htm)

VoiceMeeter routes system audio to both your speakers and Virtual Cable simultaneously (equivalent to macOS Multi-Output Device).

**Configure VoiceMeeter:**

1. Open VoiceMeeter Banana
2. In **HARDWARE INPUT 1**, select your microphone
3. In **HARDWARE OUT A1**, select your real speakers / headphones
4. In **VIRTUAL INPUTS**, enable routing to **B1** (Virtual Cable)
5. Right-click the Windows speaker icon in the taskbar → **Sound settings** → set output to **VoiceMeeter Input**

> Once configured, VoiceMeeter runs in the background. No need to change audio devices before each recording.

### Step 3: Install Claude Code CLI

```cmd
npm install -g @anthropic-ai/claude-code
claude login
```

### Step 4: Install Python dependencies

```cmd
cd meetingscribe
pip install torch torchaudio
pip install -r requirements.txt
```

### Step 5: Update config.jsonc

On Windows, audio device auto-switching is not supported — VoiceMeeter handles routing permanently. Set `output_record` and `output_restore` to empty strings, and update the device names:

```jsonc
"output_record":  "",                                    // leave empty on Windows
"output_restore": "",                                    // leave empty on Windows
"device_system_audio": "CABLE Output (VB-Audio Virtual Cable)",
"device_mic":     "Microphone (your mic name here)"
```

**How to find device names:**

```cmd
python meetingscribe.py devices
```

---

## Quick Start

### GUI

```bash
python3 meetingscribe.py ui
```

The desktop GUI is a PyQt6 + Fluent Widgets app (single window, three views in the left navigation):

1. **录音 / Recording** — click the big circular mic button to start; click again to stop. The meeting-history sidebar sits below the record controls in the left column, and the **live bilingual caption panel** owns the full-height right column: while recording, speech is transcribed and translated locally in real time (Chinese ↔ English). One local engine: sherpa-onnx streaming zipformer + opus-mt (~0.3 s latency), with each finalized line re-decoded by the offline FunASR model in the background. Each finalized line is tagged with the speaker, identified by voice print (`live_captions.speaker_id`): everyone gets a **[Speaker N]**, on both the microphone and the call audio, so colleagues sharing one mic are told apart too. Until the voice print lands (~60 ms) the line shows its audio side, **[Us]** or **[Them]**. Caption translation ships on the built-in engines (opus-mt for Fast, NLLB for Accurate). Setting `live_captions.mt_provider: "qwen"` switches to a local **Qwen 1.5B** model (llama.cpp) instead, which keeps `stt.funasr.hotword` terms verbatim in the translation and additionally repairs ASR errors on finalized lines; if `llama-cpp-python` isn't installed it falls back automatically. Models download automatically on first use (≈ 700 MB, plus ≈ 1 GB if you switch to Qwen); captions are optional and never affect the recording itself. Pick a previously recorded meeting from the history list to act on it. Three pipeline buttons below the timer:
   - **Transcribe** — runs FunASR + LLM polish, produces `.polish.txt`
   - **Generate meeting notes** — full pipeline → `.meeting.md`
   - **Generate interview report** — full pipeline → `.interview.md`
   When the corresponding output file already exists on disk, the button flips to "Open …" (blue) and just opens the file.
2. **历史 / History** — full meeting list with four filter tabs (all / summarized / transcribed / pending), inline detail pane, and the same three pipeline buttons for rerunning steps on an old recording. Right-click any row to rename or delete (with confirmation).
3. **配置 / Settings** — quick concurrency slider + a raw JSONC editor for `config.jsonc` (syntax highlighted; the editor is the canonical way to customise pipeline prompts, see `cfg["prompts"]`).

A 中文 / EN item at the bottom of the left navigation (below Settings) flips every label, dialog and message in real time.

### Command Line

```bash
# Record and auto-generate meeting notes (Ctrl+C to stop)
python3 meetingscribe.py record

# Interview mode
python3 meetingscribe.py record --mode interview

# Process an existing recording
python3 meetingscribe.py transcribe /path/to/audio.wav
python3 meetingscribe.py transcribe /path/to/audio.wav --mode interview

# Replay a recording through the live caption engine (no sound card involved) —
# the way to compare caption quality across changes; --review adds the LLM pass
python3 meetingscribe.py captions /path/to/audio.wav
python3 meetingscribe.py captions /path/to/audio.wav --start 60 --seconds 120 --review

# Backfill ASR hotwords from existing transcripts/notes into config.jsonc
# (afterwards hotwords maintain themselves after every notes generation)
python3 meetingscribe.py hotwords          # rule-based + LLM extraction
python3 meetingscribe.py hotwords --show   # print the current list
```

---

## Output Files

Recordings are saved to `~/Documents/meetingscribe/recordings/`. All output files are placed alongside the recording:

| Extension | Content |
|-----------|---------|
| `.wav` | Recording (system audio + microphone, dual stream) |
| `.raw.txt` | Raw transcript (with timestamps; also `[说话人N]` speaker labels when `stt.funasr.spk_model` is on) |
| `.polish.txt` | AI-polished transcript |
| `.meeting.md` | Meeting notes (Markdown), generated when `mode=meeting` |
| `.interview.md` | Interview report (Markdown), generated when `mode=interview` |

A meeting may also have a human-readable suffix — `<timestamp>.<custom_name>.<ext>` (e.g. `20260512_090120.客户访谈.wav`). The legacy single-`.md` filename (no `.meeting` / `.interview` infix) is still recognised for backward compatibility.

> Re-running automatically detects completed steps and skips them — no need to re-transcribe.

---

## Configuration

Edit `config.jsonc` in the project directory (supports `//` comments). Common options:

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `meeting` | `meeting` or `interview` |
| `transcribe_provider` | `funasr` | `funasr` (local, default) / `openai` / `gemini` |
| `polish_provider` | `claude` | `claude` / `openai` / `gemini` |
| `meeting_notes_provider` | `claude` | `claude` / `openai` / `gemini` |
| `stt.funasr.hotword` | `""` | Space-separated hotwords to boost recognition accuracy — **auto-maintained**: after every notes generation, new terms are extracted (rule-based + LLM, see the `hotwords` block) and merged here; `python3 meetingscribe.py hotwords` backfills from existing recordings |
| `hotwords.auto_update` | `true` | Toggle the automatic hotword extraction (schemes A rule-based + B LLM) |
| `live_captions.mt_provider` | `default` | Caption translation engine: `default`/`marian` (opus-mt), `nllb`, or `qwen` — local Qwen 1.5B via llama.cpp, which adds a line-scoped hotword glossary and ASR error correction on finalized lines (GBNF-constrained, ~1.8 s per line) |
| `live_captions.history_minutes` | `180` | How far back the caption panel can be scrolled (minutes). Retention is by time, so the window holds regardless of speech cadence; `history_max` (12000 rows) is only a memory backstop |
| `live_captions.speaker_id.enabled` | `true` | Voice-print speaker identification (cam++, ~27 MB auto-download, ~60 ms per line, own thread). Separates individual people by voice, on BOTH channels — several colleagues sharing one microphone in a meeting room are told apart just like the people on the call. Everyone gets a Speaker N; the channel only tints the tag. `threshold` (0.5) trades merging against splitting |
| `live_captions.review.enabled` | `true` | Periodic batch re-check: every `interval_minutes` (5) the captions from that window, with their translations, go to a full-size LLM (`polish_provider`) which re-checks them with the whole window as context — the only pass that can fix a term from how it was said later. Costs one LLM call per interval while recording |

**FunASR models (configured in `stt.funasr`):**

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `paraformer-zh` | ASR model — downloaded automatically on first run (~400 MB) |
| `vad_model` | `fsmn-vad` | Voice activity detection — enables long-audio support |
| `punc_model` | `ct-punc` | Punctuation restoration |
| `spk_model` | `""` | Speaker diarization; `"cam++"` turns it on and transcript lines become `[12.3s] [说话人1] …`, which the polish + notes steps carry through to the summary. **Cost:** speaker ids are only consistent within a single pass, so enabling this disables chunked parallel transcription — a long recording runs single-pass and takes roughly `workers`× longer |
| `hotword` | `""` | Domain-specific terms (space-separated) to improve accuracy |
| `chunk_secs` | `300` | Split audio into chunks of this length (seconds) for parallel transcription; `0` = always serial. Ignored when `spk_model` is set |
| `workers` | `0` | Parallel FunASR instances; `0` = auto (`max(2, cpu_count / 2)`); memory scales with this value |

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

**Q: First run is slow / downloading models**
FunASR downloads `paraformer-zh`, `fsmn-vad`, and `ct-punc` on the first run (~400 MB total). They are cached locally and not re-downloaded. Ensure a stable network connection for the first run.

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

支持 macOS 和 Windows 的会议记录助手 — 同时录制扬声器与麦克风声音，自动转写，最终由 AI 生成会议纪要或面试总结。

---

## 系统要求

### macOS

| 条件 | 说明 |
|------|------|
| macOS 12+ | 使用 CoreAudio API |
| Python 3.9+ | `python3 --version` 确认 |
| Node.js 18+ | 用于安装 Claude Code CLI |
| Homebrew | 用于安装 BlackHole |

### Windows

| 条件 | 说明 |
|------|------|
| Windows 10/11 64位 | |
| Python 3.9+ | 从 [python.org](https://www.python.org/downloads/) 下载，安装时勾选 **"Add to PATH"** |
| Node.js 18+ | 用于安装 Claude Code CLI |

---

## 安装步骤 — macOS

### 第一步：安装 Homebrew（已安装可跳过）

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 第二步：安装系统依赖

```bash
brew install --cask blackhole-2ch   # 虚拟音频驱动，用于捕获系统播放的音频
brew install ffmpeg                  # FunASR 转写所需的音频解码工具
```

> ⚠️ **安装完 BlackHole 后请重启电脑。** 虚拟音频驱动需要随系统启动加载，不重启的话第三步在「音频 MIDI 设置」中看不到 `BlackHole 2ch` 设备。

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

# 先安装 PyTorch（CPU 版；如需 GPU 版请访问 https://pytorch.org 选择对应命令）
pip3 install torch torchaudio

# 再安装其余依赖
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

## 安装步骤 — Windows

### 第一步：安装 VB-Audio Virtual Cable

下载并安装 **VB-Audio Virtual Cable**（免费）：
[https://vb-audio.com/Cable/](https://vb-audio.com/Cable/)

这是 BlackHole 的 Windows 等价工具，用于捕获系统播放的音频。

### 第二步：安装 VoiceMeeter Banana

下载并安装 **VoiceMeeter Banana**（免费）：
[https://vb-audio.com/Voicemeeter/banana.htm](https://vb-audio.com/Voicemeeter/banana.htm)

VoiceMeeter 可以将系统音频同时送到真实扬声器和 Virtual Cable，等价于 macOS 的「多输出设备」。

**VoiceMeeter 配置方法：**

1. 打开 VoiceMeeter Banana
2. 在 **HARDWARE INPUT 1** 中选择你的麦克风
3. 在 **HARDWARE OUT A1** 中选择你的扬声器 / 耳机
4. 在 **VIRTUAL INPUTS** 中，将路由启用到 **B1**（即 Virtual Cable）
5. 右键任务栏扬声器图标 → 「声音设置」→ 将输出设备改为 **VoiceMeeter Input**

> 配置完成后 VoiceMeeter 后台运行，每次录音无需手动切换音频设备。

### 第三步：安装 Claude Code CLI

```cmd
npm install -g @anthropic-ai/claude-code
claude login
```

### 第四步：安装 Python 依赖

```cmd
cd meetingscribe
pip install torch torchaudio
pip install -r requirements.txt
```

### 第五步：修改配置文件

Windows 下不需要自动切换输出设备（VoiceMeeter 已常驻处理路由），将 `output_record` 和 `output_restore` 置空，并更新设备名称：

```jsonc
"output_record":  "",                                        // Windows 下留空
"output_restore": "",                                        // Windows 下留空
"device_system_audio": "CABLE Output (VB-Audio Virtual Cable)",
"device_mic":     "麦克风 (你的麦克风名称)"
```

**查看设备名称：**

```cmd
python meetingscribe.py devices
```

---

## 快速开始

### 图形界面

```bash
python3 meetingscribe.py ui
```

桌面界面基于 PyQt6 + Fluent Widgets，单窗口三个视图（左侧导航切换）：

1. **录音** — 点中间的大麦克风按钮开始录音，再点一次停止。会议历史在左列录音控制的下方；右列整列是**实时双语字幕面板**：录音过程中本地实时转写并翻译（中英互译）。识别链路只有一条：sherpa-onnx 流式 zipformer + opus-mt（延迟约 0.3 秒），每条定稿字幕再由离线 FunASR 模型后台重解码一遍。每条定稿字幕会用声纹标注说话人（`live_captions.speaker_id`）：麦克风侧和会议对端一视同仁，所有人各自编号为 **[说话人N]**，会议室里共用一支麦克风的同事也能分开；声纹结果落地前（约 60ms）先显示声道侧 **[我方]** / **[对方]**。字幕翻译默认走内置 opus-mt；把 `live_captions.mt_provider` 改成 `"qwen"` 则换成本地 **Qwen 1.5B** 小模型（llama.cpp），它会让命中 `stt.funasr.hotword` 的术语在译文里原样保留，并对定稿行做 ASR 纠错；未安装 `llama-cpp-python` 时自动回退。模型首次使用时自动下载（约 700MB；改用 Qwen 再加约 1GB）；字幕是可选功能，任何故障都不影响录音本体。从会议历史列表挑一条已有录音作为操作对象。计时器下方三个按钮：
   - **语音转文字** — 跑 FunASR + LLM 校对，生成 `.polish.txt`
   - **生成会议纪要** — 整条流水线 → `.meeting.md`
   - **生成面试报告** — 整条流水线 → `.interview.md`
   对应输出文件已存在时，按钮会变成蓝色的「打开 X」，点击直接打开文件。
2. **历史** — 完整会议列表，四个筛选 tab（全部 / 已总结 / 已录音转文字 / 待处理），右侧详情面板，并复用同三个按钮以对旧录音重新跑流水线。任一会议右键可以重命名或删除（带二次确认）。
3. **配置** — 一键调整三个并发参数 + 一个带语法高亮的 `config.jsonc` 原文编辑器（直接在这里改 `cfg["prompts"]` 自定义流水线 prompt）。

左侧导航底部（「配置」下方）有 **中文 / EN** 切换项，实时翻转所有界面文字、对话框和提示。

### 命令行

```bash
# 录音 + 自动生成会议纪要（Ctrl+C 停止录音）
python3 meetingscribe.py record

# 面试模式
python3 meetingscribe.py record --mode interview

# 处理已有录音文件
python3 meetingscribe.py transcribe /path/to/audio.wav
python3 meetingscribe.py transcribe /path/to/audio.wav --mode interview

# 扫描已有转写/纪要，把热词回填进 config.jsonc
# （之后每次生成纪要都会自动增量维护热词，无需再手动跑）
python3 meetingscribe.py hotwords          # 规则 + LLM 提取
python3 meetingscribe.py hotwords --show   # 查看当前热词列表
```

---

## 输出文件说明

录音文件默认保存在 `~/Documents/meetingscribe/recordings/`，处理结果与录音同目录：

| 文件后缀 | 内容 |
|----------|------|
| `.wav` | 录音文件（系统音频 + 麦克风双路） |
| `.raw.txt` | 原始转写文本（含时间戳；开启 `stt.funasr.spk_model` 时还带 `[说话人N]` 发言人标注） |
| `.polish.txt` | AI 校对后的文本 |
| `.meeting.md` | 会议纪要（Markdown），`mode=meeting` 时生成 |
| `.interview.md` | 面试报告（Markdown），`mode=interview` 时生成 |

录音也可以带一个可读后缀 — `<时间戳>.<自定义名字>.<后缀>`（例如 `20260512_090120.客户访谈.wav`），可通过历史界面的右键「重命名」设置。早期的单一 `.md` 文件（没有 `.meeting` / `.interview` 中段）仍然兼容识别。

> 重复运行时会自动检测已完成的步骤并跳过，无需重新转写。

---

## 配置说明

配置文件为项目目录下的 `config.jsonc`，支持 `//` 注释。常用配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `mode` | `meeting` | 默认模式：`meeting`（会议）/ `interview`（面试） |
| `transcribe_provider` | `funasr` | 转写引擎：`funasr`（本地，默认）/ `openai` / `gemini` |
| `polish_provider` | `claude` | 校对模型：`claude` / `openai` / `gemini` |
| `meeting_notes_provider` | `claude` | 纪要模型：`claude` / `openai` / `gemini` |
| `stt.funasr.hotword` | `""` | 热词（空格分隔），提升专有名词识别率——**自动维护**：每次生成纪要后自动提取新术语（规则 + LLM，见 `hotwords` 配置块）并合并到这里；`python3 meetingscribe.py hotwords` 可从已有录音一次性回填 |
| `hotwords.auto_update` | `true` | 热词自动提取总开关（方案 A 规则 + 方案 B LLM） |
| `live_captions.mt_provider` | `default` | 字幕翻译引擎：`default`/`marian`（opus-mt）、`nllb`，或 `qwen` —— 本地 Qwen 1.5B（llama.cpp），额外提供按行命中的热词术语表和定稿行 ASR 纠错（GBNF 约束输出，每行约 1.8 秒） |
| `live_captions.history_minutes` | `180` | 字幕面板可回看的时长（分钟）。按时间保留，与语速无关；`history_max`（12000 行）只是内存兜底 |
| `live_captions.speaker_id.enabled` | `true` | 声纹说话人识别（cam++，约 27MB 自动下载，每行约 60ms，独立线程）。两个声道一视同仁地按声纹区分人——会议室里几个人共用一支麦克风也能分开，所有人统一编号为 说话人1 / 说话人2 …，声道只决定标签颜色。`threshold`（0.5）在「并成同一个人」和「拆成多个人」之间取舍 |
| `live_captions.review.enabled` | `true` | 定期批量复核：每 `interval_minutes`（5 分钟）把这段时间的字幕连同译文整批交给大模型（`polish_provider`），带整段上下文重新校对重译——这是唯一能「用后文修正前文」的一层。代价是录音期间每个周期一次 LLM 调用 |

**FunASR 配置项（`stt.funasr` 下）：**

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `model` | `paraformer-zh` | ASR 模型，首次运行自动下载（约 400 MB） |
| `vad_model` | `fsmn-vad` | 语音活动检测，支持长音频分句 |
| `punc_model` | `ct-punc` | 标点恢复模型 |
| `spk_model` | `""` | 说话人区分（声纹聚类）；填 `"cam++"` 开启后转写行变成 `[12.3s] [说话人1] …`，校对与纪要环节会把发言人区分带进总结。**代价**：说话人编号只在单次推理内一致，因此开启后放弃分块并发、单趟转写，长录音耗时约为原来的 `workers` 倍 |
| `hotword` | `""` | 热词（空格分隔），提升领域专有词识别率 |
| `chunk_secs` | `300` | 超过此时长自动分块并发转写（秒），`0` = 始终串行；设置了 `spk_model` 时此项被忽略 |
| `workers` | `0` | 并发实例数，`0` = 自动（`max(2, CPU核数/2)`），内存随之线性增长 |

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

**Q：首次运行转写很慢 / 正在下载模型**
FunASR 在首次运行时会自动下载 `paraformer-zh`、`fsmn-vad`、`ct-punc` 三个模型（合计约 400 MB），下载完成后本地缓存，后续无需重新下载。请确保首次运行时网络畅通。

**Q：如何确认设备名称是否正确**
```bash
python3 meetingscribe.py devices
```
列出的名称即为可填入 `config.jsonc` 的值。注意名称区分大小写。

**Q：macOS 要求麦克风权限**
首次运行时系统会弹出权限请求，点击「允许」即可。
如已拒绝，可在「系统设置」→「隐私与安全性」→「麦克风」中重新开启 Terminal / Python 的权限。
