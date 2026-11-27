# MeetingScribe

Meeting scribe for macOS and Windows — captures both speaker and microphone audio, auto-transcribes, and produces AI-generated meeting notes, interview reports, sharing summaries, or any output you define yourself. Optional live bilingual captions while recording.

---

## Requirements

### macOS

| Item | Detail |
|------|--------|
| macOS 12+ | Uses CoreAudio API |
| Python 3.9+ | Source install only — the macOS app bundles its own. Verify with `python3 --version` |
| Node.js 18+ | Required for Claude Code CLI |
| Homebrew | Required to install BlackHole |

### Windows

| Item | Detail |
|------|--------|
| Windows 10/11 | 64-bit |
| Python 3.9+ | Download from [python.org](https://www.python.org/downloads/) — check **"Add to PATH"** during install |
| Node.js 18+ | Required for Claude Code CLI |

---

## Installation — macOS app (DMG)

The app bundles Python and every Python library, so there is no `pip install`. Apple Silicon, macOS 13+.

> **Step-by-step guide with screenshots:** [`packaging/install-guide.en.html`](packaging/install-guide.en.html) (English) / [`packaging/install-guide.html`](packaging/install-guide.html) (中文). Both ship in the DMG as `Install Guide.html` and `安装说明.html` and opens from the app via **Help → 安装说明** and the **查看步骤** links in Settings → Environment.

1. Open `MeetingScribe-<version>.dmg` and drag **MeetingScribe** into **Applications**.
2. First launch: **right-click → Open**, then **Open** again. The app is ad-hoc signed (no Apple Developer ID), so a plain double-click is blocked the first time. If right-click is blocked too, run `xattr -dr com.apple.quarantine /Applications/MeetingScribe.app` in Terminal.
3. Allow microphone access when macOS asks (on the first recording).
4. Install what the app cannot bundle. **Settings → Environment** checks each item and shows how to fix it, and the app warns at launch when a required one is missing:

   | Item | Why | How |
   |------|-----|-----|
   | BlackHole 2ch | Records the computer's audio (the other side of a call) | `brew install --cask blackhole-2ch`, then restart — see [Step 2](#step-2-install-system-dependencies) |
   | Multi-Output Device | Lets you hear the call while it is recorded | Audio MIDI Setup — see [Step 3](#step-3-configure-macos-audio-routing) |
   | `claude` (or `codex`) CLI, logged in | Polishes the transcript and writes the notes | See [Step 4](#step-4-install-claude-code-cli); the **Test** button in Settings → Environment confirms the login |
   | ffmpeg *(optional)* | Importing non-WAV audio (mp3 / m4a …) | `brew install ffmpeg` |

**Models download on first use** — none ship with the app. The first transcription fetches ≈ 2 GB (paraformer-large, VAD, punctuation); turning on live captions fetches up to ≈ 1.7 GB more, depending on the recognition language and translation engine. The app says what it is about to download and how big it is before it starts. Each model is downloaded once and cached (`~/.cache/modelscope`, `~/.cache/huggingface`, `~/Documents/meetingscribe/models`).

**Where the app keeps its files** — everything lives in `~/Documents/meetingscribe/`, not in the app bundle (which is replaced on every update):

| Path | Content |
|------|---------|
| `config.jsonc` | Settings — created from the defaults on first launch; change the common options in **Settings → Common settings**, the rest under **Advanced** |
| `hotword.jsonc` | Hotword list (to reuse one from a source install: `cp hotword.jsonc ~/Documents/meetingscribe/`) |
| `goals.json` | Custom goals |
| `recordings/` | Recordings and every generated file |

### Building the app

```bash
python3 -m pip install pyinstaller   # once, into the Python that runs meetingscribe.py
packaging/build_dmg.sh               # → dist/MeetingScribe-<version>.dmg (full build, ~10 min)
packaging/build_dmg.sh --update      # seconds: refresh only meetingscribe.py + guides in dist/MeetingScribe.app
```

The version comes from `__version__` in `meetingscribe.py` — bump it there only. The build needs ~4 GB of free disk while it runs (intermediate files, removed afterwards) and refuses to run if `config.jsonc` still carries an inline hotword list, since that file is shipped to every user.

---

## Installation — macOS (from source)

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

1. **录音 / Recording** — click the big circular mic button to start; click again to stop.
   - **Calendar chip** (above the button, when *calendar linkage* is on): pick today's event before recording. The file is named after the event, and its attendees and linked Notion pages are handed to the notes step, so speakers get real names and the summary gets background. It reads Google Calendar through the `claude` CLI's connectors (falling back to `codex`); turn it on in Settings, which checks the connection first.
   - **Live caption panel** (right column, switch in its header; captions never affect the recording itself):
     - *Recognition* (`识别` dropdown, switchable mid-recording): English uses NVIDIA Nemotron Speech 0.6B; Chinese uses a streaming Paraformer, and each finished Chinese line is re-decoded in the background by the offline FunASR model.
     - *Translation* (`翻译` dropdown: 中文 / English / off): HY-MT 1.5 1.8B by default (`live_captions.mt_provider: "hymt"`, local via llama.cpp); `qwen`, `nllb` and opus-mt (`marian`) are alternatives, and opus-mt is the automatic fallback.
     - *Speakers*: each line is tagged by voice print as **[Speaker N]** — on both the microphone and the call audio, so colleagues sharing one mic are told apart. Until the voice print lands (~60 ms) the line shows its side, **[Us]** / **[Them]**.
     - *LLM re-check*: every two new lines, the last five (with their translations) go to a full-size LLM that fixes terms using what was said after them. This is the only part that leaves the machine; `live_captions.review.enabled: false` keeps captions fully local.
     - Models download on first use: ≈ 630 MB (English) or ≈ 230 MB (Chinese) for recognition, ≈ 1.1 GB for HY-MT. The app announces what it is about to download first.
   - **Pipeline buttons** below the timer act on the latest recording:
     - **Transcribe** — FunASR + LLM polish → `.polish.txt`
     - **Generate meeting notes** → `.meeting.md`
     - **Generate interview report** → `.interview.md`
     - **Generate sharing summary** (knowledge-sharing sessions / tech talks with Q&A) → `.sharing.md`
     - **Custom goal ▾** — your own output type. Describe what you want in a sentence (e.g. "list every customer requirement with who raised it"), let the AI expand it into a full prompt (editable), save it, then generate it for any recording → `.goal-<id>.md`. Goals live in `~/Documents/meetingscribe/goals.json`.

     When an output already exists, its button turns blue ("Open …") and opens the file.
2. **历史 / History** — full meeting list with four filter tabs (all / summarized / transcribed / pending), inline detail pane, and the same pipeline buttons (custom goals included) for running or rerunning steps on an old recording. Right-click any row to rename or delete (with confirmation).
3. **配置 / Settings**
   - **Environment** — checks BlackHole, the Multi-Output Device, the `claude` / `codex` CLI and ffmpeg, with a fix for each and a login test for the CLI.
   - **Common settings** — the options people actually change, as dropdowns and switches, saved the moment you change them: which model polishes the transcript and writes the notes, the fallback model, speaker identification in transcripts, the caption translation engine, caption re-check / speaker tags / LLM review, automatic hotword learning, an **Edit hotword list** button, and the exact location of every data file.
   - **Calendar linkage** switch (checks the connection before turning on), the concurrency setting, and — collapsed under **Advanced** — the raw `config.jsonc` editor for everything else, prompts included (`cfg["prompts"]`). Saving from the form only rewrites the changed lines, so comments in `config.jsonc` survive.

A 中文 / EN item at the bottom of the left navigation (below Settings) flips every label, dialog and message in real time.

### Command Line

```bash
# Record and auto-generate meeting notes (Ctrl+C to stop)
python3 meetingscribe.py record

# Interview / knowledge-sharing mode
python3 meetingscribe.py record --mode interview
python3 meetingscribe.py record --mode sharing

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
# One-off import of member names / project names / terminology from Notion.
# Needs an internal integration token in the environment; nothing is stored.
export NOTION_TOKEN=ntn_xxx
python3 meetingscribe.py hotwords --notion --dry-run     # preview first
python3 meetingscribe.py hotwords --notion --min-cjk 3   # skip risky 2-char CJK terms
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
| `.sharing.md` | Sharing summary (Markdown), generated when `mode=sharing` |
| `.goal-<id>.md` | Output of a custom goal (one file per goal) |
| `.meta.json` | Linked calendar event (title, time, attendees, Notion pages) — only when an event was linked |

A meeting may also have a human-readable suffix — `<timestamp>.<custom_name>.<ext>` (e.g. `20260512_090120.客户访谈.wav`). The legacy single-`.md` filename (no `.meeting` / `.interview` infix) is still recognised for backward compatibility.

> Re-running automatically detects completed steps and skips them — no need to re-transcribe.

---

## Configuration

Edit `config.jsonc` in the project directory — for the macOS app, `~/Documents/meetingscribe/config.jsonc`, also editable in Settings (supports `//` comments). Common options:

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `meeting` | `meeting` / `interview` / `sharing` |
| `transcribe_provider` | `funasr` | `funasr` (local, default) / `openai` / `gemini` |
| `polish_provider` | `claude` | `claude` / `codex_proxy` / `codex_personal` / `openai` / `gemini` |
| `meeting_notes_provider` | `claude` | same choices as `polish_provider` |
| `llm_fallback` | `["claude"]` | Providers tried in order when the main one fails; `[]` = stop at the first failure |
| `calendar.enabled` | `false` | Calendar linkage (see the calendar chip above); `calendar.provider_chain` sets which CLI reads the calendar |
| `stt.funasr.hotword` | `""` | Leave empty — the live list lives in **`hotword.jsonc`**, which is gitignored because it accumulates colleague and customer names. `cp hotword.jsonc.example hotword.jsonc` to start; after every notes generation new terms are mined (rule-based + LLM, see the `hotwords` block) and merged there, and `python3 meetingscribe.py hotwords` backfills from existing recordings |
| `hotwords.auto_update` | `true` | Toggle the automatic hotword extraction (schemes A rule-based + B LLM) |
| `live_captions.language` | `en` | Caption recognition language, `en` / `zh` (the panel's 识别 dropdown writes it) |
| `live_captions.translate_to` | `zh` | Caption translation target, `zh` / `en` / `off` (the panel's 翻译 dropdown writes it) |
| `live_captions.mt_provider` | `hymt` | Caption translation engine: `hymt` (HY-MT 1.5 1.8B, a dedicated translation model, ≈ 1.1 GB), `qwen` (Qwen 1.5B — also repairs ASR errors on Chinese lines, but unreliable at translating English), `nllb`, or `marian` (opus-mt, fastest; truncates long lines) |
| `live_captions.history_minutes` | `180` | How far back the caption panel can be scrolled (minutes). Retention is by time, so the window holds regardless of speech cadence; `history_max` (12000 rows) is only a memory backstop |
| `live_captions.speaker_id.enabled` | `true` | Voice-print speaker identification (cam++, ~27 MB auto-download, ~60 ms per line, own thread). Separates individual people by voice, on BOTH channels — several colleagues sharing one microphone in a meeting room are told apart just like the people on the call. Everyone gets a Speaker N; the channel only tints the tag. `threshold` (0.5) trades merging against splitting |
| `live_captions.review.enabled` | `true` | Rolling LLM re-check: every `trigger_lines` (2) new lines, the last `window_lines` (5), with their translations, go to a full-size LLM (`review.provider`, empty = `polish_provider`) — the only pass that can fix a term from what was said after it. Costs one LLM call per batch while recording |

**FunASR models (configured in `stt.funasr`):**

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `paraformer-zh` | ASR model — downloaded automatically on first run (≈ 950 MB; ≈ 2 GB together with the VAD and punctuation models) |
| `vad_model` | `fsmn-vad` | Voice activity detection — enables long-audio support |
| `punc_model` | `ct-punc` | Punctuation restoration |
| `spk_model` | `""` | Speaker diarization; `"cam++"` turns it on and transcript lines become `[12.3s] [说话人1] …`, which the polish + notes steps carry through to the summary. **Cost:** speaker ids are only consistent within a single pass, so enabling this disables chunked parallel transcription — a long recording runs single-pass and takes roughly `workers`× longer |
| `hotword` | `""` | Domain-specific terms (space-separated) to improve accuracy |
| `chunk_secs` | `300` | Split audio into chunks of this length (seconds) for parallel transcription; `0` = always serial. Ignored when `spk_model` is set |
| `workers` | `0` | Parallel FunASR instances; `0` = auto (`max(6, cpu_count / 2)`); memory scales with this value |

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
The first transcription downloads `paraformer-zh`, `fsmn-vad` and `ct-punc` (≈ 2 GB in total), and the first live-caption session downloads its recognition and translation models (up to ≈ 1.7 GB). The app announces each download and its size before it starts; caption downloads also show progress under the caption panel. Models are cached locally and never re-downloaded. Make sure the network is stable for the first run.

**Q: macOS says the app "cannot be opened" / the developer cannot be verified**
The app is not notarized. Right-click it in Applications → **Open** → **Open**, or run `xattr -dr com.apple.quarantine /Applications/MeetingScribe.app`.

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

支持 macOS 和 Windows 的会议记录助手 — 同时录制扬声器与麦克风声音，自动转写，最终由 AI 生成会议纪要、面试报告、分享总结，或你自定义的任何内容。录音时还可开启实时双语字幕。

---

## 系统要求

### macOS

| 条件 | 说明 |
|------|------|
| macOS 12+ | 使用 CoreAudio API |
| Python 3.9+ | 仅源码安装需要（macOS app 自带）。`python3 --version` 确认 |
| Node.js 18+ | 用于安装 Claude Code CLI |
| Homebrew | 用于安装 BlackHole |

### Windows

| 条件 | 说明 |
|------|------|
| Windows 10/11 64位 | |
| Python 3.9+ | 从 [python.org](https://www.python.org/downloads/) 下载，安装时勾选 **"Add to PATH"** |
| Node.js 18+ | 用于安装 Claude Code CLI |

---

## 安装步骤 — macOS app（DMG）

app 已经自带 Python 和所有 Python 库，不需要 `pip install`。支持 Apple Silicon、macOS 13 及以上。

> **图文安装说明**：[`packaging/install-guide.html`](packaging/install-guide.html)（英文版：[`install-guide.en.html`](packaging/install-guide.en.html)）。DMG 里中英两份并排附带（`安装说明.html` / `Install Guide.html`），app 里也能通过「帮助 → 安装说明」和「配置 → 运行环境」里的「查看步骤」打开。

1. 打开 `MeetingScribe-<版本号>.dmg`，把 **MeetingScribe** 拖进「应用程序」。
2. 第一次打开：**右键 → 打开**，在弹窗里再点一次「打开」。app 没有苹果开发者签名，第一次直接双击会被系统拦下。如果右键也被拦，在终端运行 `xattr -dr com.apple.quarantine /Applications/MeetingScribe.app`。
3. 第一次录音时 macOS 会请求麦克风权限，点「允许」。
4. 安装 app 无法自带的几样东西。**「配置 → 运行环境」**会逐项检查并给出处理方法；缺少必需项时，app 启动时也会提醒：

   | 项目 | 用途 | 安装方法 |
   |------|------|----------|
   | BlackHole 2ch | 录制电脑里播放的声音（会议对方） | `brew install --cask blackhole-2ch`，装完重启电脑，见[第二步](#第二步安装系统依赖) |
   | 多输出设备 | 录音时自己也能听到声音 | 在「音频 MIDI 设置」里创建，见[第三步](#第三步配置-macos-音频路由) |
   | `claude`（或 `codex`）命令行，且已登录 | 校对文字稿、生成纪要 | 见[第四步](#第四步安装-claude-code-cli)；可以用「运行环境」里的「测试连接」确认已登录 |
   | ffmpeg（可选） | 导入 mp3 / m4a 等非 WAV 音频 | `brew install ffmpeg` |

**模型在首次使用时下载**，app 里没有自带模型。第一次转写约下载 2 GB（paraformer-large、断句、标点模型）；第一次开实时字幕，根据识别语言和翻译引擎，最多再下载约 1.7 GB。开始下载前，app 会提示要下载哪些模型、大概多大。每个模型只下载一次，之后从本地缓存读取（`~/.cache/modelscope`、`~/.cache/huggingface`、`~/Documents/meetingscribe/models`）。

**app 的文件放在哪里**：都在 `~/Documents/meetingscribe/` 下面，不放在 app 包里，因为每次更新都会整个替换 app 包。

| 路径 | 内容 |
|------|------|
| `config.jsonc` | 配置。第一次启动时按默认值生成；常用选项在「配置 → 常用设置」里改，其余在「高级」里改 |
| `hotword.jsonc` | 热词表。想沿用源码版的热词：`cp hotword.jsonc ~/Documents/meetingscribe/` |
| `goals.json` | 自定义目标 |
| `recordings/` | 录音和所有生成的文件 |

### 打包 app

```bash
python3 -m pip install pyinstaller   # 只需一次，装到运行 meetingscribe.py 的那个 Python 里
packaging/build_dmg.sh               # → dist/MeetingScribe-<版本号>.dmg（完整构建，约 10 分钟）
packaging/build_dmg.sh --update      # 几秒：只把 meetingscribe.py 和安装说明更新进 dist/MeetingScribe.app
```

版本号只在 `meetingscribe.py` 的 `__version__` 里改。构建过程中需要约 4 GB 空闲磁盘放中间文件，结束后会自动删除。如果 `config.jsonc` 里还留着热词列表，构建会直接拒绝，因为这个文件会随 app 发给每个用户。

---

## 安装步骤 — macOS（源码运行）

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

1. **录音** — 点中间的大麦克风按钮开始录音，再点一次停止。
   - **日程条**（录音按钮上方，开启「日程关联」后显示）：录音前选中今天的某个日程。录音文件会以日程标题命名，参会人和日程里链接的 Notion 页面会交给纪要环节，用来给说话人对上真实姓名、给纪要补充背景。日程通过 `claude` 命令行的连接器读取 Google 日历（失败时改用 `codex`）；在「配置」里开启，开启时会先检查连接是否可用。
   - **实时字幕面板**（右侧整列，标题旁的开关控制开关；字幕出任何问题都不影响录音本身）：
     - *识别*（「识别」下拉框，录音中也能切换）：英文用 NVIDIA Nemotron Speech 0.6B；中文用流式 Paraformer，每条定稿的中文字幕还会在后台用离线 FunASR 模型重新识别一遍。
     - *翻译*（「翻译」下拉框：中文 / English / 不翻译）：默认用 HY-MT 1.5 1.8B（`live_captions.mt_provider: "hymt"`，通过 llama.cpp 在本地运行）；也可以改用 `qwen`、`nllb` 或 opus-mt（`marian`），HY-MT 加载失败时自动退回 opus-mt。
     - *说话人*：每条字幕按声纹标成 **[说话人N]**，麦克风和会议对方一视同仁，会议室里共用一支麦克风的同事也能分开；声纹结果出来之前（约 60 毫秒）先显示 **[我方]** / **[对方]**。
     - *大模型复核*：每新增两条字幕，就把最近五条连同译文交给大模型，借助后文修正前文里听错的术语。这是字幕里唯一需要联网调用大模型的部分；设 `live_captions.review.enabled: false` 可以让字幕完全在本地运行。
     - 模型在首次使用时下载：识别模型英文约 630 MB、中文约 230 MB，HY-MT 约 1.1 GB；下载前 app 会先提示。
   - **流水线按钮**（计时器下方，作用于最近一次录音）：
     - **语音转文字** — FunASR 转写 + 大模型校对 → `.polish.txt`
     - **生成会议纪要** → `.meeting.md`
     - **生成面试报告** → `.interview.md`
     - **生成分享总结**（知识分享、技术讲座，含观众问答）→ `.sharing.md`
     - **自定义目标 ▾** — 自己定义一种输出。用一句话描述想要什么（例如「列出客户提出的所有需求和提出人」），让 AI 扩写成完整的 Prompt（可以再改），保存后就能对任意录音生成 → `.goal-<id>.md`。目标保存在 `~/Documents/meetingscribe/goals.json`。

     对应输出文件已经存在时，按钮会变成蓝色的「打开 X」，点击直接打开文件。
2. **历史** — 完整会议列表，四个筛选 tab（全部 / 已总结 / 已录音转文字 / 待处理），右侧详情面板，以及同样的一组流水线按钮（含自定义目标），可以对旧录音生成或重新生成。任一会议右键可以重命名或删除（带二次确认）。
3. **配置**
   - **运行环境** — 检查 BlackHole、多输出设备、`claude` / `codex` 命令行和 ffmpeg，每项给出处理方法，命令行可以一键测试是否已登录。
   - **常用设置** — 常改的选项都做成了下拉框和开关，改完立即保存：校对和生成纪要用哪个模型、备选模型、转写是否区分说话人、字幕翻译引擎、字幕二次识别 / 说话人标注 / 大模型复核、自动积累热词，还有「编辑热词表」按钮和各个数据文件的具体位置。
   - **日程关联**开关（开启前先检查连接）、并发数设置，以及收在「高级」里的 `config.jsonc` 原文编辑器，用来改其余配置（包括 `cfg["prompts"]` 里的 prompt）。在表单里改设置只会改动对应的那一行，`config.jsonc` 里的注释都会保留。

左侧导航底部（「配置」下方）有 **中文 / EN** 切换项，实时翻转所有界面文字、对话框和提示。

### 命令行

```bash
# 录音 + 自动生成会议纪要（Ctrl+C 停止录音）
python3 meetingscribe.py record

# 面试 / 分享模式
python3 meetingscribe.py record --mode interview
python3 meetingscribe.py record --mode sharing

# 处理已有录音文件
python3 meetingscribe.py transcribe /path/to/audio.wav
python3 meetingscribe.py transcribe /path/to/audio.wav --mode interview

# 扫描已有转写/纪要，把热词回填进 config.jsonc
# （之后每次生成纪要都会自动增量维护热词，无需再手动跑）
python3 meetingscribe.py hotwords          # 规则 + LLM 提取
# 一次性从 Notion 导入成员名 / 项目名 / 术语（token 只从环境变量读，不落盘）
export NOTION_TOKEN=ntn_xxx
python3 meetingscribe.py hotwords --notion --dry-run     # 先预览
python3 meetingscribe.py hotwords --notion --min-cjk 3   # 剔除易误触的 2 字中文词
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
| `.sharing.md` | 分享总结（Markdown），`mode=sharing` 时生成 |
| `.goal-<id>.md` | 自定义目标的生成结果（每个目标一个文件） |
| `.meta.json` | 关联的日程信息（标题、时间、参会人、Notion 页面），仅在关联了日程时生成 |

录音也可以带一个可读后缀 — `<时间戳>.<自定义名字>.<后缀>`（例如 `20260512_090120.客户访谈.wav`），可通过历史界面的右键「重命名」设置。早期的单一 `.md` 文件（没有 `.meeting` / `.interview` 中段）仍然兼容识别。

> 重复运行时会自动检测已完成的步骤并跳过，无需重新转写。

---

## 配置说明

配置文件为项目目录下的 `config.jsonc`（macOS app 为 `~/Documents/meetingscribe/config.jsonc`，也可以在「配置」页直接编辑），支持 `//` 注释。常用配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `mode` | `meeting` | 默认模式：`meeting`（会议）/ `interview`（面试）/ `sharing`（分享） |
| `transcribe_provider` | `funasr` | 转写引擎：`funasr`（本地，默认）/ `openai` / `gemini` |
| `polish_provider` | `claude` | 校对模型：`claude` / `codex_proxy` / `codex_personal` / `openai` / `gemini` |
| `meeting_notes_provider` | `claude` | 纪要模型，可选值同 `polish_provider` |
| `llm_fallback` | `["claude"]` | 主模型失败时依次尝试的备选；`[]` 表示失败即停止 |
| `calendar.enabled` | `false` | 日程关联（见上文「日程条」）；`calendar.provider_chain` 决定用哪个命令行读取日历 |
| `stt.funasr.hotword` | `""` | 保持为空——实际热词表在 **`hotword.jsonc`**，该文件不入版本库（它会累积同事姓名与客户名）。`cp hotword.jsonc.example hotword.jsonc` 即可开始；每次生成纪要后自动提取新术语（规则 + LLM，见 `hotwords` 配置块）并合并到那里，`python3 meetingscribe.py hotwords` 可从已有录音一次性回填 |
| `hotwords.auto_update` | `true` | 热词自动提取总开关（方案 A 规则 + 方案 B LLM） |
| `live_captions.language` | `en` | 字幕识别语言，`en` / `zh`（字幕面板的「识别」下拉框会写这里） |
| `live_captions.translate_to` | `zh` | 字幕翻译目标，`zh` / `en` / `off`（字幕面板的「翻译」下拉框会写这里） |
| `live_captions.mt_provider` | `hymt` | 字幕翻译引擎：`hymt`（HY-MT 1.5 1.8B，专门的翻译模型，约 1.1 GB）、`qwen`（Qwen 1.5B，能顺带纠正中文识别错误，但翻英文不可靠）、`nllb`，或 `marian`（opus-mt，最快，长句会被截断） |
| `live_captions.history_minutes` | `180` | 字幕面板可回看的时长（分钟）。按时间保留，与语速无关；`history_max`（12000 行）只是内存兜底 |
| `live_captions.speaker_id.enabled` | `true` | 声纹说话人识别（cam++，约 27MB 自动下载，每行约 60ms，独立线程）。两个声道一视同仁地按声纹区分人——会议室里几个人共用一支麦克风也能分开，所有人统一编号为 说话人1 / 说话人2 …，声道只决定标签颜色。`threshold`（0.5）在「并成同一个人」和「拆成多个人」之间取舍 |
| `live_captions.review.enabled` | `true` | 滚动复核：每新增 `trigger_lines`（2）条，就把最近 `window_lines`（5）条字幕连同译文交给大模型（`review.provider`，留空则用 `polish_provider`）——这是唯一能「用后文修正前文」的一层。代价是录音期间每批一次 LLM 调用 |

**FunASR 配置项（`stt.funasr` 下）：**

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `model` | `paraformer-zh` | ASR 模型，首次运行自动下载（约 950 MB；连同断句、标点模型共约 2 GB） |
| `vad_model` | `fsmn-vad` | 语音活动检测，支持长音频分句 |
| `punc_model` | `ct-punc` | 标点恢复模型 |
| `spk_model` | `""` | 说话人区分（声纹聚类）；填 `"cam++"` 开启后转写行变成 `[12.3s] [说话人1] …`，校对与纪要环节会把发言人区分带进总结。**代价**：说话人编号只在单次推理内一致，因此开启后放弃分块并发、单趟转写，长录音耗时约为原来的 `workers` 倍 |
| `hotword` | `""` | 热词（空格分隔），提升领域专有词识别率 |
| `chunk_secs` | `300` | 超过此时长自动分块并发转写（秒），`0` = 始终串行；设置了 `spk_model` 时此项被忽略 |
| `workers` | `0` | 并发实例数，`0` = 自动（`max(6, CPU核数/2)`），内存随之线性增长 |

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
第一次转写会下载 `paraformer-zh`、`fsmn-vad`、`ct-punc` 三个模型（合计约 2 GB）；第一次开实时字幕会下载识别和翻译模型（最多约 1.7 GB）。开始下载前 app 会提示要下载哪些、多大；字幕模型下载时，字幕面板下方还会显示进度。模型下载后缓存在本地，之后不再重复下载。首次运行时请保持网络畅通。

**Q：macOS 提示 app「无法打开」或「无法验证开发者」**
app 没有经过苹果公证。在「应用程序」里右键 → 打开 → 再点「打开」，或在终端运行 `xattr -dr com.apple.quarantine /Applications/MeetingScribe.app`。

**Q：如何确认设备名称是否正确**
```bash
python3 meetingscribe.py devices
```
列出的名称即为可填入 `config.jsonc` 的值。注意名称区分大小写。

**Q：macOS 要求麦克风权限**
首次运行时系统会弹出权限请求，点击「允许」即可。
如已拒绝，可在「系统设置」→「隐私与安全性」→「麦克风」中重新开启 Terminal / Python 的权限。
