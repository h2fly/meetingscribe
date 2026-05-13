# MeetingScribe — Claude Code Guide

## Project Overview

Single-file Python tool: record audio → Whisper transcribe → LLM polish → meeting notes / interview summary.

- **Entry point**: `meetingscribe.py` (everything in one file)
- **Config**: `config.jsonc` (JSONC with `//` comments, sits next to the script)
- **Output dir**: `~/Documents/meetingscribe/recordings/` (created at runtime)

## Running the Tool

```bash
python3 meetingscribe.py ui                  # GUI
python3 meetingscribe.py record              # CLI recording
python3 meetingscribe.py transcribe foo.wav  # process existing file
python3 meetingscribe.py devices             # list audio devices
python3 meetingscribe.py config              # view/edit config
```

## Architecture

The pipeline has four steps, each resumable independently:

```
.wav  →  .raw.txt  →  .proofread.txt  →  .md
        (FunASR)      (LLM polish)      (LLM notes)
```

Each step checks whether its output file already exists and skips if so. This means re-running after a crash resumes from where it stopped.

## Key Functions

| Function | Purpose |
|----------|---------|
| `load_config()` | Reads `config.jsonc`, deep-merges with `DEFAULT_CONFIG` |
| `switch_output(name)` | Switches macOS audio output device via CoreAudio ctypes; **no-op on Windows/Linux** |
| `DualStreamRecorder` | Two simultaneous `sounddevice.InputStream`s — BlackHole (system audio) + mic |
| `transcribe()` | Dispatches to `_transcribe_funasr/whisper/openai/gemini` |
| `polish_transcript()` | Splits transcript into chunks, runs LLM in parallel via `ThreadPoolExecutor` |
| `generate_notes()` | Single LLM call using `PROMPTS[mode]["notes"]` |
| `_llm_run()` | Dispatches to `_llm_claude_cli / _llm_openai / _llm_gemini` |
| `cmd_ui()` | Full Tkinter GUI (~400 lines); uses `queue.Queue` + `root.after(100, _poll)` for thread→UI updates |

## Provider System

Three independently configurable providers (set in `config.jsonc` or via CLI flags):

- `transcribe_provider`: `funasr` (local, default) | `whisper` (local faster-whisper) | `openai` | `gemini`
- `polish_provider`: `claude` | `openai` | `gemini`
- `meeting_notes_provider`: `claude` | `openai` | `gemini`

The `claude` LLM type calls `claude -p <prompt>` via subprocess (Claude Code CLI must be installed).

## Config System

- `config.jsonc` is parsed with `_strip_jsonc_comments()` (regex strips `//` lines)
- `_deep_merge(DEFAULT_CONFIG, on_disk)` — nested dicts merge recursively; on-disk values win
- `load_config()` never auto-writes back; `save_config()` writes plain JSON (comments lost)
- CLI flags override config at runtime but never persist

## GUI (Tkinter)

- Dark gray-blue theme; color palette defined at the top of `cmd_ui()`
- Thread safety: background pipeline runs in a daemon thread, communicates via `log_q: queue.Queue`
- `_poll()` drains the queue every 100 ms via `root.after(100, _poll)`
- Queue message types: `("log", str)` | `("progress", int)` | `("done", path)` | `("error", str)`
- Progress bar is a `tk.Canvas` drawing a percentage fill (not `ttk.Progressbar`)
- `cancel_flag = [False]` — set by "停止任务"; pipeline thread runs to completion but `_poll` discards its result

## Platform Notes

- **macOS**: Full support. `switch_output()` uses CoreAudio ctypes to auto-switch audio devices around recording.
- **Windows**: `switch_output()` is a no-op. User sets up VoiceMeeter Banana + VB-Audio Virtual Cable manually. Use `os.startfile()` to open result files.
- **Linux**: Untested. `switch_output()` is a no-op. `xdg-open` used for result files.

## File Naming Convention

All output files share the same stem as the recording:

```
20260512_090120.wav
20260512_090120.raw.txt
20260512_090120.proofread.txt
20260512_090120.md
```

## Dependencies

```
sounddevice    # audio capture (cross-platform, wraps PortAudio)
funasr         # local FunASR inference (default transcribe provider)
modelscope     # model downloading for FunASR
torch          # PyTorch backend for FunASR (install separately via pytorch.org)
numpy          # audio buffer handling
faster-whisper # optional: only needed when transcribe_provider=whisper
```

Tkinter, wave, argparse, subprocess, ctypes — all stdlib.  
No test suite. Manual testing via `python3 meetingscribe.py ui`.

## Common Edit Patterns

**Add a new LLM provider**: add entry to `DEFAULT_CONFIG["llm"]`, add a `_llm_<name>()` function, dispatch in `_llm_run()`.

**Add a new STT provider**: add entry to `DEFAULT_CONFIG["stt"]`, add `_transcribe_<name>()`, dispatch in `transcribe()`.

**Change prompts**: edit `PROMPTS` dict — `PROMPTS["meeting"]["polish"]`, `PROMPTS["meeting"]["notes"]`, same for `"interview"`.

**Modify GUI layout**: all widgets are in `cmd_ui()`. Cards are created with the `card()` helper. Separators with `sep()`.
