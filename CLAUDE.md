# MeetingScribe — Claude Code Guide

## Project Overview

Single-file Python tool: record audio → Whisper transcribe → LLM polish → meeting notes / interview summary.

- **Entry point**: `meetingscribe.py` (everything in one file)
- **Config**: `config.jsonc` (JSONC with `//` comments, sits next to the script)
- **Output dir**: `~/Documents/meetingscribe/recordings/` (created at runtime)

## Running the Tool

```bash
python3 meetingscribe.py ui                  # GUI — Tkinter (default, no extra deps)
python3 meetingscribe.py ui-qt               # GUI — PyQt6 + Fluent (needs `python3 -m pip install PyQt6 PyQt6-Fluent-Widgets`)
python3 meetingscribe.py record              # CLI recording
python3 meetingscribe.py transcribe foo.wav  # process existing file
python3 meetingscribe.py devices             # list audio devices
python3 meetingscribe.py config              # view/edit config
```

## Architecture

The pipeline has four steps, each resumable independently:

```
.wav  →  .raw.txt  →  .polish.txt  →  .md
        (FunASR)      (LLM polish)      (LLM notes)
```

Each step checks whether its output file already exists and skips if so. This means re-running after a crash resumes from where it stopped.

## Key Functions

| Function | Purpose |
|----------|---------|
| `load_config()` | Reads `config.jsonc`, deep-merges with `DEFAULT_CONFIG` |
| `_log(category, message)` | **File-only diagnostic logger** — writes `[HH:MM:SS] [CATEGORY] msg` to the daily log file. Never touches stdout/stderr by default (use `print()` for user-facing output). `--debug` mirrors lines to stderr. Canonical categories: `REC`/`DEVICE`/`AUDIO`/`HOTPLUG`/`RESTORE`/`STREAM`/`MUTE`/`STT`/`POLISH`/`LLM`/`PIPELINE`/`CONFIG`/`MONITOR`/`ERR`/`WARN`. |
| `_QuietCapture(category)` | Context manager that redirects stdout+stderr into buffers and forwards captured lines to `_log()`. Wrapped around FunASR `m.generate()` calls to keep tqdm bars out of the console. |
| `switch_output(name)` | Switches macOS audio output device via CoreAudio ctypes; **no-op on Windows/Linux** |
| `resolve_audio_devices(query_fresh)` | **Single source of truth** for mic / system-audio source / restore target. Transport-priority (external > built-in; aggregates/virtual rejected for raw picks). Returns an `AudioPlan` dataclass. Called at start, hotplug, and stop. |
| `_restore_output_if_needed(plan)` | Stop-time restore policy: skips `switch_output` when current default is already a physical device (prevents music-app pause). Falls back through the priority list if the plan's target is missing. |
| `_install_device_listeners()` / `_remove_device_listeners()` | macOS-only CoreAudio HAL property listeners on `kAudioHardwarePropertyDevices` + `…DefaultSystemOutputDevice`. Set the module-level `_hotplug_event` so the recorder reacts within ~100 ms. |
| `AudioDeviceMonitor` | Event-driven daemon thread, started by `cmd_ui` and `cmd_record`. Blocks on `_hotplug_event.wait(timeout=_AUDIO_MONITOR_SAFETY_TIMEOUT_SEC)` (30 s) — no fixed-period polling. Runs one synchronous initial tick on `start()` so the process always syncs to the current best device on launch. **Idle branch** (when `_recording_active` is clear): calls `resolve_audio_devices(query_fresh=True)`; runs `_restore_output_if_needed` on the first tick and on every change to the resolved `(restore_output_name, mic_name, sys_source_name)` triple. **Recording branch**: never switches dOut (would pause music apps) and never refreshes PortAudio (would terminate live streams); but DOES call `resolve_audio_devices(query_fresh=False)` + `_reconcile_recording_mutes(plan)` when `(multi_output_name, restore_output_name)` changes, so the Multi-Output's inactive physical sub-devices stay silenced mid-recording on hotplug. The recorder's own `_monitor` thread remains the authoritative reconciler for input streams. Replaces the older GUI-only `_idle_dout_watchdog`. |
| `MultiStreamRecorder` | N simultaneous `sounddevice.InputStream`s; `_monitor` thread (tracked as `self._monitor_thread`) waits on `_hotplug_event` (or 1 s fallback) and on each wake calls `_monitor_iteration()` which: (1) queries the live PortAudio input device set, (2) re-derives `self.wanted` from `resolve_audio_devices(query_fresh=False)` so mic/system-source swaps mid-recording are honoured (USB headset plugged in → `_close_one` built-in mic, `_try_open` USB mic, pre-swap frames are prepended into the new device's frame list to preserve audio continuity), (3) opens newly-appearing wanted devices and closes disappeared / no-longer-wanted ones. `start()` sets the module-level `_recording_active`; `stop()` clears it in a `finally`. `stop()` **joins** `self._monitor_thread` (2 s timeout) BEFORE clearing `self._streams`, eliminating the race where `_monitor_iteration`'s `_try_open` could reset `self._frames` mid-stop and discard captured audio. `_try_open` defensively re-checks `self.recording` inside its lock and discards orphan opens with `[STREAM] discarded post-stop open of device=…`. Emits `mic-not-opened` / `system-audio-not-opened` warnings via `on_warning` and writes `<stem>.warnings.txt` sidecar. |
| `transcribe()` | Dispatches to `_transcribe_funasr/whisper/openai/gemini` |
| `polish_transcript()` | Splits transcript into chunks, runs LLM in parallel via `ThreadPoolExecutor` |
| `generate_notes()` | Single LLM call using `PROMPTS[mode]["notes"]` |
| `_llm_run()` | Dispatches to `_llm_claude_cli / _llm_openai / _llm_gemini` |
| `cmd_ui()` | Full Tkinter GUI (~400 lines); uses `queue.Queue` + `root.after(100, _poll)` for thread→UI updates |
| `cmd_ui_qt()` | PyQt6 + PyQt6-Fluent-Widgets GUI (Phase 0 MVP). Coexists with `cmd_ui()`. Lazy imports so PyQt is not required for the Tk path. Inner classes: `_PipelineWorker` (QObject on QThread), `_RecorderState` (Qt signals wrap of `MultiStreamRecorder` + dOut/mute lifecycle), `RecordingInterface` (mic button / timer / device combo / volume slider / action buttons / progress / history sidebar), `HistoryInterface` (search + filter SegmentedWidget + list + `.md` TextBrowser detail; participants/todos are mock data in Phase 0), `MainWindow` (FluentWindow with two sub-interfaces). Cross-thread UI updates use `QTimer.singleShot(0, ...)` instead of Tk's `queue.Queue + root.after` polling. Reuses every backend function (`MultiStreamRecorder`, `_reconcile_recording_mutes`, `_restore_all_recording_mutes`, `_restore_output_if_needed`, `_get_audio_monitor`, `transcribe`, `polish_transcript`, `generate_notes`, `save_minutes`). |

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

### macOS music-pause behavior on recording

Recording system audio requires routing playback through the Multi-Output Device (which tees to BlackHole for capture). The recording start/stop paths therefore change `kAudioHardwarePropertyDefaultOutputDevice` (`dOut`) twice per session: physical → Multi-Output at start, Multi-Output → physical at stop. macOS delivers a default-output-changed event on every `dOut` write, and many audio apps (Apple Music, Spotify, Safari, Chrome) react by pausing playback.

To make this workflow music-pause-free, the start path is **conditional**: if the user's permanent `dOut` is already the Multi-Output Device, the start-time switch is skipped. The `_recording_did_switch` event tracks whether the start path actually performed a switch; `_restore_output_if_needed(plan, reason="post-recording")` only runs when the event is set. Idle-state callers (`AudioDeviceMonitor`) bypass the gate via `reason="idle-event"` so hotplug-driven `dOut`+`sOut` alignment still works.

**Lifecycle ordering** (important for maintainers): `_recording_did_switch` is cleared at the **entry of the recording-start lifecycle** in both `cmd_record` and `cmd_ui` — BEFORE the start-time `switch_output` decision. The start path sets the flag only when `switch_output(multi_output_name)` actually runs. `MultiStreamRecorder.start()` does NOT touch the flag (clearing it there would wipe the bit set moments before by the start-path switch).

**Recommended user setup for zero music pauses:**
1. In *Audio MIDI Setup*, configure the Multi-Output Device to contain only `[BlackHole 2ch, your preferred physical output]` (e.g. BlackHole + External Headphones). Including multiple physical outputs causes audio to play through all of them simultaneously.
2. Right-click the Multi-Output Device in Audio MIDI Setup and set it as the macOS default output.
3. Now MeetingScribe's start/stop are silent — `_recording_did_switch` stays clear and music apps see no device change.

**Known limitation**: when the permanent `dOut` is a physical device (the default macOS setup most users have), bugs 1 and 3 of the music-pause report are intrinsic to the dOut-switching strategy and cannot be eliminated without a different recording stack (a dynamic session-scoped aggregate, rejected in `robust-audio-device-handling`; or `SCStream` screen-capture-with-audio, out of scope).

### macOS per-device mute lifecycle (Multi-Output sub-devices)

When the user's Multi-Output Device fans out to BOTH speakers AND headphones simultaneously (sub-list includes `BuiltInSpeakerDevice` + `BuiltInHeadphoneOutputDevice`), plugging in headphones during a recording causes audio to play from both at once. The fix is to mute the inactive physical sub-device(s) for the duration of the recording — leaving `dOut` unchanged (no music pause) while BlackHole capture continues uninterrupted.

**Helpers** (`meetingscribe.py`):

- `_ca_get_device_mute(name)` / `_ca_set_device_mute(name, muted)` — CoreAudio `kAudioDevicePropertyMute` accessors, output scope, master element. macOS only.
- `_get_multi_output_physical_subs(name)` — reads the Multi-Output's `kAudioAggregateDevicePropertyComposition` CFDictionary (we use `acom`, not `agdv` — the latter returns `kAudioHardwareUnknownPropertyError` on at least some macOS versions), extracts each sub-device's UID, looks up the matching device name, filters out virtual / aggregate transports.
- `_reconcile_recording_mutes(plan)` — for each physical sub of `plan.multi_output_name`, ensures `desired = (sub != plan.restore_output_name)` matches the device's current mute state. Wires the lifecycle through `_active_mutes: dict[name, original_bool]` so `_restore_all_recording_mutes()` can put each touched device back.

**Lifecycle hooks** (already wired):

- Recording start (CLI `_cmd_record_body`, GUI `_start_recording`): call `_reconcile_recording_mutes(plan)` AFTER the existing `switch_output(plan.multi_output_name)` block.
- Recording stop / abort / on_close / atexit: call `_restore_all_recording_mutes()` BEFORE `_restore_output_if_needed` so any silenced speaker comes back at the moment dOut returns to the user's physical output.
- Hotplug mid-recording: `AudioDeviceMonitor._recording_branch` re-resolves and re-reconciles when the `(multi_output_name, restore_output_name)` pair changes. Memoized via `self._prev_mute_triple` to avoid redundant work.

**Asymmetric tracking** (important): mute-state changes are recorded asymmetrically.

- When we mute a previously-unmuted device (False→True), we save the original `False` to `_active_mutes`; `_restore_all_recording_mutes()` puts it back to `False` on stop. The user's speakers come back online.
- When we unmute a previously-muted device (True→False) — which happens when the active listening target had a stale system-level mute — we DO NOT save anything. `_restore_all_recording_mutes()` will not re-mute. Rationale: pre-existing mute on a device the user is currently listening through is almost always a leftover state (macOS does not reliably clear device-level mutes across plug cycles); restoring it would put the user back into silence and re-create the same bug they're trying to escape.

**Crash recovery**: every change to `_active_mutes` is mirrored to `CONFIG_DIR / .active_mutes.json` via atomic write (`.tmp` + `os.replace`). At module load, `_recover_persisted_mutes()` reads the file and — if the recorded pid is dead (`os.kill(pid, 0)` raises `ProcessLookupError`) or matches our own pid — restores each device's recorded original then deletes the file. Live concurrent MeetingScribe processes are detected via `_pid_is_alive()` and never touched. The `_atexit_restore_mutes()` handler runs `_restore_all_recording_mutes()` as a final pass on normal exit; it is wrapped to never raise (atexit must not block process exit).

**Diagnostic log tag**: `[MUTE]`. Every reconcile / mute / unmute / persist / recover / restore-all writes a structured line so the daily log alone shows whether the lifecycle behaved correctly. Example trace from a normal start-with-headphones-plug-mid-recording-then-stop sequence:

```
[MUTE] reconcile: active='MacBook Air Speakers' multi_subs=['MacBook Air Speakers'] tracked=[]
[MUTE] device='MacBook Air Speakers' muted (original=False) ok=True
[MUTE] reconcile: active='External Headphones' multi_subs=['MacBook Air Speakers', 'External Headphones'] tracked=['MacBook Air Speakers']
[MUTE] persist: muted={'MacBook Air Speakers': False}
[MUTE] device='MacBook Air Speakers' unmuted (reverting our prior mute; not restoring at stop) ok=True
[MUTE] persist: deleted (no active mutes)
```

## File Naming Convention

All output files share the same stem as the recording:

```
20260512_090120.wav
20260512_090120.raw.txt
20260512_090120.polish.txt
20260512_090120.md
```

## Dependencies

```
sounddevice           # audio capture (cross-platform, wraps PortAudio)
funasr                # local FunASR inference (default transcribe provider)
modelscope            # model downloading for FunASR
torch                 # PyTorch backend for FunASR (install separately via pytorch.org)
numpy                 # audio buffer handling
faster-whisper        # optional: only needed when transcribe_provider=whisper
PyQt6                 # optional: only needed for `ui-qt` (Fluent GUI)
PyQt6-Fluent-Widgets  # optional: only needed for `ui-qt` (Fluent GUI). NOT PyQt-Fluent-Widgets — that one pulls PyQt5 and collides with PyQt6.
```

Tkinter, wave, argparse, subprocess, ctypes — all stdlib.  
Tests: `pytest tests/` (142 unit tests; mocked CoreAudio for cross-platform CI). Manual smoke testing via `python3 meetingscribe.py ui`.

## Common Edit Patterns

**Add a new LLM provider**: add entry to `DEFAULT_CONFIG["llm"]`, add a `_llm_<name>()` function, dispatch in `_llm_run()`.

**Add a new STT provider**: add entry to `DEFAULT_CONFIG["stt"]`, add `_transcribe_<name>()`, dispatch in `transcribe()`.

**Change prompts**: edit `PROMPTS` dict — `PROMPTS["meeting"]["polish"]`, `PROMPTS["meeting"]["notes"]`, same for `"interview"`.

**Modify GUI layout**: all widgets are in `cmd_ui()`. Cards are created with the `card()` helper. Separators with `sep()`.
