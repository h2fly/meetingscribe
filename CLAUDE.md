# MeetingScribe — Claude Code Guide

## Project Overview

Single-file Python tool: record audio → FunASR transcribe → LLM polish → meeting notes / interview summary / sharing summary.

- **Entry point**: `meetingscribe.py` (everything in one file)
- **Config**: `config.jsonc` (JSONC with `//` comments, sits next to the script)
- **Output dir**: `~/Documents/meetingscribe/recordings/` (created at runtime)

## Project Rules

These are non-negotiable conventions for this codebase. Violations are bugs even if tests pass.

- **All prompts must live in `config.jsonc`.** Every prompt the pipeline sends to any LLM (currently `polish`, and `notes_zh` / `notes_en` for each mode) is shipped as an editable block in `config.jsonc`. The `_PROMPT_DEFAULTS` dict in `meetingscribe.py` is **fallback only** — never the sole source. Rationale: prompts are the main lever for tuning product behaviour, and operations / business / non-Python contributors must be able to read and adjust them without touching code. When you add a new mode or a new prompt slot, you MUST:
  1. Add the default to `_PROMPT_DEFAULTS` (so the tool still works if the user deletes `config.jsonc`), AND
  2. Add the exact same prompt as an editable block in `config.jsonc` with a `//` comment explaining what the prompt is for, AND
  3. Verify with `python3 -c "from meetingscribe import load_config, _resolve_prompt, _PROMPT_DEFAULTS; ..."` that `_resolve_prompt` on the fresh config returns text identical to the `_PROMPT_DEFAULTS` entry — i.e. the jsonc override is a faithful copy, not a drifted variant.

## Running the Tool

```bash
python3 meetingscribe.py ui                              # GUI — PyQt6 + Fluent (needs `python3 -m pip install PyQt6 PyQt6-Fluent-Widgets`)
python3 meetingscribe.py record                          # CLI recording (mode defaults to `meeting`)
python3 meetingscribe.py record --mode interview         # interview mode
python3 meetingscribe.py record --mode sharing           # knowledge-sharing / tech-talk mode
python3 meetingscribe.py transcribe foo.wav --mode sharing  # process existing file as a sharing session
python3 meetingscribe.py devices                         # list audio devices
python3 meetingscribe.py config                          # view/edit config
```

Pipeline modes (`--mode`): `meeting` (default, 会议纪要) | `interview` (面试总结) | `sharing` (分享总结, knowledge-sharing / tech talks with audience Q&A). The set of valid modes is exported as `MODES` in `meetingscribe.py` and enforced by argparse.

## Architecture

The pipeline has four steps, each resumable independently:

```
.wav  →  .raw.txt  →  .polish.txt  →  .<mode>.md
        (FunASR)      (LLM polish)       (LLM notes)
```

`<mode>` is one of `meeting` / `interview` / `sharing` (see "Pipeline modes" above). Each mode lives at a distinct suffix (`.meeting.md` / `.interview.md` / `.sharing.md`) so the same recording can carry summaries for multiple modes side by side.

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
| `MultiStreamRecorder` | N simultaneous `sounddevice.InputStream`s; `_monitor` thread (tracked as `self._monitor_thread`) waits on `_hotplug_event` (or 1 s fallback) and on each wake calls `_monitor_iteration()` which: (1) queries the live PortAudio input device set, (2) re-derives `self.wanted` from `resolve_audio_devices(query_fresh=False)` so mic/system-source swaps mid-recording are honoured (USB headset plugged in → `_close_one` built-in mic, `_try_open` USB mic, pre-swap frames are prepended into the new device's frame list to preserve audio continuity), (3) opens newly-appearing wanted devices and closes disappeared / no-longer-wanted ones. `start()` sets the module-level `_recording_active`; `stop()` clears it in a `finally`. `stop()` **joins** `self._monitor_thread` (2 s timeout) BEFORE clearing `self._streams`, eliminating the race where `_monitor_iteration`'s `_try_open` could reset `self._frames` mid-stop and discard captured audio. `_try_open` defensively re-checks `self.recording` inside its lock and discards orphan opens with `[STREAM] discarded post-stop open of device=…`. Emits `mic-not-opened` / `system-audio-not-opened` warnings via `on_warning` and the `[WARN]` log category — warnings are log-only by design; no sidecar `.warnings.txt` is written. |
| `transcribe()` | Dispatches to `_transcribe_funasr/openai/gemini` |
| `_load_funasr_automodel(asr, vad, punc)` / `_patch_funasr_jieba_dicts()` / `_patch_one_jieba_dict(path)` / `_indexerror_came_from_jieba(exc)` | Self-heal for the FunASR ↔ jieba dictionary bug. FunASR's `iic/punc_ct-transformer_*` ships a `jieba_usr_dict` whose lines are `<word>` only; the installed jieba's module-level `load_userdict` (still at jieba/`__init__.py:307` in 0.42.1) crashes with `IndexError` on such lines. `_load_funasr_automodel` wraps `funasr.AutoModel(...)` in a `try/except IndexError`; on a jieba-originated error it calls `_patch_funasr_jieba_dicts()` which globs `<MODELSCOPE_CACHE or ~/.cache/modelscope>/hub/models/**/jieba_usr_dict` (skipping symlinks outside the cache root) and runs `_patch_one_jieba_dict` per file — append ` 1` to whitespace-free non-empty lines, atomic `tmp + os.replace`, drop a `.jieba_usr_dict.patched` sentinel — then retries `AutoModel` once. If the patch made no changes, or the retry also fails, the **original** IndexError is re-raised (with the retry failure chained via `raise ... from`). All decisions log under `[STT]`. |
| `polish_transcript()` | Splits transcript into chunks, runs LLM in parallel via `ThreadPoolExecutor`. Reads the prompt via `_resolve_prompt(cfg, "polish")` (top-level, mode-agnostic). |
| `generate_notes()` | Runs `_resolve_prompt(cfg, "notes_zh", mode=...)` + `notes_en` in parallel and concatenates the two outputs. |
| `_resolve_prompt(cfg, key, mode=None)` | Pipeline-prompt lookup. `mode=None` → top-level `cfg["prompts"][key]` (used for `polish`). `mode="meeting"/"interview"/"sharing"` → `cfg["prompts"][mode][key]` (used for `notes_zh`/`notes_en`). Accepts `str` or `list[str]` (joined on `\n`), with fallback to `DEFAULT_CONFIG["prompts"]`. `{transcript}` is substituted via `str.replace` (not `.format`). |
| `_llm_run()` | Dispatches to `_llm_claude_cli / _llm_openai / _llm_gemini` |
| `cmd_ui()` | PyQt6 + PyQt6-Fluent-Widgets desktop GUI — the project's only UI (lazy import; prints an install hint and exits if PyQt6 isn't available). Base class is `QMainWindow` (not `FluentWindow`) so macOS provides native traffic lights on the LEFT with standard hover icons; the Fluent left navigation is embedded as a `NavigationInterface` widget. Inner classes: `_PipelineWorker` (`QObject` on `QThread` — catches both `Exception` and `SystemExit`), `_RecorderState` (Qt-signal wrapper over `MultiStreamRecorder` + dOut/mute lifecycle), `RecordingInterface` (no mic selector — always system audio + resolver-chosen mic; live substring search + right-click rename / delete on the sidebar history), `HistoryInterface` (four `SegmentedWidget` tabs: 全部 / 已总结 / 已录音转文字 / 待处理, tab-specific body rendering, right-click rename / delete with confirmation, `"未获取相关信息"` placeholder for participants/todos until a `.meta.json` sidecar exists), `ConfigInterface` (SpinBox to bump `polish_max_workers` + `stt.funasr.workers` in one shot, plus a raw JSONC editor with syntax highlighting that validates via `_strip_jsonc_comments` + `json.loads` before writing back), `MainWindow`. Single 中文 / EN toggle in the top bar flips every translatable widget via per-view `apply_language()` + `_LABELS` lookup table. Cross-thread UI updates use `QTimer.singleShot(0, ...)`. Reuses every backend function (`MultiStreamRecorder`, `_reconcile_recording_mutes`, `_restore_all_recording_mutes`, `_restore_output_if_needed`, `_get_audio_monitor`, `transcribe`, `polish_transcript`, `generate_notes`, `save_minutes`, `save_config`, `_rename_meeting_files`, `_delete_meeting_files`). |
| `_split_meeting_stem(stem)` / `_rename_meeting_files(wav, name)` / `_delete_meeting_files(wav)` | Filename convention is `<timestamp>[.<custom>].<suffix>`. `_split_meeting_stem` parses the timestamp prefix (validated against `YYYYmmdd_HHMMSS`) and optional custom-name segment. `_rename_meeting_files` cascades a rename across every sibling file sharing the same stem (`.wav` + `.raw.txt` + `.polish.txt` + `.meeting.md` + `.interview.md` + `.sharing.md` + …) with collision detection. `_delete_meeting_files` removes the same sibling set, returning `(deleted_count, errors)`. Custom names are sanitised: reserved characters (`/ \\ : * ? " < > |` + C0 controls) are mapped to `_`, leading/trailing dots and whitespace are stripped. |

## Provider System

Three independently configurable providers (set in `config.jsonc` or via CLI flags):

- `transcribe_provider`: `funasr` (local, default) | `openai` | `gemini`
- `polish_provider`: `claude` | `openai` | `gemini`
- `meeting_notes_provider`: `claude` | `openai` | `gemini`

The `claude` LLM type calls `claude -p <prompt>` via subprocess (Claude Code CLI must be installed).

## Config System

- `config.jsonc` is parsed with `_strip_jsonc_comments()` (regex strips `//` lines)
- `_deep_merge(DEFAULT_CONFIG, on_disk)` — nested dicts merge recursively; on-disk values win
- `load_config()` never auto-writes back; `save_config()` writes plain JSON (comments lost)
- CLI flags override config at runtime but never persist

## GUI (PyQt6 + Fluent)

- Light theme (`setTheme(Theme.LIGHT)`); rounded cards via `_style_as_card(...)` + `_CARD_BG = "#f5f7fa"`.
- Thread safety: each pipeline runs on its own `QThread` via `_PipelineWorker` (a `QObject`). Signals: `progress(int)`, `log(str)`, `done(str)`, `failed(str)`. Cross-thread UI updates use `QTimer.singleShot(0, ...)`.
- Action buttons toggle between "Generate X" (light gray) and "Open X" (accent blue, `#0a84ff`) based on whether the corresponding artifact exists on disk — see `_apply_open_btn_style(btn, is_open)` and `_refresh_action_buttons()` / `_refresh_h_action_buttons()`.
- Internationalisation: `_LANG["current"]` holds `"zh"` / `"en"`; every translatable widget either registers a callback in `self._lang_callbacks` (static labels) or is re-rendered from a dynamic refresher inside `apply_language()` (state-dependent labels). The 中文 / EN toggle button (single button in the top bar) flips state and calls `apply_language()` on every view.
- `_confirm_dialog(parent, title, msg)` — custom Yes/No `QMessageBox` whose button labels go through `_t("ctx.confirm_yes")` / `_t("ctx.confirm_no")` so the EN toggle actually delivers an all-English UI (Qt's built-in `QMessageBox.question` would otherwise label buttons from the system locale).
- Right-click context menus on both history lists: 重命名 → 删除本次会议所有记录 → 在 Finder 中显示. Delete is gated on `self._pipeline_thread is None` (can't yank files from a running worker).

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
20260512_090120.meeting.md       # written when mode=meeting
20260512_090120.interview.md     # written when mode=interview
20260512_090120.sharing.md       # written when mode=sharing
```

A meeting may have a human-readable suffix as well — `<timestamp>.<custom_name>.<ext>` (e.g. `20260512_090120.客户访谈.wav`). The legacy single-`.md` file (no `.meeting`/`.interview`/`.sharing` infix) is still recognised for read / open / delete.

Suffix routing lives in `_NOTES_SUFFIX` (`meeting` → `.meeting.md`, `interview` → `.interview.md`, `sharing` → `.sharing.md`); `MODES = tuple(_NOTES_SUFFIX.keys())` is the single source of truth driving argparse `choices=`, the mode→label dict in `generate_notes`, and the GUI scanner.

## Dependencies

```
sounddevice           # audio capture (cross-platform, wraps PortAudio)
funasr                # local FunASR inference (default transcribe provider)
modelscope            # model downloading for FunASR
torch                 # PyTorch backend for FunASR (install separately via pytorch.org)
numpy                 # audio buffer handling
PyQt6                 # required for `ui` (the desktop GUI)
PyQt6-Fluent-Widgets  # required for `ui`. NOT PyQt-Fluent-Widgets — that one pulls PyQt5 and collides with PyQt6.
```

wave, argparse, subprocess, ctypes — all stdlib.
Tests: `pytest tests/` (185 unit tests; mocked CoreAudio for cross-platform CI). Manual smoke testing via `python3 meetingscribe.py ui`.

## Common Edit Patterns

**Add a new LLM provider**: add entry to `DEFAULT_CONFIG["llm"]`, add a `_llm_<name>()` function, dispatch in `_llm_run()`.

**Add a new STT provider**: add entry to `DEFAULT_CONFIG["stt"]`, add `_transcribe_<name>()`, dispatch in `transcribe()`.

**Change prompts**: edit `cfg["prompts"]` in `config.jsonc` (the user-facing surface, accepts both `str` and `list[str]`). The built-in defaults live in `_PROMPT_DEFAULTS` (above `DEFAULT_CONFIG`) but per the [Project Rules](#project-rules) section, `_PROMPT_DEFAULTS` is fallback only — every prompt also ships as an editable block in `config.jsonc`, and the two must stay byte-identical. Keys: `polish` is at the top level (mode-agnostic); `notes_zh` / `notes_en` live under `meeting`, `interview`, and `sharing` respectively. All call sites read via `_resolve_prompt(cfg, key, mode=None)`.

To customise just one mode without copy-pasting every default, drop the override into `config.jsonc` — `_deep_merge` keeps the other keys defaulted. For example, to ask the sharing-summary prompt to emphasise code snippets:

```jsonc
{
  "prompts": {
    "sharing": {
      // The literal "{transcript}" token is substituted via str.replace,
      // so other braces in your prompt do NOT need to be escaped.
      "notes_zh": [
        "你是一位技术分享整理助手。请在保留原有结构（分享概览 / 分享正文 / 核心要点 / 最佳实践 / 关键洞察 / 适用边界 / 风险与权衡 / 问答 / 行动建议）的基础上：",
        "- 在「分享正文」中**逐字保留代码片段、命令行、链接**，并用 ``` 围栏标注语言；",
        "- 「最佳实践」按「Do / Don't」两列罗列；",
        "- 「问答」若某问主讲人未直接回答，标注「（未直接回答 / 待跟进）」。",
        "",
        "---",
        "【分享转写】",
        "{transcript}"
      ]
    }
  }
}
```

The `notes_en` side falls back to the default — partial overrides are fine. Both `str` (literal) and `list[str]` (joined on `\n`) are accepted at every prompt slot.

**Modify GUI layout**: widgets live inside `cmd_ui()`'s three interface classes — `RecordingInterface`, `HistoryInterface`, `ConfigInterface` — all wrapped in `MainWindow`. Card surfaces via `_style_as_card(widget, padding=N, name="...")`. Pipeline-button styling via `_apply_open_btn_style(btn, is_open=bool)`.

**Add a translation key**: insert the key into both `_LABELS["zh"]` and `_LABELS["en"]` (top of `cmd_ui()`). Read in widgets via `_t("namespace.key")` or `self._i18n(widget, "namespace.key")` for the static-label callback variant.
