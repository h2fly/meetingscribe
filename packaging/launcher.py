"""Entry point of MeetingScribe.app (PyInstaller).

Kept separate from meetingscribe.py so the bundle-only concerns below run
BEFORE that module's heavy imports:

* `python -c CODE ARGS…` emulation. The bypass-audio forwarder starts a child
  with `[sys.executable, "-c", _PASSTHRU_CHILD, …]`; inside the bundle
  sys.executable is this app, which knows no `-c`. Without this the child
  would open a second GUI window instead of forwarding audio.
* multiprocessing.freeze_support(), so a spawned helper (torch / FunASR use
  multiprocessing) runs its task instead of relaunching the app.
* Finder starts an app with no arguments (and on older macOS a `-psn_…`
  process-serial argument): that means "open the GUI".
"""

import multiprocessing
import sys


def _load_meetingscribe():
    """Import meetingscribe, preferring the plain `meetingscribe.py` shipped
    next to the bundle's data files over the copy frozen into the archive.

    That is what makes `packaging/build_dmg.sh --update` possible: a change to
    meetingscribe.py alone is copied into the existing .app and re-signed in
    seconds, instead of a 10-minute PyInstaller rebuild of 1 GB of unchanged
    dependencies. The frozen copy stays as the fallback, and is also what
    PyInstaller analyses to decide which libraries to bundle.
    """
    import importlib.util
    from pathlib import Path
    src = Path(getattr(sys, "_MEIPASS", "")) / "meetingscribe.py"
    if getattr(sys, "frozen", False) and src.is_file():
        # Never write __pycache__ into the signed bundle — a new file there
        # breaks the code signature.
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location("meetingscribe", src)
        module = importlib.util.module_from_spec(spec)
        sys.modules["meetingscribe"] = module
        spec.loader.exec_module(module)
        return module
    import meetingscribe
    return meetingscribe


def _run_dash_c() -> None:
    code = sys.argv[2]
    # Same argv shape CPython gives a `-c` program.
    sys.argv = ["-c"] + sys.argv[3:]
    exec(compile(code, "<string>", "exec"), {"__name__": "__main__"})


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) >= 3 and sys.argv[1] == "-c":
        _run_dash_c()
        sys.exit(0)
    args = [a for a in sys.argv[1:] if not a.startswith("-psn_")]
    sys.argv = [sys.argv[0]] + (args or ["ui"])
    _load_meetingscribe().main()
