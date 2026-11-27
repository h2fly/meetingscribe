# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for MeetingScribe.app — built by packaging/build_dmg.sh.
#
# Models are NOT bundled: FunASR / sherpa / HY-MT weights download on first
# use into ~/.cache, exactly as when running the script, so the app stays
# ~1-2 GB instead of growing by every model.
import re
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata

ROOT = Path(SPECPATH).parent
VERSION = re.search(r'^__version__ = "([^"]+)"',
                    (ROOT / "meetingscribe.py").read_text(encoding="utf-8"),
                    re.M).group(1)

datas = [
    # Seeds ~/Documents/meetingscribe/config.jsonc on first launch.
    (str(ROOT / "config.jsonc"), "."),
    (str(ROOT / "hotword.jsonc.example"), "."),
    # Opened by Settings → Environment "查看步骤" and Help → 安装说明.
    (str(ROOT / "packaging" / "install-guide.html"), "."),
    (str(ROOT / "packaging" / "install-guide.en.html"), "."),
    # Loaded in preference to the frozen copy (launcher._load_meetingscribe),
    # so `build_dmg.sh --update` can refresh the code without a rebuild.
    (str(ROOT / "meetingscribe.py"), "."),
]
binaries = []
hiddenimports = ["meetingscribe"]

# Packages that import their own submodules by name at runtime (FunASR's
# model registry, modelscope's lazy loader) or ship data files / dylibs that
# static analysis cannot see.
for pkg in ("funasr", "modelscope", "jieba", "qfluentwidgets",
            "sherpa_onnx", "llama_cpp", "sounddevice", "_sounddevice_data",
            "kaldiio", "torchaudio"):
    try:
        d, b, h = collect_all(pkg)
    except Exception as e:  # optional backend not installed
        print(f"[spec] skip {pkg}: {e}")
        continue
    datas += d
    binaries += b
    hiddenimports += h

# transformers / modelscope / funasr verify dependency versions through
# importlib.metadata at import time; without the dist-info they raise
# PackageNotFoundError inside the app.
for pkg in ("transformers", "modelscope", "funasr", "torch", "tokenizers",
            "huggingface-hub", "safetensors", "tqdm", "regex", "requests",
            "packaging", "filelock", "numpy", "pyyaml", "sentencepiece"):
    try:
        datas += copy_metadata(pkg)
    except Exception as e:
        print(f"[spec] no metadata for {pkg}: {e}")

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "pytest", "IPython", "notebook", "jupyter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MeetingScribe",
    console=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=None,
)

coll = COLLECT(exe, a.binaries, a.datas, name="MeetingScribe")

app = BUNDLE(
    coll,
    name="MeetingScribe.app",
    icon=str(ROOT / "packaging" / "MeetingScribe.icns")
    if (ROOT / "packaging" / "MeetingScribe.icns").exists() else None,
    bundle_identifier="app.meetingscribe.desktop",
    version=VERSION,
    info_plist={
        "CFBundleName": "MeetingScribe",
        "CFBundleDisplayName": "MeetingScribe",
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "LSMinimumSystemVersion": "13.0",
        "NSHighResolutionCapable": True,
        # Without this key macOS denies microphone access outright — no
        # prompt is ever shown — and every recording is silent.
        "NSMicrophoneUsageDescription":
            "MeetingScribe 需要使用麦克风来录制会议音频。",
    },
)
