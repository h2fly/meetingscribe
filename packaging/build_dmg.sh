#!/bin/bash
# Build MeetingScribe.app and wrap it in a DMG.
#
#   packaging/build_dmg.sh            → dist/MeetingScribe-<version>.dmg
#   packaging/build_dmg.sh --app-only → dist/MeetingScribe.app (no DMG; for
#                                       testing before cutting a release)
#   packaging/build_dmg.sh --update   → refresh meetingscribe.py, the install
#                                       guides and the seed config inside the
#                                       existing dist/MeetingScribe.app and
#                                       re-sign: seconds instead of ~10 min.
#                                       Only valid when no new third-party
#                                       library and no version bump happened.
#
# Uses the python3 on PATH, which must already have the runtime dependencies
# (requirements.txt + torch) and PyInstaller installed. The version comes from
# `__version__` in meetingscribe.py — bump it there, nowhere else.
set -euo pipefail

APP_ONLY=0
UPDATE=0
case "${1:-}" in
    --app-only) APP_ONLY=1 ;;
    --update) UPDATE=1 ;;
    "") ;;
    *) echo "usage: $0 [--app-only | --update]" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

VERSION="$("$PY" - <<'PYEOF'
import re
src = open("meetingscribe.py", encoding="utf-8").read()
print(re.search(r'^__version__ = "([^"]+)"', src, re.M).group(1))
PYEOF
)"
echo "==> MeetingScribe $VERSION"

# Never ship the developer's personal data. config.jsonc is the seed config
# every user gets; the hotword list (colleague / customer names) must be empty.
"$PY" - <<'PYEOF'
import json, sys
sys.path.insert(0, ".")
from meetingscribe import _strip_jsonc_comments
cfg = json.loads(_strip_jsonc_comments(open("config.jsonc", encoding="utf-8").read()))
hw = ((cfg.get("stt") or {}).get("funasr") or {}).get("hotword") or ""
if hw.strip():
    sys.exit("config.jsonc carries an inline hotword list — clear it before packaging")
PYEOF

# The install guide ships in the DMG and inside the app. It is generated
# (docs/build_install_guide.py), so fail early rather than ship without it.
for g in packaging/install-guide.html packaging/install-guide.en.html; do
    [ -f "$g" ] || {
        echo "$g missing — run: python3 docs/build_install_guide.py" >&2
        exit 1
    }
done

if [ "$UPDATE" = 1 ]; then
    APP=dist/MeetingScribe.app
    RES="$APP/Contents/Resources"
    [ -f "$RES/meetingscribe.py" ] || {
        echo "no updatable app in dist/ — run a full build (--app-only) once first" >&2
        exit 1
    }
    BUILT="$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "$APP/Contents/Info.plist")"
    [ "$BUILT" = "$VERSION" ] || {
        echo "version changed ($BUILT → $VERSION): Info.plist needs a full build" >&2
        exit 1
    }
    for f in meetingscribe.py config.jsonc; do cp "$f" "$RES/$f"; done
    cp packaging/install-guide.html packaging/install-guide.en.html "$RES/"
    for f in meetingscribe.py config.jsonc install-guide.html install-guide.en.html; do
        [ -e "$APP/Contents/Frameworks/$f" ] || ln -s "../Resources/$f" "$APP/Contents/Frameworks/$f"
    done
    # Re-seal only: nested code is untouched and keeps its signatures.
    codesign --force --sign - "$APP"
    codesign --verify --deep --strict "$APP"
    echo "==> updated $APP (code + guides) — a new third-party import still needs a full build"
    exit 0
fi

# Intermediate files go to a temp dir that is always removed: they are as
# large as the app itself.
WORK="$(mktemp -d -t meetingscribe-build)"
trap 'rm -rf "$WORK"' EXIT

"$PY" -m PyInstaller packaging/MeetingScribe.spec \
    --noconfirm --clean \
    --workpath "$WORK/build" \
    --distpath "$WORK/dist"

APP="$WORK/dist/MeetingScribe.app"
[ -d "$APP" ] || { echo "build produced no app" >&2; exit 1; }

# Ad-hoc signature over the whole bundle (no Developer ID). Required on
# Apple Silicon for the app to launch at all, and for macOS to remember the
# microphone permission across launches.
codesign --force --deep --sign - "$APP"

if [ "$APP_ONLY" = 1 ]; then
    mkdir -p dist
    rm -rf dist/MeetingScribe.app
    # ditto keeps the bundle's symlinks and extended attributes intact, so
    # the signature made above still verifies after the copy.
    ditto "$APP" dist/MeetingScribe.app
    codesign --verify --deep --strict dist/MeetingScribe.app
    echo "==> $(du -sh dist/MeetingScribe.app | cut -f1)  dist/MeetingScribe.app"
    exit 0
fi

STAGE="$WORK/dmg"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
cp packaging/install-guide.html "$STAGE/安装说明.html"
cp packaging/install-guide.en.html "$STAGE/Install Guide.html"

mkdir -p dist
DMG="dist/MeetingScribe-$VERSION.dmg"
rm -f "$DMG"
hdiutil create -volname "MeetingScribe $VERSION" -srcfolder "$STAGE" \
    -ov -format UDZO "$DMG" >/dev/null

echo "==> $(du -h "$DMG" | cut -f1)  $DMG"
