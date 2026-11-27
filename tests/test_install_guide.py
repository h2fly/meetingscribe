"""
Tests for the install guide (安装说明): the generated HTML carries every
section the app links to, the app opens the right section, and the guide is
wired into the bundle and the DMG.
Run: pytest tests/
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import meetingscribe as ms

ROOT = Path(ms.__file__).parent
GUIDES = {"zh": ROOT / "packaging" / "install-guide.html",
          "en": ROOT / "packaging" / "install-guide.en.html"}


def test_guide_has_every_anchor_the_app_links_to():
    for lang, path in GUIDES.items():
        text = path.read_text(encoding="utf-8")
        for anchor in ms._GUIDE_ANCHORS.values():
            assert f'id="{anchor}"' in text, (lang, anchor)


def test_english_guide_is_english():
    text = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "",
                  GUIDES["en"].read_text(encoding="utf-8"))
    body = re.sub(r"<style>.*?</style>|<script>.*?</script>", "", text, flags=re.S)
    # Only the language switch link and the Chinese-Mac screenshot note may be CJK.
    cjk = re.findall(r"[\u4e00-\u9fff]+", body)
    assert set(cjk) <= {"中文"}, cjk


def test_url_follows_ui_language():
    assert "install-guide.en.html" in ms._install_guide_url("", "en")
    assert ms._install_guide_url("", "zh").endswith("install-guide.html")


def test_guide_text_carries_no_local_paths_or_accounts():
    for path in GUIDES.values():
        # Screenshots are base64; strip them and check what a reader sees.
        text = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", path.read_text(encoding="utf-8"))
        assert "/Users/" not in text
        assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)   # no e-mail addresses


def test_section_url_goes_through_redirect_with_anchor():
    url = ms._install_guide_url("multi_output")
    assert url.startswith("file://")
    page = Path(url[len("file://"):]).read_text(encoding="utf-8")
    assert "install-guide.html#multi-output" in page
    page_en = Path(ms._install_guide_url("multi_output", "en")[len("file://"):]).read_text(encoding="utf-8")
    assert "install-guide.en.html#multi-output" in page_en


def test_unknown_section_opens_top_of_guide():
    assert ms._install_guide_url("../../etc").endswith("install-guide.html")


def test_missing_guide_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(ms, "_BUNDLE_DIR", tmp_path)
    assert ms._install_guide_url("blackhole") is None


def test_guide_is_bundled_and_shipped_in_dmg():
    spec = (ROOT / "packaging" / "MeetingScribe.spec").read_text(encoding="utf-8")
    assert '"install-guide.html"' in spec and '"install-guide.en.html"' in spec
    build = (ROOT / "packaging" / "build_dmg.sh").read_text(encoding="utf-8")
    assert "安装说明.html" in build and "Install Guide.html" in build
