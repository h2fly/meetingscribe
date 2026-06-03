"""Tests for the FunASR jieba_usr_dict self-heal path.

Pure-Python: no FunASR / modelscope / jieba imports are required, and the
``_load_funasr_automodel`` retry tests fake the ``funasr`` module via
``sys.modules``.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import meetingscribe as ms


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_jieba_indexerror(tmp_path: Path, msg: str = "list index out of range") -> IndexError:
    """Synthesize an IndexError whose traceback contains a frame from a file
    named ``<some>/jieba/__init__.py`` (mirroring what jieba raises in the wild).
    """
    jieba_dir = tmp_path / "fake_site_packages" / "jieba"
    jieba_dir.mkdir(parents=True, exist_ok=True)
    init_py = jieba_dir / "__init__.py"
    init_py.write_text(
        f"def boom():\n"
        f"    tup = ['only-one-token']\n"
        f"    word, freq = tup[0], tup[1]  # raises IndexError\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("ms_test_fake_jieba", init_py)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        mod.boom()
    except IndexError as e:
        if msg != "list index out of range":
            return IndexError(msg).with_traceback(e.__traceback__)
        return e
    raise AssertionError("expected boom() to raise IndexError")


def _make_plain_indexerror() -> IndexError:
    """An IndexError raised from THIS test file — no jieba frame in the tb."""
    try:
        empty: list = []
        empty[0]  # noqa: B018
    except IndexError as e:
        return e
    raise AssertionError("expected IndexError")


# ── _patch_one_jieba_dict ────────────────────────────────────────────────────


def test_patch_one_jieba_dict_fixes_single_token_lines(tmp_path: Path) -> None:
    dict_path = tmp_path / "jieba_usr_dict"
    dict_path.write_text(
        "alpha\nbeta\ngamma\nfoo 5 nz\ndelta\n\nepsilon\n",
        encoding="utf-8",
    )
    modified, fixed, total = ms._patch_one_jieba_dict(dict_path)
    assert modified is True
    assert fixed == 5
    assert total == 6  # six non-empty lines (one is already well-formed)
    text = dict_path.read_text(encoding="utf-8")
    assert text == "alpha 1\nbeta 1\ngamma 1\nfoo 5 nz\ndelta 1\n\nepsilon 1\n"
    sentinel = tmp_path / ".jieba_usr_dict.patched"
    assert sentinel.exists()
    assert sentinel.read_text(encoding="utf-8").startswith("v1")


def test_patch_one_jieba_dict_no_op_when_already_well_formed(tmp_path: Path) -> None:
    dict_path = tmp_path / "jieba_usr_dict"
    original = "alpha 1\nbeta 2 n\ngamma 3\n"
    dict_path.write_text(original, encoding="utf-8")
    before_bytes = dict_path.read_bytes()
    modified, fixed, total = ms._patch_one_jieba_dict(dict_path)
    assert modified is False
    assert fixed == 0
    assert total == 3
    assert dict_path.read_bytes() == before_bytes
    sentinel = tmp_path / ".jieba_usr_dict.patched"
    assert sentinel.exists()


def test_patch_one_jieba_dict_short_circuits_on_sentinel(tmp_path: Path) -> None:
    dict_path = tmp_path / "jieba_usr_dict"
    dict_path.write_text("alpha\nbeta\n", encoding="utf-8")
    sentinel = tmp_path / ".jieba_usr_dict.patched"
    sentinel.write_text("v1\n", encoding="utf-8")
    # Make the sentinel newer than the dict so the short-circuit fires.
    future = dict_path.stat().st_mtime + 10
    os.utime(sentinel, (future, future))

    before = dict_path.read_bytes()
    modified, fixed, total = ms._patch_one_jieba_dict(dict_path)
    assert (modified, fixed, total) == (False, 0, 0)
    assert dict_path.read_bytes() == before
    assert not (tmp_path / "jieba_usr_dict.tmp").exists()


def test_patch_one_jieba_dict_atomic_on_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dict_path = tmp_path / "jieba_usr_dict"
    original = "alpha\nbeta\ngamma\n"
    dict_path.write_text(original, encoding="utf-8")
    before = dict_path.read_bytes()

    real_replace = os.replace

    def boom_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(ms.os, "replace", boom_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        ms._patch_one_jieba_dict(dict_path)

    assert dict_path.read_bytes() == before
    assert not (tmp_path / "jieba_usr_dict.tmp").exists()
    assert not (tmp_path / ".jieba_usr_dict.patched").exists()
    # Sanity: the un-monkeypatched os.replace still works after the test.
    assert real_replace is os.replace.__wrapped__ if hasattr(os.replace, "__wrapped__") else True


# ── _patch_funasr_jieba_dicts ────────────────────────────────────────────────


def _make_modelscope_layout(cache_root: Path, dict_content: str = "alpha\nbeta\n") -> Path:
    dict_dir = cache_root / "hub" / "models" / "iic" / "punc-model"
    dict_dir.mkdir(parents=True)
    dict_path = dict_dir / "jieba_usr_dict"
    dict_path.write_text(dict_content, encoding="utf-8")
    return dict_path


def test_patch_funasr_jieba_dicts_honours_modelscope_cache_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "ms-cache"
    dict_path = _make_modelscope_layout(cache)
    monkeypatch.setenv("MODELSCOPE_CACHE", str(cache))

    n = ms._patch_funasr_jieba_dicts()

    assert n == 1
    assert dict_path.read_text(encoding="utf-8") == "alpha 1\nbeta 1\n"
    assert (dict_path.parent / ".jieba_usr_dict.patched").exists()


def test_patch_funasr_jieba_dicts_returns_zero_when_no_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MODELSCOPE_CACHE", raising=False)
    # Redirect Path.home() so ~/.cache/modelscope resolves into tmp_path
    # (where no such directory exists).
    monkeypatch.setattr(ms.Path, "home", classmethod(lambda cls: tmp_path))

    n = ms._patch_funasr_jieba_dicts()

    assert n == 0
    assert list(tmp_path.iterdir()) == []  # nothing got created anywhere


def test_patch_funasr_jieba_dicts_skips_symlinks_out_of_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks not supported on this platform")
    cache = tmp_path / "ms-cache"
    real_dict = _make_modelscope_layout(cache)

    # A malformed dict OUTSIDE the cache.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    outside_dict = elsewhere / "jieba_usr_dict"
    outside_dict.write_text("xray\nyankee\n", encoding="utf-8")
    outside_before = outside_dict.read_bytes()

    # A symlink INSIDE the cache that points OUT of the cache.
    bad_link_dir = cache / "hub" / "models" / "evil"
    bad_link_dir.mkdir(parents=True)
    bad_link = bad_link_dir / "jieba_usr_dict"
    try:
        os.symlink(outside_dict, bad_link)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlink on this platform / filesystem")

    monkeypatch.setenv("MODELSCOPE_CACHE", str(cache))

    n = ms._patch_funasr_jieba_dicts()

    assert n == 1  # only the real cache dict was patched
    assert real_dict.read_text(encoding="utf-8") == "alpha 1\nbeta 1\n"
    assert outside_dict.read_bytes() == outside_before
    assert not (elsewhere / ".jieba_usr_dict.patched").exists()


# ── _indexerror_came_from_jieba ──────────────────────────────────────────────


def test_indexerror_came_from_jieba_true_for_jieba_frame(tmp_path: Path) -> None:
    err = _make_jieba_indexerror(tmp_path)
    assert ms._indexerror_came_from_jieba(err) is True


def test_indexerror_came_from_jieba_false_for_unrelated_frame() -> None:
    err = _make_plain_indexerror()
    assert ms._indexerror_came_from_jieba(err) is False


# ── _load_funasr_automodel (retry semantics) ─────────────────────────────────


class _FakeAutoModel:
    """A stand-in for funasr.AutoModel driven by a class-level script.

    Each call to __init__ pops one entry off ``__class__.script``. If the entry
    is an exception instance, it is raised; otherwise it is stored on ``self``
    so the test can assert on it later.
    """

    script: list = []
    calls: list[dict] = []

    def __init__(self, **kwargs):
        type(self).calls.append(dict(kwargs))
        action = type(self).script.pop(0)
        if isinstance(action, BaseException):
            raise action
        self.tag = action

    def generate(self, **kwargs):
        return [{"text": ""}]


def _install_fake_funasr(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeAutoModel.script = []
    _FakeAutoModel.calls = []
    fake = types.ModuleType("funasr")
    fake.AutoModel = _FakeAutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "funasr", fake)


def test_load_funasr_automodel_retries_after_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "ms-cache"
    dict_path = _make_modelscope_layout(cache)
    monkeypatch.setenv("MODELSCOPE_CACHE", str(cache))
    _install_fake_funasr(monkeypatch)

    _FakeAutoModel.script = [_make_jieba_indexerror(tmp_path), "second-call-ok"]

    model = ms._load_funasr_automodel("asr-x", "vad-y", "punc-z")

    assert isinstance(model, _FakeAutoModel)
    assert model.tag == "second-call-ok"
    assert len(_FakeAutoModel.calls) == 2
    assert _FakeAutoModel.calls[0] == _FakeAutoModel.calls[1]
    # The dict was patched in between the two AutoModel calls.
    assert dict_path.read_text(encoding="utf-8") == "alpha 1\nbeta 1\n"
    assert (dict_path.parent / ".jieba_usr_dict.patched").exists()


def test_load_funasr_automodel_does_not_patch_for_unrelated_indexerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "ms-cache"
    dict_path = _make_modelscope_layout(cache)
    dict_before = dict_path.read_bytes()
    monkeypatch.setenv("MODELSCOPE_CACHE", str(cache))
    _install_fake_funasr(monkeypatch)

    _FakeAutoModel.script = [_make_plain_indexerror()]

    with pytest.raises(IndexError):
        ms._load_funasr_automodel("a", "b", "c")

    assert len(_FakeAutoModel.calls) == 1  # no retry
    assert dict_path.read_bytes() == dict_before
    assert not (dict_path.parent / ".jieba_usr_dict.patched").exists()


def test_load_funasr_automodel_reraises_original_when_patch_yields_zero_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "ms-cache"
    dict_path = _make_modelscope_layout(cache)
    sentinel = dict_path.parent / ".jieba_usr_dict.patched"
    # Pre-create the sentinel with a future mtime so the patcher short-circuits.
    sentinel.write_text("v1\n", encoding="utf-8")
    future = dict_path.stat().st_mtime + 100
    os.utime(sentinel, (future, future))

    monkeypatch.setenv("MODELSCOPE_CACHE", str(cache))
    _install_fake_funasr(monkeypatch)

    original = _make_jieba_indexerror(tmp_path, msg="ORIGINAL-MARKER")
    _FakeAutoModel.script = [original]

    with pytest.raises(IndexError) as excinfo:
        ms._load_funasr_automodel("a", "b", "c")

    assert excinfo.value.args == ("ORIGINAL-MARKER",)
    assert excinfo.value is original
    assert len(_FakeAutoModel.calls) == 1  # no retry


def test_load_funasr_automodel_retry_failure_chains_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "ms-cache"
    _make_modelscope_layout(cache)
    monkeypatch.setenv("MODELSCOPE_CACHE", str(cache))
    _install_fake_funasr(monkeypatch)

    original = _make_jieba_indexerror(tmp_path, msg="ORIGINAL")
    retry_failure = RuntimeError("retry-blew-up")
    _FakeAutoModel.script = [original, retry_failure]

    with pytest.raises(IndexError) as excinfo:
        ms._load_funasr_automodel("a", "b", "c")

    assert excinfo.value is original
    # Per `raise original_err from retry_err`, __cause__ is the retry failure.
    assert excinfo.value.__cause__ is retry_failure
