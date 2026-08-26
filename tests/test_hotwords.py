"""
Tests for the hotword automation (Schemes A+B: rule-based + LLM extraction
merged into config.jsonc's stt.funasr.hotword) and the Qwen caption MT
backend (Scheme C: llama.cpp + hotword glossary).
Run: pytest tests/
"""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import meetingscribe as ms


# ── Scheme A: _extract_hotword_candidates ────────────────────────────────────

# Enough Chinese padding that the text registers as CJK-dominant.
_ZH_PAD = "今天我们开会讨论了很多内容，包括架构设计、成本优化和后续的排期安排。" * 4


class TestExtractHotwordCandidates:
    def test_acronym_camelcase_digit_always_kept(self):
        text = _ZH_PAD + " 我们用 GKE 跑 BigQuery，机型换成 N4D。"
        out = ms._extract_hotword_candidates(text)
        assert "GKE" in out and "BigQuery" in out and "N4D" in out

    def test_stopwords_excluded(self):
        text = _ZH_PAD + " the and with from this that " * 3
        out = ms._extract_hotword_candidates(text)
        assert out == []

    def test_lowercase_needs_repetition_in_cjk_text(self):
        text = _ZH_PAD + " 先建一个 sandbox，然后在 sandbox 里验证。另外提了一次 webhook。"
        out = ms._extract_hotword_candidates(text)
        assert "sandbox" in out        # appears twice
        assert "webhook" not in out    # appears once

    def test_plain_words_skipped_in_english_dominant_text(self):
        text = ("We deployed the sandbox environment and the sandbox works. "
                "The BigQuery cluster on GKE is fine. " * 5)
        out = ms._extract_hotword_candidates(text)
        assert "GKE" in out and "BigQuery" in out
        # English prose: plain lowercase words AND Capitalized-only words
        # are skipped (sentence-initial capitals would be pure noise) —
        # only acronym / mixed-case / digit-bearing tokens qualify.
        assert "sandbox" not in out

    def test_capitalized_single_occurrence_kept_in_cjk_text(self):
        text = _ZH_PAD + " 基础设施统一由 Terraform 管理。"
        assert "Terraform" in ms._extract_hotword_candidates(text)

    def test_ordered_by_frequency(self):
        text = _ZH_PAD + " Kong Kong Kong 网关，GKE GKE，还有 ADK。"
        out = ms._extract_hotword_candidates(text)
        assert out.index("Kong") < out.index("GKE") < out.index("ADK")

    def test_cased_variant_preferred_for_display(self):
        text = _ZH_PAD + " 我们聊 kong 的配置，Kong 的插件，kong 网关。"
        out = ms._extract_hotword_candidates(text)
        assert "Kong" in out

    def test_length_bounds_and_empty(self):
        assert ms._extract_hotword_candidates("") == []
        text = _ZH_PAD + " X " + "A" * 40
        out = ms._extract_hotword_candidates(text)
        assert "X" not in out
        assert all(len(t) <= 32 for t in out)


# ── _merge_hotwords ──────────────────────────────────────────────────────────

class TestMergeHotwords:
    def test_appends_new_preserving_existing_order(self):
        assert ms._merge_hotwords("Kong GKE", ["ADK", "MCP"]) == "Kong GKE ADK MCP"

    def test_dedup_case_insensitive_first_casing_wins(self):
        assert ms._merge_hotwords("Kong", ["kong", "KONG", "GKE"]) == "Kong GKE"

    def test_eviction_drops_oldest_over_cap(self):
        merged = ms._merge_hotwords("w1 w2 w3", ["w4", "w5"], max_count=3)
        assert merged == "w3 w4 w5"

    def test_sanitizes_tokens(self):
        merged = ms._merge_hotwords("", ["Kong,", "（GKE）", "42", "a", "the"])
        assert merged == "Kong GKE"

    def test_zero_cap_means_unlimited(self):
        terms = [f"term{i}" for i in range(150)]
        assert len(ms._merge_hotwords("", terms, max_count=0).split()) == 150


# ── Scheme B: _parse_llm_hotwords ────────────────────────────────────────────

class TestParseLlmHotwords:
    def test_single_line_space_separated(self):
        assert ms._parse_llm_hotwords("Kong GKE API网关") == ["Kong", "GKE", "API网关"]

    def test_tolerates_bullets_and_cjk_separators(self):
        out = ms._parse_llm_hotwords("- Kong, GKE、API网关\n* sandbox；N4D")
        assert out == ["Kong", "GKE", "API网关", "sandbox", "N4D"]

    def test_drops_preamble_line_ending_with_colon(self):
        out = ms._parse_llm_hotwords("以下是提取的热词：\nKong GKE")
        assert out == ["Kong", "GKE"]

    def test_empty_and_whitespace(self):
        assert ms._parse_llm_hotwords("") == []
        assert ms._parse_llm_hotwords("\n  \n") == []

    def test_caps_at_50(self):
        out = ms._parse_llm_hotwords(" ".join(f"term{i}" for i in range(80)))
        assert len(out) == 50


# ── persistence into config.jsonc ────────────────────────────────────────────

_CONFIG_TEMPLATE = """{
  // top comment survives
  "stt": {
    "funasr": {
      // hotword comment survives
      "hotword": "%s"
    }
  }
}
"""


@pytest.fixture
def tmp_config(monkeypatch, tmp_path):
    cfg_file = tmp_path / "config.jsonc"
    cfg_file.write_text(_CONFIG_TEMPLATE % "", encoding="utf-8")
    monkeypatch.setattr(ms, "CONFIG_FILE", cfg_file)
    monkeypatch.setattr(ms, "CONFIG_DIR", tmp_path)
    return cfg_file


class TestPersistHotwords:
    def test_writes_merged_hotword_and_keeps_comments(self, tmp_config):
        cfg = {"stt": {"funasr": {"hotword": "Kong"}}}
        added = ms._persist_hotwords(["GKE", "kong"], cfg)
        assert added == ["GKE"]
        text = tmp_config.read_text(encoding="utf-8")
        assert "// hotword comment survives" in text
        on_disk = json.loads(ms._strip_jsonc_comments(text))
        assert on_disk["stt"]["funasr"]["hotword"] == "Kong GKE"
        # runtime cfg updated in place too
        assert cfg["stt"]["funasr"]["hotword"] == "Kong GKE"

    def test_no_change_returns_empty_and_leaves_file(self, tmp_config):
        before = tmp_config.read_text(encoding="utf-8")
        cfg = {"stt": {"funasr": {"hotword": "Kong GKE"}}}
        assert ms._persist_hotwords(["kong", "GKE"], cfg) == []
        assert tmp_config.read_text(encoding="utf-8") == before

    def test_cap_enforced_on_persist(self, tmp_config):
        cfg = {"stt": {"funasr": {"hotword": "w1 w2"}}}
        ms._persist_hotwords(["w3", "w4"], cfg, max_count=3)
        assert cfg["stt"]["funasr"]["hotword"] == "w2 w3 w4"


class TestAutoUpdateHotwords:
    def test_rule_plus_llm_merged(self, tmp_config, monkeypatch):
        monkeypatch.setattr(ms, "_llm_run", lambda *a, **k: "API网关 灰度发布")
        cfg = {"stt": {"funasr": {"hotword": ""}}}
        transcript = _ZH_PAD + " 我们在 GKE 上部署，GKE 的配置由 Terraform 管理。"
        added = ms._auto_update_hotwords(transcript, "claude", cfg)
        merged = cfg["stt"]["funasr"]["hotword"].split()
        assert "GKE" in merged and "API网关" in merged and "灰度发布" in merged
        assert set(added) == set(merged)

    def test_disabled_is_noop(self, tmp_config, monkeypatch):
        monkeypatch.setattr(
            ms, "_llm_run",
            lambda *a, **k: pytest.fail("llm must not run when disabled"))
        cfg = {"hotwords": {"auto_update": False},
               "stt": {"funasr": {"hotword": ""}}}
        assert ms._auto_update_hotwords("GKE GKE " + _ZH_PAD, "claude", cfg) == []
        assert cfg["stt"]["funasr"]["hotword"] == ""

    def test_llm_failure_keeps_rule_terms(self, tmp_config, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("cli exploded")
        monkeypatch.setattr(ms, "_llm_run", boom)
        cfg = {"stt": {"funasr": {"hotword": ""}}}
        transcript = _ZH_PAD + " 迁移到 N4D 机型，N4D 的成本更低。"
        added = ms._auto_update_hotwords(transcript, "claude", cfg)
        assert "N4D" in added

    def test_never_raises(self, tmp_config, monkeypatch):
        monkeypatch.setattr(ms, "_persist_hotwords",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
        assert ms._auto_update_hotwords("GKE GKE " + _ZH_PAD, "claude",
                                        {"stt": {"funasr": {}}}) == []


# ── Scheme C: _QwenCaptionMT ─────────────────────────────────────────────────

class _FakeLlama:
    def __init__(self, **kwargs):
        self.kwargs = kwargs          # constructor kwargs (n_gpu_layers, …)
        self.prompts = []
        self.call_kwargs = []         # per-completion kwargs (grammar, …)

    def create_chat_completion(self, messages, **kwargs):
        self.prompts.append(messages[0]["content"])
        self.call_kwargs.append(kwargs)
        return {"choices": [{"message": {"content": "  TRANSLATED  "}}]}


@pytest.fixture
def fake_llama(monkeypatch, tmp_path):
    mod = types.ModuleType("llama_cpp")
    mod.Llama = _FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    gguf = tmp_path / "fake.gguf"
    gguf.write_bytes(b"GGUF")
    return gguf


@pytest.fixture(autouse=True)
def clear_backend_cache():
    ms._caption_backend_cache.clear()
    yield
    ms._caption_backend_cache.clear()


def _qwen(fake_llama, hotword=""):
    lc = {"qwen": {"model_path": str(fake_llama)}}
    return ms._QwenCaptionMT({}, lc, hotword=hotword)


class TestQwenCaptionMT:
    def test_gpu_offload_enabled_by_default(self, fake_llama):
        mt = _qwen(fake_llama)
        assert mt._llm.kwargs["n_gpu_layers"] == -1

    def test_zh_to_en_direction_and_output_stripped(self, fake_llama):
        mt = _qwen(fake_llama)
        out = mt.translate("我们今天讨论架构方案")
        assert out == "TRANSLATED"
        prompt = mt._llm.prompts[-1]
        assert "中文" in prompt and "翻译成英文" in prompt
        assert "我们今天讨论架构方案" in prompt

    def test_en_to_zh_direction(self, fake_llama):
        mt = _qwen(fake_llama)
        mt.translate("let's review the deployment plan")
        assert "翻译成中文" in mt._llm.prompts[-1]

    def test_glossary_injected_on_hotword_hit(self, fake_llama):
        mt = _qwen(fake_llama, hotword="Kong GKE API网关")
        mt.translate("我们把流量切到Kong和API网关上")
        prompt = mt._llm.prompts[-1]
        assert "必须原样保留" in prompt
        assert "Kong" in prompt and "API网关" in prompt
        assert "GKE" not in prompt   # not hit → not in glossary

    def test_hit_matching_is_case_insensitive(self, fake_llama):
        mt = _qwen(fake_llama, hotword="Kong")
        mt.translate("kong 的插件机制")
        assert "必须原样保留" in mt._llm.prompts[-1]

    def test_no_glossary_without_hits(self, fake_llama):
        mt = _qwen(fake_llama, hotword="Kong GKE")
        mt.translate("我们聊聊排期")
        assert "必须原样保留" not in mt._llm.prompts[-1]

    def test_missing_model_path_raises(self, fake_llama, tmp_path):
        lc = {"qwen": {"model_path": str(tmp_path / "nope.gguf")}}
        with pytest.raises(RuntimeError, match="不存在"):
            ms._QwenCaptionMT({}, lc)

    def test_missing_llama_cpp_raises_pip_hint(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "llama_cpp", None)
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"GGUF")
        with pytest.raises(RuntimeError, match="llama-cpp-python"):
            ms._QwenCaptionMT({}, {"qwen": {"model_path": str(gguf)}})


class TestParseFixMt:
    def test_two_line_format(self):
        out = "修正：我们把流量切到GKE\n翻译：We route traffic to GKE"
        assert ms._parse_fix_mt(out, "我们把流量切到G K E") == (
            "我们把流量切到GKE", "We route traffic to GKE")

    def test_ascii_colon_tolerated(self):
        assert ms._parse_fix_mt("修正: A词\n翻译: B", "A词") == ("A词", "B")

    def test_missing_translation_falls_back_to_plain_translation(self):
        # Model ignored the format → whole reply treated as translation,
        # source untouched.
        assert ms._parse_fix_mt("Just a translation.", "原文") == (
            "原文", "Just a translation.")

    def test_suspicious_fixed_length_distrusted(self):
        out = "修正：好\n翻译：ok"
        fixed, trans = ms._parse_fix_mt(out, "这是一个相当长的原始句子内容")
        assert fixed == "这是一个相当长的原始句子内容" and trans == "ok"

    def test_empty_reply(self):
        assert ms._parse_fix_mt("", "原文") == ("原文", "")

    def test_dropped_clause_rejected(self):
        # A bare prefix scores high on similarity but loses content —
        # the length guard must keep the original line.
        original = "我们把流量切到新集群上，机器人会自动通知大家"
        out = "修正：我们把流量切到新集群上\n翻译：We switch traffic."
        assert ms._parse_fix_mt(out, original) == (original, "We switch traffic.")

    def test_rewrite_masquerade_rejected(self):
        # An English translation smuggled into the 修正 line is discarded.
        original = "这个方案明天再讨论"
        out = ("修正：Let's discuss the plan tomorrow\n"
               "翻译：Let's discuss the plan tomorrow")
        fixed, _ = ms._parse_fix_mt(out, original)
        assert fixed == original


class _FixFakeLlama(_FakeLlama):
    def create_chat_completion(self, messages, **kwargs):
        self.prompts.append(messages[0]["content"])
        self.call_kwargs.append(kwargs)
        return {"choices": [{"message": {
            "content": "修正：我们把流量切到GKE集群\n翻译：We route traffic to the GKE cluster"}}]}


class TestQwenCorrectAndTranslate:
    def _fix_qwen(self, monkeypatch, tmp_path, hotword="", correct=True):
        mod = types.ModuleType("llama_cpp")
        mod.Llama = _FixFakeLlama
        monkeypatch.setitem(sys.modules, "llama_cpp", mod)
        gguf = tmp_path / "fake.gguf"
        gguf.write_bytes(b"GGUF")
        lc = {"qwen": {"model_path": str(gguf), "correct": correct}}
        return ms._QwenCaptionMT({}, lc, hotword=hotword)

    def test_returns_fixed_and_translation(self, monkeypatch, tmp_path):
        mt = self._fix_qwen(monkeypatch, tmp_path)
        fixed, trans = mt.correct_and_translate("我们把流量切到G K E集群", [])
        assert fixed == "我们把流量切到GKE集群"
        assert trans == "We route traffic to the GKE cluster"

    def test_context_in_prompt(self, monkeypatch, tmp_path):
        mt = self._fix_qwen(monkeypatch, tmp_path, hotword="GKE ChatOps")
        mt.correct_and_translate("测试句子", ["上一句A", "上一句B"])
        prompt = mt._llm.prompts[-1]
        assert "上一句A" in prompt and "上一句B" in prompt
        assert "修正：" in prompt and "测试句子" in prompt

    def test_glossary_is_line_scoped_not_whole_table(self, monkeypatch, tmp_path):
        """Only hotwords this line plausibly contains get injected — dumping
        the whole table is what made the model apply unrelated terms."""
        mt = self._fix_qwen(monkeypatch, tmp_path, hotword="GKE ChatOps Kong")
        mt.correct_and_translate("我们把流量切到 GKE 集群", [])
        prompt = mt._llm.prompts[-1]
        assert "GKE" in prompt                       # literal hit
        assert "ChatOps" not in prompt and "Kong" not in prompt

    def test_glossary_includes_fuzzy_asr_miss(self, monkeypatch, tmp_path):
        # The correction use case: ASR mangled ChatOps into 拆特ops, so the
        # term must still reach the prompt even without a literal match.
        mt = self._fix_qwen(monkeypatch, tmp_path, hotword="GKE ChatOps")
        mt.correct_and_translate("拆特ops的审批流程", [])
        prompt = mt._llm.prompts[-1]
        assert "ChatOps" in prompt and "GKE" not in prompt

    def test_no_glossary_block_when_nothing_matches(self, monkeypatch, tmp_path):
        mt = self._fix_qwen(monkeypatch, tmp_path, hotword="GKE ChatOps")
        mt.correct_and_translate("今天天气不错", [])
        prompt = mt._llm.prompts[-1]
        assert "术语表" not in prompt

    def test_empty_context_placeholder(self, monkeypatch, tmp_path):
        mt = self._fix_qwen(monkeypatch, tmp_path)
        mt.correct_and_translate("测试", [])
        assert "（无）" in mt._llm.prompts[-1]

    def test_correct_disabled_falls_back_to_translate(self, monkeypatch, tmp_path):
        mt = self._fix_qwen(monkeypatch, tmp_path, correct=False)
        fixed, trans = mt.correct_and_translate("测试句子", [])
        assert fixed == "测试句子"
        # plain translate path used → prompt is the caption_mt template
        assert "同声传译引擎" in mt._llm.prompts[-1]
        assert "修正：" not in mt._llm.prompts[-1]


class TestLoadMtBackendQwen:
    def _engine(self, fake_llama, hotword="Kong"):
        cfg = {
            "live_captions": {"mt_provider": "qwen",
                              "qwen": {"model_path": str(fake_llama)}},
            "stt": {"funasr": {"hotword": hotword}},
        }
        return ms.LiveCaptionEngine(cfg, lambda ev: None)

    def test_selects_qwen_backend(self, fake_llama):
        backend = self._engine(fake_llama)._load_mt_backend()
        assert isinstance(backend, ms._QwenCaptionMT)
        assert backend._hotwords == ["Kong"]

    def test_cached_backend_gets_hotword_refresh(self, fake_llama):
        first = self._engine(fake_llama, hotword="Kong")._load_mt_backend()
        second = self._engine(fake_llama, hotword="Kong GKE")._load_mt_backend()
        assert second is first
        assert second._hotwords == ["Kong", "GKE"]

    def test_falls_back_to_default_backend_on_qwen_failure(self, monkeypatch):
        class _Boom:
            def __init__(self, *a, **k):
                raise RuntimeError("no llama")

        marian = object()
        monkeypatch.setattr(ms, "_QwenCaptionMT", _Boom)
        monkeypatch.setattr(ms, "_MarianCaptionMT", lambda lc: marian)
        eng = ms.LiveCaptionEngine({"live_captions": {"mt_provider": "qwen"}}, lambda ev: None)
        assert eng._load_mt_backend() is marian

    def test_default_provider_skips_qwen(self, monkeypatch):
        marian = object()
        monkeypatch.setattr(
            ms, "_QwenCaptionMT",
            lambda *a, **k: pytest.fail("qwen must not load for default"))
        monkeypatch.setattr(ms, "_MarianCaptionMT", lambda lc: marian)
        eng = ms.LiveCaptionEngine({"live_captions": {"mt_provider": "default"}}, lambda ev: None)
        assert eng._load_mt_backend() is marian


# ── backfill CLI (`hotwords` subcommand body) ────────────────────────────────

class TestCmdHotwordsBody:
    def test_rule_only_backfill(self, tmp_config, monkeypatch, tmp_path):
        rec = tmp_path / "recordings"
        rec.mkdir()
        (rec / "20260101_090000.polish.txt").write_text(
            _ZH_PAD + " 我们在 GKE 上部署，GKE 由 Terraform 管理。",
            encoding="utf-8")
        cfg = {"stt": {"funasr": {"hotword": ""}}}
        args = types.SimpleNamespace(no_llm=True, show=False)
        ms._cmd_hotwords_body(args, cfg)
        merged = cfg["stt"]["funasr"]["hotword"].split()
        assert "GKE" in merged and "Terraform" in merged
        on_disk = json.loads(
            ms._strip_jsonc_comments(ms.CONFIG_FILE.read_text(encoding="utf-8")))
        assert on_disk["stt"]["funasr"]["hotword"] == cfg["stt"]["funasr"]["hotword"]

    def test_show_does_not_write(self, tmp_config):
        before = tmp_config.read_text(encoding="utf-8")
        cfg = {"stt": {"funasr": {"hotword": "Kong"}}}
        ms._cmd_hotwords_body(types.SimpleNamespace(no_llm=False, show=True), cfg)
        assert tmp_config.read_text(encoding="utf-8") == before


# ── prompt parity (project rule: config.jsonc mirrors _PROMPT_DEFAULTS) ─────

class TestPromptParity:
    @pytest.mark.parametrize(
        "key", ["polish", "hotwords_extract", "caption_mt", "caption_fix_mt"])
    def test_config_jsonc_matches_defaults(self, key):
        repo_cfg = Path(ms.__file__).parent / "config.jsonc"
        cfg = json.loads(
            ms._strip_jsonc_comments(repo_cfg.read_text(encoding="utf-8")))
        assert ms._resolve_prompt(cfg, key).rstrip() == \
            ms._PROMPT_DEFAULTS[key].rstrip()

    def test_hotwords_extract_has_transcript_token(self):
        assert "{transcript}" in ms._resolve_prompt({}, "hotwords_extract")

    def test_caption_mt_has_its_placeholders(self):
        tpl = ms._resolve_prompt({}, "caption_mt")
        for token in ("{src_lang}", "{dst_lang}", "{glossary}", "{text}"):
            assert token in tpl

    def test_caption_fix_mt_has_its_placeholders(self):
        tpl = ms._resolve_prompt({}, "caption_fix_mt")
        for token in ("{src_lang}", "{dst_lang}", "{glossary}",
                      "{context}", "{text}"):
            assert token in tpl

    def test_polish_has_hotwords_placeholder(self):
        assert "{hotwords}" in ms._resolve_prompt({}, "polish")


class TestGlossaryCandidates:
    """Line-scoped glossary selection. Injecting the whole hotword table
    (auto-grows to hotwords.max_count = 100) is what made the 1.5B model
    invent relations between unrelated terms."""

    HW = ["ChatOps", "GKE", "gVisor", "LangGraph", "李雷", "AE"]

    def test_literal_hit(self):
        assert ms._glossary_candidates("切到 GKE 集群", self.HW) == ["GKE"]

    def test_unrelated_terms_excluded(self):
        out = ms._glossary_candidates("切到 GKE 集群", self.HW)
        assert "ChatOps" not in out and "LangGraph" not in out

    def test_fuzzy_hit_on_mangled_ascii(self):
        # The correction use case: ASR wrote 拆特ops for ChatOps.
        assert ms._glossary_candidates("拆特ops的审批", self.HW) == ["ChatOps"]

    def test_case_insensitive_literal(self):
        assert ms._glossary_candidates("用 chatops 通知", self.HW) == ["ChatOps"]

    def test_no_match_returns_empty(self):
        assert ms._glossary_candidates("今天天气不错", self.HW) == []

    def test_pure_cjk_term_is_literal_only(self):
        # 里雷 is a homophone of 李雷 but shares no character; matching it
        # would need pinyin, so it is deliberately not a hit.
        assert ms._glossary_candidates("里雷说下周", self.HW) == []
        assert ms._glossary_candidates("李雷说下周", self.HW) == ["李雷"]

    def test_literal_hits_rank_before_fuzzy(self):
        out = ms._glossary_candidates("拆特ops 和 GKE", self.HW)
        assert out[0] == "GKE" and "ChatOps" in out

    def test_capped_at_limit(self):
        hw = [f"Term{i}" for i in range(20)]
        out = ms._glossary_candidates("term0 term1 term2 term3", hw, limit=2)
        assert len(out) == 2

    def test_short_hotwords_ignored(self):
        assert ms._glossary_candidates("a b c", ["a", "b"]) == []

    def test_empty_inputs(self):
        assert ms._glossary_candidates("", self.HW) == []
        assert ms._glossary_candidates("GKE", []) == []
        assert ms._glossary_candidates("GKE", None) == []

    def test_two_char_span_needs_literal_match(self):
        # Spans shorter than 3 chars are too noisy to fuzzy-match against
        # (an "AE" hotword would otherwise hit any 2-letter token).
        assert ms._glossary_candidates("这个 ok 吗", ["AE"]) == []


class _GrammarSpy:
    """Fake llama_cpp.LlamaGrammar recording the GBNF handed to it."""
    seen: list = []

    def __init__(self, src):
        self.src = src

    @classmethod
    def from_string(cls, src, verbose=True):
        cls.seen.append(src)
        return cls(src)


class TestQwenGrammar:
    def _make(self, monkeypatch, tmp_path, grammar_cls=_GrammarSpy,
              grammar_cfg=True):
        mod = types.ModuleType("llama_cpp")
        mod.Llama = _FixFakeLlama
        if grammar_cls is not None:
            mod.LlamaGrammar = grammar_cls
        monkeypatch.setitem(sys.modules, "llama_cpp", mod)
        gguf = tmp_path / "fake.gguf"
        gguf.write_bytes(b"GGUF")
        lc = {"qwen": {"model_path": str(gguf), "grammar": grammar_cfg}}
        return ms._QwenCaptionMT({}, lc, hotword="GKE")

    def test_grammar_built_and_passed_to_completion(self, monkeypatch, tmp_path):
        _GrammarSpy.seen = []
        mt = self._make(monkeypatch, tmp_path)
        assert mt._fix_grammar is not None
        assert any("修正：" in g for g in _GrammarSpy.seen)
        mt.correct_and_translate("测试句子", [])
        assert isinstance(mt._llm.call_kwargs[-1].get("grammar"), _GrammarSpy)

    def test_plain_translate_stays_unconstrained(self, monkeypatch, tmp_path):
        """The translate path also serves in-flight partials: GBNF sampling
        measured +0.30 s/line there and has no structure to enforce, so the
        grammar must stay off it even when enabled."""
        mt = self._make(monkeypatch, tmp_path)
        mt.translate("测试句子")
        assert "grammar" not in mt._llm.call_kwargs[-1]

    def test_grammar_selects_zero_shot_prompt(self, monkeypatch, tmp_path):
        mt = self._make(monkeypatch, tmp_path)
        mt.correct_and_translate("测试句子", [])
        prompt = mt._llm.prompts[-1]
        # zero-shot: instructions only, no worked example to copy from
        assert "先输出「修正：」行" in prompt
        assert "预算下周才能确认" not in prompt

    def test_grammar_disabled_by_config_uses_few_shot(self, monkeypatch, tmp_path):
        mt = self._make(monkeypatch, tmp_path, grammar_cfg=False)
        assert mt._fix_grammar is None
        mt.correct_and_translate("测试句子", [])
        assert "预算下周才能确认" in mt._llm.prompts[-1]
        assert "grammar" not in mt._llm.call_kwargs[-1]

    def test_old_llama_without_grammar_falls_back(self, monkeypatch, tmp_path):
        # Bounded repetition {m,n} needs a recent llama.cpp; a build that
        # can't parse it must degrade to the few-shot prompt, not crash.
        class _Boom:
            @staticmethod
            def from_string(src, verbose=True):
                raise ValueError("expecting ')' at {1,400}")

        mt = self._make(monkeypatch, tmp_path, grammar_cls=_Boom)
        assert mt._fix_grammar is None
        mt.correct_and_translate("测试句子", [])
        assert "预算下周才能确认" in mt._llm.prompts[-1]
        assert "grammar" not in mt._llm.call_kwargs[-1]

    def test_missing_grammar_class_falls_back(self, monkeypatch, tmp_path):
        mt = self._make(monkeypatch, tmp_path, grammar_cls=None)
        assert mt._fix_grammar is None
        mt.correct_and_translate("测试句子", [])
        assert "预算下周才能确认" in mt._llm.prompts[-1]
