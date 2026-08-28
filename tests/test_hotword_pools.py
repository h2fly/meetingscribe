"""Tests for hotword pooling and hit-recency eviction (P4).

The old policy was pure FIFO — over `max_count` the oldest entries went,
regardless of whether they were still in daily use. That evicts a colleague's
name imported from Notion in favour of a token mined once from one meeting and
never heard again. Two things change it: `pinned` terms are never evicted, and
the rest are ranked by (last_seen, hits).

Every test redirects HOTWORD_FILE. Forgetting to do that once overwrote the
developer's real 100-term list with a two-word fixture.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import meetingscribe as ms


@pytest.fixture(autouse=True)
def isolate_hotword_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "HOTWORD_FILE", tmp_path / "hotword.jsonc")
    monkeypatch.setattr(ms, "CONFIG_FILE", tmp_path / "config.jsonc")
    return tmp_path / "hotword.jsonc"


def _write(path: Path, payload: dict) -> None:
    path.write_text("// header comment\n" +
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8")


# ── store round-trip and backward compatibility ──────────────────────────────

class TestStore:
    def test_missing_file_is_empty_not_an_error(self, isolate_hotword_file):
        store = ms._load_hotword_store()
        assert store == {"terms": [], "phrases": [], "pinned": set(),
                         "hits": {}, "last_seen": {}, "epoch": 0}

    def test_flat_legacy_file_loads_as_all_rolling(self, isolate_hotword_file):
        """Every file written before this existed, and anything hand-edited."""
        _write(isolate_hotword_file, {"hotword": "GKE ChatOps 李雷"})
        store = ms._load_hotword_store()
        assert store["terms"] == ["GKE", "ChatOps", "李雷"]
        assert store["pinned"] == set() and store["hits"] == {}

    def test_list_form_is_tolerated(self, isolate_hotword_file):
        _write(isolate_hotword_file, {"hotword": ["GKE", "ChatOps"]})
        assert ms._load_hotword_store()["terms"] == ["GKE", "ChatOps"]

    def test_broken_file_is_ignored_not_fatal(self, isolate_hotword_file):
        isolate_hotword_file.write_text("{not json", encoding="utf-8")
        assert ms._load_hotword_store()["terms"] == []

    def test_metadata_round_trips(self, isolate_hotword_file):
        store = {"terms": ["GKE", "Acme"], "pinned": {"acme"},
                 "hits": {"gke": 5}, "last_seen": {"gke": 3}, "epoch": 7}
        ms._save_hotword_file("GKE Acme", store)
        back = ms._load_hotword_store()
        assert back["terms"] == ["GKE", "Acme"]
        assert back["pinned"] == {"acme"}
        assert back["hits"] == {"gke": 5}
        assert back["last_seen"] == {"gke": 3}
        assert back["epoch"] == 7

    def test_hotword_stays_the_first_key_and_readable(self, isolate_hotword_file):
        ms._save_hotword_file("GKE Acme", {"pinned": {"acme"}, "epoch": 1})
        text = isolate_hotword_file.read_text(encoding="utf-8")
        assert text.startswith("//")                     # header preserved
        body = json.loads(ms._strip_jsonc_comments(text))
        assert list(body)[0] == "hotword"
        assert body["hotword"] == "GKE Acme"

    def test_metadata_is_omitted_when_there_is_none(self, isolate_hotword_file):
        ms._save_hotword_file("GKE", None)
        body = json.loads(ms._strip_jsonc_comments(
            isolate_hotword_file.read_text(encoding="utf-8")))
        assert body == {"hotword": "GKE"}

    def test_stale_metadata_for_dropped_terms_is_not_written(
            self, isolate_hotword_file):
        store = {"pinned": {"gone"}, "hits": {"gone": 9},
                 "last_seen": {"gone": 2}, "epoch": 3}
        ms._save_hotword_file("GKE", store)          # "gone" is not in the list
        body = json.loads(ms._strip_jsonc_comments(
            isolate_hotword_file.read_text(encoding="utf-8")))
        assert "pinned" not in body and "hits" not in body

    def test_hand_pinned_term_joins_the_list_without_being_in_hotword(
            self, isolate_hotword_file):
        """So a user can pin a name by adding one line, not two."""
        _write(isolate_hotword_file, {"hotword": "GKE", "pinned": ["李雷"]})
        store = ms._load_hotword_store()
        assert store["terms"] == ["GKE", "李雷"]
        assert store["pinned"] == {"李雷"}

    def test_garbage_metadata_values_are_skipped(self, isolate_hotword_file):
        _write(isolate_hotword_file,
               {"hotword": "GKE", "hits": {"gke": "many"}, "epoch": "soon"})
        store = ms._load_hotword_store()
        assert store["hits"] == {} and store["epoch"] == 0

    def test_load_hotword_file_still_returns_a_flat_string(
            self, isolate_hotword_file):
        _write(isolate_hotword_file, {"hotword": "GKE ChatOps"})
        assert ms._load_hotword_file() == "GKE ChatOps"


# ── hit detection ────────────────────────────────────────────────────────────

class TestHits:
    def test_case_insensitive_substring_match(self):
        got = ms._hotword_hits("我们用 chatops 发布到 gke",
                               ["ChatOps", "GKE", "Spanner"])
        assert got == {"chatops", "gke"}

    def test_cjk_terms_match(self):
        assert ms._hotword_hits("李雷说要巡检", ["李雷", "王小美"]) == {"李雷"}

    def test_empty_inputs(self):
        assert ms._hotword_hits("", ["GKE"]) == set()
        assert ms._hotword_hits("GKE", []) == set()


# ── eviction policy ──────────────────────────────────────────────────────────

class TestMergeEviction:
    def test_default_behaviour_is_unchanged_fifo(self):
        """With neither pinned nor rank, a plain string merge must behave
        exactly as before — oldest out."""
        got = ms._merge_hotwords("a1 b1 c1", ["d1"], max_count=3)
        assert got == "b1 c1 d1"

    def test_pinned_terms_survive_the_cap(self):
        got = ms._merge_hotwords("Acme b1 c1", ["d1"], max_count=2,
                                 pinned={"acme"})
        assert "Acme" in got.split()
        assert len(got.split()) == 2

    def test_rank_evicts_least_recently_used_not_oldest(self):
        """old_but_used was added first but is still in daily use; new_junk
        arrived yesterday and was never heard again."""
        rank = {"oldbutused": (9, 40), "middle": (8, 5), "newjunk": (1, 0)}
        got = ms._merge_hotwords("oldbutused middle newjunk", [],
                                 max_count=2, rank=rank)
        assert got.split() == ["oldbutused", "middle"]

    def test_ties_fall_back_to_insertion_order(self):
        rank = {"aa": (1, 0), "bb": (1, 0), "cc": (1, 0)}
        got = ms._merge_hotwords("aa bb cc", [], max_count=2, rank=rank)
        assert got.split() == ["bb", "cc"]

    def test_unranked_terms_are_evicted_first(self):
        # Fixture words must not be in _HOTWORD_STOPWORDS ("known" is), or
        # they get filtered before eviction ever runs.
        rank = {"tracked": (5, 5)}
        got = ms._merge_hotwords("untracked tracked", [], max_count=1,
                                 rank=rank)
        assert got.split() == ["tracked"]

    def test_pinned_alone_over_cap_is_kept(self):
        """Silently dropping authoritative names is worse than a list that is
        slightly too long; the caller logs the overshoot."""
        got = ms._merge_hotwords("Acme Bcme Ccme", [], max_count=2,
                                 pinned={"acme", "bcme", "ccme"})
        assert got.split() == ["Acme", "Bcme", "Ccme"]

    def test_output_keeps_original_order(self):
        rank = {"aa": (9, 9), "bb": (1, 1), "cc": (9, 9)}
        got = ms._merge_hotwords("aa bb cc", [], max_count=2, rank=rank)
        assert got.split() == ["aa", "cc"]       # not ["cc", "aa"]

    def test_no_cap_keeps_everything(self):
        got = ms._merge_hotwords("aa bb cc", ["dd"], max_count=0)
        assert got.split() == ["aa", "bb", "cc", "dd"]


# ── persist wiring ───────────────────────────────────────────────────────────

def _cfg(hotword=""):
    return {"stt": {"funasr": {"hotword": hotword}}}


class TestPersist:
    def test_pin_marks_new_terms(self, isolate_hotword_file):
        cfg = _cfg()
        ms._persist_hotwords(["Acme", "李雷"], cfg, max_count=100, pin=True)
        assert ms._load_hotword_store()["pinned"] == {"acme", "李雷"}

    def test_transcript_bumps_hits_and_last_seen(self, isolate_hotword_file):
        cfg = _cfg("GKE ChatOps")
        ms._persist_hotwords(["Spanner"], cfg, max_count=100,
                             transcript="今天在 GKE 上跑了一遍")
        store = ms._load_hotword_store()
        assert store["hits"]["gke"] == 1
        assert store["last_seen"]["gke"] == store["epoch"]
        assert "chatops" not in store["hits"]     # not mentioned

    def test_epoch_advances_each_update(self, isolate_hotword_file):
        cfg = _cfg("GKE")
        ms._persist_hotwords(["A1"], cfg, max_count=100, transcript="GKE")
        first = ms._load_hotword_store()["epoch"]
        ms._persist_hotwords(["B1"], cfg, max_count=100, transcript="GKE")
        assert ms._load_hotword_store()["epoch"] == first + 1

    def test_new_terms_start_at_the_current_epoch(self, isolate_hotword_file):
        """Otherwise a freshly mined term ranks at epoch 0 and is first out."""
        cfg = _cfg()
        ms._persist_hotwords(["Fresh"], cfg, max_count=100)
        store = ms._load_hotword_store()
        assert store["last_seen"]["fresh"] == store["epoch"]

    def test_pinned_term_survives_pressure_from_new_terms(
            self, isolate_hotword_file):
        cfg = _cfg()
        ms._persist_hotwords(["Acme"], cfg, max_count=3, pin=True)
        for i in range(10):
            ms._persist_hotwords([f"junk{i}"], cfg, max_count=3)
        assert "Acme" in cfg["stt"]["funasr"]["hotword"].split()

    def test_used_term_survives_where_fifo_would_have_dropped_it(
            self, isolate_hotword_file):
        """The whole point of the change, end to end."""
        cfg = _cfg()
        ms._persist_hotwords(["Veteran"], cfg, max_count=3)
        for i in range(6):
            # Veteran keeps showing up in the transcripts; the junk does not.
            ms._persist_hotwords([f"junk{i}"], cfg, max_count=3,
                                 transcript="又提到了 Veteran 这个词")
        terms = cfg["stt"]["funasr"]["hotword"].split()
        assert "Veteran" in terms
        assert len(terms) == 3

    def test_returns_only_the_newly_added(self, isolate_hotword_file):
        cfg = _cfg("GKE")
        added = ms._persist_hotwords(["GKE", "Spanner"], cfg, max_count=100)
        assert added == ["Spanner"]

    def test_drop_is_logged(self, isolate_hotword_file, monkeypatch):
        logged = []
        monkeypatch.setattr(ms, "_log", lambda cat, msg: logged.append(msg))
        cfg = _cfg("aa bb cc")
        ms._persist_hotwords(["dd"], cfg, max_count=3)
        line = next(m for m in logged if "hotword updated" in m)
        assert "-1" in line and "epoch=" in line and "pinned=" in line

    def test_pinned_over_cap_warns(self, isolate_hotword_file, monkeypatch):
        logged = []
        monkeypatch.setattr(ms, "_log", lambda cat, msg: logged.append(msg))
        cfg = _cfg()
        ms._persist_hotwords(["A1", "B1", "C1"], cfg, max_count=2, pin=True)
        assert any("exceed" in m and "max_count" in m for m in logged)

    def test_hit_only_update_still_records_history(self, isolate_hotword_file):
        """No new terms, but the hit history is worth persisting — it is what
        the next eviction reads."""
        cfg = _cfg("GKE")
        ms._persist_hotwords([], cfg, max_count=100, transcript="用了 GKE")
        assert ms._load_hotword_store()["hits"]["gke"] == 1

    def test_nothing_to_do_is_a_no_op(self, isolate_hotword_file):
        cfg = _cfg("GKE")
        assert ms._persist_hotwords([], cfg, max_count=100) == []
        assert not isolate_hotword_file.exists()

    def test_runtime_cfg_is_updated_in_place(self, isolate_hotword_file):
        cfg = _cfg("GKE")
        ms._persist_hotwords(["Spanner"], cfg, max_count=100)
        assert "Spanner" in cfg["stt"]["funasr"]["hotword"].split()

    def test_cold_start_gives_every_existing_term_the_benefit_of_the_doubt(
            self, isolate_hotword_file):
        """First run after the upgrade: the whole list has no history. Seeding
        at epoch 0 would let a token mined today outrank CockroachDB just
        because CockroachDB wasn't mentioned in this one meeting."""
        cfg = _cfg("CockroachDB Spanner BigQuery")
        ms._persist_hotwords(["JunkToday"], cfg, max_count=100,
                             transcript="今天只提到了 JunkToday")
        store = ms._load_hotword_store()
        assert store["last_seen"]["cockroachdb"] == store["epoch"]
        assert store["last_seen"]["junktoday"] == store["epoch"]
        # Only the mentioned one has a hit, which is what breaks the tie later.
        assert store["hits"].get("cockroachdb", 0) == 0

    def test_usage_differentiates_from_the_second_meeting_on(
            self, isolate_hotword_file):
        # "Used" would be filtered as a stopword before eviction ever sees it.
        cfg = _cfg("Spanner Unused")
        ms._persist_hotwords([], cfg, max_count=100, transcript="提到 Spanner")
        ms._persist_hotwords([], cfg, max_count=100, transcript="又提到 Spanner")
        store = ms._load_hotword_store()
        assert store["last_seen"]["spanner"] > store["last_seen"]["unused"]


class TestHitBoundaries:
    """A false hit protects exactly the junk eviction exists to remove, so
    boundary behaviour is load-bearing, not cosmetic. All measured on one real
    15.5k-character transcript."""

    @pytest.mark.parametrize("text,term", [
        ("我们需要怎么去 check 一下", "ch"),
        ("主动地找他们 face to face", "ac"),
        ("这个 nodejs 的问题", "node"),
        ("这个 background 任务", "back"),
    ])
    def test_ascii_term_needs_word_boundaries(self, text, term):
        assert ms._hotword_hits(text, [term]) == set()

    @pytest.mark.parametrize("text,term", [
        ("想切换到 claude code 自动", "code"),
        ("跑在 N4D 机型上", "N4D"),
        ("CockroachDB 的优化", "CockroachDB"),
    ])
    def test_standalone_ascii_term_hits(self, text, term):
        assert ms._hotword_hits(text, [term]) == {term.lower()}

    def test_ascii_term_flanked_by_cjk_still_hits(self):
        r"""\b sits between a letter and a CJK character, so a \bgke\b test
        would pass here — but it would FAIL on 「用GKE集群」 with no spaces,
        which is how these actually appear in Chinese speech."""
        assert ms._hotword_hits("用GKE集群跑", ["GKE"]) == {"gke"}

    @pytest.mark.parametrize("text,term", [
        ("李雷说要巡检", "李雷"),
        ("成本上涨了", "成本"),
        ("做冷热存储分离", "冷热存储"),
    ])
    def test_cjk_term_matches_as_substring(self, text, term):
        assert ms._hotword_hits(text, [term]) == {term}

    def test_newly_added_terms_earn_hits_from_the_same_transcript(
            self, isolate_hotword_file):
        """A name imported now that is ALREADY spoken in this transcript has
        earned a hit. Without this the first eviction cannot separate it from
        a name that never appears — both sit at hits=0 on the same seeded
        last_seen. Measured on a 212-name workspace import: 0 of 427 new terms
        showed a hit before the scan included them."""
        cfg = _cfg("GKE")
        ms._persist_hotwords(["Spanner", "NeverSaid"], cfg, max_count=100,
                             transcript="今天在 Spanner 上跑了一遍")
        store = ms._load_hotword_store()
        assert store["hits"].get("spanner") == 1
        assert store["hits"].get("neversaid", 0) == 0
        # ...which is what makes the unmentioned one the eviction candidate.
        rank = lambda t: (store["last_seen"].get(t, 0), store["hits"].get(t, 0))
        assert rank("neversaid") < rank("spanner")


class TestPhrases:
    """Multi-word terms. The flat `hotword` string is space-separated and so
    cannot hold one, yet most GCP service names are two words. sherpa encodes
    a whole hotwords-file line, so a phrase biases correctly — verified
    against the real model for `Cloud SQL`, `error budget`, `node pool`."""

    def test_phrases_round_trip(self, isolate_hotword_file):
        ms._save_hotword_file("GKE Spanner",
                              {"phrases": ["Cloud SQL", "error budget"]})
        back = ms._load_hotword_store()
        assert back["phrases"] == ["Cloud SQL", "error budget"]
        assert back["terms"] == ["GKE", "Spanner"]

    def test_flat_key_is_untouched_by_phrases(self, isolate_hotword_file):
        """Every existing consumer keeps reading the same space-separated
        string; only phrase-capable ones read the new array."""
        ms._save_hotword_file("GKE", {"phrases": ["Cloud SQL"]})
        body = json.loads(ms._strip_jsonc_comments(
            isolate_hotword_file.read_text(encoding="utf-8")))
        assert body["hotword"] == "GKE"
        assert body["phrases"] == ["Cloud SQL"]

    def test_single_word_in_phrases_is_dropped_on_load(self, isolate_hotword_file):
        """It belongs in `hotword`; two places for one term is a bug factory."""
        _write(isolate_hotword_file, {"hotword": "GKE", "phrases": ["Spanner"]})
        assert ms._load_hotword_store()["phrases"] == []

    def test_phrases_absent_when_there_are_none(self, isolate_hotword_file):
        ms._save_hotword_file("GKE", {"phrases": []})
        body = json.loads(ms._strip_jsonc_comments(
            isolate_hotword_file.read_text(encoding="utf-8")))
        assert "phrases" not in body

    def test_string_form_is_tolerated(self, isolate_hotword_file):
        _write(isolate_hotword_file, {"hotword": "GKE", "phrases": "Cloud SQL"})
        assert ms._load_hotword_store()["phrases"] == ["Cloud SQL"]

    def test_inner_whitespace_is_collapsed(self, isolate_hotword_file):
        _write(isolate_hotword_file,
               {"hotword": "GKE", "phrases": ["Cloud    SQL"]})
        assert ms._load_hotword_store()["phrases"] == ["Cloud SQL"]

    def test_duplicates_dropped_case_insensitively(self, isolate_hotword_file):
        _write(isolate_hotword_file, {"hotword": "GKE",
                                      "phrases": ["Cloud SQL", "cloud sql"]})
        assert ms._load_hotword_store()["phrases"] == ["Cloud SQL"]


class TestTermList:
    def test_words_and_phrases_combine(self):
        assert ms._hotword_term_list("GKE Spanner", ["Cloud SQL"]) == [
            "GKE", "Spanner", "Cloud SQL"]

    def test_deduplicated_across_both_stores(self):
        assert ms._hotword_term_list("GKE", ["gke", "Cloud SQL"]) == [
            "GKE", "Cloud SQL"]

    def test_empty_inputs(self):
        assert ms._hotword_term_list("", None) == []


class TestSherpaEncodable:
    """One stray character makes the recognizer skip the WHOLE term, and it
    complains on C-level stderr that the log tee never captured — so a term
    looked configured while biasing nothing. All measured against the real
    model."""

    @pytest.mark.parametrize("term", [
        "GKE", "n2d", "CockroachDB", "K8s", "P0", "巡检", "GKE 集群",
        "Cloud SQL", "error budget", "node pool", "Spot VM", "AI Native",
    ])
    def test_encodable(self, term):
        assert ms._sherpa_encodable(term)

    @pytest.mark.parametrize("term", [
        "Pub/Sub", "blue-green", "AI-Native", "Agent-Ready", "Semi-Annual",
        "one-liner", "Node.js", "C++", "snake_case", "it's", "3.5",
    ])
    def test_not_encodable(self, term):
        assert not ms._sherpa_encodable(term)

    def test_blank_is_not_encodable(self):
        assert not ms._sherpa_encodable("")
        assert not ms._sherpa_encodable("   ")
        assert not ms._sherpa_encodable(None)


class TestPromptGlossary:
    """`_cfg_hotword` feeds the polish and caption-review prompts. It must not
    lose multi-word terms the way the sherpa hotwords file used to."""

    def test_phrases_survive_with_their_boundaries(self):
        cfg = {"stt": {"funasr": {"hotword": "GKE Spanner",
                                  "hotword_phrases": ["Cloud SQL"]}}}
        got = ms._cfg_hotword(cfg)
        assert "Cloud SQL" in got
        # Space-joining would make it indistinguishable from two terms.
        assert got == "GKE、Spanner、Cloud SQL"

    def test_words_only_still_works(self):
        cfg = {"stt": {"funasr": {"hotword": "GKE Spanner"}}}
        assert ms._cfg_hotword(cfg) == "GKE、Spanner"

    def test_empty_config(self):
        assert ms._cfg_hotword({}) == ""
        assert ms._cfg_hotword({"stt": {"funasr": {"hotword": ""}}}) == ""

    def test_funasr_still_reads_the_raw_space_separated_string(self):
        """The offline recognizer's own hotword format is space-separated, so
        it must keep reading the config value directly, not this rendering."""
        src = Path(ms.__file__).read_text(encoding="utf-8")
        assert 'hotword     = pcfg.get("hotword", "")' in src
