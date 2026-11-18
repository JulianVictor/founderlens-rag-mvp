"""Frozen retrieval policy (Commit 4.3): identity, verification and the selected BM25 path.

Synthetic project only: a V1 baseline plus a V2 collection built with a fake
V2-identified tokenizer. No model, no network, no real corpus, no reports.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]  # rank-bm25 0.2.2 has no type information

from founderlens.config import Settings
from founderlens.retrieval import bm25
from founderlens.retrieval import embeddings as em
from founderlens.retrieval import policy as pol
from founderlens.retrieval.bm25 import DEFAULT_BM25_CONFIG, BM25Retriever
from founderlens.retrieval.collections import (
    CHUNKING_V2_CONFIG,
    collection_path,
    load_named_collection,
)
from founderlens.retrieval.policy import (
    SELECTED_POLICY,
    V2_ARTIFACT_SHA256,
    RetrievalPolicy,
    load_policy_retriever,
)
from founderlens.retrieval.schema import RetrievalError
from tests.test_chunking_v2 import V2Encoder
from tests.test_chunking_v2 import _project as _v1_project

Factory = Callable[..., Path]
QUERIES = ("revenue", "Google Cloud revenue grew", "zyzzyva", "", "$ % ( )")


def _v2_project(make_project: Factory) -> tuple[Settings, str]:
    root = _v1_project(make_project)
    settings = Settings(root)
    em.run_build_collection(settings, V2Encoder(), expected_document_count=4)
    shutil.rmtree(settings.reports_dir)  # benchmark (and other) reports are not runtime inputs
    sha = hashlib.sha256(collection_path(settings, "v2").read_bytes()).hexdigest()
    return settings, sha


# --- the policy itself ------------------------------------------------------------------------


def test_selected_policy_is_bm25_on_v2_with_ten_candidates() -> None:
    p = SELECTED_POLICY
    assert (p.method, p.collection, p.candidate_top_k) == ("bm25", "v2", 10)
    assert (p.score_threshold, p.fallback, p.page_deduplication) == (None, None, False)
    assert p.collection_config_id == CHUNKING_V2_CONFIG.config_id == "0c2f87411627"
    assert p.bm25_config_id == DEFAULT_BM25_CONFIG.config_id == "c21bdd0691f8"
    assert p.candidate_top_k != bm25.DEFAULT_BM25_CONFIG.top_k  # not the CLI default of 5


def test_policy_identity_is_stable_and_separate_from_artifact_identities() -> None:
    assert SELECTED_POLICY.policy_id == RetrievalPolicy().policy_id == "f20913f1c1ec"
    canonical = json.dumps(SELECTED_POLICY.to_dict(), sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest()[:12] == SELECTED_POLICY.policy_id
    assert RetrievalPolicy(candidate_top_k=5).policy_id != SELECTED_POLICY.policy_id
    assert set(SELECTED_POLICY.to_dict()) == {
        "schema_version",
        "method",
        "collection",
        "collection_config_id",
        "bm25_config_id",
        "candidate_top_k",
        "score_threshold",
        "fallback",
        "page_deduplication",
        "matching",
        "tie_breaking",
    }  # no paths, timestamps or artifact hashes
    assert V2_ARTIFACT_SHA256 not in canonical
    assert len(V2_ARTIFACT_SHA256) == 64
    others = {"0c2f87411627", "c21bdd0691f8", "3fd7b205dc81", "f6982b8eb438", V2_ARTIFACT_SHA256}
    assert SELECTED_POLICY.policy_id not in others


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "dense"}, "method must be 'bm25'"),
        ({"method": "hybrid"}, "method must be 'bm25'"),
        ({"collection": "v3"}, "unknown collection"),
        ({"candidate_top_k": 0}, "candidate_top_k"),
        ({"candidate_top_k": True}, "candidate_top_k"),
        ({"score_threshold": 0.5}, "score_threshold must be None"),
        ({"fallback": "dense"}, "fallback must be None"),
        ({"page_deduplication": True}, "page_deduplication must be False"),
        ({"schema_version": 2}, "schema_version"),
    ],
)
def test_invalid_policies_are_rejected(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RetrievalPolicy(**kwargs)


# --- the generation-facing retriever ----------------------------------------------------------


def test_policy_retriever_equals_direct_bm25_v2_top_ten(make_project: Factory) -> None:
    settings, sha = _v2_project(make_project)
    retriever = load_policy_retriever(settings, expected_artifact_sha256=sha)
    v2 = load_named_collection(settings, "v2")
    direct = BM25Retriever(v2.chunks)
    for query in QUERIES:
        got = retriever.search(query)
        assert [r.to_dict() for r in got] == [r.to_dict() for r in direct.search(query, top_k=10)]
        assert all(r.method == "bm25" for r in got)  # never another retriever
    hits = retriever.search("revenue")
    assert hits
    assert [r.rank for r in hits] == list(range(1, len(hits) + 1))
    for hit in hits:  # the original, validated chunk records (text and provenance)
        assert hit.chunk in v2.chunks
        assert hit.chunk.chunking_config_id == CHUNKING_V2_CONFIG.config_id
    assert retriever.search("") == () == direct.search("", top_k=10)
    assert not settings.reports_dir.exists()
    assert not (settings.project_root / "artifacts").exists()  # no embedding artifacts needed


def test_identities_report_policy_and_artifacts_separately(make_project: Factory) -> None:
    settings, sha = _v2_project(make_project)
    ids = load_policy_retriever(settings, expected_artifact_sha256=sha).identities()
    assert ids["policy_id"] == SELECTED_POLICY.policy_id
    assert ids["policy"] == SELECTED_POLICY.to_dict()
    assert ids["collection_config_id"] == CHUNKING_V2_CONFIG.config_id
    assert ids["collection_artifact_sha256"] == sha
    assert ids["bm25_config_id"] == SELECTED_POLICY.bm25_config_id
    assert ids["bm25_config"] == DEFAULT_BM25_CONFIG.scoring_dict()
    assert len({ids["policy_id"], ids["collection_config_id"], ids["bm25_config_id"]}) == 3
    retriever = load_policy_retriever(settings, expected_artifact_sha256=sha)
    ids["policy"]["candidate_top_k"] = 99  # a returned copy, not the policy
    assert retriever.identities()["policy"]["candidate_top_k"] == 10
    with pytest.raises(AttributeError):
        retriever.policy = RetrievalPolicy(candidate_top_k=5)  # type: ignore[misc]  # read-only by design
    with pytest.raises(AttributeError):
        retriever.policy.candidate_top_k = 5  # type: ignore[misc]  # frozen dataclass


def test_index_is_built_once_and_requests_ten(
    make_project: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, sha = _v2_project(make_project)
    builds: list[int] = []

    def counting(*args: Any, **kwargs: Any) -> Any:
        builds.append(1)
        return BM25Okapi(*args, **kwargs)

    monkeypatch.setattr("founderlens.retrieval.bm25.BM25Okapi", counting)
    requested: list[int | None] = []
    search = BM25Retriever.search

    def spy(self: BM25Retriever, query: str, top_k: int | None = None) -> Any:
        requested.append(top_k)
        return search(self, query, top_k)

    monkeypatch.setattr(BM25Retriever, "search", spy)
    retriever = load_policy_retriever(settings, expected_artifact_sha256=sha)
    for query in QUERIES:
        retriever.search(query)
    assert builds == [1]
    assert requested == [10] * len(QUERIES)


# --- verification and no fallback -------------------------------------------------------------


def test_artifact_hash_mismatch_is_rejected(make_project: Factory) -> None:
    settings, _ = _v2_project(make_project)
    with pytest.raises(RetrievalError, match="differs from the evaluated artifact"):
        load_policy_retriever(settings)  # the synthetic V2 is not the evaluated artifact
    before = collection_path(settings, "v2").read_bytes()
    with pytest.raises(RetrievalError):
        load_policy_retriever(settings, expected_artifact_sha256="0" * 64)
    assert collection_path(settings, "v2").read_bytes() == before  # never rebuilt


def test_missing_v2_fails_without_falling_back_to_v1(make_project: Factory) -> None:
    settings, sha = _v2_project(make_project)
    collection_path(settings, "v2").unlink()
    assert collection_path(settings, "v1").is_file()
    with pytest.raises(RetrievalError, match=r"collection v2 not found.*build-collection"):
        load_policy_retriever(settings, expected_artifact_sha256=sha)
    assert not collection_path(settings, "v2").exists()


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (RetrievalPolicy(collection_config_id="a8a0288e0348"), "policy expects collection v2"),
        (RetrievalPolicy(collection="v1"), "policy expects collection v1"),
        (RetrievalPolicy(bm25_config_id="000000000000"), "BM25 scoring configuration"),
    ],
)
def test_incompatible_identities_are_rejected(
    make_project: Factory, policy: RetrievalPolicy, message: str
) -> None:
    settings, sha = _v2_project(make_project)
    with pytest.raises(RetrievalError, match=message):
        load_policy_retriever(settings, policy, expected_artifact_sha256=sha)


def test_changed_bm25_library_is_rejected(
    make_project: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, sha = _v2_project(make_project)
    monkeypatch.setattr(bm25, "library_version", lambda: "9.9.9")
    with pytest.raises(RetrievalError, match="rank-bm25 version"):
        load_policy_retriever(settings, expected_artifact_sha256=sha)


def test_policy_cli_reports_failures(
    make_project: Factory, capsys: pytest.CaptureFixture[str]
) -> None:
    settings, _ = _v2_project(make_project)
    assert pol.main(["--project-root", str(settings.project_root)]) == 1
    assert "differs from the evaluated artifact" in capsys.readouterr().err


# --- lightweight runtime and unchanged commands -----------------------------------------------


def test_selected_path_initializes_no_transformer_or_dense_code(make_project: Factory) -> None:
    settings, sha = _v2_project(make_project)
    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        from founderlens.config import Settings
        from founderlens.retrieval.policy import load_policy_retriever
        r = load_policy_retriever(Settings(Path(sys.argv[1])), expected_artifact_sha256={sha!r})
        assert r.search("revenue")
        heavy = ("torch", "transformers", "sentence_transformers", "huggingface_hub")
        local = ("dense", "hybrid", "embeddings")
        print(sorted(m for m in sys.modules if m.split(".")[0] in heavy
                     or m in {{"founderlens.retrieval." + n for n in local}}))
        """
    )
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1"}
    done = subprocess.run(
        [sys.executable, "-c", script, str(settings.project_root)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_standalone_bm25_cli_still_defaults_to_v1(
    make_project: Factory, capsys: pytest.CaptureFixture[str]
) -> None:
    settings, _ = _v2_project(make_project)
    assert (
        bm25.main(["--project-root", str(settings.project_root), "--query", "revenue", "--json"])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    v1_sha = hashlib.sha256(collection_path(settings, "v1").read_bytes()).hexdigest()
    assert payload["chunk_artifact_sha256"] == v1_sha
    assert len(payload["results"]) <= DEFAULT_BM25_CONFIG.top_k == 5
