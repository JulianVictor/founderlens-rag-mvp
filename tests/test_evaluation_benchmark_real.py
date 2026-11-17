"""Explicit real benchmark: BM25 and dense on V2, development split only.

Skipped by default: needs the corpus PDFs (``FOUNDERLENS_REAL_CORPUS=1``) and
the pinned model in the local Hugging Face cache (``FOUNDERLENS_REAL_MODEL=1``).
Only ``dataset.json`` and ``questions.dev.jsonl`` are copied into ``tmp_path``,
so the held-out file is not even present. Checks function, identities,
denominators and reproducibility; no quality threshold is asserted. Run::

    $env:FOUNDERLENS_REAL_CORPUS = "1"; $env:FOUNDERLENS_REAL_MODEL = "1"
    python -m pytest tests/test_evaluation_benchmark_real.py -q
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import pytest

from founderlens.config import Settings, load_settings
from founderlens.corpus.manifest import load_manifest
from founderlens.evaluation.benchmark import (
    DEFAULT_EVALUATION_CONFIG,
    build_systems,
    deterministic_view,
    load_dev_split,
    run_benchmark,
    write_reports,
)
from founderlens.ingestion.chunk import run_chunking
from founderlens.ingestion.extract import run_extraction
from founderlens.ingestion.normalize import run_normalization
from founderlens.retrieval.collections import collection_path, load_named_collection
from founderlens.retrieval.embeddings import (
    EmbeddingConfig,
    SentenceTransformerEncoder,
    run_build,
    run_build_collection,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("FOUNDERLENS_REAL_CORPUS") != "1"
    or os.environ.get("FOUNDERLENS_REAL_MODEL") != "1",
    reason="set FOUNDERLENS_REAL_CORPUS=1 and FOUNDERLENS_REAL_MODEL=1 (model must be cached)",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_development_benchmark(tmp_path: Path) -> None:
    source = load_settings()
    manifest = load_manifest(source.corpus_manifest_path)
    originals = {d.filename: _sha(source.raw_data_dir / d.filename) for d in manifest.documents}
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "evaluation").mkdir()
    shutil.copyfile(source.corpus_manifest_path, tmp_path / "data" / "manifest.json")
    for filename in originals:
        shutil.copyfile(source.raw_data_dir / filename, tmp_path / "data" / "raw" / filename)
    for name in ("dataset.json", "questions.dev.jsonl"):  # never the held-out split
        shutil.copyfile(source.evaluation_dir / name, tmp_path / "evaluation" / name)
    settings = Settings(tmp_path)
    run_extraction(settings)
    run_normalization(settings)
    run_chunking(settings)
    encoder = SentenceTransformerEncoder(EmbeddingConfig())
    run_build_collection(settings, encoder)
    v2 = load_named_collection(settings, "v2")
    run_build(settings, encoder, EmbeddingConfig(), v2, collection_path(settings, "v2"))

    split = load_dev_split(settings.evaluation_dir, settings.corpus_manifest_path)
    assert all(q.question_id.startswith("dev-") for q in split.questions)
    collection, systems, load_ms = build_systems(settings)
    first = run_benchmark(split, systems, collection, load_timings_ms=load_ms)
    second = run_benchmark(split, systems, collection)
    assert first.complete
    assert second.complete
    assert deterministic_view(first.report) == deterministic_view(second.report)

    report = first.report
    assert report["dataset"]["questions"] == len(split.questions) == 24
    assert report["dataset"]["split_sha256"] == _sha(source.evaluation_dir / "questions.dev.jsonl")
    identities = report["systems"]
    assert {i["collection_sha256"] for i in identities.values()} == {v2.artifact_sha256}
    assert identities["bm25"]["embedding_config_id"] == "not_applicable"
    repository_matrix = (
        source.artifacts_dir / "embeddings" / v2.chunking_config_id / "embeddings.npy"
    )
    if repository_matrix.is_file():
        assert identities["dense"]["matrix_sha256"] == _sha(repository_matrix)

    answerable = [q for q in split.questions if q.answerable]
    multi_fact = [q for q in answerable if len(q.required_facts) >= 2]
    agg = report["aggregates"]
    assert agg["answerable_n"] == len(answerable)
    for method in ("bm25", "dense"):
        for k in DEFAULT_EVALUATION_CONFIG.ks:
            assert agg["overall"][method]["recall"][str(k)]["n"] == len(answerable)
            assert agg["overall"][method]["complete_coverage"][str(k)]["n"] == len(multi_fact)
            value = agg["overall"][method]["fact_coverage"][str(k)]["mean"]
            assert 0.0 <= value <= 1.0
    pages = {(c.document_id, c.page_number) for c in collection.chunks}
    for record in report["per_question"]:
        for gold_pages in (record["gold"] or {}).values():
            for label in gold_pages:
                document_id, page = label.rsplit(":p", 1)
                assert (document_id, int(page)) in pages  # labels resolve to real pages
        for entry in record["systems"].values():
            assert entry["status"] == "ok"
            assert entry["returned"] == 10

    json_path, md_path = write_reports(report, tmp_path / "reports")
    assert json_path.is_file()
    assert "Primary results" in md_path.read_text("utf-8")
    for filename, digest in originals.items():
        assert _sha(source.raw_data_dir / filename) == digest
