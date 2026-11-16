"""Smoke tests: package import, metadata, and the ``python -m founderlens`` entry point."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

import pytest

import founderlens
from founderlens.__main__ import main
from founderlens.config import PROJECT_ROOT_ENV_VAR

REPO_ROOT = Path(__file__).resolve().parents[1]


def _clean_env(**extra: str) -> dict[str, str]:
    """Process environment without credentials or a project-root override."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("OPENAI_") and key != PROJECT_ROOT_ENV_VAR
    }
    env.update(extra)
    return env


def _run(args: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_package_version_matches_installed_metadata() -> None:
    assert founderlens.__version__ == "0.1.0"
    assert importlib.metadata.version("founderlens") == founderlens.__version__


def test_main_reports_status_for_explicit_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["--project-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert f"FounderLens AI RAG MVP {founderlens.__version__}" in out
    assert str(tmp_path.resolve()) in out
    assert "not created" in out
    assert "Status: OK" in out


def test_main_returns_error_for_invalid_root(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--project-root", "relative-path"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "configuration error" in captured.err
    assert captured.out == ""


def test_main_creates_no_files_or_directories(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--project-root", str(tmp_path)]) == 0
    capsys.readouterr()
    assert list(tmp_path.iterdir()) == []


def test_module_entry_point_from_different_directory(tmp_path: Path) -> None:
    # Launched from an unrelated cwd, without credentials: discovery must
    # still find the repository, and nothing may be written to the cwd.
    result = _run(["-m", "founderlens"], cwd=tmp_path, env=_clean_env())
    assert result.returncode == 0, result.stderr
    assert f"Project root:   {REPO_ROOT}" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_module_entry_point_reports_config_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    result = _run(
        ["-m", "founderlens"],
        cwd=tmp_path,
        env=_clean_env(**{PROJECT_ROOT_ENV_VAR: str(missing)}),
    )
    assert result.returncode == 1
    assert "does not exist" in result.stderr


def test_import_has_no_filesystem_side_effects(tmp_path: Path) -> None:
    result = _run(
        ["-c", "import founderlens, founderlens.config, founderlens.__main__"],
        cwd=tmp_path,
        env=_clean_env(**{PROJECT_ROOT_ENV_VAR: str(tmp_path)}),
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
