"""Tests for founderlens.config: resolution order, validation, and side effects."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from founderlens import config
from founderlens.config import (
    PROJECT_ROOT_ENV_VAR,
    ConfigError,
    Settings,
    discover_project_root,
    load_settings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clear_root_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Isolate tests from a developer's local override.
    monkeypatch.delenv(PROJECT_ROOT_ENV_VAR, raising=False)


def _snapshot(root: Path) -> set[Path]:
    return set(root.rglob("*"))


def test_default_discovers_repository_root() -> None:
    settings = load_settings()
    assert settings.project_root == REPO_ROOT
    assert (settings.project_root / "pyproject.toml").is_file()


def test_derived_paths_are_relative_to_root(tmp_path: Path) -> None:
    settings = load_settings(tmp_path)
    root = tmp_path.resolve()
    assert settings.project_root == root
    assert settings.raw_data_dir == root / "data" / "raw"
    assert settings.processed_data_dir == root / "data" / "processed"
    assert settings.artifacts_dir == root / "artifacts"
    assert settings.reports_dir == root / "reports"
    for path in (
        settings.raw_data_dir,
        settings.processed_data_dir,
        settings.artifacts_dir,
        settings.reports_dir,
    ):
        assert path.is_absolute()


def test_explicit_root_accepts_str(tmp_path: Path) -> None:
    assert load_settings(str(tmp_path)).project_root == tmp_path.resolve()


def test_explicit_root_overrides_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_root = tmp_path / "env"
    explicit_root = tmp_path / "explicit"
    env_root.mkdir()
    explicit_root.mkdir()
    monkeypatch.setenv(PROJECT_ROOT_ENV_VAR, str(env_root))
    assert load_settings(explicit_root).project_root == explicit_root.resolve()


def test_env_var_overrides_discovery(tmp_path: Path) -> None:
    settings = load_settings(environ={PROJECT_ROOT_ENV_VAR: str(tmp_path)})
    assert settings.project_root == tmp_path.resolve()


def test_env_var_is_read_from_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV_VAR, str(tmp_path))
    assert load_settings().project_root == tmp_path.resolve()


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_explicit_root_is_rejected(value: str) -> None:
    with pytest.raises(ConfigError, match="must not be empty"):
        load_settings(value)


def test_empty_env_var_is_rejected() -> None:
    with pytest.raises(ConfigError, match=PROJECT_ROOT_ENV_VAR):
        load_settings(environ={PROJECT_ROOT_ENV_VAR: ""})


def test_relative_root_is_rejected() -> None:
    with pytest.raises(ConfigError, match="absolute"):
        load_settings(Path("relative") / "root")


def test_missing_root_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_settings(tmp_path / "missing")


def test_file_root_is_rejected(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ConfigError, match="not a directory"):
        load_settings(file_path)


def test_direct_construction_is_validated() -> None:
    with pytest.raises(ConfigError, match="absolute"):
        Settings(Path("relative"))


def test_settings_are_immutable(tmp_path: Path) -> None:
    settings = load_settings(tmp_path)
    field_name = "project_root"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(settings, field_name, Path.home())


def test_resolution_is_independent_of_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_settings().project_root == REPO_ROOT


def test_discover_returns_none_outside_repository(tmp_path: Path) -> None:
    assert discover_project_root(tmp_path) is None


def test_undiscoverable_root_without_override_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulates a non-editable install, where the package lives outside the repo.
    monkeypatch.setattr(config, "discover_project_root", lambda start: None)
    with pytest.raises(ConfigError, match=PROJECT_ROOT_ENV_VAR):
        load_settings(environ={})


def test_discover_finds_root_from_nested_directory(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    package = tmp_path / "src" / "founderlens"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    nested = package / "deeper"
    nested.mkdir()
    assert discover_project_root(nested) == tmp_path.resolve()


def test_loading_settings_creates_no_files_or_directories(tmp_path: Path) -> None:
    before = _snapshot(tmp_path)
    settings = load_settings(tmp_path)
    _ = (
        settings.raw_data_dir,
        settings.processed_data_dir,
        settings.artifacts_dir,
        settings.reports_dir,
    )
    assert _snapshot(tmp_path) == before
    assert not settings.raw_data_dir.exists()
    assert not settings.artifacts_dir.exists()
