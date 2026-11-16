"""Typed, immutable project configuration.

Resolution order for the project root:

1. An explicit ``project_root`` argument.
2. The ``FOUNDERLENS_PROJECT_ROOT`` environment variable.
3. Automatic discovery: walk upward from this module's location looking for
   the repository layout (``pyproject.toml`` + ``src/founderlens``). This works
   for source checkouts and editable installs, never via the working directory.

Non-editable installs live in ``site-packages``, where discovery is impossible;
those must use option 1 or 2. Explicit roots must be absolute existing
directories so the result never depends on the process working directory.

This module has no import-time side effects: it creates no directories, reads
no secrets and makes no network requests.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT_ENV_VAR = "FOUNDERLENS_PROJECT_ROOT"

RAW_DATA_DIRNAME = Path("data") / "raw"
PROCESSED_DATA_DIRNAME = Path("data") / "processed"
ARTIFACTS_DIRNAME = Path("artifacts")
REPORTS_DIRNAME = Path("reports")


class ConfigError(ValueError):
    """Raised when configuration inputs are invalid or cannot be resolved."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved, absolute project paths. Construct via :func:`load_settings`."""

    project_root: Path

    def __post_init__(self) -> None:
        # Guard direct construction as well as load_settings().
        _validate_root(self.project_root, source="project_root")

    @property
    def raw_data_dir(self) -> Path:
        return self.project_root / RAW_DATA_DIRNAME

    @property
    def processed_data_dir(self) -> Path:
        return self.project_root / PROCESSED_DATA_DIRNAME

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / ARTIFACTS_DIRNAME

    @property
    def reports_dir(self) -> Path:
        return self.project_root / REPORTS_DIRNAME


def _validate_root(path: Path, *, source: str) -> None:
    if not path.is_absolute():
        raise ConfigError(f"{source} must be an absolute path, got {str(path)!r}.")
    if not path.exists():
        raise ConfigError(f"{source} does not exist: {path}")
    if not path.is_dir():
        raise ConfigError(f"{source} is not a directory: {path}")


def _coerce_root(value: str | os.PathLike[str], *, source: str) -> Path:
    if isinstance(value, str) and not value.strip():
        raise ConfigError(f"{source} must not be empty.")
    path = Path(value)
    _validate_root(path, source=source)
    return path.resolve()


def discover_project_root(start: Path) -> Path | None:
    """Return the nearest ancestor of ``start`` with the repository layout, if any."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "founderlens" / "__init__.py"
        ).is_file():
            return candidate
    return None


def load_settings(
    project_root: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Resolve project settings. See the module docstring for resolution order."""
    if project_root is not None:
        return Settings(_coerce_root(project_root, source="project_root"))

    env = os.environ if environ is None else environ
    env_value = env.get(PROJECT_ROOT_ENV_VAR)
    if env_value is not None:
        return Settings(_coerce_root(env_value, source=PROJECT_ROOT_ENV_VAR))

    discovered = discover_project_root(Path(__file__).parent)
    if discovered is None:
        raise ConfigError(
            "Could not discover the FounderLens project root (this is expected for "
            "non-editable installs). Pass project_root explicitly or set "
            f"{PROJECT_ROOT_ENV_VAR} to the repository's absolute path."
        )
    return Settings(discovered)
