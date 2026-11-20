"""Local generation-token counting (Commit 5.2).

Usage::

    python -m founderlens.generation.token_budget prepare               # fetch once (network)
    python -m founderlens.generation.token_budget prepare --from-file F # or copy a local file
    python -m founderlens.generation.token_budget check                 # offline verification

Counts use ``tiktoken`` with ``o200k_base``, the encoding tiktoken maps to the
GPT-5 family (``tiktoken.encoding_name_for_model("gpt-5.4-mini-2026-03-17")``).
They are **local estimates** for budgeting. Provider-reported usage
(:class:`~founderlens.generation.schema.TokenUsage`) is the actual count, and it
also includes request framing the local count cannot see.

**Assets.** ``o200k_base`` needs a vocabulary file (about 3.6 MB). By default
tiktoken downloads it on first use into a temporary-directory cache. This
module never lets that happen implicitly:

- ``prepare`` fetches the official file once (or copies one given with
  ``--from-file``), verifies its SHA-256 and writes it to
  ``artifacts/tokenizers/tiktoken/`` (git-ignored), under the file name
  tiktoken's cache uses for that URL.
- :func:`load_generation_tokenizer` reads that file once and checks its
  SHA-256. A missing or altered file raises :class:`TokenizerUnavailableError`.
  The ranks are parsed from exactly those verified bytes, and tiktoken's
  official ``o200k_base`` definition (split pattern, special tokens) is used
  with its file loader replaced by one that returns these ranks. That loader
  first checks that tiktoken asked for the pinned URL and hash. tiktoken
  therefore never reads, caches or downloads a file. Nothing is downloaded
  during imports, tests or query handling, and the default temporary cache
  is not used.

Strings that look like special tokens (``<|endoftext|>``) are encoded as
ordinary text (``disallowed_special=()``), never as control tokens.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import importlib.metadata
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from founderlens.config import ConfigError, Settings, load_settings
from founderlens.generation.errors import GenerationError

if TYPE_CHECKING:
    import tiktoken

ENCODING_NAME = "o200k_base"
ASSET_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
# The hash tiktoken 0.14.0 itself pins for this file (tiktoken_ext.openai_public).
ASSET_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
TIKTOKEN_DISTRIBUTION = "tiktoken"
_CACHE_SUBDIR = Path("tokenizers") / "tiktoken"
_LOAD_LOCK = threading.Lock()


class TokenizerUnavailableError(GenerationError):
    """The local tokenizer asset is missing, altered or cannot be loaded."""

    category = "tokenizer_unavailable"


class TokenCounter(Protocol):
    """Counts tokens of a text. ``identity`` names the tokenizer exactly."""

    @property
    def identity(self) -> str: ...

    def count(self, text: str) -> int: ...


def tiktoken_version() -> str:
    return importlib.metadata.version(TIKTOKEN_DISTRIBUTION)


def tokenizer_identity() -> str:
    """``tiktoken-<version>/o200k_base``: the identity the default context config expects."""
    return f"tiktoken-{tiktoken_version()}/{ENCODING_NAME}"


def asset_cache_dir(settings: Settings) -> Path:
    return settings.artifacts_dir / _CACHE_SUBDIR


def asset_path(settings: Settings) -> Path:
    # tiktoken's cache file name for a URL is the SHA-1 of the URL.
    return asset_cache_dir(settings) / hashlib.sha1(ASSET_URL.encode("utf-8")).hexdigest()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_verified_asset(settings: Settings) -> bytes:
    """The asset's bytes after a full SHA-256 check (never downloads)."""
    path = asset_path(settings)
    if not path.is_file():
        raise TokenizerUnavailableError(
            f"{ENCODING_NAME} tokenizer asset not prepared; run "
            "python -m founderlens.generation.token_budget prepare"
        )
    data = path.read_bytes()
    if _sha256(data) != ASSET_SHA256:
        raise TokenizerUnavailableError(
            f"{ENCODING_NAME} tokenizer asset does not match SHA-256 {ASSET_SHA256}; "
            "delete it and run prepare again"
        )
    return data


def verify_asset(settings: Settings) -> Path:
    """The verified asset path, or :class:`TokenizerUnavailableError` (never downloads)."""
    read_verified_asset(settings)
    return asset_path(settings)


def _parse_ranks(data: bytes) -> dict[bytes, int]:
    """The same parsing as ``tiktoken.load.load_tiktoken_bpe``: base64 token, rank."""
    ranks: dict[bytes, int] = {}
    for line in data.splitlines():
        if line:
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
    return ranks


class TiktokenCounter:
    """A loaded tiktoken encoding. Build once (see :func:`load_generation_tokenizer`), reuse."""

    __slots__ = ("_encoding", "_identity")

    def __init__(self, encoding: tiktoken.Encoding, identity: str) -> None:
        self._encoding = encoding
        self._identity = identity

    @property
    def identity(self) -> str:
        return self._identity

    def count(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        return len(self._encoding.encode(text, allowed_special=set(), disallowed_special=()))


def load_generation_tokenizer(settings: Settings) -> TiktokenCounter:
    """Build ``o200k_base`` from the verified local asset bytes only.

    Call once per process and reuse the counter (parsing takes about 0.2 s).
    tiktoken's official definition supplies the split pattern and special
    tokens. Its file loader is swapped, under a lock and only for this call,
    for one that returns ranks parsed from the bytes just verified, after
    checking that tiktoken requested the pinned URL and hash. No file is read
    twice and no network path exists.
    """
    data = read_verified_asset(settings)
    import tiktoken  # imported here: importing this module never loads tiktoken

    # tiktoken_ext ships no py.typed marker; import_module keeps it out of the type check.
    openai_public = importlib.import_module("tiktoken_ext.openai_public")
    ranks = _parse_ranks(data)

    def verified_ranks(url: str, expected_hash: str | None = None) -> dict[bytes, int]:
        if url != ASSET_URL or expected_hash != ASSET_SHA256:
            raise TokenizerUnavailableError(
                f"tiktoken {tiktoken_version()} expects a different {ENCODING_NAME} asset than "
                "the pinned one; update the pin deliberately"
            )
        return ranks

    namespace = vars(openai_public)  # the module's own globals, read by o200k_base()
    with _LOAD_LOCK:
        original = namespace["load_tiktoken_bpe"]
        namespace["load_tiktoken_bpe"] = verified_ranks
        try:
            spec = openai_public.o200k_base()
        finally:
            namespace["load_tiktoken_bpe"] = original
    try:
        encoding = tiktoken.Encoding(**spec)
    except Exception as exc:  # tiktoken raises plain exceptions; keep the cause short
        raise TokenizerUnavailableError(
            f"cannot build {ENCODING_NAME} from the prepared asset ({type(exc).__name__})"
        ) from None
    return TiktokenCounter(encoding, tokenizer_identity())


def prepare_asset(settings: Settings, *, from_file: Path | None = None) -> Path:
    """Fetch (or copy) the official asset, verify its SHA-256 and store it atomically."""
    target = asset_path(settings)
    if target.is_file() and _sha256(target.read_bytes()) == ASSET_SHA256:
        return target
    if from_file is not None:
        data = from_file.read_bytes()
    else:
        from tiktoken.load import read_file  # explicit, user-requested network fetch

        data = read_file(ASSET_URL)
    if _sha256(data) != ASSET_SHA256:
        raise TokenizerUnavailableError(
            f"{ENCODING_NAME} asset does not match the pinned SHA-256 {ASSET_SHA256}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(data)
    temporary.replace(target)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.generation.token_budget",
        description="Prepare or check the local generation tokenizer asset (o200k_base).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Fetch or copy and verify the asset.")
    prepare.add_argument("--from-file", default=None, help="Copy this local file instead.")
    commands.add_parser("check", help="Verify and load the prepared asset (offline).")
    parser.add_argument("--project-root", default=None, help="Absolute project root override.")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.project_root)
        if args.command == "prepare":
            source = None if args.from_file is None else Path(args.from_file)
            path = prepare_asset(settings, from_file=source)
            print(f"Tokenizer asset ready: {path} (sha256 {ASSET_SHA256})")
            return 0
        counter = load_generation_tokenizer(settings)
        sample = counter.count("FounderLens tokenizer check")
        print(f"Tokenizer OK: {counter.identity}, asset sha256 {ASSET_SHA256}, sample {sample}")
        return 0
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except TokenizerUnavailableError as exc:
        print(f"Tokenizer FAILED: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Tokenizer FAILED: cannot read or write ({type(exc).__name__})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
