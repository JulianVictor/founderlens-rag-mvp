"""Acquire the approved corpus PDFs listed in the manifest.

Usage::

    python -m founderlens.corpus.download [--document-id ID ...] [--replace-invalid]

Safety properties:

- Only manifest documents with a ``download_url`` on an approved host are fetched.
- HTTPS only; every redirect target is re-checked against the host allowlist.
- Explicit timeout, bounded retries for transient failures, maximum size.
- Bytes stream to a ``*.part`` temp file in the target directory, are verified
  against the manifest's SHA-256 and size, and only then atomically renamed.
  Failed or mismatched downloads are deleted; no partial PDF is left behind.
- Idempotent: a present file whose hash already matches is never re-downloaded
  or overwritten. A present file that does *not* match is left untouched
  unless ``--replace-invalid`` is given.

Nothing here runs on import. Unit tests inject a fake opener; no network access.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from email.message import Message
from enum import Enum
from http.client import HTTPMessage
from pathlib import Path
from typing import IO, Protocol

from founderlens.config import ConfigError, load_settings
from founderlens.corpus.manifest import (
    ALLOWED_DOWNLOAD_HOSTS,
    DocumentEntry,
    ManifestError,
    check_https_url,
    is_safe_filename,
    load_manifest,
)

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_BYTES = 20 * 1024 * 1024
DEFAULT_ATTEMPTS = 3
MAX_REDIRECTS = 3
_CHUNK_SIZE = 64 * 1024
_USER_AGENT = "founderlens-corpus-downloader/0.1 (+python-urllib)"
_RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class DownloadError(RuntimeError):
    """A document could not be acquired or verified. The message is user-facing."""


class _Response(Protocol):
    @property
    def headers(self) -> Message: ...

    def geturl(self) -> str: ...

    def read(self, amt: int, /) -> bytes: ...

    def close(self) -> None: ...


Opener = Callable[[urllib.request.Request, float], _Response]
Sleeper = Callable[[float], None]


class _AllowlistRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to an approved HTTPS host, and at most MAX_REDIRECTS times."""

    max_redirections = MAX_REDIRECTS

    def __init__(self, allowed_hosts: frozenset[str]) -> None:
        super().__init__()
        self.allowed_hosts = allowed_hosts

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        issue = check_https_url(newurl, allowed_hosts=self.allowed_hosts)
        if issue:
            raise DownloadError(f"refusing redirect to {newurl!r}: {issue}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_opener(allowed_hosts: frozenset[str] = ALLOWED_DOWNLOAD_HOSTS) -> Opener:
    """Real network opener with HTTPS-only handlers and the redirect allowlist."""
    director = urllib.request.build_opener(_AllowlistRedirectHandler(allowed_hosts))

    def open_url(request: urllib.request.Request, timeout: float) -> _Response:
        response: _Response = director.open(request, timeout=timeout)
        return response

    return open_url


class Outcome(Enum):
    DOWNLOADED = "downloaded"
    ALREADY_VALID = "already valid"


@dataclass(frozen=True, slots=True)
class DownloadResult:
    document_id: str
    path: Path
    outcome: Outcome


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def target_path(raw_dir: Path, filename: str) -> Path:
    """Resolve ``raw_dir/filename``, rejecting any name that could escape ``raw_dir``."""
    if not is_safe_filename(filename):
        raise DownloadError(f"unsafe filename {filename!r}")
    base = raw_dir.resolve()
    path = (base / filename).resolve()
    if path.parent != base:
        raise DownloadError(f"filename {filename!r} escapes {base}")
    return path


@contextmanager
def _temporary_part_file(dest: Path) -> Iterator[Path]:
    part = dest.with_name(f".{dest.name}.{os.getpid()}.part")
    try:
        yield part
    finally:
        part.unlink(missing_ok=True)


def _stream_to_file(response: _Response, part: Path, max_bytes: int) -> tuple[int, str]:
    declared = response.headers.get("Content-Length")
    if declared is not None and declared.isdigit() and int(declared) > max_bytes:
        raise DownloadError(f"declared size {declared} bytes exceeds limit of {max_bytes}")
    digest = hashlib.sha256()
    total = 0
    with part.open("wb") as handle:
        while chunk := response.read(_CHUNK_SIZE):
            total += len(chunk)
            if total > max_bytes:
                raise DownloadError(f"download exceeded size limit of {max_bytes} bytes")
            digest.update(chunk)
            handle.write(chunk)
    return total, digest.hexdigest()


def _fetch_once(
    url: str, part: Path, *, opener: Opener, timeout: float, max_bytes: int
) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    response = opener(request, timeout)
    try:
        # Defence in depth: the final URL must still be an approved host.
        final = response.geturl()
        if issue := check_https_url(final, allowed_hosts=ALLOWED_DOWNLOAD_HOSTS):
            raise DownloadError(f"final URL rejected: {issue}")
        return _stream_to_file(response, part, max_bytes)
    finally:
        response.close()


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _RETRYABLE_HTTP_STATUS
    return isinstance(exc, urllib.error.URLError | TimeoutError | ConnectionError)


def _fetch_with_retries(
    document_id: str,
    url: str,
    part: Path,
    *,
    opener: Opener,
    timeout: float,
    max_bytes: int,
    attempts: int,
    sleep: Sleeper,
) -> tuple[int, str]:
    attempt = 1
    while True:
        try:
            return _fetch_once(url, part, opener=opener, timeout=timeout, max_bytes=max_bytes)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if not _is_transient(exc) or attempt >= attempts:
                raise DownloadError(
                    f"failed to download {document_id} after {attempt} attempt(s): {exc}"
                ) from exc
        sleep(min(2.0 ** (attempt - 1), 8.0))
        attempt += 1


def download_document(
    entry: DocumentEntry,
    raw_dir: Path,
    *,
    opener: Opener | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    attempts: int = DEFAULT_ATTEMPTS,
    sleep: Sleeper = time.sleep,
    replace_invalid: bool = False,
) -> DownloadResult:
    """Ensure ``entry`` exists in ``raw_dir`` with the manifest's exact bytes."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    dest = target_path(raw_dir, entry.filename)

    if dest.exists():
        if dest.is_file() and sha256_file(dest) == entry.sha256:
            return DownloadResult(entry.document_id, dest, Outcome.ALREADY_VALID)
        if not replace_invalid:
            raise DownloadError(
                f"{dest} exists but does not match the manifest SHA-256. It was left untouched; "
                "remove it or rerun with --replace-invalid."
            )

    if entry.download_url is None:
        raise DownloadError(
            f"no verified download URL for {entry.document_id}; download it manually from "
            f"{entry.source_url} and save it as {dest}"
        )
    if issue := check_https_url(entry.download_url, allowed_hosts=ALLOWED_DOWNLOAD_HOSTS):
        raise DownloadError(issue)

    open_url = opener or build_opener()
    raw_dir.mkdir(parents=True, exist_ok=True)
    with _temporary_part_file(dest) as part:
        size, digest = _fetch_with_retries(
            entry.document_id,
            entry.download_url,
            part,
            opener=open_url,
            timeout=timeout,
            max_bytes=max_bytes,
            attempts=attempts,
            sleep=sleep,
        )
        if size != entry.size_bytes:
            raise DownloadError(
                f"{entry.document_id}: downloaded {size} bytes, manifest expects {entry.size_bytes}"
            )
        if digest != entry.sha256:
            raise DownloadError(
                f"{entry.document_id}: SHA-256 mismatch (got {digest}, expected {entry.sha256}). "
                "The publisher may have changed the file; the download was discarded."
            )
        part.replace(dest)  # atomic on the same filesystem
    return DownloadResult(entry.document_id, dest, Outcome.DOWNLOADED)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.corpus.download",
        description="Download and verify the approved corpus PDFs listed in data/manifest.json.",
    )
    parser.add_argument("--project-root", default=None, help="Absolute project root override.")
    parser.add_argument(
        "--document-id",
        action="append",
        default=None,
        help="Restrict to this document (repeatable). Default: all documents.",
    )
    parser.add_argument(
        "--replace-invalid",
        action="store_true",
        help="Replace an existing file whose SHA-256 does not match the manifest.",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.project_root)
        manifest = load_manifest(settings.corpus_manifest_path)
    except (ConfigError, ManifestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    entries = list(manifest.documents)
    if args.document_id:
        unknown = sorted(set(args.document_id) - {e.document_id for e in entries})
        if unknown:
            print(f"error: unknown document id(s): {unknown}", file=sys.stderr)
            return 1
        entries = [e for e in entries if e.document_id in set(args.document_id)]

    failures = 0
    for entry in entries:
        try:
            result = download_document(
                entry,
                settings.raw_data_dir,
                timeout=args.timeout,
                replace_invalid=args.replace_invalid,
            )
        except DownloadError as exc:
            failures += 1
            print(f"FAIL  {entry.document_id}: {exc}", file=sys.stderr)
        else:
            print(f"OK    {entry.document_id}: {result.outcome.value} ({entry.filename})")

    if failures:
        print(
            f"{failures} document(s) failed. See data/README.md for manual download instructions.",
            file=sys.stderr,
        )
        return 1
    print(f"All {len(entries)} document(s) present and verified against manifest SHA-256.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
