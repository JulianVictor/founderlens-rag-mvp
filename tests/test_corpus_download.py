"""Download utility behaviour with an injected fake opener (no network access)."""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import date
from email.message import Message
from http.client import HTTPMessage
from pathlib import Path

import pytest

from founderlens.corpus import download as dl
from founderlens.corpus.download import (
    DownloadError,
    Outcome,
    download_document,
    target_path,
)
from founderlens.corpus.manifest import DocumentEntry
from tests.conftest import make_pdf_bytes, sha256

URL = "https://s206.q4cdn.com/479360582/files/example.pdf"
PAYLOAD = make_pdf_bytes(2)


def _entry(data: bytes = PAYLOAD, *, download_url: str | None = URL) -> DocumentEntry:
    return DocumentEntry(
        document_id="aaa-2024-q1",
        company="Example Corp",
        ticker="EXMP",
        title="Example",
        document_type="earnings_release",
        fiscal_period="FY2024 Q1",
        period_end=date(2024, 3, 31),
        publication_date=date(2024, 4, 25),
        source_url="https://example.com/investors",
        download_url=download_url,
        acquisition_date=date(2024, 5, 1),
        filename="aaa-2024-q1.pdf",
        sha256=sha256(data),
        size_bytes=len(data),
        page_count=2,
    )


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        url: str = URL,
        content_length: int | None = None,
        fail_after: int | None = None,
    ) -> None:
        self._stream = io.BytesIO(body)
        self._url = url
        self._fail_after = fail_after
        self._read = 0
        self.closed = False
        self._headers = Message()
        if content_length is not None:
            self._headers["Content-Length"] = str(content_length)

    @property
    def headers(self) -> Message:
        return self._headers

    def geturl(self) -> str:
        return self._url

    def read(self, amt: int, /) -> bytes:
        if self._fail_after is not None and self._read >= self._fail_after:
            raise ConnectionResetError("connection reset mid-stream")
        chunk = self._stream.read(min(amt, 1024))
        self._read += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


class FakeOpener:
    """Returns (or raises) scripted results in order and records each call."""

    def __init__(self, *results: FakeResponse | BaseException) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, float]] = []

    def __call__(self, request: urllib.request.Request, timeout: float) -> FakeResponse:
        self.calls.append((request.full_url, timeout))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _no_network(request: urllib.request.Request, timeout: float) -> FakeResponse:
    raise AssertionError("network must not be used")


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(URL, code, "error", HTTPMessage(), None)


def _files(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


Recorder = Callable[[float], None]


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def sleep(sleeps: list[float]) -> Recorder:
    return sleeps.append


def test_downloads_verifies_and_finalizes(tmp_path: Path, sleep: Recorder) -> None:
    opener = FakeOpener(FakeResponse(PAYLOAD, content_length=len(PAYLOAD)))
    result = download_document(_entry(), tmp_path, opener=opener, timeout=7.5, sleep=sleep)
    assert result.outcome is Outcome.DOWNLOADED
    assert (tmp_path / "aaa-2024-q1.pdf").read_bytes() == PAYLOAD
    assert _files(tmp_path) == ["aaa-2024-q1.pdf"]  # no .part leftovers
    assert opener.calls == [(URL, 7.5)]


def test_repeated_run_is_idempotent_and_offline(tmp_path: Path) -> None:
    (tmp_path / "aaa-2024-q1.pdf").write_bytes(PAYLOAD)
    before = (tmp_path / "aaa-2024-q1.pdf").stat().st_mtime_ns
    result = download_document(_entry(), tmp_path, opener=_no_network)
    assert result.outcome is Outcome.ALREADY_VALID
    assert (tmp_path / "aaa-2024-q1.pdf").stat().st_mtime_ns == before


def test_mismatched_existing_file_is_not_overwritten(tmp_path: Path) -> None:
    existing = tmp_path / "aaa-2024-q1.pdf"
    existing.write_bytes(b"%PDF-manually placed but different")
    with pytest.raises(DownloadError, match="left untouched"):
        download_document(_entry(), tmp_path, opener=_no_network)
    assert existing.read_bytes() == b"%PDF-manually placed but different"


def test_replace_invalid_replaces_mismatched_file(tmp_path: Path, sleep: Recorder) -> None:
    (tmp_path / "aaa-2024-q1.pdf").write_bytes(b"stale")
    opener = FakeOpener(FakeResponse(PAYLOAD))
    result = download_document(_entry(), tmp_path, opener=opener, sleep=sleep, replace_invalid=True)
    assert result.outcome is Outcome.DOWNLOADED
    assert (tmp_path / "aaa-2024-q1.pdf").read_bytes() == PAYLOAD


def test_checksum_mismatch_discards_download(tmp_path: Path, sleep: Recorder) -> None:
    tampered = PAYLOAD[:-1] + b"X"
    opener = FakeOpener(FakeResponse(tampered))
    with pytest.raises(DownloadError, match="SHA-256 mismatch"):
        download_document(_entry(), tmp_path, opener=opener, sleep=sleep)
    assert _files(tmp_path) == []


def test_size_mismatch_discards_download(tmp_path: Path, sleep: Recorder) -> None:
    opener = FakeOpener(FakeResponse(PAYLOAD + b"extra"))
    with pytest.raises(DownloadError, match="manifest expects"):
        download_document(_entry(), tmp_path, opener=opener, sleep=sleep)
    assert _files(tmp_path) == []


def test_declared_size_over_limit_rejected_before_reading(tmp_path: Path, sleep: Recorder) -> None:
    response = FakeResponse(PAYLOAD, content_length=10_000_000)
    with pytest.raises(DownloadError, match="exceeds limit"):
        download_document(
            _entry(), tmp_path, opener=FakeOpener(response), max_bytes=1000, sleep=sleep
        )
    assert response.closed
    assert _files(tmp_path) == []


def test_streamed_size_over_limit_aborts_and_cleans_up(tmp_path: Path, sleep: Recorder) -> None:
    opener = FakeOpener(FakeResponse(PAYLOAD))  # no Content-Length header
    with pytest.raises(DownloadError, match="exceeded size limit"):
        download_document(_entry(), tmp_path, opener=opener, max_bytes=100, sleep=sleep)
    assert _files(tmp_path) == []


def test_timeouts_retry_with_bounded_attempts(
    tmp_path: Path, sleep: Recorder, sleeps: list[float]
) -> None:
    opener = FakeOpener(TimeoutError("timed out"), TimeoutError("timed out"), TimeoutError("t"))
    with pytest.raises(DownloadError, match="after 3 attempt"):
        download_document(_entry(), tmp_path, opener=opener, attempts=3, sleep=sleep)
    assert len(opener.calls) == 3
    assert sleeps == [1.0, 2.0]
    assert _files(tmp_path) == []


def test_transient_failure_then_success(tmp_path: Path, sleep: Recorder) -> None:
    opener = FakeOpener(_http_error(503), FakeResponse(PAYLOAD))
    result = download_document(_entry(), tmp_path, opener=opener, sleep=sleep)
    assert result.outcome is Outcome.DOWNLOADED
    assert len(opener.calls) == 2


def test_mid_stream_failure_cleans_up_then_retries(tmp_path: Path, sleep: Recorder) -> None:
    opener = FakeOpener(FakeResponse(PAYLOAD, fail_after=200), FakeResponse(PAYLOAD))
    result = download_document(_entry(), tmp_path, opener=opener, sleep=sleep)
    assert result.outcome is Outcome.DOWNLOADED
    assert _files(tmp_path) == ["aaa-2024-q1.pdf"]


def test_client_errors_are_not_retried(
    tmp_path: Path, sleep: Recorder, sleeps: list[float]
) -> None:
    opener = FakeOpener(_http_error(404))
    with pytest.raises(DownloadError, match="after 1 attempt"):
        download_document(_entry(), tmp_path, opener=opener, sleep=sleep)
    assert sleeps == []


def test_final_url_on_unapproved_host_rejected(tmp_path: Path, sleep: Recorder) -> None:
    response = FakeResponse(PAYLOAD, url="https://evil.example.com/x.pdf")
    with pytest.raises(DownloadError, match="final URL rejected"):
        download_document(_entry(), tmp_path, opener=FakeOpener(response), sleep=sleep)
    assert _files(tmp_path) == []


@pytest.mark.parametrize(
    "target",
    ["http://s206.q4cdn.com/479360582/files/x.pdf", "https://evil.example.com/x.pdf"],
)
def test_unsafe_redirect_rejected(target: str) -> None:
    handler = dl._AllowlistRedirectHandler(frozenset({"s206.q4cdn.com"}))
    request = urllib.request.Request(URL)
    with pytest.raises(DownloadError, match="refusing redirect"):
        handler.redirect_request(request, io.BytesIO(), 302, "Found", HTTPMessage(), target)


def test_redirect_within_allowlist_followed() -> None:
    handler = dl._AllowlistRedirectHandler(frozenset({"s206.q4cdn.com"}))
    request = urllib.request.Request(URL)
    new_url = "https://s206.q4cdn.com/479360582/files/moved.pdf"
    followed = handler.redirect_request(request, io.BytesIO(), 302, "Found", HTTPMessage(), new_url)
    assert followed is not None
    assert followed.full_url == new_url


def test_missing_download_url_gives_manual_instructions(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="download it manually"):
        download_document(_entry(download_url=None), tmp_path, opener=_no_network)


def test_unapproved_download_url_rejected_without_network(tmp_path: Path) -> None:
    entry = _entry(download_url="https://evil.example.com/x.pdf")
    with pytest.raises(DownloadError, match="not an approved download host"):
        download_document(entry, tmp_path, opener=_no_network)


@pytest.mark.parametrize("filename", ["../escape.pdf", "sub/x.pdf", "..\\x.pdf"])
def test_target_path_rejects_traversal(tmp_path: Path, filename: str) -> None:
    with pytest.raises(DownloadError, match="unsafe filename"):
        target_path(tmp_path, filename)


def test_cli_rejects_unknown_document_id(
    make_project: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_project()
    code = dl.main(["--project-root", str(root), "--document-id", "nope"])
    assert code == 1
    assert "unknown document id" in capsys.readouterr().err


def test_cli_reports_already_valid_without_network(
    make_project: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_project()
    monkeypatch.setattr(dl, "build_opener", lambda *a, **k: _no_network)
    assert dl.main(["--project-root", str(root)]) == 0
    out = capsys.readouterr().out
    assert out.count("already valid") == 2
