# Document corpus

**Corpus:** `founderlens-financial-research` **v1.0.0** — four official quarterly
earnings-release PDFs, 10 pages each (40 pages total). The authoritative record is
[`manifest.json`](manifest.json).

| document_id | Company | Fiscal period | Period end | Published | Pages | Bytes | SHA-256 |
|-------------|---------|---------------|-----------:|----------:|------:|------:|---------|
| `goog-2024-q3` | Alphabet Inc. (GOOG) | FY2024 Q3 | 2024-09-30 | 2024-10-29 | 10 | 127,666 | `54c89309d04aacb1674da28bf2aa3789a3a21a4675c24d837d4ccb7ea84a7ea3` |
| `goog-2024-q4` | Alphabet Inc. (GOOG) | FY2024 Q4 | 2024-12-31 | 2025-02-04 | 10 | 148,726 | `f6779037aa896d961bd14fa415eff46d56213407b9a6f78b4ddf86faab8697b8` |
| `nvda-fy2025-q3` | NVIDIA Corporation (NVDA) | FY2025 Q3 | 2024-10-27 | 2024-11-20 | 10 | 85,389 | `fb0b8b714aa9ec690e7955821b4d81eafb9c7ce8d69e448bc21734b8e01bfea4` |
| `nvda-fy2025-q4` | NVIDIA Corporation (NVDA) | FY2025 Q4 | 2025-01-26 | 2025-02-26 | 10 | 84,498 | `1c716d4fb428ad371801ecc2f71f4ec0d71d87e77649852d6d1449b37b0c8c9e` |

NVIDIA's fiscal year ends in late January: its "fiscal 2025 Q4" ended
2025-01-26, while Alphabet's calendar Q4 2024 ended 2024-12-31. The two are
**not** the same period.

## Provenance (verified 2026-11-16)

| Document | Official page | PDF download URL |
|----------|---------------|------------------|
| NVIDIA Q3 FY25 | [investor.nvidia.com press release](https://investor.nvidia.com/news/press-release-details/2024/NVIDIA-Announces-Financial-Results-for-Third-Quarter-Fiscal-2025/default.aspx) | `https://nvidianews.nvidia.com/_gallery/download_pdf/673e52d63d6332ce4bb72f56/` |
| NVIDIA Q4 FY25 | [investor.nvidia.com press release](https://investor.nvidia.com/news/press-release-details/2025/NVIDIA-Announces-Financial-Results-for-Fourth-Quarter-and-Fiscal-2025/) | `https://nvidianews.nvidia.com/_gallery/download_pdf/67bf85d73d633281c6875f57/` |
| Alphabet Q3 2024 | [abc.xyz/investor/earnings](https://abc.xyz/investor/earnings/) | `https://s206.q4cdn.com/479360582/files/doc_financials/2024/q3/2024q3-alphabet-earnings-release.pdf` |
| Alphabet Q4 2024 | [abc.xyz/investor/earnings](https://abc.xyz/investor/earnings/) | `https://s206.q4cdn.com/479360582/files/doc_financials/2024/q4/2024q4-alphabet-earnings-release.pdf` |

How each URL was established:

- **NVIDIA.** The investor-relations press-release pages have no PDF attached;
  their download link is hidden and has no target. NVIDIA's official
  *Quarterly Results* page links each release to the **NVIDIA Newsroom**
  (`nvidianews.nvidia.com`). The URLs above are that page's own *Download Press
  Release* links. `investor.nvidia.com` serves a bot challenge to scripted
  clients, so it was viewed in an ordinary browser and **no access control was
  bypassed**. The newsroom PDF URLs return the same bytes on repeated requests.
  The PDFs' embedded creation dates (2024-11-20, 2025-02-26) match the
  publication dates.
- **Alphabet.** The `abc.xyz/assets/...` PDF URLs originally proposed now
  redirect (HTTP 301) to the abc.xyz site map, so they are stale. The current
  official links are the *Earnings Release* links on
  `abc.xyz/investor/earnings/`, which point to Alphabet's investor-relations
  CDN (`s206.q4cdn.com`). The PDFs are titled "GOOG Exhibit 99.1 Q3/Q4 2024".

The downloader only contacts `nvidianews.nvidia.com` and `s206.q4cdn.com`
(`ALLOWED_DOWNLOAD_HOSTS` in `src/founderlens/corpus/manifest.py`), over HTTPS,
including after any redirect.

## Reproducing the corpus

```powershell
python -m founderlens.corpus.download   # fetches into data/raw/, verifies SHA-256 + size
python -m founderlens.corpus.validate   # read-only: signature, parse, encryption, pages, hash
```

The download command is idempotent: files that already match are not fetched
again or rewritten. A file that is present but does not match is **left
untouched** and reported. Rerun with `--replace-invalid` to replace it. Bytes
are written to a hidden `.part` file and renamed only after verification, so a
failed download never leaves a partial PDF.

Optional explicit check against the real PDFs (the test is skipped by default
and in CI):

```powershell
$env:FOUNDERLENS_REAL_CORPUS = "1"; python -m pytest tests/test_corpus_real.py -q
```

## Manual download (if automated download fails)

1. Open the official page from the provenance table in a browser.
2. NVIDIA: on the linked NVIDIA Newsroom release, click **Download Press
   Release**. Alphabet: on *abc.xyz/investor/earnings*, choose year 2024 and
   click **Earnings Release** for Q3 or for Q4 & Fiscal Year.
3. Save the file **unchanged** as `data/raw/<document_id>.pdf`, for example
   `data/raw/nvda-fy2025-q3.pdf`.
4. Check the hash in PowerShell. It must equal the manifest value:

   ```powershell
   (Get-FileHash data\raw\nvda-fy2025-q3.pdf -Algorithm SHA256).Hash.ToLower()
   ```

5. Run `python -m founderlens.corpus.validate`.

If the hash differs, **the publisher's file has changed** and the corpus
cannot be reproduced exactly. Do not edit the manifest to match. Record the
discrepancy, and create a new corpus version only by a deliberate, documented
decision, since the evaluation labels refer to v1.0.0 pages.

## Versioning

- The corpus is immutable per version. Adding, removing or replacing any
  document, or any byte change, requires a new `corpus_version` (semantic
  versioning) and re-review of the evaluation evidence.
- Validation never regenerates the manifest. Hashes, sizes, page counts and
  acquisition dates were recorded once, when the files were acquired.

## Redistribution

The PDFs are **not committed** (`data/raw/` is git-ignored). Public availability
does not by itself grant redistribution rights, and no separate permission has
been established, so this repository commits only metadata and acquisition
instructions.
