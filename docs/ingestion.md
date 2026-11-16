# Ingestion: page-aware extraction (Commit 2.1)

> **Scope.** This commit turns the four verified corpus PDFs into raw,
> page-level text records. It does **not** clean or normalize text, chunk it,
> index it, or reconstruct tables. Those are later commits (see
> [Remaining ingestion work](#remaining-ingestion-work)).

## Running it

```powershell
python -m founderlens.corpus.download     # once, if data/raw is empty
python -m founderlens.ingestion.extract
```

The command:
1. resolves paths through `founderlens.config.load_settings()`;
2. validates the corpus against `data/manifest.json`;
3. extracts every manifest page and validates the records;
4. publishes the artifact and writes the report;
5. prints a summary that never includes page text.

It exits `1` with an actionable message on any failure. It makes no network
requests and calls no LLM.

| Output | Path | Git |
|--------|------|-----|
| Page records (deterministic) | `data/processed/pages.raw.jsonl` | ignored |
| Diagnostics report (includes run timing) | `reports/extraction_report.json` | ignored |

## Procedure

1. **Corpus validation first.** `founderlens.corpus.validate.validate_corpus`
   checks every manifest document: presence, size, SHA-256, PDF signature,
   readability, encryption and page count. Any issue aborts the run before
   extraction.
2. **Manifest documents only, in manifest order** (sorted by `document_id`).
   No directory is scanned for PDFs.
3. **Per document:**
   - read the bytes once and re-check the SHA-256;
   - open the bytes from memory with PyMuPDF, so the file on disk is never
     opened for writing;
   - reject encrypted or repaired PDFs;
   - confirm the physical page count equals the manifest's;
   - extract pages `1..n` in physical order.
4. **Failures are explicit.** A parser exception on any page aborts the whole
   run with `"<document_id> page <n>: text extraction failed: …"`. It is never
   recorded as an empty page.
5. **Record validation:**
   - the record count equals the sum of the manifest page counts (40 for
     corpus v1.0.0, derived rather than hard-coded);
   - records follow document order, then physical page order;
   - page IDs are unique;
   - each record's metadata matches its manifest entry.
6. **Atomic publish:**
   - the JSONL is written to a temporary file in `data/processed/`;
   - it is re-read, re-parsed and re-validated;
   - only then is it renamed over `pages.raw.jsonl`.

   A failed run leaves any previous artifact and report byte-for-byte
   unchanged. Output directories are created only at this point, never on
   import.

## Extraction configuration (baseline)

| Setting | Value | Why |
|---------|-------|-----|
| Parser | PyMuPDF 1.28.2 (`pymupdf`) | The approved parser, pinned. No other PDF library, no OCR. |
| Call | `Page.get_text("text", flags=67, sort=True)` | Plain text, one output line per text line. |
| Reading order | `sort=True`: lines ordered by position, top-to-bottom then left-to-right | Content-stream order (`sort=False`) put each table cell on its own line and split `$` from its value on every financial table inspected. Position order keeps a row's label and values on one line. |
| Flags | `TEXT_PRESERVE_LIGATURES \| TEXT_PRESERVE_WHITESPACE \| TEXT_MEDIABOX_CLIP` (= 67) | PyMuPDF's default `text` flags (195) minus `TEXT_CID_FOR_UNKNOWN_UNICODE`. A glyph with no Unicode mapping therefore appears as U+FFFD and is counted, instead of becoming an arbitrary character code. On corpus v1.0.0 both flag sets give identical text. |
| Dehyphenation, image placeholders, OCR | Off | Raw output only. |

The whole configuration is recorded in the report (`extraction_config`).

## Page record schema (`extraction_schema_version` 1)

One JSON object per physical page, with keys sorted:

| Field | Meaning |
|-------|---------|
| `document_id` | Manifest document ID, unchanged |
| `document_sha256` | SHA-256 of the source PDF from the manifest, re-verified at extraction |
| `company`, `fiscal_period`, `source_url` | Copied from the manifest entry |
| `page_number` | One-based **physical** page number, not the printed label |
| `page_id` | `{document_id}:p{page_number:04d}`, for example `nvda-fy2025-q3:p0001` |
| `text` | Raw extractor output, unmodified |
| `character_count` | `len(text)` in Unicode code points, whitespace included |
| `extraction_status` | `ok`, or `empty` when the text has no non-whitespace character |

Records are validated on construction. A wrong `page_id`, a
`character_count` that disagrees with the text, or a status that contradicts
the text cannot be created or parsed. Records contain no absolute paths and
no timestamps.

**Raw text.** No whitespace collapsing, lowercasing, header or footer
removal, or value correction is applied. Currency symbols, parentheses for
negative values, decimal points, percent signs and line breaks are kept
exactly as PyMuPDF returns them. Normalization is Commit 2.2.

## Determinism

The serialization is fixed:
- UTF-8 JSON with sorted keys and compact separators;
- non-ASCII characters written literally;
- one record per line with LF endings and a final LF;
- documents in manifest order, pages in physical order.

Given the same PDF bytes, the same PyMuPDF version and the same
`ExtractionConfig`, two runs produce byte-identical `pages.raw.jsonl`. This is
verified by tests on synthetic PDFs and on the real corpus.

**Boundaries.** A different PyMuPDF or MuPDF version, or any configuration
change, may change the text and therefore the artifact hash. The report's
`operational.extraction_duration_seconds` varies between runs by design; the
rest of the report is deterministic.

## Diagnostics

**Automatic diagnostic signals** (`warnings`, each with
`"source": "automatic_heuristic"`) are signals to check, not proof of a bad
extraction:
- `empty_page`: no text was extracted. The page may be blank or image-only,
  and no OCR is applied.
- `low_text`: a non-empty page with fewer than 100 non-whitespace characters.
- `replacement_characters`: the page contains U+FFFD.

Each page entry also records its character, non-whitespace and line counts.

Baseline run on corpus v1.0.0 (`python -m founderlens.ingestion.extract`,
PyMuPDF 1.28.2):

| Document | Pages | Empty | Characters | Mean / page |
|----------|------:|------:|-----------:|------------:|
| `goog-2024-q3` | 10 | 0 | 36,137 | 3,613.7 |
| `goog-2024-q4` | 10 | 0 | 38,414 | 3,841.4 |
| `nvda-fy2025-q3` | 10 | 0 | 32,946 | 3,294.6 |
| `nvda-fy2025-q4` | 10 | 0 | 34,773 | 3,477.3 |
| **Total** | **40** | **0** | **142,270** | |

It raised 0 automatic warnings and found no U+FFFD characters. Character
counts include the layout padding spaces that `sort=True` produces.

## Manual extraction observations

These come from reading representative extracted pages of all four documents:
- opening narrative pages;
- summary results tables;
- the condensed statements of income, balance sheets and cash flows;
- GAAP-to-non-GAAP reconciliations.

They were assessed for document quality only, independently of the
evaluation questions. **The heuristics above do not detect any of them.** The
presence of an expected number in the text does not mean the table was
extracted correctly.

| Area | Observation |
|------|-------------|
| Row alignment | Simple statement rows (balance sheets, cash flows, Alphabet's summary table) come out as one line: label, then values in column order, for example `Revenues  $  76,693  $  88,268` (padding shortened here). Column positions are conveyed only by runs of spaces, and the spacing is not consistent from row to row. |
| Values ↔ periods | Period headers are often split across lines. On NVIDIA reconciliation pages, a header such as `October 27, 2024` becomes three fragments on three lines. One header (`July 28,`) is placed at the far right of the line, out of column order. The values themselves are in the right order, but matching a value to its period from the text alone is unreliable there. |
| Empty cells | In Alphabet's revenue-by-geography reconciliation, cells left blank (for example a hedging-effect column) leave no placeholder. The remaining values shift left, so only the spacing suggests which column each belongs to. |
| Displaced labels | On NVIDIA reconciliation pages some row labels appear **after** their values (`$ 292 … $ 633  Sales, general and administrative`) or on the next line (`Cost of revenue` below its values). Wrapped labels can enclose their values, for example `Interest expense related to amortization of debt` / values / `discount`. |
| Tables across pages | NVIDIA's condensed statement of income starts on physical page 3 and continues on page 4. The page-4 rows (from `Operating Income` down) have no period headers or `(In millions)` unit line on their own page. Some NVIDIA statement titles (`NVIDIA CORPORATION`, `CONDENSED CONSOLIDATED BALANCE SHEETS`) sit physically at the bottom of a page, with the table on the next page. A page-level record therefore separates the title from its table. |
| Wrapped cells | In NVIDIA's summary tables, cells such as `Down 0.5 pts` are split across the lines above and below their row. |
| Currency and units | `$` is kept but separated from the number by padding (`$   35,082`). Unit statements (`(In millions)`, `($ in millions, except earnings per share)`) are present as separate lines. The second one is split around the column-header line. |
| Negative values | Shown in parentheses as in the PDFs, for example `(146)` and `(1)%`, and preserved. No U+2212 minus signs occur. NVIDIA shows a missing comparison as `--`. |
| Reading order (prose) | The narrative pages inspected read in the correct order, and none has side-by-side text columns. Within table rows, a label printed slightly below its values' baseline is placed after them (see *Displaced labels*). Justified lines sometimes contain doubled spaces, and line-end hyphenation (`non-` / `affiliated`) is kept. |
| Headers, footers, footnotes | Every Alphabet page ends with a standalone page-number line (`1` … `10`); NVIDIA pages have none. Footnote markers stay attached to labels (`year over year(1)`, `Total revenues(1)`, `(A)`), and footnote text follows its table. |
| Bullets and symbols | Alphabet's bullet points are extracted as `•`. No bullet character appears in the text of NVIDIA's highlight lines. Curly quotes, dashes, ™, ® and © are preserved. |

## Financial table limitations

- The extractor produces lines of text, not a table structure. Cells, column
  headers and periods are not linked, and nothing in this commit tries to
  link them.
- Downstream steps must not assume that a number's position on a line
  identifies its period or column. That is unreliable for reconciliation
  tables and tables that span pages.
- Normalization (Commit 2.2) must not collapse the padding spaces blindly
  without considering their effect on table rows.

## Remaining ingestion work

- **Commit 2.2:** deterministic text normalization from `pages.raw.jsonl`.
- **Later Stage 2 commits:** configurable chunking that keeps document and
  page spans, with output in `data/processed/`.

Retrieval, generation and evaluation are not implemented.
