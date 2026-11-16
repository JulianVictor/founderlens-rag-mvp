# Ingestion: page-aware extraction and normalization

> **Scope.**
> - **Commit 2.1** turns the four verified corpus PDFs into raw, page-level
>   text records.
> - **Commit 2.2** adds conservative text normalization and word-coordinate
>   table candidates; see
>   [Commit 2.2](#commit-22-layout-aware-normalization).
>
> Neither commit chunks or indexes text (see
> [Remaining ingestion work](#remaining-ingestion-work)).

## Running it

```powershell
python -m founderlens.corpus.download     # once, if data/raw is empty
python -m founderlens.ingestion.extract
python -m founderlens.ingestion.normalize # Commit 2.2; needs the raw artifact
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

## Financial table limitations (raw text)

- The raw extractor produces lines of text, not a table structure. In
  `pages.raw.jsonl`, cells, column headers and periods are not linked.
- Downstream steps must not assume that a number's position on a line
  identifies its period or column. That is unreliable for reconciliation
  tables and tables that span pages.
- Commit 2.2 adds a separate, conservative table view (below). It does not
  change the raw text.

## Commit 2.2: layout-aware normalization

```powershell
python -m founderlens.ingestion.normalize
```

**Inputs:**
- the corpus, validated first;
- `data/processed/pages.raw.jsonl`, validated against the manifest.

For every page, the command re-extracts the raw text with the Commit 2.1
configuration and requires it to equal the raw record. A stale raw artifact,
or changed extraction settings, therefore fail the run with
"re-run `python -m founderlens.ingestion.extract`".

**Outputs**, both git-ignored and published atomically after validation:
- `data/processed/pages.normalized.jsonl`, one record per physical page;
- `reports/normalization_report.json`.

The raw artifact is read only.

### Text normalization rules

`normalize_text` is deterministic and idempotent:
- CR and CRLF are converted to LF.
- Unicode is normalized to **NFC**. This only composes canonical
  equivalents; compatibility forms (NFKC/NFKD) are not used, so `™`,
  ligatures and similar characters are untouched.
- Trailing whitespace is removed from each line.
- Runs of blank lines are reduced to one, and leading and trailing blank lines
  are removed.

Leading and internal spaces are **kept**, because the layout-sorted text uses
them for column alignment. Nothing is lowercased, dehyphenated, removed as a
header or footer, or rewritten.

On corpus v1.0.0 the raw text has no CR characters, no non-NFC text and no
trailing whitespace. The whole change, 142,270 → 142,067 characters, comes
from collapsing redundant blank lines. That reduction is not evidence of
better extraction quality.

### Table investigation: why not `Page.find_tables()`

On every corpus page, PyMuPDF 1.28.2's `find_tables` was compared with the
word-coordinate approach.

- **`strategy="lines"` (default)** found few and wrong tables.
  - Alphabet statements have only horizontal rules, so each row's label and
    values merged into a single cell (`Marketable securities 86,868`), with
    the other cells `None`.
  - On NVIDIA pages, decorative page boxes became "tables" whose header cell
    held about 30 lines of prose.
  - Many statement pages returned none.
- **`strategy="text"`** returned exactly one table per page on all 40 pages,
  prose included.
  - It cut words across columns (`Cost of revenu` | `e`).
  - It merged a value with the next column's `$` (`20,687 $`).
  - It did place NVIDIA's `July 28,` header in the correct column, which shows
    the word coordinates carry the column structure.
- `find_tables` also prints an advisory line about another package on every
  call. It is not used.

### Selected method: word-coordinate table candidates (`founderlens.ingestion.tables`)

This is a bounded method, not a general table parser. The thresholds are
fixed in `TableConfig` and recorded in the report.

1. Words from `Page.get_text("words")` are grouped into visual lines by
   vertical overlap against the line's first word. A label printed a few
   points below its values therefore stays on their row.
2. Lines are split into segments at gaps wider than 5 pt. A lone `$` joins
   the following number. Words with no gap are joined without a space
   (`(1)%`).
3. A line with a long run of closely spaced words is prose and ends a table.
4. A candidate starts at a line that has a stub label in the left 30% of the
   page and a numeric cell. It grows over non-prose lines and stops after a
   large gap or three lines without values.
5. **Columns** are the union of overlapping x-ranges of the value cells. A
   row with two cells in one column makes the table `fallback`.
6. **Header lines** are assigned bottom-up, by x-overlap with the columns.
   - A segment covering several columns becomes a spanning header, recorded
     with the columns it physically overlaps.
   - Collection stops at prose, a large gap, a page-centred title, a
     stub-only line above the header block, or a line with value-like text.
7. **Unit statements** (`in millions/billions/thousands`) are taken from
   inside the table, or from the nearest one at most 110 pt above it. Only
   the matching text segment is kept. Units are never inferred.

**Table record:**
- `table_id` (`{page_id}:tNN`, ordered top-down), `page_id`, `document_id`,
  `page_number`, `bounding_box`;
- `structure_status`, `column_count`, `column_headers` (multi-line headers
  joined with `\n`, or `null` when none was found), `spanning_headers`,
  `stub_header`, `unit_statements`;
- `rows`, each a `label` plus `cells`. `null` is an empty position, never
  zero; `""` as a label means no stub text was found;
- `warnings`.

**Status values** come from automatic structural checks, not semantic
verification:

| Status | Meaning |
|--------|---------|
| `structured` | Rows and columns assigned without collisions, every column has header text, every row with values has a label, and a nearby unit statement exists. Financial associations are **not** independently verified. |
| `uncertain` | Rows and columns assigned, but at least one check failed. `warnings` gives the codes: `no_header_rows`, `columns_without_header`, `rows_without_label`, `values_above_first_row`, `spanning_header_ranges_ambiguous`, `header_text_between_columns`, `no_unit_statement_on_page`. Not verified evidence. |
| `fallback` | No consistent column structure (`column_collision`). Only the bounding box and warnings are kept; `normalized_text` is the source. |

**Normalized page record:**
- the raw identity and metadata fields;
- `extraction_status` and `raw_character_count`;
- `normalized_text` and `normalized_character_count`;
- `tables` and `table_text_overlap`;
- `normalization_warnings`, one entry for each page with an empty page,
  uncertain tables or fallback tables.

`normalized_text` always holds the whole page, table text included.
`tables` is an additional view. `table_text_overlap` is `true` when a table
carries rows, so consumers must not concatenate the two without
deduplicating.

### Results on corpus v1.0.0 (`python -m founderlens.ingestion.normalize`)

| Document | Pages | Characters raw → normalized | Table candidates | Structured | Uncertain | Fallback |
|----------|------:|---------------------------:|-----------------:|-----------:|----------:|---------:|
| `goog-2024-q3` | 10 | 36,137 → 36,105 | 14 | 7 | 7 | 0 |
| `goog-2024-q4` | 10 | 38,414 → 38,376 | 16 | 7 | 9 | 0 |
| `nvda-fy2025-q3` | 10 | 32,946 → 32,876 | 15 | 2 | 13 | 0 |
| `nvda-fy2025-q4` | 10 | 34,773 → 34,710 | 13 | 5 | 6 | 2 |
| **Total** | **40** | **142,270 → 142,067** | **58** | **21** | **35** | **2** |

There are no empty pages. As an automated consistency check, all 296
structured rows with values match a raw layout line containing the same
label followed by the same values in the same order.

There is no labelled table ground truth, so **no table-accuracy percentage
is claimed**.

### Commit 2.2 before-and-after inspection

This inspection was performed by the implementer (Claude Code), not an
independent review. Selected page regions were rendered and compared with
the Commit 2.1 raw text and the Commit 2.2 table records. Pages were chosen by
layout type, not by evaluation questions:
- Alphabet Q3 pages 1 and 9;
- Alphabet Q4 pages 1, 6 and 10;
- NVIDIA Q3 pages 1, 4 and 8;
- NVIDIA Q4 pages 1 and 7.

| Problem | Page | Raw text (2.1) | Normalized / table view (2.2) | Verdict |
|---------|------|----------------|-------------------------------|---------|
| A. Period headers out of column order | `nvda-fy2025-q3` p8 | `October  October  October  October … July 28,` on one line, `27, 29, 27, 29,` on the next | `column_headers` = `October\n27,\n2024`, `July 28,\n2024`, `October\n29,\n2023`, …; spans `Three Months Ended` (cols 0–2) and `Nine Months Ended` (cols 3–4), matching the render | **Improved.** The table stays `uncertain` only because no unit statement is near it. |
| B. Label after its values | `nvda-fy2025-q3` p8 | `$ 292 … $ 633  Sales, general and administrative` | row `Sales, general and administrative` → `$ 292, $ 282, $ 240, $ 822, $ 633` | **Improved.** The label is 5 pt below its values in the PDF, and grouping now keeps them together. |
| C. Empty cells | `goog-2024-q3` p9 | `$ 76,693  $ 88,268   $ 89,268  15 %  0 %  (1)%  16 %`, with no gap marker | `Total revenues(1)` → `…, null, "$ 89,268", "15 %", "0 %", "(1)%", "16 %"`, with the blank "Less FX Effect" cell kept as `null` | **Improved for cells.** Spanning headers here physically overlap columns 3–6 and 4–6, while the true spans are 2–7 and 4–7, so the table is `uncertain` (`spanning_header_ranges_ambiguous`). |
| D. Table continues across pages | `nvda-fy2025-q3` p4 | Rows from `Operating Income` with no headers or units | Table `uncertain` with `no_header_rows` and `no_unit_statement_on_page`; headers `null` | **Detected, not resolved.** Headers from page 3 are not inherited, by design. |
| E. Wrapped cells | `nvda-fy2025-q3` p1 | `Down 0.5   Up 0.6` above `Gross margin`, `pts  pts` below | Separate label-less rows with `Down 0.5`/`Up 0.6` and `pts`/`pts` in the Q/Q and Y/Y columns; `rows_without_label` → `uncertain` | **Unresolved but contained.** Positions are kept and nothing is merged. On `nvda-fy2025-q4` p1, a wrapped `Down 1.6 pts` sits within 5 pt of `76.0%`, which makes both summary tables `fallback`. |
| F. Layout whitespace | All pages | Column padding with runs of spaces; many blank lines | Padding kept for alignment; blank-line runs collapsed | **Partly changed, on purpose.** Internal spaces are preserved. |
| Titles absorbed into headers (found during this commit) | `goog-2024-q4` p6 | n/a | An early prototype read `Alphabet Inc.` and `(In millions)` as the first column's header. Page-centred titles now end header collection. | **Fixed before publishing.** |

Financial notation is preserved in cells: `$`, thousands separators,
decimals, `%`, parentheses for negatives, `-` and `--` placeholders, and
footnote markers in labels (`Operating margin(2)`, `(A)`).

### Known unresolved layouts

- Some table candidates have no header lines on their own page:
  - NVIDIA statements continued from a previous page (Q3 pages 4, 5, 7 and 9;
    Q4 pages 7, 8 and 9);
  - the lower parts of Alphabet tables that a large vertical gap splits into
    a second candidate (pages 4 and 8 of both quarters).

  Headers and units are not inherited.
- Wrapped row labels are left as label-only rows plus label-less value rows.
  For example, `Weighted average shares used in per share` / `computation:`
  and NVIDIA cash flow wrapped labels.
- Alphabet constant-currency reconciliation tables (pages 9–10) have
  multi-level headers whose column ranges cannot be established from
  positions alone.
- NVIDIA Q4 summary tables (page 1) are `fallback` because of wrapped cells.
- Spanning header column ranges are positional, even in `structured` tables.

## Remaining ingestion work

- **Commit 2.3, chunking:** build retrieval chunks from `normalized_text`
  and the table view without duplicating content, since
  `table_text_overlap` marks pages where both carry the same values.
  - Keep document and page provenance.
  - Use only `structured` tables as row-level evidence, if at all; treat
    `uncertain` and `fallback` regions as plain text.
  - Do not split table rows across chunks.

Retrieval, generation and evaluation are not implemented.
