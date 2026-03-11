# MySwat Code Expert Index Design

## Context

MySwat already has a good foundation for:

- project-scoped long-term memory in TiDB
- AI-distilled knowledge storage and vector search
- persistent task state and workflow recovery

That foundation is strong for workflow memory and document knowledge, but it is
not enough to answer deep source-code questions reliably across large repos such
as `tidb-expert` or `tikv-expert`.

The missing layer is a **project-local exact code index**.

## Goal

The finalized design adds a local source index so MySwat can answer code-heavy
questions using:

1. exact source retrieval from a local project index
2. project memory and document knowledge already stored in TiDB
3. later, AI-generated code summaries stored in TiDB

Phase A scope is intentionally narrow:

- source files
- selected `*.proto` files
- selected config files explicitly included by project config

For now, the following stay on the existing `myswat feed` / TiDB knowledge path
and are **not** part of the local code index:

- documentation
- RFC / design notes
- issue discussions
- review notes

## Final Architecture

The finalized architecture is:

- **Local project files**: exact source index and exact retrieval
- **TiDB**: project memory, document knowledge, and later semantic code summaries
- **AI agent**: answer synthesis and, in later phases, code summarization

This keeps the architecture split clear:

- MySwat does **not** put a full IDE-style source graph into TiDB.
- MySwat does **not** rely on vector search alone for source-code questions.
- Phase A is **local-only** for indexing and retrieval augmentation.

## Non-Goals

This design does not aim to build:

- a full IDE
- a full LSP replacement
- a JetBrains-equivalent semantic graph
- automatic background indexing in Phase A
- local indexing for docs/RFC/issues in Phase A
- TiDB schema changes or TiDB summary writes in Phase A

The target is a lightweight, deterministic, code-aware index that is good
enough to power expert-style retrieval for large repos.

## Ownership and Configuration

Index configuration is owned per project in `Project.config_json`, not in global
settings. The runtime should materialize this into a typed project index config
with defaults.

Typed defaults:

```json
{
  "code_index": {
    "enabled": false,
    "repo_root": null,
    "include_globs": [],
    "extra_include_globs": [],
    "exclude_globs": [
      "**/.git/**",
      "**/vendor/**",
      "**/node_modules/**",
      "**/target/**",
      "**/build/**",
      "**/dist/**"
    ],
    "max_chunk_lines": 200,
    "max_chunk_bytes": 16384,
    "enable_ctags": true,
    "lsp_tools": []
  }
}
```

Config rules:

- `include_globs` should cover project source files.
- `extra_include_globs` is the explicit allow-list for selected config / proto
  files.
- docs, RFCs, and issue exports remain outside this config because they stay on
  the existing `feed` / TiDB path.
- missing config falls back to typed defaults rather than ad hoc dict handling.

## Local Prerequisites

Phase A requires local tooling on the machine where indexing runs:

- `tree-sitter`
- `universal-ctags`
- SQLite (bundled with Python is acceptable for the local DB)

Optional enrichers for later phases:

- `gopls`
- `rust-analyzer`

Operational rules:

- indexing commands fail fast with actionable errors if required local tools are
  missing
- optional LSP tools never block Phase A
- Phase A does not depend on network services other than the repo already being
  present locally

## Storage Split

### TiDB Responsibilities

TiDB remains the system of record for:

- project metadata
- agent/workflow state
- project-shared knowledge
- document knowledge from the existing `feed` path
- later, semantic code summaries produced in Phase B+
- embeddings and vector search for semantic knowledge

TiDB stores the **semantic layer**, not the raw local source index.

### Local Index Responsibilities

Each expert project keeps a project-local code index on disk.

Default location:

```text
<repo_root>/.myswat/index/
```

If the project has no repo path:

```text
~/.myswat/projects/<project-slug>/index/
```

The local index stores the **exact retrieval layer**:

- file inventory
- content fingerprints
- deterministic cross-store UIDs
- symbols
- code chunks
- exact text search data
- chunk location metadata for on-demand source loading

## Identity Model

SQLite row ids are local implementation details only. They are never used as the
cross-store identity.

The finalized design uses deterministic identifiers:

- `file_uid`: stable per project-relative file path
- `chunk_uid`: stable per file chunk boundary and split metadata

UID derivation:

- `file_uid = sha256(normalized_repo_relative_path)`
- `chunk_uid = sha256(file_uid + chunk_kind + start_line + end_line + parent_chunk_uid + chunk_sequence)`

Rules:

- UIDs must be reproducible across rebuilds of the same project state.
- TiDB metadata must reference `file_uid` / `chunk_uid`, never SQLite row ids.
- SQLite may still use internal integer keys for joins, but those keys do not
  escape the local DB.

## Local Index Layout

Final layout:

```text
.myswat/
  index/
    manifest.json
    code_index.sqlite
    runs/
      2026-03-09T12-00-00Z.json
    cache/
      parser/
      symbol/
```

### `manifest.json`

Tracks:

- project slug
- repo path
- repo HEAD commit at last build
- index version
- last full build time
- last incremental refresh time
- enabled local parsers / enrichers

### `code_index.sqlite`

Single local SQLite DB, because it is:

- simple to back up
- easy to query
- portable
- much lighter than a full IDE cache

Final Phase A tables:

#### `files`

- `file_uid` text primary key
- `path`
- `language`
- `size_bytes`
- `mtime`
- `sha256`
- `repo_commit`
- `parse_status`

#### `symbols`

- `symbol_id` integer primary key
- `file_uid`
- `name`
- `qualified_name`
- `kind`
- `parent_symbol_id` nullable
- `start_line`
- `end_line`
- `signature_text`

#### `chunks`

- `chunk_uid` text primary key
- `file_uid`
- `symbol_id` nullable
- `chunk_kind` (`function`, `method`, `type`, `module`, `fallback`)
- `start_line`
- `end_line`
- `text_hash`
- `parent_chunk_uid` nullable
- `chunk_sequence` nullable
- `source_byte_start` nullable
- `source_byte_end` nullable
- `raw_text_cache` nullable

Raw source storage policy:

- canonical source remains the repo file on disk
- chunk rows store file + range (and optional byte offsets) as the source of
  truth
- `raw_text_cache` is optional and can be size-limited for hot chunks
- split chunks can be reassembled by `parent_chunk_uid` + `chunk_sequence`

#### `relations` (optional Phase D only)

- `src_symbol_id`
- `dst_symbol_id`
- `relation_type`

Examples:

- `defines`
- `calls`
- `implements`
- `references_import`

This table is intentionally **not** part of Phase A. If enabled later, it
should be populated only from LSP outputs (`gopls` / `rust-analyzer`), not from
`tree-sitter` heuristics.

#### `fts_chunks`

SQLite FTS5 virtual table over:

- `path`
- `qualified_name`
- `search_text`

This enables fast exact lexical search without needing TiDB vector search.

`search_text` normalization for Phase A is intentionally minimal and explicit:

- keep original code/comments as text source
- normalize line endings and tabs
- collapse repeated whitespace
- keep identifier punctuation such as `_` for tokenizer compatibility

No comments stripping or aggressive rewrites are part of Phase A.

## Parsing and Indexing Stack

### Phase A Required Tools

- `tree-sitter`
  - primary code-aware chunking
  - syntax-aware boundaries
- `universal-ctags`
  - symbol metadata enrichment

### Phase D Optional Enrichers

- `gopls` for Go projects such as TiDB
- `rust-analyzer` for Rust projects such as TiKV
- relation edges populated from LSP outputs only

These are optional enrichers, not prerequisites for the first workable
version.

## Phase A Boundary

Phase A is explicitly local-only.

Phase A includes:

- file discovery
- syntax-aware chunking
- symbol extraction
- SQLite persistence
- FTS-based exact retrieval
- retrieval-time integration that prepends local hits ahead of existing TiDB
  retrieval

Phase A does **not** include:

- TiDB schema changes
- TiDB writes for code summaries
- embeddings for source chunks
- automatic semantic summarization
- automatic background indexing

## Phase A Ingestion Pipeline

### Step 1: File Discovery

Discover files by project config.

Expected input classes:

- configured source-code globs
- selected `*.proto` files
- selected config files explicitly listed in `extra_include_globs`

Rules:

- skip generated/vendor/build outputs unless explicitly configured
- record hash and mtime
- only re-index changed files during incremental refresh
- do not index docs, RFCs, issues, or review notes through this path

### Step 2: Code-Aware Chunking

For each indexed file:

1. detect language
2. parse with `tree-sitter`
3. extract top-level code units

Preferred chunk boundaries:

- function
- method
- impl block
- struct / enum / interface / trait
- module-level constant block
- fallback fixed-size chunk when parsing fails

Each chunk must preserve:

- `chunk_uid`
- source location (file + line range, optional byte range)
- line range
- symbol anchor if available

Chunk size limits are required for model context safety:

- `max_chunk_lines = 200` (default)
- `max_chunk_bytes = 16 KiB` (default)

When a syntax unit exceeds limits:

1. split by child syntax blocks where possible
2. otherwise split into fixed windows (for example `120` lines with `20` line overlap)
3. set `parent_chunk_uid` and `chunk_sequence` for each child chunk

### Step 3: Symbol Extraction

Run `ctags` to enrich chunk/file metadata with:

- symbol name
- symbol kind
- scope / qualified name

If optional LSP enrichers are enabled in later phases, add:

- definition targets
- references
- hover summary
- interface / impl relationships

### Step 4: Persist Local Results

Persist to local SQLite only:

- file metadata keyed by `file_uid`
- symbol metadata
- chunk ranges keyed by `chunk_uid`
- exact-search index

Phase A writes no semantic code knowledge to TiDB.

## Phase B Semantic Summary Persistence

Phase B adds semantic code summaries on top of the local exact index.

### Summary Generation

Summarization is introduced in Phase B, not Phase A.

Allowed triggers in Phase B:

- on-demand summarization for top retrieved chunks without summaries
- explicit CLI mode such as `myswat index summarize`
- optional operator-triggered backfill

The raw chunk remains the ground truth. AI summaries are a semantic layer on top
of local source retrieval.

### TiDB Upsert Contract

TiDB summary persistence must be idempotent.

Logical uniqueness key:

- `project_id`
- `chunk_uid`
- `repo_commit`
- `scope`

For code summaries, `scope = source_summary`.

Contract:

- repeated writes with the same `(project_id, chunk_uid, repo_commit, scope)`
  update or replace the same semantic record
- retries must not create duplicate summaries
- a new repo commit creates a new summary version for the same `chunk_uid`
- summary metadata must include `file_uid`, `chunk_uid`, `path`, line range,
  symbol identity when available, and the repo commit used for summarization

This is a logical contract regardless of whether TiDB uses dedicated columns,
metadata JSON, or a later schema refinement. Phase A does not depend on any
TiDB schema extension.

## Retrieval Flow

### Phase A Heuristic Router

Phase A uses a simple deterministic router, not an open-ended classifier.

Route a query to the local code index when any of these signals are present:

- backticked identifiers or code fragments
- file paths or filename extensions such as `.go`, `.rs`, `.proto`
- identifier-like tokens such as `CamelCase`, `snake_case`, `pkg/type.Func`,
  `::`, or `->`
- code-navigation phrases such as `where is`, `defined in`, `implements`,
  `calls`, `references`, or `which file`

Otherwise, stay on the existing TiDB retrieval path.

The router is intentionally heuristic, transparent, and cheap to run in Phase A.

### Composite Retriever

When the heuristic router selects the code path, MySwat runs a composite
retriever:

1. local symbol/path/FTS retrieval against the SQLite index
2. the current TiDB-first retriever already used for project memory and fed
   knowledge
3. merge results with local-index hits first, then TiDB results

Merge rules:

- local exact hits are ranked ahead of TiDB semantic hits
- de-duplicate by `chunk_uid` when a TiDB summary points back to the same chunk
- preserve TiDB-only results for document context and prior project memory
- if the local index is missing or stale, fall back to the current TiDB path and
  surface that limitation in the answer flow

This keeps the existing retriever architecture intact while inserting the local
exact index ahead of it for code-heavy questions.

## Answer Construction

The answering agent should receive:

- local exact hits with file path and line range
- relevant TiDB knowledge from the existing retrieval pipeline
- later, linked code summaries from TiDB keyed by `chunk_uid`
- source citations including:
  - file path
  - line range
  - symbol
  - repo commit when available
  - `chunk_uid` for internal tracing

This makes answers:

- more exact than summary-only retrieval
- more grounded than TiDB-only semantic retrieval
- compatible with the existing project-memory architecture

## Manual Indexing UX

Phase A indexing is manual-only.

Commands added over the phased rollout are operator-invoked commands such as:

```bash
myswat index build -p tidb-expert
myswat index refresh -p tidb-expert
myswat index stats -p tidb-expert
myswat index doctor -p tidb-expert
myswat index summarize -p tidb-expert
```

Operational rules:

- MySwat does not auto-build or auto-refresh the index during query handling in
  Phase A.
- If the index is missing, MySwat reports that and falls back to the existing
  TiDB retrieval path.
- `index summarize` is a Phase B command, not part of the Phase A implementation
  surface.

### `index build`

- full rebuild
- parse all configured files
- recreate SQLite FTS if needed
- local-only in Phase A

### `index refresh`

- only changed files
- compare file hash + current commit
- preserve unchanged chunks
- local-only in Phase A

### `index stats`

Show:

- indexed files
- indexed symbols
- indexed chunks
- stale chunks vs HEAD
- local index size on disk
- whether required local tools are available

### `index doctor`

Check:

- missing parser dependencies
- invalid manifest
- stale chunk references
- missing repo path or unusable config

### `index summarize` (Phase B)

- generate missing summaries in batch or on demand
- write semantic entries to TiDB using the idempotent upsert contract
- does not rebuild the local index unless explicitly paired with refresh/build

## Continuous Refresh and Staleness

Retrieval should compare indexed commit metadata with current repo HEAD and
attach staleness signals to each hit:

- `fresh`: chunk/file commit equals current HEAD
- `stale`: commit differs from HEAD
- `unknown`: non-git project or commit unavailable

The answering layer should surface this explicitly in citations, especially
when stale chunks are used.

## Space / Cost Target

This design remains much lighter than JetBrains indexing because it does not
store full IDE semantic state.

Target:

- local index size significantly below full IDE cache size
- TiDB stores document knowledge and later code summaries, not a whole IDE graph
- raw source retained locally; SQLite stores offsets and optional hot-cache text
- TiDB does not store full source bodies by default

## Why This Fits MySwat

This keeps MySwat aligned with its strengths:

- AI-first workflow and memory
- project-scoped persistent knowledge
- pragmatic local tooling

It does **not** force MySwat to become a full IDE, but it gives it enough
indexing capability to become a real project expert.

## Phased Rollout

### Phase A: Local Exact Index

- add local `.myswat/index/`
- build SQLite `files/symbols/chunks/fts_chunks` schema
- use deterministic `file_uid` / `chunk_uid` as the cross-store identity
- skip `relations` and other LSP-derived graph tables
- implement `tree-sitter` chunking + `ctags` enrichment
- implement `myswat index build`, `myswat index refresh`, `myswat index stats`,
  and `myswat index doctor`
- keep indexing local-only with no TiDB schema changes and no TiDB code-summary
  writes
- use the heuristic router and composite retriever to prepend local hits ahead
  of the existing TiDB retriever

### Phase B: Hybrid Semantic Retrieval

- add lazy or explicit code summarization
- persist TiDB summaries with the idempotent key `(project_id, chunk_uid, repo_commit, scope)`
- teach answering prompts to combine local exact hits + TiDB semantic summaries

### Phase C: Incremental Robustness

- improve changed-file detection and refresh ergonomics
- strengthen retrieval-time staleness reporting vs repo HEAD
- improve operational visibility for stale or missing local indexes

### Phase D: Semantic Enrichment

- optional `gopls` / `rust-analyzer`
- optional relation edges from LSP output only

## Final Summary

The finalized `tidb-expert` / `tikv-expert` design is:

- **Local project files** for exact indexing of source plus selected config / proto files
- **TiDB** for existing project memory and fed document knowledge, plus later semantic code summaries
- **AI agents** for code understanding, answer synthesis, and later summary generation

The first shipped phase stays deliberately small and local. It improves code
retrieval immediately without changing TiDB schema, without adding model-based
query classification, and without moving docs/RFC/issues off the existing
`feed` path.
