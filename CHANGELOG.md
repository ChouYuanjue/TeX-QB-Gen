## [0.4.1] - 2025-10-07
### Fixed
- Addressed issues with Chinese PDF segmentation and garbled text handling.

### Changed
- Updated TO-DO list to reflect completed tasks and new priorities.

## [0.4.0] - 2025-10-07
### Added
- Asynchronous pipeline entry point with configurable concurrency (`TEXBANK_CONCURRENCY`) to parallelise image/PDF/URL processing.
- Thread-safe, per-question TeX emission that regenerates `master.tex` immediately after each problem is rendered.
- Test coverage for the async workflow to ensure incremental rendering remains stable under concurrent workloads.

### Documentation
- README/README_zh-CN expanded with bilingual cross-links, banner, and usage notes aligned with the new processing flow.


## [0.3.0] - 2025-10-06
### Added
- Verbose/debug logging switches and granular progress logs for PDF parsing, multimodal extraction, fallbacks, and rendering.
- Hint-aware multimodal extraction for text-based PDFs, including batched page image requests and smarter aggregation.

### Changed
- Text-PDF pipeline prioritises multimodal extraction, with OCR/text fallbacks triggered automatically on failure.
- Expanded keyword dictionary and capped page-group attempts to reduce redundant LLM calls.

### Documentation
- README updated with processing pipeline walkthrough, diagnostics guidance, and CLI logging options.

## [0.2.0] - 2025-10-06
### Added
- Configurable output language control (`--language`, `TEXBANK_DEFAULT_LANGUAGE`) with automatic translation of exercise/solution fields and metadata tagging.
- Language-aware solution generation pipeline that preserves LaTeX syntax while translating content when needed.

### Changed
- Pipeline helpers now normalise multilingual content and ensure consistent metadata for translated items.

### Documentation
- README + `.env.example` refreshed to highlight language defaults and configuration instructions.

## [0.1.1] - 2025-10-05
### Added
- Unified multimodal/text pipeline covering images, PDFs, and URLs with caching-aware OpenRouter client integration.
- StackExchange ingestion flow with keyword search, answer classification, and structured `ProblemItem` generation.
- Master TeX rendering utility that composes per-question files and a consolidated `master.tex`.

### Changed
- Refactored configuration module to centralise model selection, cache paths, and API credentials.
- Hardened PDF utilities with scan detection, keyword span extraction, and OCR fallbacks.

### Testing
- Added pytest suite with stubbed OpenRouter client to validate pipeline behaviour without external calls.

## [0.1.0] - 2025-10-05
- Initial scaffold.
