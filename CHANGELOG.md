## [0.5.0] - 2025-10-10
### Added
- LLM-assisted segmentation for chapter-end exercise blocks, allowing recovery of problems even without explicit numbering.
- Unit coverage for the new segmentation flow to ensure unclear numbering still yields structured problems and metadata.
- Dedicated paired-sequence pipeline for searchable PDFs with question/answer page pairs, plus a `--paired-sequence` CLI flag to drive it.
- Paired-sequence extraction now supports multi-level numbering (for example `{chapter}.{section}.{n}`) with range configuration, covering chapter/section patterns such as 1.x.x and 2.x.x.
- CLI flag `--omit-answer-field` to request only `exercise` and `solution` from multimodal extraction when answer classification is unreliable.

### Changed
- Enhanced OCR preprocessing with contrast/denoise/sharpen steps before invoking Tesseract or PaddleOCR, improving recognition on blurry scans.
- Introduced configurable segmentation chunk/overlap limits to fine-tune large-text handling in `PipelineConfig`.
- Adjusted window fallback logic so detected spans (including LLM-derived ones) are preserved even when legibility is low.
- In paired mode, if only a single page matches, the pipeline now appends the next page before multimodal extraction to capture adjacent answers.
- Improved CLI input handling by trimming leading and trailing whitespace so PDF paths are not misidentified as unsupported.
- Added `--paired-latest-only` flag to keep only the final occurrence and its next page when multiple hits exist, and to skip generation whenever the target `.tex` already exists.
- Multimodal extraction now retries once on invalid JSON responses before falling back to local OCR.

## [0.4.6] - 2025-10-08
### Fixed
- Added math-aware cleaning for LLM responses to preserve math environments and avoid over-escaping in formulas.
- Improved Markdown-to-LaTeX conversion (better table -> tabular conversion) and robust JSON wrapper extraction from LLM outputs.
- Fixed scanned-PDF fallback path (indentation/variable bugs) and ensured page labeling is correct for scanned pages.
- Improved post-processing pipeline to balance math delimiters and restore math alignment characters inside math mode.

### Changed
- Switched to a normalization pipeline for LLM solution text that first strips structured wrappers, converts markdown, then sanitizes LaTeX.
- Added helper utilities for math segmentation and sanitization in `pipeline.py`.
- Updated `render_master` to use natural sorting so numbered questions appear in intuitive order.

## [0.4.5] - 2025-10-08
### Fixed
- Improved LLM solution generation to return pure LaTeX text instead of JSON format.
- Enhanced prompt engineering to explicitly require LaTeX syntax and avoid Markdown formatting.
- Added comprehensive text cleaning functions to remove JSON wrappers, fix markdown tables, and escape special characters.
- Fixed common LaTeX issues like undefined control sequences (\n), unbalanced math delimiters, and special character escaping.
- Ensured LLM-generated solutions are complete and properly formatted with balanced $ and \[ \] delimiters.

### Changed
- Modified `_generate_solution` method to include stricter instructions for LaTeX-only output.
- Added `_clean_json_from_text` and `_fix_latex_issues` functions for post-processing LLM responses.
- Enhanced `_clean_markdown_from_text` to convert markdown tables to LaTeX tabular environments.

## [0.4.4] - 2025-10-08
### Added
- Implemented hierarchical master.tex generation with automatic directory structure mapping.
- Added folder-based chapter/section organization in generated TeX documents.
- Created script-based master.tex generation that extracts content from individual .tex files and organizes them into a cohesive document.
- Added table of contents generation for the master document.
- Added \usepackage{ctex} to support Chinese text rendering in both individual and master TeX files.
- Added directory traversal support for input processing - directories are recursively scanned for supported file types (images and PDFs).
- Added --generate-master option to create master.tex from existing .tex files in a directory.

### Changed
- Modified `render_master` function to read and clean individual TeX files instead of using `\input`.
- Updated file rendering to preserve folder hierarchy in output directory.
- Enhanced document structure to use `book` class with proper chapter/section/subsection levels.
- Updated CLI to support directory inputs and master document generation.

## [0.4.3] - 2025-10-07
### Added
- Enhanced error handling for LLM responses with automatic JSON parsing fixes.
- Markdown-to-LaTeX conversion for LLM-generated content to ensure proper TeX output.
- Improved prompts to explicitly require LaTeX syntax instead of Markdown.

### Fixed
- Better handling of malformed JSON responses from LLMs.
- Automatic cleanup of markdown formatting in extracted mathematical content.

## [0.4.2] - 2025-10-07
### Added
- Integrated PaddleOCR as an alternative OCR engine for better Chinese text recognition.
- Added `TEXBANK_OCR_ENGINE` configuration to choose between 'tesseract' and 'paddle' OCR engines.
- Updated Tesseract to include Chinese language support by default.

### Changed
- Enhanced OCR module to support multiple engines with configurable selection.

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
