from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

from bs4 import BeautifulSoup  # type: ignore[import]

from bs4 import BeautifulSoup  # type: ignore[import]

from .config import PipelineConfig, get_pipeline_config
from .llm_client import OpenRouterClient
from .models import ProblemItem
from .ocr import image_to_text
from .pdf_utils import (TextSpan, detect_scan_ratio, extract_keyword_spans, extract_text_pages,
                        iter_pdf_images, is_text_pdf)
from .stackexchange import fetch_full_questions, fetch_questions_by_keyword
from .texgen import render_master, render_single_tex
from .web_utils import fetch_url


logger = logging.getLogger(__name__)


_MATH_BLOCK_PATTERN = re.compile(
    r'(\$\$.*?\$\$|\$[^$]*\$|\\\[.*?\\\]|\\\(.*?\\\)|'
    r'\\begin\{(?P<env>(?:align\*?|equation\*?|gather\*?|multline\*?|cases|array|pmatrix|bmatrix|vmatrix|Vmatrix|smallmatrix|matrix))\}.*?\\end\{(?P=env)\})',
    re.DOTALL,
)

_QUESTION_HEADING_PATTERN = re.compile(
    r'^\s*(?:第\s*[一二三四五六七八九十百千]+\s*题|[（(]?[一二三四五六七八九十]{1,3}[)）．．、:]|[（(]?\d{1,3}[)）．．、:]|(?:Problem|Exercise|Question)\s*\d+)',
    re.IGNORECASE,
)

_SOLUTION_MARKER_PATTERN = re.compile(r'(?:^|\n)\s*(?:解|证|Solution|Proof)\b')

def _split_math_segments(text: str) -> List[Tuple[str, str]]:
    segments: List[Tuple[str, str]] = []
    last = 0
    for match in _MATH_BLOCK_PATTERN.finditer(text):
        start, end = match.span()
        if start > last:
            segments.append(('text', text[last:start]))
        segments.append(('math', match.group(0)))
        last = end
    if last < len(text):
        segments.append(('text', text[last:]))
    return segments


def _normalize_math_segment(segment: str) -> str:
    segment = segment.replace(r'\&', '&')
    segment = segment.replace(r'\%', '%')
    segment = segment.replace(r'\n', ' ')
    return segment


def _escape_text_segment(segment: str) -> str:
    def repl(match: re.Match[str]) -> str:
        mapping = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
        }
        return mapping.get(match.group(0), match.group(0))

    segment = re.sub(r'(?<!\\)[&%$#_]', repl, segment)
    segment = segment.replace('~', r'\textasciitilde{}')
    segment = re.sub(r'(?<!\\)\^', r'\^{}', segment)
    return segment


def _convert_markdown_tables(segment: str) -> str:
    if '|' not in segment:
        return segment
    lines = segment.splitlines()
    output: List[str] = []
    i = 0
    while i < len(lines):
        if _is_markdown_table_header(lines, i):
            j = i
            table_lines: List[str] = []
            while j < len(lines) and lines[j].strip() and '|' in lines[j]:
                table_lines.append(lines[j])
                j += 1
            output.append(_markdown_table_to_tabular(table_lines))
            i = j
            continue
        output.append(lines[i])
        i += 1
    return '\n'.join(output)


def _is_markdown_table_header(lines: List[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    header = lines[index]
    separator = lines[index + 1]
    if '|' not in header or '|' not in separator:
        return False
    if not header.strip() or not separator.strip():
        return False
    if not re.match(r'^\s*\|?\s*[:\-\s\|]+\|?\s*$', separator):
        return False
    return True


def _markdown_table_to_tabular(table_lines: List[str]) -> str:
    rows: List[List[str]] = []
    for line in table_lines:
        stripped = line.strip()
        if not stripped or '|' not in stripped:
            continue
        cells = [cell.strip() for cell in stripped.strip('|').split('|')]
        rows.append(cells)
    if not rows:
        return '\\textit{表格内容解析失败}'
    max_cols = max(len(row) for row in rows)
    for row in rows:
        if len(row) < max_cols:
            row.extend([''] * (max_cols - len(row)))
    align = 'c' * max_cols
    latex_lines = ['\\begin{tabular}{' + align + '}', '\\hline']
    for idx, row in enumerate(rows):
        latex_row = ' & '.join(row) + ' \\\\'
        latex_lines.append(latex_row)
        if idx == 0:
            latex_lines.append('\\hline')
    latex_lines.append('\\hline')
    latex_lines.append('\\end{tabular}')
    return '\n'.join(latex_lines)


def _balance_math_delimiters(text: str) -> str:
    if text.count('$') % 2 != 0:
        text += '$'
    open_brackets = text.count(r'\[')
    close_brackets = text.count(r'\]')
    if open_brackets > close_brackets:
        text += r'\]'
    return text


def _extract_text_from_json(obj: Any) -> Optional[str]:
    if isinstance(obj, dict):
        for key in ('solution', 'answer', 'exercise', 'content', 'text'):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value
        for value in obj.values():
            nested = _extract_text_from_json(value)
            if nested:
                return nested
    elif isinstance(obj, list):
        for item in obj:
            nested = _extract_text_from_json(item)
            if nested:
                return nested
    return None


def _strip_structured_wrappers(text: str) -> str:
    candidate = text.strip()
    if not candidate:
        return text
    decoder = json.JSONDecoder()
    index = 0
    while index < len(candidate):
        try:
            obj, end = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            next_object = candidate.find('{', index + 1)
            next_array = candidate.find('[', index + 1)
            options = [pos for pos in (next_object, next_array) if pos != -1]
            if not options:
                break
            index = min(options)
            continue
        extracted = _extract_text_from_json(obj)
        if extracted:
            return extracted
        index += end
    return text

class Pipeline:
    IMAGE_SYSTEM_WITH_ANSWER = (
        "你是一个数学题目整理助手，所有输出都必须是严格的JSON对象，仅包含exercise, answer, solution三个字段。"
        "所有数学内容必须使用LaTeX语法，不要使用Markdown格式。"
        "如果内容包含数学公式，请确保使用正确的LaTeX命令如$...$或\\[...\\]。"
    )

    IMAGE_SYSTEM_WITHOUT_ANSWER = (
        "你是一个数学题目整理助手，所有输出都必须是严格的JSON对象，仅包含exercise, solution两个字段。"
        "所有数学内容必须使用LaTeX语法，不要使用Markdown格式。"
        "如果内容包含数学公式，请确保使用正确的LaTeX命令如$...$或\\[...\\]。"
    )

    def __init__(self, config: Optional[PipelineConfig] = None, *, enable_llm_solution: bool = True, language: Optional[str] = None, paired_sequence: Optional[str] = None, paired_latest_only: bool = False, include_answer_field: bool = True):
        self.config = config or get_pipeline_config()
        self.client = OpenRouterClient(self.config.openrouter)
        self.enable_llm_solution = enable_llm_solution
        lang = (language or self.config.default_language or 'auto').lower()
        if lang not in {'auto', 'zh', 'en'}:
            lang = 'auto'
        self.language = lang
        self.concurrency = max(1, self.config.concurrency_limit)
        self._out_dir: Optional[Path] = None
        self._rendered_rel_paths: List[str] = []
        self._render_lock = threading.Lock()
        self._paired_config = self._parse_paired_sequence_config(paired_sequence, latest_only=paired_latest_only)
        self.include_answer_field = include_answer_field
        self._image_system_prompt = (
            self.IMAGE_SYSTEM_WITH_ANSWER if self.include_answer_field else self.IMAGE_SYSTEM_WITHOUT_ANSWER
        )
        self._image_retry_attempts = 2

    # region image ---------------------------------------------------------------------------------
    def process_image(self, image_path: str, ask_llm_for_solution: bool = True) -> ProblemItem:
        logger.info("Processing image: %s", image_path)
        item = self._extract_problem_from_images([image_path], identifier=Path(image_path).stem)
        if self.enable_llm_solution and ask_llm_for_solution and self._needs_llm_solution(item):
            item.llm_solution = self._generate_solution(item.exercise)
        item.metadata['source'] = image_path
        item.metadata['type'] = 'image'
        self._ensure_language(item)
        return item

    # endregion ------------------------------------------------------------------------------------

    # region pdf -----------------------------------------------------------------------------------
    def process_pdf(self, pdf_path: str, keywords: Optional[List[str]] = None, *, on_item: Optional[Callable[[ProblemItem], None]] = None) -> List[ProblemItem]:
        logger.info("Processing PDF: %s", pdf_path)
        if self._paired_config:
            return self._process_paired_sequence_pdf(pdf_path, on_item=on_item)
        keywords = keywords or self._default_pdf_keywords()
        if is_text_pdf(pdf_path):
            logger.info("Detected text-mode PDF, extracting spans (max preview %d pages)", self.config.max_pdf_preview_pages)
            page_text = extract_text_pages(pdf_path, limit=self.config.max_pdf_preview_pages)
            legibility = self._compute_page_legibility(page_text)
            if legibility:
                avg_legibility = sum(legibility.values()) / max(1, len(legibility))
            else:
                avg_legibility = 0.0
            spans = extract_keyword_spans(page_text, keywords)
            logger.info("Found %d candidate spans in %s", len(spans), pdf_path)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Keyword list: %s", ', '.join(keywords))
            if self._spans_need_llm_segmentation(spans, page_text):
                logger.info("Keyword segmentation insufficient for %s; invoking LLM assisted segmentation", pdf_path)
                llm_spans = self._segment_exercises_with_llm(pdf_path, page_text)
                if llm_spans:
                    spans = llm_spans
                    logger.info("LLM segmentation produced %d spans for %s", len(spans), pdf_path)
            force_windows = False
            if avg_legibility < self.config.text_legibility_threshold:
                logger.info(
                    "Average legibility %.2f below threshold %.2f; falling back to windowed image extraction",
                    avg_legibility,
                    self.config.text_legibility_threshold,
                )
                force_windows = True
            if not spans:
                logger.info("No keyword spans detected; will approximate using page windows")
                force_windows = True
            if force_windows and page_text and not spans:
                page_numbers = sorted(page_text.keys())
                spans = self._build_window_spans(page_numbers)
            if not spans and page_text:
                page_numbers = sorted(page_text.keys())
                combined = '\n\n'.join(page_text[p] for p in page_numbers)
                spans = [TextSpan(start_page=page_numbers[0], end_page=page_numbers[-1], body=combined)]
            items = self._process_text_pdf_via_images(pdf_path, spans, on_item=on_item)
            if not items:
                logger.info("Falling back to text parsing for %s", pdf_path)
                items = self._process_text_pdf_via_text(pdf_path, spans, on_item=on_item)
            return items

        # scanning fallback
        ratio = detect_scan_ratio(pdf_path)
        candidate_pages = [page for page, value in ratio.items() if value >= self.config.scan_detection_ratio]
        if not candidate_pages:
            candidate_pages = list(sorted(ratio.keys()))
        logger.info("Processing scanned PDF %s; candidate pages: %s", pdf_path, candidate_pages)
        images = list(iter_pdf_images(pdf_path, pages=candidate_pages))
        items: List[ProblemItem] = []
        for idx, image_path in enumerate(images):
            item = self.process_image(image_path)
            item.identifier = f"{Path(pdf_path).stem}_scan_{idx+1}"
            page_label = candidate_pages[idx] if idx < len(candidate_pages) else '?'
            item.metadata.update({'source_pdf': pdf_path, 'page': str(page_label)})
            self._ensure_language(item)
            items.append(item)
            if on_item:
                on_item(item)
        return items

    # endregion ------------------------------------------------------------------------------------

    def _render_problem(self, item: ProblemItem) -> str:
        with self._render_lock:
            # Create folder structure based on metadata or something
            # For now, use source or type
            source = item.metadata.get('source', 'unknown')
            source_path = Path(source)
            if source_path.is_absolute():
                rel_dir = source_path.parent.relative_to(Path.cwd()) if source_path.parent != Path.cwd() else Path('.')
            else:
                rel_dir = Path('.')
            out_dir = self._out_dir / rel_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{item.identifier}.tex"
            render_single_tex(item, str(out_path))
            rel_path = out_path.relative_to(self._out_dir)
            return str(rel_path)

    def process_inputs(self, inputs: List[str], out_dir: str, keyword: Optional[str] = None, max_items: Optional[int] = None, site: Optional[str] = None) -> List[str]:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._rendered_rel_paths = []

        def render_sink(item: ProblemItem) -> None:
            rel_path = self._render_problem(item)
            self._rendered_rel_paths.append(rel_path)

        for inp in inputs:
            if inp.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                logger.info("Processing image: %s", inp)
                item = self.process_image(inp)
                render_sink(item)
            elif inp.lower().endswith('.pdf'):
                logger.info("Processing PDF: %s", inp)
                items = self.process_pdf(inp, on_item=render_sink)
                if not items:
                    logger.warning("No problems extracted from PDF %s", inp)
            elif inp.startswith('http://') or inp.startswith('https://'):
                logger.info("Processing URL: %s", inp)
                items = self.process_url(inp, keyword=keyword, max_items=max_items, site=site)
                for item in items:
                    render_sink(item)
            else:
                raise ValueError(f'Unsupported input type: {inp}')

        # Generate master.tex
        master_path = self._out_dir / 'master.tex'
        render_master(str(self._out_dir), str(master_path))
        generated_paths = [str(master_path)] + [str(self._out_dir / rel) for rel in self._rendered_rel_paths]
        logger.info("Generated %d TeX files and master.tex in %s", len(self._rendered_rel_paths), out_dir)
        return generated_paths

    # region url -----------------------------------------------------------------------------------
    def process_url(self, url: str, *, keyword: Optional[str] = None, max_items: Optional[int] = None, site: Optional[str] = None) -> List[ProblemItem]:
        parsed = urlparse(url)
        host = parsed.hostname or ''
        if 'stackexchange' in host or 'mathoverflow.net' in host or host.endswith('math.stackexchange.com'):
            return self._process_stackexchange(url, keyword=keyword, max_items=max_items, site=site)
        return self._process_generic_url(url)

    def _process_stackexchange(self, url: str, *, keyword: Optional[str], max_items: Optional[int], site: Optional[str]) -> List[ProblemItem]:
        cfg = self.config.stackexchange
        site_slug = site or cfg.default_site
        max_items = max_items or cfg.page_size
        key = cfg.api_key

        parsed = urlparse(url)
        ids: List[int] = []
        path_parts = [part for part in parsed.path.split('/') if part]
        if path_parts and path_parts[-1].isdigit():
            ids = [int(path_parts[-1])]
        elif keyword:
            hits = fetch_questions_by_keyword(keyword, site=site_slug, pagesize=max_items, key=key)
            ids = [hit['question_id'] for hit in hits][:max_items]

        questions = fetch_full_questions(ids, site=site_slug, key=key)
        items: List[ProblemItem] = []
        for q in questions:
            identifier = f"se_{q.get('id')}"
            exercise = f"{q.get('title', '')}\n\n{self._strip_html(q.get('body', ''))}"
            answers = q.get('answers', [])
            detailed = [self._strip_html(ans) for ans in answers if len(self._strip_html(ans)) > self.config.minimal_answer_tokens]
            short = [self._strip_html(ans) for ans in answers if len(self._strip_html(ans)) <= self.config.minimal_answer_tokens]
            item = ProblemItem(identifier=identifier, exercise=exercise)
            if detailed:
                item.solution = '\n\n'.join(detailed)
            if short and not item.solution:
                item.answer = '\n\n'.join(short)
            if self.enable_llm_solution and self._needs_llm_solution(item):
                item.llm_solution = self._generate_solution(item.exercise)
            item.metadata.update({'source': url, 'type': 'stackexchange', 'site': site_slug})
            self._ensure_language(item)
            items.append(item)
        return items

    def _process_generic_url(self, url: str) -> List[ProblemItem]:
        html = fetch_url(url)
        soup = BeautifulSoup(html, 'lxml')
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        text = soup.get_text('\n', strip=True)
        system_prompt = (
            "你是一名数学题目整理助手，请根据用户提供的网页正文提取题目。"
            "输出必须是JSON数组，每项包含title, body, answers(列表，可空)。"
            "所有数学内容必须使用LaTeX语法，不要使用Markdown格式。"
            "公式使用$...$或\\[...\\]。"
        )
        user_prompt = (
            "请从以下网页正文中提取所有数学题目及可能的解答。"
            "保持原有数学符号与LaTeX表达，可根据上下文补全标题。"
            "所有数学内容必须使用LaTeX语法，不要使用Markdown格式。"
            f"\n\n{text}"
        )
        schema = '[{"title": "string", "body": "string", "answers": ["string"]}]'
        try:
            data = self.client.generate_structured(user_prompt, self.config.models.text_reasoning, schema_hint=schema, system=system_prompt)
        except ValueError:
            data = []
        items: List[ProblemItem] = []
        if isinstance(data, list):
            for idx, entry in enumerate(data):
                exercise = f"{entry.get('title', '').strip()}\n\n{entry.get('body', '').strip()}".strip()
                answers = entry.get('answers') or []
                detailed = [ans for ans in answers if len(ans) > self.config.minimal_answer_tokens]
                item = ProblemItem(identifier=f"url_{idx+1}", exercise=exercise)
                if detailed:
                    item.solution = '\n\n'.join(detailed)
                else:
                    item.answer = '\n\n'.join(answers)
                if self.enable_llm_solution and self._needs_llm_solution(item):
                    item.llm_solution = self._generate_solution(item.exercise)
                item.metadata.update({'source': url, 'type': 'url'})
                self._ensure_language(item)
                items.append(item)
        if not items:
            fallback = ProblemItem(identifier='url_1', exercise=text)
            if self.enable_llm_solution:
                fallback.llm_solution = self._generate_solution(fallback.exercise)
            fallback.metadata.update({'source': url, 'type': 'url'})
            self._ensure_language(fallback)
            items.append(fallback)
        return items

    # endregion ------------------------------------------------------------------------------------

    # region rendering -----------------------------------------------------------------------------
    def _reset_render_state(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._rendered_rel_paths = []

    def _write_master(self) -> str:
        if self._out_dir is None:
            raise ValueError('Output directory has not been initialised')
        master_path = self._out_dir / 'master.tex'
        render_master(self._rendered_rel_paths, str(master_path))
        return str(master_path)

    def _render_problem(self, item: ProblemItem) -> str:
        if self._out_dir is None:
            raise ValueError('Output directory has not been initialised')
        # Create folder structure based on source
        source = item.metadata.get('source', '')
        if source:
            source_path = Path(source)
            parent = source_path.parent
            if source_path.is_absolute():
                try:
                    rel_parts = parent.relative_to(Path.cwd()).parts
                    rel_dir_parts = rel_parts[1:] if len(rel_parts) > 1 else []
                except ValueError:
                    rel_dir_parts = [parent.name]
            else:
                rel_parts = parent.parts
                rel_dir_parts = rel_parts[1:] if len(rel_parts) > 1 else []
            out_dir = self._out_dir
            for part in rel_dir_parts:
                out_dir = out_dir / part
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = self._out_dir
        path = out_dir / f"{item.identifier}.tex"
        with self._render_lock:
            render_single_tex(item, str(path))
            rel_path = path.relative_to(self._out_dir)
            rel_name = str(rel_path)
            if rel_name not in self._rendered_rel_paths:
                self._rendered_rel_paths.append(rel_name)
        return str(path)

    def render_items(self, items: Iterable[ProblemItem], out_dir: str) -> List[str]:
        out_dir_path = Path(out_dir)
        self._reset_render_state(out_dir_path)
        paths: List[str] = []
        for item in items:
            paths.append(self._render_problem(item))
        master_path = self._write_master()
        paths.append(master_path)
        return paths

    def process_inputs(self, inputs: List[str], out_dir: str, keyword: Optional[str] = None, max_items: Optional[int] = None, site: Optional[str] = None) -> List[str]:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._rendered_rel_paths = []
        
        generated_paths = []
        for inp in inputs:
            if inp.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                item = self.process_image(inp)
                generated_paths.append(self._render_problem(item))
            elif inp.lower().endswith('.pdf'):
                items = self.process_pdf(inp, keywords=None)  # or default
                for item in items:
                    generated_paths.append(self._render_problem(item))
            else:
                # Assume URL or other
                items = self.process_url(inp, keyword=keyword, max_items=max_items, site=site)
                for item in items:
                    generated_paths.append(self._render_problem(item))
        
        master_path = self._write_master()
        generated_paths.append(master_path)
        return generated_paths

    # region helpers -------------------------------------------------------------------------------
    def _write_master(self) -> str:
        master_path = self._out_dir / 'master.tex'
        render_master(str(self._out_dir), str(master_path))
        return str(master_path)

    def _default_pdf_keywords(self) -> List[str]:
        return [
            'Exercise', 'Exercises', 'Problem', 'Problems', 'Question', 'Questions', 'Example', 'Examples',
            'Proof', 'Lemma', 'Theorem', 'Corollary', 'Practice', 'Homework', 'Review', 'Quiz', 'Test',
            '题目', '问题', '练习', '习题', '例题', '例', '例子', '证明', '解答', '答案', '思考', '讨论'
        ]

    def _extract_problem_from_images(self, image_paths: List[str], *, identifier: Optional[str] = None, hint: Optional[str] = None) -> ProblemItem:
        if not image_paths:
            raise ValueError('No images provided for extraction')
        logger.info(
            "Submitting %d image(s) for multimodal extraction (identifier=%s)",
            len(image_paths),
            identifier,
        )
        if hint and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Extraction hint: %s", hint)
        structured: Optional[Dict[str, Any]] = None
        last_error: Optional[Exception] = None
        prompt = self._build_image_prompt(hint)
        for attempt in range(1, self._image_retry_attempts + 1):
            try:
                structured = self.client.generate_from_image(
                    image_paths,
                    model=self.config.models.ocr_multimodal_image,
                    prompt=prompt,
                    system=self._image_system_prompt,
                )
                break
            except ValueError as exc:
                message = str(exc)
                if 'not valid JSON' in message and attempt < self._image_retry_attempts:
                    logger.warning(
                        "Multimodal extraction attempt %d/%d for %s returned invalid JSON; retrying.",
                        attempt,
                        self._image_retry_attempts,
                        identifier,
                    )
                    last_error = exc
                    continue
                last_error = exc
                break
            except Exception as exc:
                last_error = exc
                break

        if structured is None:
            logger.warning(
                "Multimodal extraction failed for %s; falling back to OCR. Reason: %s",
                identifier,
                last_error,
            )
            fallback_texts = []
            for path in image_paths:
                try:
                    fallback_texts.append(image_to_text(path, self.config.ocr_engine))
                except Exception as ocr_exc:
                    logger.debug("OCR fallback failed for %s: %s", path, ocr_exc, exc_info=True)
                    continue
            structured = {'exercise': '\n\n'.join(t for t in fallback_texts if t)}
        normalized = self._normalize_parts(structured)
        resolved_identifier = identifier or Path(image_paths[0]).stem
        logger.info(
            "Completed extraction for %s (exercise length: %d characters)",
            resolved_identifier,
            len(normalized.get('exercise') or ''),
        )
        return ProblemItem(
            identifier=resolved_identifier,
            exercise=normalized.get('exercise') or '',
            answer=normalized.get('answer'),
            solution=normalized.get('solution'),
        )

    def _process_text_pdf_via_images(self, pdf_path: str, spans: List[TextSpan], *, on_item: Optional[Callable[[ProblemItem], None]] = None) -> List[ProblemItem]:
        if not spans:
            return []

        def worker(span_index: int, span: TextSpan) -> Tuple[int, ProblemItem]:
            try:
                item = self._extract_span_via_images(pdf_path, span, span_index)
            except Exception as exc:  # pragma: no cover - logging path
                logger.warning(
                    "Span #%d in %s fell back to text extraction due to error: %s",
                    span_index + 1,
                    pdf_path,
                    exc,
                )
                item = self._extract_span_via_text(pdf_path, span, span_index)
            if on_item:
                on_item(item)
            return span_index, item

        if self.concurrency > 1 and len(spans) > 1:
            logger.debug("Processing %d spans concurrently (max_workers=%d)", len(spans), self.concurrency)
            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = [executor.submit(worker, idx, span) for idx, span in enumerate(spans)]
                results = [future.result() for future in futures]
        else:
            results = [worker(idx, span) for idx, span in enumerate(spans)]

        results.sort(key=lambda pair: pair[0])
        return [item for _, item in results]

    def _process_text_pdf_via_text(self, pdf_path: str, spans: List[TextSpan], *, on_item: Optional[Callable[[ProblemItem], None]] = None) -> List[ProblemItem]:
        items: List[ProblemItem] = []
        for idx, span in enumerate(spans):
            item = self._extract_span_via_text(pdf_path, span, idx)
            items.append(item)
            if on_item:
                on_item(item)
        return items

    def _extract_span_via_images(self, pdf_path: str, span: TextSpan, idx: int) -> ProblemItem:
        page_groups = self._select_page_groups(span)
        if not page_groups:
            raise ValueError('No page groups available for span')
        identifier = f"{Path(pdf_path).stem}_q{idx+1}"
        aggregate: Optional[ProblemItem] = None
        applied_groups: List[List[int]] = []
        hint = self._make_question_hint(span.body)
        logger.info(
            "Processing span #%d for %s (pages %s-%s) with %d candidate group(s)",
            idx + 1,
            pdf_path,
            span.start_page,
            span.end_page,
            len(page_groups),
        )
        if hint and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Span #%d hint: %s", idx + 1, hint)
        for group in page_groups:
            logger.info("Span #%d trying page group: %s", idx + 1, group)
            image_paths = self._render_pdf_page_group(pdf_path, group)
            if not image_paths:
                logger.warning("Failed to render pages %s for %s; skipping group", group, pdf_path)
                continue
            partial = self._extract_problem_from_images(image_paths, identifier=identifier, hint=hint)
            if not partial.exercise and not partial.answer and not partial.solution:
                logger.info("Group %s produced empty result, continuing", group)
                continue
            if aggregate is None:
                aggregate = partial
            else:
                self._merge_problem_items(aggregate, partial)
            applied_groups.append(group)
            if aggregate.exercise and (aggregate.solution or aggregate.answer):
                logger.info("Span #%d satisfied with current aggregation; stopping group attempts", idx + 1)
                break
        if aggregate is None or not aggregate.exercise.strip():
            logger.warning("Span #%d failed to extract via multimodal path", idx + 1)
            raise ValueError('Failed to extract problem via multimodal pipeline')
        if self.enable_llm_solution and self._needs_llm_solution(aggregate):
            logger.info("Generating LLM solution for span #%d (%s)", idx + 1, identifier)
            aggregate.llm_solution = self._generate_solution(aggregate.exercise)
        aggregate.metadata.update({
            'source': pdf_path,
            'type': 'pdf-image',
            'page_start': str(span.start_page),
            'page_end': str(span.end_page),
            'extraction_mode': 'multimodal',
        })
        if applied_groups:
            aggregate.metadata['image_groups'] = ';'.join(' '.join(str(p) for p in grp) for grp in applied_groups)
        if hint:
            aggregate.metadata['question_hint'] = hint[:200]
        self._ensure_language(aggregate)
        return aggregate

    def _extract_span_via_text(self, pdf_path: str, span: TextSpan, idx: int) -> ProblemItem:
        identifier = f"{Path(pdf_path).stem}_q{idx+1}"
        logger.info(
            "Text fallback span #%d for %s (pages %s-%s)",
            idx + 1,
            pdf_path,
            span.start_page,
            span.end_page,
        )
        item = ProblemItem(identifier=identifier, exercise=span.body, answer=span.answer, solution=span.solution)

        self._refine_text_fallback(item)

        if self.enable_llm_solution and self._needs_llm_solution(item):
            item.llm_solution = self._generate_solution(item.exercise)
        item.metadata.update({
            'source': pdf_path,
            'type': 'pdf-text',
            'page_start': str(span.start_page),
            'page_end': str(span.end_page),
            'extraction_mode': 'text-fallback',
        })
        origin = getattr(span, 'origin', None)
        if origin:
            item.metadata['segmentation_origin'] = origin
        self._ensure_language(item)
        return item

    def _process_paired_sequence_pdf(self, pdf_path: str, *, on_item: Optional[Callable[[ProblemItem], None]] = None) -> List[ProblemItem]:
        cfg = self._paired_config
        if not cfg:
            return []
        logger.info(
            "Processing paired sequence PDF %s with template %s",
            pdf_path,
            cfg['template'],
        )
        page_text = extract_text_pages(pdf_path, limit=None)
        if not page_text:
            logger.warning("No text content available in %s for paired-sequence extraction", pdf_path)
            return []

        results: List[ProblemItem] = []
        terminal_key = cfg['terminal_key']
        for prefix_context in self._iter_paired_prefix_contexts(cfg):
            if len(results) >= cfg['max_questions']:
                break
            current = cfg['start']
            consecutive_misses = 0
            while len(results) < cfg['max_questions']:
                format_args = dict(prefix_context)
                format_args[terminal_key] = current
                try:
                    label = cfg['template'].format(**format_args)
                except KeyError as exc:
                    logger.warning("缺少配对序列占位符 %s，模板: %s", exc, cfg['template'])
                    break
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("格式化配对标签失败 %s: %s", format_args, exc)
                    break

                hit_pages = self._search_pdf_pages_for_label(page_text, label)
                if not hit_pages:
                    consecutive_misses += 1
                    logger.debug("Paired label %s not found (miss %d)", label, consecutive_misses)
                    if consecutive_misses > cfg['max_gap']:
                        logger.info(
                            "停止在前缀%s后续，因为连续缺失 %d 次",
                            self._format_prefix_context(prefix_context),
                            consecutive_misses,
                        )
                        break
                    current += 1
                    continue

                consecutive_misses = 0
                if cfg.get('latest_only'):
                    target_page = sorted(hit_pages)[-1]
                    pages = [target_page]
                else:
                    pages = sorted(dict.fromkeys(hit_pages))[: cfg['max_pages']]
                if len(pages) == 1 and cfg['max_pages'] > 1:
                    candidate = pages[0] + 1
                    max_page = max(page_text.keys())
                    if candidate <= max_page and candidate not in pages:
                        pages.append(candidate)
                if not pages:
                    current += 1
                    continue

                identifier = self._build_paired_identifier(pdf_path, label, len(results) + 1)
                if self._out_dir is not None:
                    candidate_path = self._out_dir / f"{identifier}.tex"
                    if candidate_path.exists():
                        logger.info("Skipping label %s (prefix %s) because %s already exists", label, self._format_prefix_context(prefix_context), candidate_path)
                        current += 1
                        continue

                image_paths = self._render_pdf_page_group(pdf_path, pages)
                if not image_paths:
                    logger.warning("Failed to render pages %s for label %s", pages, label)
                    current += 1
                    continue

                if cfg.get('latest_only'):
                    hint = (
                        "我们只关注题号 {label} 的题目。选择了扫描结果中最后一次出现的该题目所在页面"
                        "及其紧接着的一页，这通常包含答案。请忽略图片中与该题无关的内容。"
                    ).format(label=label)
                else:
                    hint = f"题号为 {label} 的题目及其配套解答。"
                item = self._extract_problem_from_images(image_paths, identifier=identifier, hint=hint)
                if self.enable_llm_solution and self._needs_llm_solution(item):
                    item.llm_solution = self._generate_solution(item.exercise)
                prefix_repr = self._format_prefix_context(prefix_context)
                meta = {
                    'source': pdf_path,
                    'type': 'pdf-paired',
                    'paired_label': label,
                    'paired_pages': ','.join(str(p) for p in pages),
                    'extraction_mode': 'paired-sequence',
                }
                if prefix_repr:
                    meta['paired_prefix'] = prefix_repr
                item.metadata.update(meta)
                self._ensure_language(item)
                results.append(item)
                if on_item:
                    on_item(item)
                current += 1

        logger.info("Paired sequence extraction produced %d item(s) from %s", len(results), pdf_path)
        return results

    def _spans_need_llm_segmentation(self, spans: List[TextSpan], page_text: Dict[int, str]) -> bool:
        if not page_text:
            return False
        if not spans:
            return True
        suspicious = 0
        for span in spans:
            body = (span.body or '').strip()
            if not body:
                continue
            if len(body) > 1600 and self._count_question_headings(body) <= 1:
                suspicious += 1
                continue
            if self._contains_multiple_solution_markers(body) and self._count_question_headings(body) == 0:
                suspicious += 1
        return suspicious >= max(1, len(spans) // 2)

    def _segment_exercises_with_llm(self, pdf_path: str, page_text: Dict[int, str]) -> List[TextSpan]:
        if not page_text:
            return []
        ordered_pages = sorted(page_text.items())
        limit = max(1200, self.config.llm_segmentation_chunk_chars)
        overlap = max(0, min(self.config.llm_segmentation_overlap_chars, limit // 2))
        chunk: List[Tuple[int, str]] = []
        chunk_chars = 0
        spans: List[TextSpan] = []
        for page, text in ordered_pages:
            excerpt = self._prepare_page_excerpt(text)
            if not excerpt:
                continue
            chunk.append((page, excerpt))
            chunk_chars += len(excerpt)
            if chunk_chars >= limit:
                spans.extend(self._call_llm_segmentation_chunk(pdf_path, chunk))
                if overlap and chunk:
                    tail_page, tail_text = chunk[-1]
                    tail_excerpt = tail_text[-overlap:]
                    chunk = [(tail_page, tail_excerpt)] if tail_excerpt.strip() else []
                    chunk_chars = len(tail_excerpt)
                else:
                    chunk = []
                    chunk_chars = 0
        if chunk:
            spans.extend(self._call_llm_segmentation_chunk(pdf_path, chunk))

        unique: List[TextSpan] = []
        seen: Set[Tuple[int, int, str]] = set()
        for span in spans:
            key = (span.start_page, span.end_page, (span.body or '').strip()[:80])
            if key in seen:
                continue
            seen.add(key)
            unique.append(span)
        return unique

    def _call_llm_segmentation_chunk(self, pdf_path: str, chunk: List[Tuple[int, str]]) -> List[TextSpan]:
        if not chunk:
            return []
        start_page = chunk[0][0]
        end_page = chunk[-1][0]
        body_sections = []
        for page, text in chunk:
            body_sections.append(f"### Page {page}\n{text}")
        prompt = (
            "我们从一本教材的章节末尾提取了OCR文本，请识别其中真正的数学习题。"
            "正文中的讲解或概念部分请忽略，只保留题目本身以及其后紧跟的解答或答案。"
            f"这些文本来自PDF {Path(pdf_path).name} 的第 {start_page} 到 {end_page} 页。"
            "请返回一个JSON数组，严格按照schema输出。每个条目需要包含题干exercise，可选的answer和solution，以及对应的起止页码。"
            "如果原文中缺少答案或解答，请用空字符串。"
            "务必保留原有的数学符号并使用LaTeX格式。"
            "\n\n"
            + "\n\n".join(body_sections)
        )
        schema = '[{"exercise":"string","answer":"string","solution":"string","page_start":1,"page_end":1,"confidence":0.0}]'
        system = (
            "你是一名数学题目抽取助手，请仅返回真正的习题。"
            "忽略正文叙述、定义或例题的讲解，只保留题号、题干以及紧随其后的解或证。"
            "输出时保证所有数学内容为LaTeX格式，不要包含Markdown。"
        )
        try:
            response = self.client.generate_structured(
                prompt,
                self.config.models.text_reasoning,
                schema_hint=schema,
                system=system,
                temperature=0.0,
                max_tokens=1500,
            )
        except Exception as exc:  # pragma: no cover - logging path
            logger.warning(
                "LLM segmentation request failed for %s pages %s-%s: %s",
                pdf_path,
                start_page,
                end_page,
                exc,
            )
            return []

        if not isinstance(response, list):
            logger.warning("Unexpected segmentation payload for %s: %s", pdf_path, type(response))
            return []

        results: List[TextSpan] = []
        for entry in response[: self.config.llm_segmentation_max_questions]:
            exercise = self._ensure_text(entry.get('exercise'))
            if not exercise or len(exercise) < 6:
                continue
            answer = self._ensure_text(entry.get('answer'))
            solution = self._ensure_text(entry.get('solution'))
            page_start = self._safe_int(entry.get('page_start'), start_page)
            page_end = self._safe_int(entry.get('page_end'), end_page)
            span = TextSpan(
                start_page=page_start,
                end_page=page_end,
                body=exercise,
                answer=answer or None,
                solution=solution or None,
            )
            setattr(span, 'origin', 'llm-segmentation')
            results.append(span)
        return results

    @staticmethod
    def _prepare_page_excerpt(text: str, max_chars: int = 2400) -> str:
        cleaned = re.sub(r'\n{3,}', '\n\n', text or '').strip()
        if not cleaned:
            return ''
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars]
        return cleaned

    @staticmethod
    def _count_question_headings(text: str) -> int:
        return sum(1 for line in text.splitlines() if _QUESTION_HEADING_PATTERN.match(line.strip()))

    @staticmethod
    def _contains_multiple_solution_markers(text: str) -> bool:
        return len(_SOLUTION_MARKER_PATTERN.findall(text)) >= 2

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _refine_text_fallback(self, item: ProblemItem) -> None:
        if not item.exercise or not item.exercise.strip():
            return

        sections = [
            "以下是通过OCR/文本回退得到的内容，请判断其本意并还原为正确的LaTeX排版。",
            "请使用JSON对象返回，字段包括exercise, answer, solution。",
            "所有数学内容必须使用LaTeX语法，不要使用Markdown，也不要省略关键符号。",
            "如果原文没有答案或解答，请用空字符串。",
            "---",
            "# Exercise (raw)",
            item.exercise.strip(),
            "# Answer (raw)",
            (item.answer or '').strip() or '(empty)',
            "# Solution (raw)",
            (item.solution or '').strip() or '(empty)',
        ]
        prompt = '\n'.join(sections)

        schema = '{"exercise":"string","answer":"string","solution":"string"}'
        system = (
            "你是一名数学排版助手，请根据OCR或文本提取的粗糙内容恢复成正确的LaTeX格式。"
            "务必保持题目含义不变，保留数学公式，返回JSON对象。"
        )

        try:
            response = self.client.generate_structured(
                prompt,
                self.config.models.text_reasoning,
                schema_hint=schema,
                system=system,
                temperature=0.0,
                max_tokens=900,
            )
        except Exception as exc:  # pragma: no cover - logging path
            logger.warning("LLM refinement failed for %s: %s", item.identifier, exc)
            return

        if isinstance(response, list):
            response = response[0] if response else {}
        if not isinstance(response, dict):
            logger.warning("LLM refinement returned unexpected payload for %s", item.identifier)
            return

        normalized = self._normalize_parts(response)
        if normalized.get('exercise'):
            item.exercise = normalized['exercise']
        if normalized.get('answer') is not None:
            item.answer = normalized.get('answer')
        if normalized.get('solution'):
            item.solution = normalized['solution']

        item.metadata['llm_text_refined'] = 'true'

    def _render_pdf_page_group(self, pdf_path: str, pages: List[int]) -> List[str]:
        normalized = sorted({page for page in pages if page and page > 0})
        if not normalized:
            return []
        try:
            return list(iter_pdf_images(pdf_path, pages=normalized))
        except Exception as exc:
            logger.warning("Failed to render PDF %s pages %s: %s", pdf_path, normalized, exc, exc_info=True)
            return []

    def _select_page_groups(self, span: TextSpan) -> List[List[int]]:
        pages = list(range(span.start_page, span.end_page + 1))
        if not pages:
            return []
        candidates: List[List[int]] = []
        if len(pages) == 1:
            candidates.append([pages[0]])
            candidates.append([pages[0], pages[0] + 1])
        else:
            head_pair = pages[:2]
            tail_pair = pages[-2:]
            candidates.append(head_pair)
            if tail_pair != head_pair:
                candidates.append(tail_pair)
            candidates.append([pages[0]])
            candidates.append([pages[-1]])
            candidates.append([pages[-1], pages[-1] + 1])
        unique: List[List[int]] = []
        seen = set()
        for group in candidates:
            ordered = tuple(sorted({p for p in group if p > 0}))
            if not ordered:
                continue
            if ordered in seen:
                continue
            seen.add(ordered)
            unique.append(list(ordered))
        return unique[:3]

    def _merge_problem_items(self, base: ProblemItem, incoming: ProblemItem) -> None:
        base.exercise = self._append_unique_text(base.exercise, incoming.exercise)
        for field in ['answer', 'solution']:
            current = getattr(base, field)
            addition = getattr(incoming, field)
            if not addition:
                continue
            if not current:
                setattr(base, field, addition)
            else:
                setattr(base, field, self._append_unique_text(current, addition))

    @staticmethod
    def _append_unique_text(original: Optional[str], addition: Optional[str]) -> str:
        base = (original or '').strip()
        extra = (addition or '').strip()
        if not extra:
            return base
        if not base:
            return extra
        if extra in base:
            return base
        return f"{base}\n\n{extra}"

    def _build_image_prompt(self, hint: Optional[str]) -> str:
        if self.include_answer_field:
            header = (
                "请识别图片中的数学题目与解答，并以JSON返回：{\"exercise\":题干, \"answer\":原答案(可选), \"solution\":原详细解答(可选)}。"
                "所有数学内容必须使用LaTeX语法，不要使用Markdown格式如**bold**或*italic*。"
                "如果没有解答或答案，请将对应字段设置为空字符串\"\"。"
                "保持原有LaTeX语法，公式使用$...$或\\[...\\]。"
            )
        else:
            header = (
                "请识别图片中的数学题目，并以JSON返回：{\"exercise\":题干, \"solution\":原详细解答(可选)}。"
                "所有数学内容必须使用LaTeX语法，不要使用Markdown格式如**bold**或*italic*。"
                "如果没有解答，请将solution字段设置为空字符串\"\"。"
                "保持原有LaTeX语法，公式使用$...$或\\[...\\]。"
            )
        if hint:
            return (
                f"{header}我们只关注与下述提示最匹配的题目，请仅整理这一题：{hint}"
                "。如果图片中包含多道题目，请选择与提示最契合的一道。"
            )
        return header

    def _parse_paired_sequence_config(self, spec: Optional[str], *, latest_only: bool = False) -> Optional[Dict[str, Any]]:
        if not spec:
            return None
        parts = [segment.strip() for segment in spec.split('|') if segment.strip()]
        if not parts:
            return None
        template = parts[0]
        placeholders = re.findall(r'{([a-zA-Z0-9_]+)}', template)
        if not placeholders:
            logger.warning("Paired-sequence template必须至少包含一个占位符: %s", template)
            return None
        config: Dict[str, Any] = {
            'template': template,
            'placeholders': placeholders,
            'terminal_key': placeholders[-1],
            'prefix_keys': placeholders[:-1],
            'ranges': {},
            'prefix_limit': 9,
            'start': 1,
            'max_gap': 0,
            'max_questions': 200,
            'max_pages': 2,
            'latest_only': latest_only,
        }
        standard_keys = {'start', 'max_gap', 'max_questions', 'max_pages', 'prefix_limit'}
        for extra in parts[1:]:
            if '=' not in extra:
                continue
            key, value = extra.split('=', 1)
            key = key.strip()
            value = value.strip()
            low_key = key.lower()
            if low_key in standard_keys:
                try:
                    parsed = int(value)
                    if low_key == 'max_gap':
                        config['max_gap'] = max(0, parsed)
                    elif low_key == 'prefix_limit':
                        config['prefix_limit'] = max(1, parsed)
                    else:
                        config[low_key] = max(1, parsed)
                except ValueError:
                    logger.warning("Invalid integer for %s in paired-sequence config: %s", key, value)
                continue
            placeholder_key = key
            if placeholder_key in placeholders:
                parsed_range = self._parse_sequence_range(value)
                if parsed_range:
                    config['ranges'][placeholder_key] = parsed_range
                else:
                    logger.warning("空的范围定义 %s=%s 被忽略", key, value)
            else:
                logger.debug("忽略未知的配对序列项: %s=%s", key, value)
        return config

    def _iter_paired_prefix_contexts(self, cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        prefix_keys = cfg.get('prefix_keys', [])
        if not prefix_keys:
            yield {}
            return
        ranges = cfg.get('ranges', {})
        limit = cfg.get('prefix_limit', 9)
        value_lists: List[List[Any]] = []
        for key in prefix_keys:
            explicit_values = ranges.get(key)
            if explicit_values:
                values = list(explicit_values)
            else:
                values = list(range(1, limit + 1))
                if len(values) > limit:
                    values = values[:limit]
            if not values:
                values = list(range(1, limit + 1))
                if len(values) > limit:
                    values = values[:limit]
            value_lists.append(values)
        if not value_lists:
            yield {}
            return
        for combo in itertools.product(*value_lists):
            yield dict(zip(prefix_keys, combo))

    @staticmethod
    def _format_prefix_context(context: Dict[str, Any]) -> str:
        if not context:
            return ''
        return ','.join(f"{key}={value}" for key, value in sorted(context.items()))

    @staticmethod
    def _parse_sequence_range(value: str) -> List[Any]:
        cleaned = value.strip()
        if not cleaned:
            return []
        if ',' in cleaned:
            parts = [part.strip() for part in cleaned.split(',') if part.strip()]
            return [Pipeline._coerce_range_value(part) for part in parts]
        interval = re.match(r'^(-?\d+)\s*-\s*(-?\d+)$', cleaned)
        if interval:
            start = int(interval.group(1))
            end = int(interval.group(2))
            step = 1 if end >= start else -1
            return list(range(start, end + step, step))
        return [Pipeline._coerce_range_value(cleaned)]

    @staticmethod
    def _coerce_range_value(token: str) -> Any:
        try:
            return int(token)
        except ValueError:
            return token

    @staticmethod
    def _search_pdf_pages_for_label(page_text: Dict[int, str], label: str) -> List[int]:
        hits: List[int] = []
        target = label.strip()
        if not target:
            return hits
        for page in sorted(page_text.keys()):
            if target in (page_text.get(page) or ''):
                hits.append(page)
        return hits

    @staticmethod
    def _build_paired_identifier(pdf_path: str, label: str, index: int) -> str:
        safe = re.sub(r'[^0-9A-Za-z]+', '_', label).strip('_')
        base = Path(pdf_path).stem
        suffix = safe or f'item{index}'
        return f"{base}_{suffix}"

    @staticmethod
    def _make_question_hint(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        compact = ' '.join(text.strip().split())
        if not compact:
            return None
        if len(compact) <= 160:
            return compact
        return compact[:160] + '…'

    @staticmethod
    def _ensure_text(value):
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            parts = [Pipeline._ensure_text(v) for v in value]
            return '\n'.join(p for p in parts if p)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, indent=2)
        return str(value)

    def _normalize_parts(self, parts: dict) -> dict:
        normalized = {
            'exercise': self._ensure_text(parts.get('exercise')),
            'answer': self._ensure_text(parts.get('answer')),
            'solution': self._ensure_text(parts.get('solution')),
        }
        
        # Clean markdown from all text fields
        for key in ['exercise', 'answer', 'solution']:
            if normalized.get(key):
                normalized[key] = self._clean_markdown_from_text(normalized[key])
        
        exercise = normalized.get('exercise') or ''
        if exercise.startswith('{') and exercise.endswith('}'):
            try:
                embedded = json.loads(exercise)
                normalized['exercise'] = self._ensure_text(embedded.get('exercise')) or normalized['exercise']
                normalized['answer'] = normalized['answer'] or self._ensure_text(embedded.get('answer'))
                normalized['solution'] = normalized['solution'] or self._ensure_text(embedded.get('solution'))
                # Clean markdown from embedded fields too
                for key in ['exercise', 'answer', 'solution']:
                    if normalized.get(key):
                        normalized[key] = self._clean_markdown_from_text(normalized[key])
            except json.JSONDecodeError:
                pass

        if not normalized.get('solution') and exercise:
            split = self._split_solution_from_exercise(exercise)
            if split:
                normalized['exercise'], normalized['solution'] = split

        if normalized.get('solution') and normalized.get('answer') == normalized.get('solution'):
            normalized['answer'] = None

        return normalized

    @staticmethod
    def _split_solution_from_exercise(text: str) -> Optional[tuple]:
        markers = [r'(?:^|\n)\s*(?:Solution|Proof|解答|证明)\s*[:：]?']
        for pattern in markers:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                exercise = text[:match.start()].strip()
                solution = text[match.start():].strip()
                if exercise and solution:
                    return exercise, solution
        return None

    def _ensure_language(self, item: ProblemItem) -> None:
        if self.language == 'auto':
            item.metadata.setdefault('target_language', 'auto')
            return
        translated = False
        for field in ['exercise', 'answer', 'solution', 'llm_solution']:
            value = getattr(item, field)
            if not value:
                continue
            if not self._needs_translation(value):
                continue
            translated_value = self._translate_text(value)
            if translated_value:
                setattr(item, field, translated_value)
                translated = True
        item.metadata['target_language'] = self.language
        if translated:
            item.metadata['translated'] = 'true'

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    def _needs_translation(self, text: str) -> bool:
        if self.language == 'auto':
            return False
        has_chinese = self._contains_chinese(text)
        if self.language == 'zh':
            return not has_chinese and text.strip() != ''
        if self.language == 'en':
            return has_chinese
        return False

    def _translate_text(self, text: str) -> str:
        target = '中文' if self.language == 'zh' else '英文'
        system = "你是一名精准的专业翻译，需要保持原始数学公式 (如 $...$ 或 \\[ ... \\]) 与符号不变。"
        prompt = f"请将下面的内容翻译成{target}，保持数学公式与符号原样：\n\n{text}"
        result = self.client.generate_text(
            prompt,
            self.config.models.text_detailed,
            system=system,
            temperature=0.0,
            max_tokens=1200,
        )
        return result.text.strip()

    @staticmethod
    def _strip_html(value: str) -> str:
        soup = BeautifulSoup(value, 'lxml')
        return soup.get_text('\n', strip=True)

    @staticmethod
    def _simplify_answer(answer: Optional[str]) -> Optional[str]:
        if not answer:
            return None
        cleaned = answer.strip()
        if len(cleaned) <= 6:
            return cleaned
        if cleaned.lower() in {'obvious', 'trivial', 'clear', 'self-evident'}:
            return cleaned
        return None

    def _needs_llm_solution(self, item: ProblemItem) -> bool:
        if item.solution:
            return False
        if item.answer and len(item.answer) > self.config.minimal_answer_tokens:
            return False
        return True

    def _generate_solution(self, exercise: str) -> str:
        system = (
            "你是一名数学教师，请给出严谨详细的解答，并以中文描述，保留原始符号的LaTeX写法。"
            "所有数学内容必须使用LaTeX语法，不要使用Markdown格式。"
            "如果需要强调，请使用\\textbf{}而不是**bold**。"
            "公式使用$...$内联或\\[...\\]显示。"
            "直接返回LaTeX格式的解答文本，不要返回JSON或其他格式。"
            "确保所有LaTeX命令正确，避免未定义的控制序列。"
            "对于表格，使用LaTeX的tabular环境，而不是Markdown表格。"
            "转义特殊字符，如&写成\\&。"
            "确保公式完整，不缺少$符号。"
        )
        prompt = (
            "请为下面的题目写出详细解答，并在开头加入句子：'由LLM生成的解答可能不准确，请自行验证。'"
            f"\n\n{exercise}\n\n请直接返回LaTeX格式的解答，不要包含JSON包装。"
        )
        result = self.client.generate_text(
            prompt,
            self.config.models.text_reasoning,
            system=system,
            max_tokens=1200,
            temperature=0.1,
        )
        raw_text = result.text.strip()
        cleaned = self._clean_json_from_text(raw_text)
        cleaned = self._clean_markdown_from_text(cleaned)
        cleaned = self._fix_latex_issues(cleaned)
        return cleaned.strip()

    @staticmethod
    def _clean_markdown_from_text(text: str) -> str:
        segments = _split_math_segments(text)
        cleaned: List[str] = []
        for kind, segment in segments:
            if kind == 'math':
                cleaned.append(_normalize_math_segment(segment))
                continue
            segment = _convert_markdown_tables(segment)
            segment = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', segment)
            segment = re.sub(r'(?<!\*)\*(?!\s)(.*?)(?<!\s)\*', r'\\textit{\1}', segment)
            segment = re.sub(r'^#+\s+', '', segment, flags=re.MULTILINE)
            segment = re.sub(r'^\s*[-*+]\s+', r'\\textbullet\ ', segment, flags=re.MULTILINE)
            segment = re.sub(r'^\s*\d+\.\s+', r'\\textbf{\g<0>}', segment, flags=re.MULTILINE)
            segment = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', segment)
            segment = re.sub(r'```[^\n]*\n', '', segment)
            segment = segment.replace('```', '')
            segment = re.sub(r'`([^`]+)`', r'\\texttt{\1}', segment)
            cleaned.append(_escape_text_segment(segment))
        return ''.join(cleaned)

    @staticmethod
    def _clean_json_from_text(text: str) -> str:
        return _strip_structured_wrappers(text)

    @staticmethod
    def _fix_latex_issues(text: str) -> str:
        segments = _split_math_segments(text)
        fixed_parts: List[str] = []
        for kind, segment in segments:
            if kind == 'math':
                fixed_parts.append(_normalize_math_segment(segment))
            else:
                fixed_parts.append(_escape_text_segment(segment))
        merged = ''.join(fixed_parts)
        merged = merged.replace('\\n', ' ')
        return _balance_math_delimiters(merged)

    def process_inputs(self, inputs: List[str], out_dir: str, keyword: Optional[str] = None, max_items: Optional[int] = None, site: Optional[str] = None) -> List[str]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.process_inputs_async(inputs, out_dir, keyword=keyword, max_items=max_items, site=site))
        else:
            raise RuntimeError('process_inputs cannot be used when an event loop is running; use await process_inputs_async instead.')

    async def process_inputs_async(self, inputs: List[str], out_dir: str, keyword: Optional[str] = None, max_items: Optional[int] = None, site: Optional[str] = None) -> List[str]:
        out_dir_path = Path(out_dir)
        self._reset_render_state(out_dir_path)

        semaphore = asyncio.Semaphore(self.concurrency)

        async def dispatch(inp: str) -> List[str]:
            async with semaphore:
                return await asyncio.to_thread(self._process_and_render_input, inp, keyword, max_items, site)

        tasks = [asyncio.create_task(dispatch(inp)) for inp in inputs]
        generated_paths: List[str] = []
        for task in asyncio.as_completed(tasks):
            generated_paths.extend(await task)

        master_path = self._write_master()
        if master_path not in generated_paths:
            generated_paths.append(master_path)
        logger.info("Rendered %d problem(s) to %s (async)", len(self._rendered_rel_paths), out_dir)
        return generated_paths

    def _process_and_render_input(self, inp: str, keyword: Optional[str], max_items: Optional[int], site: Optional[str]) -> List[str]:
        raw_inp = inp
        inp = inp.strip()
        generated_paths: List[str] = []

        if not inp:
            logger.warning("Skipping empty input entry derived from %s", raw_inp)
            return generated_paths

        def sink(item: ProblemItem) -> None:
            generated_paths.append(self._render_problem(item))

        lower_inp = inp.lower()
        if lower_inp.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            logger.info("Dispatching image input: %s", inp)
            item = self.process_image(inp)
            sink(item)
        elif lower_inp.endswith('.pdf'):
            logger.info("Dispatching PDF input: %s", inp)
            items = self.process_pdf(inp, on_item=sink)
            if not items:
                logger.warning("No problems extracted from PDF %s", inp)
        elif inp.startswith('http://') or inp.startswith('https://'):
            logger.info("Dispatching URL input: %s", inp)
            for item in self.process_url(inp, keyword=keyword, max_items=max_items, site=site):
                sink(item)
        else:
            raise ValueError(f'Unsupported input type: {inp}')
        return generated_paths

    def _compute_page_legibility(self, page_text: Dict[int, str]) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        for page, text in page_text.items():
            scores[page] = self._legibility_score(text)
        return scores

    @staticmethod
    def _legibility_score(text: str) -> float:
        meaningful = 0
        total = 0
        for ch in text:
            if ch.isspace():
                continue
            total += 1
            code = ord(ch)
            if '0' <= ch <= '9' or 'a' <= ch <= 'z' or 'A' <= ch <= 'Z':
                meaningful += 1
            elif 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0x20000 <= code <= 0x2A6DF:
                meaningful += 1
        if total == 0:
            return 0.0
        return meaningful / total

    def _build_window_spans(self, page_numbers: List[int]) -> List[TextSpan]:
        if not page_numbers:
            return []
        window = max(1, self.config.window_span_pages)
        spans: List[TextSpan] = []
        total = len(page_numbers)
        for idx, start_page in enumerate(page_numbers):
            end_index = min(idx + window - 1, total - 1)
            spans.append(TextSpan(start_page=start_page, end_page=page_numbers[end_index], body=''))
        return spans


def process_inputs(inputs: List[str], out_dir: str, keyword: Optional[str] = None, max_items: Optional[int] = None, site: Optional[str] = None) -> List[str]:
    pipeline = Pipeline()
    return pipeline.process_inputs(inputs, out_dir, keyword=keyword, max_items=max_items, site=site)


__all__ = ['Pipeline', 'process_inputs']
