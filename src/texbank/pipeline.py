from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

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


class Pipeline:
    IMAGE_SYSTEM = (
        "你是一个数学题目整理助手，所有输出都必须是 JSON 对象，仅包含 exercise, answer, solution 三个字段。"
    )

    def __init__(self, config: Optional[PipelineConfig] = None, *, enable_llm_solution: bool = True, language: Optional[str] = None):
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
            if force_windows and page_text:
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
            item.metadata.update({'source_pdf': pdf_path, 'page': str(candidate_pages[idx] if idx < len(candidate_pages) else '?')})
            self._ensure_language(item)
            items.append(item)
            if on_item:
                on_item(item)
        return items

    # endregion ------------------------------------------------------------------------------------

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
            "输出必须是 JSON 数组，每项包含 title, body, answers(列表，可空)。"
        )
        user_prompt = (
            "请从以下网页正文中提取所有数学题目及可能的解答。"
            "保持原有数学符号与 LaTeX 表达，可根据上下文补全标题。"
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
        path = self._out_dir / f"{item.identifier}.tex"
        with self._render_lock:
            render_single_tex(item, str(path))
            rel_name = path.name
            if rel_name not in self._rendered_rel_paths:
                self._rendered_rel_paths.append(rel_name)
            self._write_master()
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

    # endregion ------------------------------------------------------------------------------------

    # region helpers -------------------------------------------------------------------------------
    def _default_pdf_keywords(self) -> List[str]:
        return [
            'Exercise', 'Exercises', 'Problem', 'Problems', 'Question', 'Questions', 'Example', 'Examples',
            'Proof', 'Lemma', 'Theorem', 'Corollary', 'Practice', 'Homework', 'Review', 'Quiz', 'Test',
            '题目', '问题', '练习', '习题', '例题', '例', '例子', '证明', '解答', '答案', '思考', '讨论'
        ]

    def _extract_problem_from_images(self, image_paths: List[str], *, identifier: Optional[str] = None, hint: Optional[str] = None) -> ProblemItem:
        if not image_paths:
            raise ValueError('No images provided for extraction')
        try:
            logger.info(
                "Submitting %d image(s) for multimodal extraction (identifier=%s)",
                len(image_paths),
                identifier,
            )
            if hint and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Extraction hint: %s", hint)
            structured = self.client.generate_from_image(
                image_paths,
                model=self.config.models.ocr_multimodal_image,
                prompt=self._build_image_prompt(hint),
                system=self.IMAGE_SYSTEM,
            )
        except Exception as exc:
            logger.warning(
                "Multimodal extraction failed for %s; falling back to OCR. Reason: %s",
                identifier,
                exc,
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
        if self.enable_llm_solution and self._needs_llm_solution(item):
            item.llm_solution = self._generate_solution(item.exercise)
        item.metadata.update({
            'source': pdf_path,
            'type': 'pdf-text',
            'page_start': str(span.start_page),
            'page_end': str(span.end_page),
            'extraction_mode': 'text-fallback',
        })
        self._ensure_language(item)
        return item

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
        header = (
            "请识别图片中的数学题目与解答，并以 JSON 返回：{\"exercise\":题干, \"answer\":原答案(可选), \"solution\":原详细解答(可选)}。"
            "若没有解答，请留空。保持原有 LaTeX 语法。"
        )
        if hint:
            return (
                f"{header}我们只关注与下述提示最匹配的题目，请仅整理这一题：{hint}"
                "。如果图片中包含多道题目，请选择与提示最契合的一道。"
            )
        return header

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
        exercise = normalized.get('exercise') or ''
        if exercise.startswith('{') and exercise.endswith('}'):
            try:
                embedded = json.loads(exercise)
                normalized['exercise'] = self._ensure_text(embedded.get('exercise')) or normalized['exercise']
                normalized['answer'] = normalized['answer'] or self._ensure_text(embedded.get('answer'))
                normalized['solution'] = normalized['solution'] or self._ensure_text(embedded.get('solution'))
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
        system = "你是一名数学教师，请给出严谨详细的解答，并以中文描述，保留原始符号的 LaTeX 写法。"
        prompt = f"请为下面的题目写出详细解答，并在开头加入句子：'由LLM生成的解答可能不准确，请自行验证。'\n\n{exercise}"
        result = self.client.generate_text(prompt, self.config.models.text_reasoning, system=system, max_tokens=1200, temperature=0.1)
        return result.text.strip()

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
        generated_paths: List[str] = []

        def sink(item: ProblemItem) -> None:
            generated_paths.append(self._render_problem(item))

        if inp.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            logger.info("Dispatching image input: %s", inp)
            item = self.process_image(inp)
            sink(item)
        elif inp.lower().endswith('.pdf'):
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
