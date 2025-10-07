from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF
from pdf2image import convert_from_path


@dataclass
class TextSpan:
    start_page: int
    end_page: int
    body: str
    answer: Optional[str] = None
    solution: Optional[str] = None


def is_text_pdf(pdf_path: str) -> bool:
    doc = fitz.open(pdf_path)
    text_pages = 0
    for page in doc:
        txt = page.get_text().strip()
        if len(txt) > 80:
            text_pages += 1
    ratio = text_pages / max(1, doc.page_count)
    return ratio > 0.55


def detect_scan_ratio(pdf_path: str) -> Dict[int, float]:
    doc = fitz.open(pdf_path)
    ratios: Dict[int, float] = {}
    for idx, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        image_count = len(page.get_images())
        ratios[idx] = image_count / max(1, len(text) + image_count)
    return ratios


def extract_text_pages(pdf_path: str, pages: Optional[Iterable[int]] = None, limit: Optional[int] = None) -> Dict[int, str]:
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    if pages is None:
        page_range = range(1, total_pages + 1)
    else:
        page_range = pages
    results: Dict[int, str] = {}
    for idx, page_number in enumerate(page_range, start=1):
        if limit is not None and idx > limit:
            break
        page = doc[page_number - 1]
        results[page_number] = page.get_text()
    return results


def extract_keyword_spans(page_text: Dict[int, str], keywords: List[str]) -> List[TextSpan]:
    keyword_pattern = re.compile(r'(?:^|\n)\s*(%s)[\s:：]+' % '|'.join(re.escape(k) for k in keywords), re.IGNORECASE)
    spans: List[TextSpan] = []
    current: Optional[Dict[str, object]] = None

    def finalize(entry: Dict[str, object]) -> None:
        text = '\n'.join(entry['text']) if isinstance(entry.get('text'), list) else str(entry.get('text', ''))
        body, answer, solution = _split_sections(text)
        spans.append(TextSpan(start_page=entry['start_page'], end_page=entry['end_page'], body=body, answer=answer, solution=solution))

    for page in sorted(page_text.keys()):
        text = page_text[page]
        matches = list(keyword_pattern.finditer(text))
        if not matches:
            if current:
                current['text'].append(text)
                current['end_page'] = page
            continue

        segments: List[str] = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            segments.append(text[start:end])

        prefix = text[:matches[0].start()]
        if current and prefix.strip():
            current['text'].append(prefix)
            current['end_page'] = page
            finalize(current)
            current = None
        elif current:
            finalize(current)
            current = None

        for segment in segments[:-1]:
            entry = {'start_page': page, 'end_page': page, 'text': [segment]}
            finalize(entry)
        current = {'start_page': page, 'end_page': page, 'text': [segments[-1]]}

    if current:
        finalize(current)

    return spans


def _split_sections(text: str) -> Tuple[str, Optional[str], Optional[str]]:
    cleaned = text.strip()
    if not cleaned:
        return '', None, None
    answer_pattern = re.compile(r'(?:^|\n)\s*(?:Answer|答案)\s*[:：]?', re.IGNORECASE)
    solution_pattern = re.compile(r'(?:^|\n)\s*(?:Solution|Proof|解答|证明)\s*[:：]?', re.IGNORECASE)

    solution_match = solution_pattern.search(cleaned)
    answer_match = answer_pattern.search(cleaned)

    solution = None
    answer = None
    body = cleaned

    if solution_match:
        solution = cleaned[solution_match.start():].strip()
        body = cleaned[:solution_match.start()].strip()
    if answer_match and (not solution_match or answer_match.start() < solution_match.start()):
        answer = cleaned[answer_match.start():solution_match.start() if solution_match else None].strip()
        body = cleaned[:answer_match.start()].strip()
    elif answer_match:
        answer = cleaned[answer_match.start():].strip()

    return body, answer, solution


def iter_pdf_images(pdf_path: str, pages: Optional[List[int]] = None, dpi: int = 200) -> Iterable[str]:
    output_folder = Path(pdf_path).with_suffix('').as_posix() + '_images'
    os.makedirs(output_folder, exist_ok=True)
    first_page = min(pages) if pages else None
    last_page = max(pages) if pages else None
    images = convert_from_path(pdf_path, dpi=dpi, first_page=first_page, last_page=last_page)
    desired = set(pages) if pages else None
    current_page = first_page if first_page is not None else 1
    for img in images:
        page_number = current_page
        current_page += 1
        if desired and page_number not in desired:
            continue
        save_path = os.path.join(output_folder, f'page_{page_number}.png')
        img.save(save_path)
        yield save_path
