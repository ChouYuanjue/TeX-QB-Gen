import os
import unicodedata
from pathlib import Path
from typing import Iterable

from .models import ProblemItem

TEX_TEMPLATE = r"""\documentclass[12pt]{{article}}
\usepackage{{amsmath,amssymb}}
\begin{{document}}

\section*{{Exercise}}
{exercise}

{answer_block}

{solution_block}

{llm_solution_block}

\end{{document}}
"""


_ALLOWED_PUNCT = set(" -_.,;:?!/\\()[]{}+=*&^$#@~'\"|<>")


def _sanitize_comment(text: str) -> str:
    cleaned = text.replace('\r', ' ').replace('\n', ' ')
    sanitized_chars = []
    for ch in cleaned:
        if ch == '%':
            sanitized_chars.append(r'\%')
            continue
        category = unicodedata.category(ch)
        if category.startswith('C') and ch not in {'\t', ' '}:
            sanitized_chars.append(' ')
        else:
            sanitized_chars.append(ch)
    sanitized = ''.join(sanitized_chars)
    while '  ' in sanitized:
        sanitized = sanitized.replace('  ', ' ')
    sanitized = sanitized.strip()

    def _allowed(ch: str) -> bool:
        if ch in _ALLOWED_PUNCT:
            return True
        if '0' <= ch <= '9' or 'a' <= ch <= 'z' or 'A' <= ch <= 'Z':
            return True
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0x20000 <= code <= 0x2A6DF:
            return True
        return False

    filtered = ''.join(ch for ch in sanitized if _allowed(ch))
    if filtered:
        total = len(filtered)
        signal = sum(1 for ch in filtered if ch.isalnum() or (0x4E00 <= ord(ch) <= 0x9FFF))
        if total == 0 or signal / total < 0.3:
            return '[unavailable]'
        return filtered
    return '[unavailable]'


def render_single_tex(item: ProblemItem, out_path: str) -> None:
    answer_block = ''
    if item.answer:
        answer_block = '\\subsection*{Answer}\n' + item.answer
    solution_block = ''
    if item.solution:
        solution_block = '\\subsection*{Solution}\n' + item.solution
    llm_solution_block = ''
    if item.llm_solution:
        llm_solution_block = '\\subsection*{Solution (by LLM)}\n' + item.llm_solution

    header_lines = []
    for key, value in item.metadata.items():
        header_lines.append(f"% {key}: {_sanitize_comment(str(value))}")
    header = '\n'.join(header_lines)
    content = header + ('\n' if header else '') + TEX_TEMPLATE.format(
        exercise=item.exercise,
        answer_block=answer_block,
        solution_block=solution_block,
        llm_solution_block=llm_solution_block
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(content)


def render_master(rel_paths: Iterable[str], out_path: str) -> None:
    master = [
        '\\documentclass[12pt]{article}',
        '\\usepackage{amsmath,amssymb}',
        '\\begin{document}',
    ]
    for rel in rel_paths:
        master.append('\\input{%s}' % rel.replace('\\', '/'))
    master.append('\\end{document}')
    Path(out_path).write_text('\n'.join(master), encoding='utf-8')
