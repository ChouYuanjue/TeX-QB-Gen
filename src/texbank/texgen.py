import os
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


def _escape_comment(text: str) -> str:
    return text.replace('%', r'\%')


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
        header_lines.append(f"% {key}: {_escape_comment(value)}")
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
