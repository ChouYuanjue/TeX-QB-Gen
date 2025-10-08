import os
import unicodedata
from pathlib import Path
from typing import Iterable

from .models import ProblemItem

TEX_TEMPLATE = r"""\documentclass[12pt]{{article}}
\usepackage{{amsmath,amssymb}}
\usepackage{{ctex}}
\begin{{document}}

\section*{{Exercise}}
{exercise}

{answer_block}

{solution_block}

{llm_solution_block}

\end{{document}}
"""


_ALLOWED_PUNCT = set(" -_.,;:?!/\\()[]{}+=*&^$#@~'\"|<>")


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


def _clean_tex_content(content: str) -> str:
    # Remove everything before \begin{document}
    begin_doc = r'\begin{document}'
    start = content.find(begin_doc)
    if start != -1:
        content = content[start + len(begin_doc):]
    
    # Remove \end{document} and everything after
    end_doc = r'\end{document}'
    end = content.find(end_doc)
    if end != -1:
        content = content[:end]
    
    # Remove leading/trailing whitespace
    return content.strip()


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


def render_master(out_dir: str = 'out/', master_path: str = 'master.tex') -> None:
    out_path = Path(out_dir)
    if not out_path.exists():
        raise FileNotFoundError(f"Output directory {out_dir} does not exist")
    
    tex_files = sorted(out_path.rglob('*.tex'))
    if not tex_files:
        raise FileNotFoundError(f"No .tex files found in {out_dir}")
    
    master_lines = [
        r'\documentclass[12pt]{book}',
        r'\usepackage{amsmath,amssymb}',
        r'\usepackage{ctex}',
        r'\begin{document}',
        r'\tableofcontents',
        r'\newpage',
    ]
    
    current_chapter = None
    current_section = None
    
    for tex_file in tex_files:
        rel_path = tex_file.relative_to(out_path)
        parts = list(rel_path.parts)
        if parts[-1].endswith('.tex'):
            parts[-1] = parts[-1][:-4]  # remove .tex
        
        # Generate titles from parts
        titles = [part.replace('_', ' ').title() for part in parts]
        
        # Determine levels: folder levels as chapters/sections
        num_parts = len(parts)
        if num_parts >= 1:
            chapter_title = titles[0]
            if chapter_title != current_chapter:
                master_lines.append(f'\\chapter{{{chapter_title}}}')
                current_chapter = chapter_title
                current_section = None  # reset section
        
        if num_parts >= 2:
            section_title = titles[1]
            if section_title != current_section:
                master_lines.append(f'\\section{{{section_title}}}')
                current_section = section_title
        
        if num_parts >= 3:
            subsection_title = ' '.join(titles[2:])
            master_lines.append(f'\\subsection{{{subsection_title}}}')
        
        # Read and clean content
        content = tex_file.read_text(encoding='utf-8')
        cleaned_content = _clean_tex_content(content)
        if cleaned_content:
            master_lines.append(cleaned_content)
            master_lines.append(r'\newpage')
    
    master_lines.append(r'\end{document}')
    
    Path(master_path).write_text('\n'.join(master_lines), encoding='utf-8')
