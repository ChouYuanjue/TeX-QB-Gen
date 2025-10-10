import argparse
import logging
import re
from pathlib import Path
from typing import List

from .pipeline import Pipeline


def _parse_inputs(values: List[str]) -> List[str]:
    inputs: List[str] = []
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.pdf'}
    
    for value in values:
        if value.startswith('@'):
            with open(value[1:], 'r', encoding='utf-8') as fh:
                inputs.extend([line.strip() for line in fh if line.strip()])
        else:
            path = Path(value)
            if path.is_dir():
                # Recursively find all supported files in the directory
                for ext in supported_extensions:
                    inputs.extend([str(p) for p in path.rglob(f'*{ext}')])
            else:
                inputs.append(value)
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate TeX problems from images, PDFs, or URLs.')
    parser.add_argument('--input', '-i', nargs='+', required=True, help='Input paths, directories, or URLs. Directories will be recursively scanned for images and PDFs. Use @file to read a list.')
    parser.add_argument('--out', '-o', required=True, help='Output directory for TeX files.')
    parser.add_argument('--keyword', '-k', help='Keyword for StackExchange search when processing StackExchange URLs.')
    parser.add_argument('--max-items', '-m', type=int, help='Maximum number of StackExchange questions to fetch.')
    parser.add_argument('--site', '-s', choices=['math', 'mathoverflow'], help='StackExchange site slug.')
    parser.add_argument('--no-llm-solution', action='store_true', help='Disable automatic LLM-generated solutions.')
    parser.add_argument('--omit-answer-field', action='store_true', help='When set, multimodal extraction only requests exercise and solution fields.')
    parser.add_argument('--language', choices=['auto', 'zh', 'en'], help='Target language for outputs (auto, zh, en).')
    parser.add_argument('--generate-master', action='store_true', help='Generate master.tex from existing .tex files in the input directory instead of processing inputs.')
    parser.add_argument('--paired-sequence', help="Template for paired question/answer PDFs, supports multi-level placeholders, e.g. '{chapter}.{section}.{n}|chapter=1-5|section=1-3'.")
    parser.add_argument('--paired-start', type=int, help='Override the starting question index when iterating paired labels.')
    parser.add_argument('--paired-max-gap', type=int, help='Maximum number of consecutive missing labels allowed before stopping a prefix traversal.')
    parser.add_argument('--paired-max-questions', type=int, help='Limit the total number of questions extracted for the paired sequence traversal.')
    parser.add_argument('--paired-max-pages', type=int, help='Maximum distinct pages to fetch per matched label (before auto-appending the next page).')
    parser.add_argument('--paired-prefix-limit', type=int, help='Limit the number of values generated for non-terminal placeholders when no explicit range is provided.')
    parser.add_argument('--paired-latest-only', action='store_true', help='When searching a label yields multiple pages, use only the last occurrence and the following page for multimodal extraction.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable info-level logging output.')
    parser.add_argument('--debug', action='store_true', help='Enable debug-level logging output.')
    args = parser.parse_args()

    log_level = logging.WARNING
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    logging.basicConfig(level=log_level, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')

    if args.generate_master:
        # Generate master.tex from existing .tex files
        from .texgen import render_master
        import os
        if len(args.input) != 1:
            print("Error: --generate-master requires exactly one input directory")
            return
        input_dir = args.input[0]
        if not os.path.isdir(input_dir):
            print(f"Error: {input_dir} is not a directory")
            return
        master_path = os.path.join(args.out, 'master.tex')
        os.makedirs(args.out, exist_ok=True)
        render_master(input_dir, master_path)
        print(f'Generated master.tex: {master_path}')
    else:
        inputs = _parse_inputs(args.input)
        paired_spec = args.paired_sequence
        paired_overrides = {
            'start': args.paired_start,
            'max_gap': args.paired_max_gap,
            'max_questions': args.paired_max_questions,
            'max_pages': args.paired_max_pages,
            'prefix_limit': args.paired_prefix_limit,
        }
        if any(value is not None for value in paired_overrides.values()) and not paired_spec:
            parser.error('Paired sequence traversal overrides require --paired-sequence to be set.')
        if paired_spec:
            for key, value in paired_overrides.items():
                if value is None:
                    continue
                pattern = rf'(?:^|\|)\s*{key}\s*='
                if re.search(pattern, paired_spec):
                    logging.warning('Ignoring CLI override for %s because it is already specified in --paired-sequence.', key)
                    continue
                paired_spec += f"|{key}={value}"
        pipeline = Pipeline(
            enable_llm_solution=not args.no_llm_solution,
            language=args.language,
            paired_sequence=paired_spec,
            paired_latest_only=args.paired_latest_only,
            include_answer_field=not args.omit_answer_field,
        )
        results = pipeline.process_inputs(inputs, args.out, keyword=args.keyword, max_items=args.max_items, site=args.site)
        print('Generated TeX files:')
        for path in results:
            print(f' - {path}')


if __name__ == '__main__':
    main()
