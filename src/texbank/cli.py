import argparse
import logging
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
    parser.add_argument('--language', choices=['auto', 'zh', 'en'], help='Target language for outputs (auto, zh, en).')
    parser.add_argument('--generate-master', action='store_true', help='Generate master.tex from existing .tex files in the input directory instead of processing inputs.')
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
        pipeline = Pipeline(enable_llm_solution=not args.no_llm_solution, language=args.language)
        results = pipeline.process_inputs(inputs, args.out, keyword=args.keyword, max_items=args.max_items, site=args.site)
        print('Generated TeX files:')
        for path in results:
            print(f' - {path}')


if __name__ == '__main__':
    main()
