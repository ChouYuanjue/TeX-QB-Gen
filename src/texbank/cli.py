import argparse
import logging
from typing import List

from .pipeline import Pipeline


def _parse_inputs(values: List[str]) -> List[str]:
    inputs: List[str] = []
    for value in values:
        if value.startswith('@'):
            with open(value[1:], 'r', encoding='utf-8') as fh:
                inputs.extend([line.strip() for line in fh if line.strip()])
        else:
            inputs.append(value)
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate TeX problems from images, PDFs, or URLs.')
    parser.add_argument('--input', '-i', nargs='+', required=True, help='Input paths or URLs. Use @file to read a list.')
    parser.add_argument('--out', '-o', required=True, help='Output directory for TeX files.')
    parser.add_argument('--keyword', '-k', help='Keyword for StackExchange search when processing StackExchange URLs.')
    parser.add_argument('--max-items', '-m', type=int, help='Maximum number of StackExchange questions to fetch.')
    parser.add_argument('--site', '-s', choices=['math', 'mathoverflow'], help='StackExchange site slug.')
    parser.add_argument('--no-llm-solution', action='store_true', help='Disable automatic LLM-generated solutions.')
    parser.add_argument('--language', choices=['auto', 'zh', 'en'], help='Target language for outputs (auto, zh, en).')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable info-level logging output.')
    parser.add_argument('--debug', action='store_true', help='Enable debug-level logging output.')
    args = parser.parse_args()

    log_level = logging.WARNING
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    logging.basicConfig(level=log_level, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')

    inputs = _parse_inputs(args.input)
    pipeline = Pipeline(enable_llm_solution=not args.no_llm_solution, language=args.language)
    results = pipeline.process_inputs(inputs, args.out, keyword=args.keyword, max_items=args.max_items, site=args.site)
    print('Generated TeX files:')
    for path in results:
        print(f' - {path}')


if __name__ == '__main__':
    main()
