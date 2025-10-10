import asyncio
from pathlib import Path
from typing import List

import pytest  # type: ignore[import]

from texbank.models import ProblemItem  # type: ignore[import]
from texbank.pdf_utils import TextSpan, extract_keyword_spans  # type: ignore[import]
from texbank.pipeline import Pipeline  # type: ignore[import]
from texbank.texgen import render_master  # type: ignore[import]


class DummyClient:
    def __init__(self, *_args, **_kwargs):
        self.captured_prompts = []

    def generate_from_image(self, *_args, **_kwargs):
        return {
            "exercise": "示例题目",
            "answer": "答案是42",
            "solution": "详细证明"
        }

    def generate_text(self, prompt, *_, **__):
        self.captured_prompts.append(prompt)
        return type('obj', (), {'text': '由LLM生成的解答可能不准确，请自行验证。\n详细解答'})

    def generate_structured(self, prompt, *_, **kwargs):
        self.captured_prompts.append(prompt)
        schema_hint = kwargs.get('schema_hint', '') or ''
        if isinstance(schema_hint, str) and schema_hint.strip().startswith('['):
            return [{
                "exercise": "示例题目(LaTeX)",
                "answer": "",
                "solution": "",
                "page_start": 1,
                "page_end": 1,
                "confidence": 0.95,
            }]
        return {
            "exercise": "示例题目(LaTeX)",
            "answer": "",
            "solution": ""
        }


@pytest.fixture(autouse=True)
def patch_client(monkeypatch):
    monkeypatch.setattr('texbank.pipeline.OpenRouterClient', lambda *_args, **_kwargs: DummyClient())


def test_process_image_creates_problem(tmp_path):
    pipeline = Pipeline(enable_llm_solution=False)
    image = tmp_path / 'sample.png'
    image.write_bytes(b'fake image data')
    item = pipeline.process_image(str(image))
    assert item.exercise == '示例题目'
    assert item.answer.startswith('答案')
    assert item.solution.startswith('详细')
    assert item.metadata['type'] == 'image'


def test_render_items_produces_master(tmp_path):
    pipeline = Pipeline(enable_llm_solution=False)
    items = [
        ProblemItem(identifier='q1', exercise='题干1'),
        ProblemItem(identifier='q2', exercise='题干2', solution='证明'),
    ]
    paths = pipeline.render_items(items, tmp_path.as_posix())
    master = Path(tmp_path) / 'master.tex'
    assert master.exists()
    content = master.read_text(encoding='utf-8')
    assert '题干1' in content
    assert '题干2' in content
    assert '证明' in content
    for path in paths:
        assert Path(path).exists()


def test_extract_keyword_spans_handles_multiple_questions():
    pages = {
        1: 'Exercise 1\n内容A\nSolution: 解答A',
        2: 'Exercise 2\n内容B\nAnswer: 略',
    }
    spans = extract_keyword_spans(pages, ['Exercise'])
    assert len(spans) == 2
    assert isinstance(spans[0], TextSpan)
    assert spans[0].solution.startswith('Solution')
    assert spans[1].answer.startswith('Answer')


def test_process_inputs_async(tmp_path):
    pipeline = Pipeline(enable_llm_solution=False)
    images = []
    for idx in range(2):
        image = tmp_path / f'sample_{idx}.png'
        image.write_bytes(b'fake image data')
        images.append(image.as_posix())
    results = asyncio.run(pipeline.process_inputs_async(images, tmp_path.as_posix()))
    master = Path(tmp_path) / 'master.tex'
    assert master.exists()
    assert any(Path(p) == master for p in results)
    assert len(results) == 3  # two problems + master


def test_process_pdf_streams_items(tmp_path, monkeypatch):
    pipeline = Pipeline(enable_llm_solution=False)
    pipeline._reset_render_state(tmp_path)

    problems = [
        ProblemItem(identifier='p1', exercise='题干1'),
        ProblemItem(identifier='p2', exercise='题干2'),
    ]

    def fake_process_pdf(self, pdf_path, keywords=None, *, on_item=None):
        assert on_item is not None
        for item in problems:
            on_item(item)
        return list(problems)

    monkeypatch.setattr(Pipeline, 'process_pdf', fake_process_pdf)

    paths = pipeline._process_and_render_input('dummy.pdf', None, None, None)
    assert len(paths) == len(problems)
    for item in problems:
        tex_path = Path(tmp_path) / f"{item.identifier}.tex"
        assert tex_path.exists()
        assert tex_path.read_text(encoding='utf-8').strip().startswith(r'\documentclass')


def test_low_legibility_pdf_falls_back(monkeypatch):
    pipeline = Pipeline(enable_llm_solution=False)

    monkeypatch.setattr('texbank.pipeline.is_text_pdf', lambda _path: True)
    monkeypatch.setattr('texbank.pipeline.extract_text_pages', lambda *_args, **_kwargs: {1: '???' * 50, 2: '???' * 50, 3: '???' * 50})
    monkeypatch.setattr('texbank.pipeline.extract_keyword_spans', lambda *_args, **_kwargs: [])

    captured = {}

    def fake_process_text_pdf_via_images(self, _pdf_path, spans, *, on_item=None):
        captured['count'] = len(spans)
        return []

    monkeypatch.setattr(Pipeline, '_process_text_pdf_via_images', fake_process_text_pdf_via_images)
    monkeypatch.setattr(Pipeline, '_process_text_pdf_via_text', lambda *args, **kwargs: [])

    result = pipeline.process_pdf('dummy.pdf')
    assert result == []
    assert captured.get('count', 0) >= 1


def test_llm_segmentation_handles_missing_numbering(monkeypatch):
    pipeline = Pipeline(enable_llm_solution=False)

    monkeypatch.setattr('texbank.pipeline.is_text_pdf', lambda _path: True)
    page_map = {
        10: "本章习题\n第一题 证明题内容较长\n解 首先将...",
        11: "继续的证明文字以及\n证 另一道题目的开头没有题号\n解 该题的详细步骤",
    }
    monkeypatch.setattr('texbank.pipeline.extract_text_pages', lambda *_args, **_kwargs: page_map)
    monkeypatch.setattr('texbank.pipeline.extract_keyword_spans', lambda *_args, **_kwargs: [])

    captured = {}

    def fake_process_images(self, _pdf_path, spans, *, on_item=None):
        captured['span_count'] = len(spans)
        return []

    monkeypatch.setattr(Pipeline, '_process_text_pdf_via_images', fake_process_images)

    items = pipeline.process_pdf('dummy.pdf')
    assert items
    assert captured['span_count'] == len(items)
    for item in items:
        assert item.metadata.get('segmentation_origin') == 'llm-segmentation'
        assert '示例题目' in item.exercise


def test_paired_sequence_pdf(monkeypatch):
    page_map = {
        1: "题号 1.1.1 内容",
        2: "答案 1.1.1",
        3: "题号 1.1.2 内容",
        4: "答案 1.1.2",
        5: "题号 2.1.1 内容",
        6: "答案 2.1.1",
    }
    monkeypatch.setattr('texbank.pipeline.extract_text_pages', lambda *_args, **_kwargs: page_map)

    def fake_render(self, pdf_path, pages):
        return [f"{pdf_path}_p{page}.png" for page in pages]

    def fake_extract(self, image_paths, identifier, hint):
        return ProblemItem(identifier=identifier, exercise=f"{hint} -> {','.join(image_paths)}")

    monkeypatch.setattr(Pipeline, '_render_pdf_page_group', fake_render)
    monkeypatch.setattr(Pipeline, '_extract_problem_from_images', fake_extract)

    pipeline = Pipeline(enable_llm_solution=False, paired_sequence='{chapter}.{section}.{n}|chapter=1-2|section=1|max_pages=2|max_questions=3')
    items = pipeline.process_pdf('dummy.pdf')

    assert len(items) == 3
    labels = [item.metadata['paired_label'] for item in items]
    assert labels == ['1.1.1', '1.1.2', '2.1.1']
    for item in items:
        assert item.metadata['type'] == 'pdf-paired'
        assert '题号为' in item.exercise


def test_paired_sequence_single_hit_appends_next_page(monkeypatch):
    page_map = {
        10: "题号 1.2.5 内容",
        11: "答案 1.2.5",
    }
    monkeypatch.setattr('texbank.pipeline.extract_text_pages', lambda *_args, **_kwargs: page_map)

    captured: List[List[int]] = []

    def fake_render(self, pdf_path, pages):
        captured.append(list(pages))
        return [f"{pdf_path}_p{page}.png" for page in pages]

    def fake_extract(self, image_paths, identifier, hint):
        return ProblemItem(identifier=identifier, exercise=f"{hint} -> {','.join(image_paths)}")

    monkeypatch.setattr(Pipeline, '_render_pdf_page_group', fake_render)
    monkeypatch.setattr(Pipeline, '_extract_problem_from_images', fake_extract)

    pipeline = Pipeline(enable_llm_solution=False, paired_sequence='1.2.{n}|start=5|max_pages=2|max_questions=1')
    items = pipeline.process_pdf('dummy.pdf')

    assert len(items) == 1
    assert captured and captured[0] == [10, 11]


def test_process_input_trims_trailing_whitespace(tmp_path, monkeypatch):
    pipeline = Pipeline(enable_llm_solution=False)
    pipeline._reset_render_state(tmp_path)

    def fake_process_pdf(self, pdf_path, keywords=None, *, on_item=None):
        assert pdf_path == 'tests/abstract_algebra.pdf'
        item = ProblemItem(identifier='p_trim', exercise='题干')
        if on_item:
            on_item(item)
        return [item]

    monkeypatch.setattr(Pipeline, 'process_pdf', fake_process_pdf)

    paths = pipeline._process_and_render_input('tests/abstract_algebra.pdf   ', None, None, None)
    assert len(paths) == 1
    assert Path(paths[0]).exists()


def test_paired_sequence_latest_only(monkeypatch, tmp_path):
    page_map = {
        20: "题号 1.5.1",
        21: "题号 1.5.1",
        22: "答案 1.5.1",
        23: "附录",
    }
    monkeypatch.setattr('texbank.pipeline.extract_text_pages', lambda *_args, **_kwargs: page_map)

    captured: List[List[int]] = []

    def fake_render(self, pdf_path, pages):
        captured.append(list(pages))
        return [f"{pdf_path}_p{page}.png" for page in pages]

    def fake_extract(self, image_paths, identifier, hint):
        return ProblemItem(identifier=identifier, exercise='题干')

    monkeypatch.setattr(Pipeline, '_render_pdf_page_group', fake_render)
    monkeypatch.setattr(Pipeline, '_extract_problem_from_images', fake_extract)

    pipeline = Pipeline(enable_llm_solution=False, paired_sequence='1.5.{n}|start=1|max_pages=2', paired_latest_only=True)
    pipeline._reset_render_state(tmp_path)
    items = pipeline.process_pdf('dummy.pdf')

    assert len(items) == 1
    assert captured[0] == [22, 23]


def test_paired_sequence_skips_existing(monkeypatch, tmp_path):
    page_map = {5: "题号 1.9.1", 6: "答案 1.9.1"}
    monkeypatch.setattr('texbank.pipeline.extract_text_pages', lambda *_args, **_kwargs: page_map)

    def fake_render(self, pdf_path, pages):
        return [f"{pdf_path}_p{page}.png" for page in pages]

    def fake_extract(self, image_paths, identifier, hint):
        return ProblemItem(identifier=identifier, exercise='题干')

    monkeypatch.setattr(Pipeline, '_render_pdf_page_group', fake_render)
    monkeypatch.setattr(Pipeline, '_extract_problem_from_images', fake_extract)

    pipeline = Pipeline(enable_llm_solution=False, paired_sequence='1.9.{n}|start=1|max_pages=2', paired_latest_only=False)
    pipeline._reset_render_state(tmp_path)

    existing = tmp_path / 'dummy_1_9_1.tex'
    existing.write_text('already there', encoding='utf-8')

    items = pipeline.process_pdf('dummy.pdf')

    assert items == []


def test_text_fallback_triggers_llm_refinement(monkeypatch):
    pipeline = Pipeline(enable_llm_solution=False)
    span = TextSpan(start_page=1, end_page=1, body='raw OCR 内容', answer=None, solution=None)
    item = pipeline._extract_span_via_text('dummy.pdf', span, 0)
    assert item.exercise.startswith('示例题目')
    assert item.metadata.get('llm_text_refined') == 'true'


def test_render_master_uses_natural_sort(tmp_path):
    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    def write_tex(name: str, content: str) -> None:
        path = out_dir / f'{name}.tex'
        path.write_text(
            '\\documentclass{article}\n\\begin{document}\n' + content + '\n\\end{document}',
            encoding='utf-8',
        )

    write_tex('topic_q1', 'CONTENT-1')
    write_tex('topic_q10', 'CONTENT-10')
    write_tex('topic_q2', 'CONTENT-2')

    master_path = tmp_path / 'master.tex'
    render_master(out_dir.as_posix(), master_path.as_posix())
    compiled = master_path.read_text(encoding='utf-8')

    idx1 = compiled.index('CONTENT-1')
    idx2 = compiled.index('CONTENT-2')
    idx10 = compiled.index('CONTENT-10')

    assert idx1 < idx2 < idx10


def test_multimodal_retries_invalid_json(tmp_path):
    pipeline = Pipeline(enable_llm_solution=False)

    class FlakyClient(DummyClient):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def generate_from_image(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ValueError('Multimodal response is not valid JSON. Received: ...')
            return super().generate_from_image(*args, **kwargs)

    pipeline.client = FlakyClient()
    image = tmp_path / 'retry.png'
    image.write_bytes(b'fake image data')

    item = pipeline.process_image(image.as_posix())

    assert pipeline.client.calls == 2
    assert item.exercise == '示例题目'


def test_image_prompt_without_answer(monkeypatch, tmp_path):
    class NoAnswerClient(DummyClient):
        def __init__(self):
            super().__init__()
            self.last_prompt = None

        def generate_from_image(self, *args, **kwargs):
            self.last_prompt = kwargs.get('prompt', '')
            return {
                "exercise": "题干",
                "solution": "解答",
            }

    pipeline = Pipeline(enable_llm_solution=False, include_answer_field=False)
    pipeline.client = NoAnswerClient()
    image = tmp_path / 'sample.png'
    image.write_bytes(b'fake image data')

    item = pipeline.process_image(image.as_posix())

    assert 'answer' not in pipeline.client.last_prompt
    assert item.answer is None or item.answer == ''
    assert item.solution == '解答'
