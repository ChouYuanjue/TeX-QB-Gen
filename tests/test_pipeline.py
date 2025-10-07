import asyncio
from pathlib import Path

import pytest  # type: ignore[import]

from texbank.models import ProblemItem  # type: ignore[import]
from texbank.pdf_utils import TextSpan, extract_keyword_spans  # type: ignore[import]
from texbank.pipeline import Pipeline  # type: ignore[import]


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

    def generate_structured(self, *_args, **__kwargs):
        return []


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
    assert '\\input{q1.tex}' in content
    assert '\\input{q2.tex}' in content
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
    master = Path(tmp_path) / 'master.tex'
    assert master.exists()
    master_content = master.read_text(encoding='utf-8')
    for item in problems:
        tex_path = Path(tmp_path) / f"{item.identifier}.tex"
        assert tex_path.exists()
        assert tex_path.read_text(encoding='utf-8').strip().startswith(r'\documentclass')
        assert f"{item.identifier}.tex" in master_content


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
    assert captured.get('count', 0) >= 3
