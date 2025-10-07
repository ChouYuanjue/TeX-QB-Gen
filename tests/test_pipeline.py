from pathlib import Path

import pytest

from texbank.models import ProblemItem
from texbank.pdf_utils import TextSpan, extract_keyword_spans
from texbank.pipeline import Pipeline


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
