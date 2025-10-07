import os
from dataclasses import dataclass, field
from typing import Optional

import importlib


def _load_dotenv():
    spec = importlib.util.find_spec('dotenv')
    if spec is None:  # pragma: no cover - optional dependency
        return lambda *_args, **_kwargs: False
    module = importlib.import_module('dotenv')
    return getattr(module, 'load_dotenv')


load_dotenv = _load_dotenv()

load_dotenv()


@dataclass(frozen=True)
class ModelConfig:
    ocr_multimodal: str = os.getenv('TEXBANK_OCR_MODEL', 'google/gemini-2.5-flash-lite')
    ocr_multimodal_image: str = os.getenv('TEXBANK_OCR_IMAGE_MODEL', 'google/gemini-2.5-flash-image-preview')
    text_reasoning: str = os.getenv('TEXBANK_TEXT_MODEL', 'deepseek/deepseek-chat')
    text_detailed: str = os.getenv('TEXBANK_DETAILED_MODEL', 'meta-llama/llama-3.1-70b-instruct')
    fallback_completion: str = os.getenv('TEXBANK_FALLBACK_MODEL', 'deepseek/deepseek-r1')


@dataclass(frozen=True)
class CacheConfig:
    root: str = os.getenv('TEXBANK_CACHE_DIR', '.texbank_llm_cache')
    ttl_seconds: Optional[int] = field(default=None)


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: Optional[str] = os.getenv('OPENROUTER_API_KEY')
    api_url: str = os.getenv('OPENROUTER_API_URL', 'https://openrouter.ai/api/v1/chat/completions')
    app_title: str = os.getenv('OPENROUTER_APP_TITLE', 'TeX Test Bank Generator')
    http_referer: Optional[str] = os.getenv('OPENROUTER_HTTP_REFERER')
    request_timeout: int = int(os.getenv('OPENROUTER_TIMEOUT', '180'))
    max_retries: int = int(os.getenv('OPENROUTER_MAX_RETRIES', '3'))
    retry_backoff: float = float(os.getenv('OPENROUTER_RETRY_BACKOFF', '1.5'))


@dataclass(frozen=True)
class StackExchangeConfig:
    api_key: Optional[str] = os.getenv('STACKEXCHANGE_KEY')
    default_site: str = os.getenv('STACKEXCHANGE_SITE', 'math')
    page_size: int = int(os.getenv('STACKEXCHANGE_PAGE_SIZE', '10'))


@dataclass(frozen=True)
class PipelineConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    stackexchange: StackExchangeConfig = field(default_factory=StackExchangeConfig)
    max_pdf_preview_pages: int = int(os.getenv('TEXBANK_PDF_PREVIEW_PAGES', '200'))
    scan_detection_ratio: float = float(os.getenv('TEXBANK_SCAN_THRESHOLD', '0.2'))
    minimal_answer_tokens: int = int(os.getenv('TEXBANK_MIN_ANSWER_TOKENS', '20'))
    llm_timeout: int = int(os.getenv('TEXBANK_LLM_TIMEOUT', '210'))
    default_language: str = os.getenv('TEXBANK_DEFAULT_LANGUAGE', 'auto')


def get_pipeline_config() -> PipelineConfig:
    return PipelineConfig()
