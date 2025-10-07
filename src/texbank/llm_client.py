from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests

from .cache import DiskCache
from .config import OpenRouterConfig, get_pipeline_config


@dataclass
class LLMResult:
    text: str
    raw: Dict[str, Any]
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None


class OpenRouterClient:
    """OpenRouter REST client with caching, retry, and multimodal support."""

    def __init__(self, config: Optional[OpenRouterConfig] = None):
        cfg = config or get_pipeline_config().openrouter
        self.api_key = cfg.api_key
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set in environment or .env")
        self.api_url = self._normalize_endpoint(cfg.api_url)
        self.app_title = cfg.app_title
        self.http_referer = cfg.http_referer
        self.timeout = cfg.request_timeout
        self.max_retries = cfg.max_retries
        self.backoff = cfg.retry_backoff
        self.cache = DiskCache('openrouter')

    @staticmethod
    def _normalize_endpoint(url: str) -> str:
        url = (url or 'https://openrouter.ai/api/v1/chat/completions').strip().strip("'\"")
        if not url.endswith('/chat/completions'):
            url = url.rstrip('/')
            if url.endswith('/api/v1'):
                url = url + '/chat/completions'
            elif url.endswith('/api/v1/chat'):
                url = url + '/completions'
            else:
                url = url + '/chat/completions'
        return url

    def _headers(self) -> Dict[str, str]:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }
        if self.app_title:
            headers['X-Title'] = self.app_title
        if self.http_referer:
            headers['HTTP-Referer'] = self.http_referer
        return headers

    def _call(self, payload: Dict[str, Any], cache_key_parts: Iterable[str]) -> LLMResult:
        key = DiskCache.make_key(*cache_key_parts)
        cached = self.cache.load(key)
        if cached:
            return LLMResult(text=cached.get('text', ''), raw=cached.get('raw', {}), tokens_in=cached.get('tokens_in'), tokens_out=cached.get('tokens_out'))

        retries = self.max_retries
        delay = 1.0
        last_exc: Optional[Exception] = None
        while retries > 0:
            try:
                response = requests.post(self.api_url, headers=self._headers(), json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                text = self._extract_text(data)
                result = LLMResult(text=text, raw=data, tokens_in=self._token_usage(data, 'prompt'), tokens_out=self._token_usage(data, 'completion'))
                self.cache.save(key, {
                    'text': result.text,
                    'raw': result.raw,
                    'tokens_in': result.tokens_in,
                    'tokens_out': result.tokens_out,
                })
                return result
            except requests.HTTPError as err:
                status = err.response.status_code if err.response is not None else None
                if status == 429 and retries > 1:
                    time.sleep(delay)
                    delay *= self.backoff
                    retries -= 1
                    last_exc = err
                    continue
                raise
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(delay)
                delay *= self.backoff
                retries -= 1
        if last_exc:
            raise last_exc
        raise RuntimeError('OpenRouter call failed without exception context')

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        choices = data.get('choices') or []
        if not choices:
            return json.dumps(data, ensure_ascii=False)
        first = choices[0]
        if isinstance(first, dict):
            message = first.get('message')
            if isinstance(message, dict):
                content = message.get('content')
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            parts.append(part.get('text', ''))
                    if parts:
                        return ''.join(parts)
                if isinstance(content, str):
                    return content
        return first.get('text', '') if isinstance(first, dict) else str(first)

    @staticmethod
    def _token_usage(data: Dict[str, Any], key: str) -> Optional[int]:
        usage = data.get('usage')
        if not isinstance(usage, dict):
            return None
        value = usage.get(key)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def generate_text(self, prompt: str, model: str, *, max_tokens: int = 800, temperature: float = 0.2, system: Optional[str] = None) -> LLMResult:
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})
        payload = {
            'model': model,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
        }
        return self._call(payload, [model, prompt, str(max_tokens), str(temperature), system or ''])

    def generate_structured(self, prompt: str, model: str, *, schema_hint: Optional[str] = None, system: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 1200) -> Dict[str, Any]:
        guidance = prompt
        if schema_hint:
            guidance = f"{prompt}\n\n请严格返回 JSON，对象 schema 为: {schema_hint}."
        result = self.generate_text(guidance, model, max_tokens=max_tokens, temperature=temperature, system=system)
        parsed = self._safe_json(result.text)
        if parsed is None:
            raise ValueError('LLM response is not valid JSON. Received: %s' % result.text[:200])
        return parsed

    def generate_from_image(self, image_path: Sequence[str] | str, model: str, *, prompt: str, system: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 1200) -> Dict[str, Any]:
        if isinstance(image_path, str):
            image_paths: Sequence[str] = [image_path]
        else:
            image_paths = list(image_path)
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({'role': 'system', 'content': system})
        content: List[Dict[str, Any]] = [{'type': 'text', 'text': prompt}]
        key_parts: List[str] = [model, prompt]
        for path in image_paths:
            with open(path, 'rb') as fh:
                image_bytes = fh.read()
            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type:
                mime_type = 'image/png'
            data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
            content.append({'type': 'image_url', 'image_url': {'url': data_url, 'detail': 'auto'}})
            key_parts.extend([os.path.basename(path), str(len(image_bytes))])
        messages.append({'role': 'user', 'content': content})
        payload = {
            'model': model,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
        }
        if not key_parts:
            key_parts = [model, prompt]
        result = self._call(payload, key_parts)
        parsed = self._safe_json(result.text)
        if parsed is None:
            raise ValueError('Multimodal response is not valid JSON. Received: %s' % result.text[:200])
        return parsed

    @staticmethod
    def _safe_json(text: str) -> Optional[Dict[str, Any]]:
        candidate = text.strip()
        if candidate.startswith('```'):
            candidate = candidate.split('\n', 1)[-1]
            if candidate.endswith('```'):
                candidate = candidate[:-3]
        start = candidate.find('{')
        end = candidate.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(candidate[start:end + 1])
        except json.JSONDecodeError:
            return None


__all__ = ['OpenRouterClient', 'LLMResult']
