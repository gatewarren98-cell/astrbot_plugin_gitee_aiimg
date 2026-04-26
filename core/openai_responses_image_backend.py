from __future__ import annotations

import asyncio
import base64
import inspect
import json
import re
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

from astrbot.api import logger

from .gitee_sizes import normalize_size_text, size_to_ratio
from .image_format import decode_base64_image_payload, guess_image_mime_and_ext


_RESPONSES_SUFFIX_RE = re.compile(r"/(?:v1/)?responses/?$", re.IGNORECASE)
_SUPPORTED_IMAGE_TOOL_SIZES = {"1024x1024", "1536x1024", "1024x1536"}
_DATA_IMAGE_RE = re.compile(r"(data:image/[^\s)\"']+)", re.IGNORECASE)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")
_IMAGE_URL_RE = re.compile(
    r"(https?://[^\s<>\"')\]]+?\.(?:png|jpg|jpeg|webp|gif)(?:\?[^\s<>\"')\]]*)?)",
    re.IGNORECASE,
)


def normalize_responses_base_url(raw: str) -> str:
    s = str(raw or "").strip().rstrip("/")
    if not s:
        return ""
    s = _RESPONSES_SUFFIX_RE.sub("", s).rstrip("/")
    if re.search(r"/v1($|/)", s):
        return s
    try:
        parts = urlsplit(s)
        if parts.scheme and parts.netloc:
            path = (parts.path or "").rstrip("/") + "/v1"
            return urlunsplit((parts.scheme, parts.netloc, path, "", "")).rstrip("/")
    except Exception:
        pass
    return f"{s}/v1"


def _resolution_to_size(resolution: str | None) -> str:
    r = str(resolution or "").strip().upper()
    if r in {"1K", "1024", "2K", "2048", "4K", "4096"}:
        return "1024x1024"
    return ""


def _decode_base64_bytes(text: str) -> bytes:
    s = re.sub(r"\s+", "", str(text or "").strip())
    if not s:
        return b""
    candidates = [s, s.replace("-", "+").replace("_", "/")]
    for cand in candidates:
        pad = "=" * ((4 - len(cand) % 4) % 4)
        try:
            raw = base64.b64decode(cand + pad, validate=False)
            if raw:
                return raw
        except Exception:
            continue
    try:
        return base64.urlsafe_b64decode(s + ("=" * ((4 - len(s) % 4) % 4)))
    except Exception:
        return b""


class OpenAIResponsesImageBackend:
    """OpenAI Responses API image_generation tool backend."""

    def __init__(
        self,
        *,
        imgr,
        base_url: str,
        api_keys: list[str],
        timeout: int = 300,
        max_retries: int = 2,
        default_model: str = "gpt-5.3-codex",
        default_size: str = "auto",
        supports_edit: bool = True,
        extra_body: dict | None = None,
        proxy_url: str | None = None,
    ):
        self.imgr = imgr
        self.base_url = normalize_responses_base_url(base_url)
        self.api_keys = [str(k).strip() for k in (api_keys or []) if str(k).strip()]
        self.timeout = int(timeout or 300)
        self.max_retries = int(max_retries or 2)
        self.default_model = str(default_model or "").strip()
        self.default_size = normalize_size_text(default_size) or "auto"
        self.supports_edit = bool(supports_edit)
        self.extra_body = extra_body or {}
        self.proxy_url = str(proxy_url or "").strip() or None

        self._key_index = 0
        self._client: httpx.AsyncClient | None = None

    def _responses_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/responses"

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client

        kwargs: dict = {
            "timeout": httpx.Timeout(float(self.timeout)),
            "follow_redirects": True,
        }
        if self.proxy_url:
            for proxy_kwargs in ({"proxy": self.proxy_url}, {"proxies": self.proxy_url}):
                try:
                    self._client = httpx.AsyncClient(**kwargs, **proxy_kwargs)
                    return self._client
                except TypeError:
                    continue
        self._client = httpx.AsyncClient(**kwargs)
        return self._client

    def _next_key(self) -> str:
        if not self.api_keys:
            raise RuntimeError("未配置 API Key")
        key = self.api_keys[self._key_index]
        self._key_index = (self._key_index + 1) % len(self.api_keys)
        return key

    def _resolve_tool_size(self, size: str | None, resolution: str | None) -> str:
        raw = normalize_size_text(size)
        if not raw:
            raw = _resolution_to_size(resolution)
        if not raw:
            raw = self.default_size
        if not raw or raw == "auto":
            return ""
        if raw in _SUPPORTED_IMAGE_TOOL_SIZES:
            return raw

        ratio = size_to_ratio(raw)
        if not ratio or ":" not in ratio:
            logger.warning("[ResponsesImage] 不支持的 size='%s'，改由模型默认决定", raw)
            return ""
        try:
            w, h = (int(x) for x in ratio.split(":", 1))
        except Exception:
            return ""
        if w > h:
            return "1536x1024"
        if h > w:
            return "1024x1536"
        return "1024x1024"

    def _build_tools(
        self,
        *,
        size: str | None,
        resolution: str | None,
    ) -> list[dict]:
        tool: dict = {"type": "image_generation"}
        final_size = self._resolve_tool_size(size, resolution)
        if final_size:
            tool["size"] = final_size
        return [tool]

    @staticmethod
    def _build_generate_input(prompt: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": str(prompt or ""),
                    }
                ],
            }
        ]

    @staticmethod
    def _build_edit_input(prompt: str, images: list[bytes]) -> list[dict]:
        content: list[dict] = [
            {
                "type": "input_text",
                "text": str(prompt or ""),
            }
        ]
        for image_bytes in images:
            mime, _ext = guess_image_mime_and_ext(image_bytes)
            b64 = base64.b64encode(image_bytes).decode()
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{mime};base64,{b64}",
                }
            )
        return [{"role": "user", "content": content}]

    def _build_payload(
        self,
        *,
        prompt: str,
        images: list[bytes] | None,
        model: str | None,
        size: str | None,
        resolution: str | None,
        extra_body: dict | None,
    ) -> dict:
        final_model = str(model or self.default_model or "").strip()
        if not final_model:
            raise RuntimeError("未配置 model")

        payload: dict = {
            "model": final_model,
            "input": (
                self._build_edit_input(prompt, images)
                if images is not None
                else self._build_generate_input(prompt)
            ),
            "tools": self._build_tools(size=size, resolution=resolution),
            "tool_choice": {"type": "image_generation"},
        }
        payload.update(self.extra_body)
        payload.update(extra_body or {})
        return payload

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        return status_code in {408, 409, 425, 429} or status_code >= 500

    async def _post_responses(self, key: str, payload: dict, *, log_tag: str) -> dict:
        client = self._get_client()
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        url = self._responses_url()
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                resp = await client.post(url, headers=headers, json=payload)
                body_text = resp.text
                if resp.status_code < 200 or resp.status_code >= 300:
                    message = (
                        f"responses request failed HTTP {resp.status_code}: "
                        f"{body_text[:500]}"
                    )
                    if (
                        attempt < self.max_retries
                        and self._is_retryable_status(resp.status_code)
                    ):
                        last_error = RuntimeError(message)
                        await asyncio.sleep(0.5 * (2**attempt))
                        continue
                    raise RuntimeError(message)

                try:
                    data = resp.json()
                except Exception as e:
                    raise RuntimeError(f"responses 返回非 JSON: {body_text[:500]}") from e

                logger.info(
                    "[ResponsesImage][%s] API 响应耗时: %.2fs",
                    log_tag,
                    time.time() - t0,
                )
                return data
            except Exception as e:
                last_error = e
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(0.5 * (2**attempt))

        raise RuntimeError(f"responses API 调用失败: {last_error}") from last_error

    async def _resolve_awaitable(self, value: object) -> object:
        while inspect.isawaitable(value):
            value = await value
        return value

    def _iter_image_candidates(self, obj: object):
        if obj is None:
            return
        if isinstance(obj, str):
            yield obj
            return
        if isinstance(obj, dict):
            item_type = str(obj.get("type") or "").strip()
            if item_type == "image_generation_call":
                for key in (
                    "result",
                    "image",
                    "data",
                    "url",
                    "image_url",
                    "b64_json",
                    "b64",
                    "base64",
                    "partial_image_b64",
                    "partial_image",
                ):
                    if key in obj:
                        yield from self._iter_image_candidates(obj.get(key))
            for key in (
                "b64_json",
                "b64",
                "base64",
                "image_base64",
                "image_b64",
                "partial_image_b64",
                "partial_image",
                "url",
                "image_url",
                "file_url",
                "text",
            ):
                if key in obj:
                    yield from self._iter_image_candidates(obj.get(key))
            for key in ("output", "content", "result", "data", "images", "message"):
                if key in obj:
                    yield from self._iter_image_candidates(obj.get(key))
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                yield from self._iter_image_candidates(item)

    @staticmethod
    def _response_debug_snippet(response: object) -> str:
        def scrub(value: object) -> object:
            if isinstance(value, dict):
                out: dict = {}
                for key, inner in value.items():
                    key_text = str(key)
                    if key_text.lower() in {
                        "result",
                        "b64_json",
                        "b64",
                        "base64",
                        "image_base64",
                        "image_b64",
                        "partial_image_b64",
                        "image_url",
                    }:
                        text = str(inner or "")
                        out[key_text] = f"<redacted len={len(text)}>"
                    else:
                        out[key_text] = scrub(inner)
                return out
            if isinstance(value, list):
                return [scrub(item) for item in value[:6]]
            if isinstance(value, str) and len(value) > 300:
                return f"{value[:300]}..."
            return value

        try:
            return json.dumps(scrub(response), ensure_ascii=False)[:1200]
        except Exception:
            return str(response)[:1200]

    @staticmethod
    def _extract_ref_from_text(text: str) -> str:
        s = str(text or "").strip()
        if not s:
            return ""
        if s.startswith(("http://", "https://", "data:image/", "base64://")):
            return s
        m = _MARKDOWN_IMAGE_RE.search(s)
        if m:
            return m.group(1).strip().strip('"').strip("'")
        m = _DATA_IMAGE_RE.search(s)
        if m:
            return m.group(1).strip()
        m = _IMAGE_URL_RE.search(s)
        if m:
            return m.group(1).strip()
        return ""

    async def _save_response_image(self, response: dict) -> Path:
        response = await self._resolve_awaitable(response)
        for candidate in self._iter_image_candidates(response):
            ref = self._extract_ref_from_text(str(candidate or "").strip())
            if not ref:
                ref = str(candidate or "").strip()
            if not ref:
                continue
            if ref.startswith(("http://", "https://")):
                return await self.imgr.download_image(ref)
            if ref.startswith("data:image/") or ref.startswith("base64://"):
                return await self.imgr.save_image(decode_base64_image_payload(ref))
            raw = _decode_base64_bytes(ref)
            if raw:
                try:
                    return await self.imgr.save_image(decode_base64_image_payload(ref))
                except Exception:
                    continue
        snippet = self._response_debug_snippet(response)
        logger.warning("[ResponsesImage] 未解析到图片，响应摘要: %s", snippet)
        raise RuntimeError("responses 未返回图片数据，请检查响应摘要")

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str | None = None,
        resolution: str | None = None,
        extra_body: dict | None = None,
    ) -> Path:
        key = self._next_key()
        payload = self._build_payload(
            prompt=prompt,
            images=None,
            model=model,
            size=size,
            resolution=resolution,
            extra_body=extra_body,
        )
        response = await self._post_responses(key, payload, log_tag="generate")
        return await self._save_response_image(response)

    async def edit(
        self,
        prompt: str,
        images: list[bytes],
        *,
        model: str | None = None,
        size: str | None = None,
        resolution: str | None = None,
        extra_body: dict | None = None,
    ) -> Path:
        if not self.supports_edit:
            raise RuntimeError("该后端不支持改图/图生图")
        if not images:
            raise ValueError("至少需要一张图片")

        key = self._next_key()
        payload = self._build_payload(
            prompt=prompt,
            images=images,
            model=model,
            size=size,
            resolution=resolution,
            extra_body=extra_body,
        )
        response = await self._post_responses(key, payload, log_tag="edit")
        return await self._save_response_image(response)
