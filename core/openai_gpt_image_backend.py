from __future__ import annotations

import asyncio
import base64
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from astrbot.api import logger

from .gitee_sizes import normalize_size_text
from .image_format import guess_image_mime_and_ext


def _normalize_base_url(raw: str) -> str:
    s = str(raw or "").strip().rstrip("/")
    if not s:
        return ""

    lower = s.lower()
    for suffix in (
        "/v1/images/generations",
        "/images/generations",
        "/v1/images/edits",
        "/images/edits",
        "/v1/images",
        "/images",
    ):
        if lower.endswith(suffix):
            s = s[: -len(suffix)].rstrip("/")
            break

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
    if not r or r == "AUTO":
        return ""
    if r in {"1K", "1024"}:
        return "1024x1024"
    if r in {"2K", "2048"}:
        return "2048x2048"
    if r in {"4K", "4096"}:
        return "4096x4096"
    if re.fullmatch(r"\d{2,5}X\d{2,5}", r):
        return r.lower()
    return ""


def _decode_base64_image(text: str) -> bytes:
    s = re.sub(r"\s+", "", str(text or "").strip())
    if not s:
        return b""
    if s.startswith("data:image/"):
        _header, _sep, s = s.partition(",")
    pad = "=" * ((4 - len(s) % 4) % 4)
    for candidate in (s, s.replace("-", "+").replace("_", "/")):
        try:
            out = base64.b64decode(candidate + pad, validate=False)
            if out:
                return out
        except Exception:
            continue
    return b""


class OpenAIGPTImageBackend:
    """REST backend for OpenAI gpt-image-2 Images API.

    The official image edits endpoint accepts multiple reference images as
    repeated multipart fields. The generic OpenAI SDK backend keeps one-image
    compatibility by packing multiple inputs into a collage; this backend keeps
    each reference image separate for identity/reference-heavy workflows.
    """

    def __init__(
        self,
        *,
        imgr,
        base_url: str,
        api_keys: list[str],
        default_model: str = "gpt-image-2",
        default_size: str = "auto",
        quality: str = "auto",
        output_format: str = "png",
        output_compression: int | None = None,
        moderation: str = "auto",
        supports_edit: bool = True,
        timeout: int = 300,
        max_retries: int = 2,
        max_input_images: int = 16,
        extra_body: dict | None = None,
        proxy_url: str | None = None,
    ):
        self.imgr = imgr
        self.base_url = _normalize_base_url(base_url or "https://api.openai.com/v1")
        self.api_keys = [str(k).strip() for k in (api_keys or []) if str(k).strip()]
        self.default_model = str(default_model or "gpt-image-2").strip()
        self.default_size = self._normalize_size(default_size) or "auto"
        self.quality = self._normalize_choice(
            quality,
            allowed={"auto", "low", "medium", "high"},
            default="auto",
        )
        self.output_format = self._normalize_choice(
            output_format,
            allowed={"png", "jpeg", "webp"},
            default="png",
        )
        self.output_compression = self._normalize_compression(output_compression)
        self.moderation = self._normalize_choice(
            moderation,
            allowed={"auto", "low"},
            default="auto",
        )
        self.supports_edit = bool(supports_edit)
        self.timeout = max(1, int(timeout or 300))
        self.max_retries = max(0, int(max_retries or 0))
        self.max_input_images = max(1, min(16, int(max_input_images or 16)))
        self.extra_body = extra_body if isinstance(extra_body, dict) else {}
        self.proxy_url = str(proxy_url or "").strip() or None

        self._key_index = 0
        self._client: httpx.AsyncClient | None = None

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        kwargs: dict[str, Any] = {
            "timeout": float(self.timeout),
            "follow_redirects": True,
        }
        if self.proxy_url:
            kwargs["proxy"] = self.proxy_url
        self._client = httpx.AsyncClient(**kwargs)
        return self._client

    def _next_key(self) -> str:
        if not self.api_keys:
            raise RuntimeError("未配置 API Key")
        key = self.api_keys[self._key_index]
        self._key_index = (self._key_index + 1) % len(self.api_keys)
        return key

    @staticmethod
    def _normalize_choice(value: Any, *, allowed: set[str], default: str) -> str:
        text = str(value or "").strip().lower()
        return text if text in allowed else default

    @staticmethod
    def _normalize_compression(value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            num = int(value)
        except (TypeError, ValueError):
            return None
        return max(0, min(100, num))

    @staticmethod
    def _normalize_size(value: Any) -> str:
        text = str(value or "").strip()
        if text.lower() == "auto":
            return "auto"
        return normalize_size_text(text)

    def _resolve_size(self, size: str | None, resolution: str | None) -> str:
        return (
            self._normalize_size(size)
            or self._normalize_size(_resolution_to_size(resolution))
            or self.default_size
            or "auto"
        )

    def _endpoint(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _common_payload(
        self,
        prompt: str,
        *,
        model: str | None,
        size: str | None,
        resolution: str | None,
        extra_body: dict | None,
    ) -> dict[str, Any]:
        final_model = str(model or self.default_model or "").strip()
        if not final_model:
            raise RuntimeError("未配置 model")

        payload: dict[str, Any] = {
            "model": final_model,
            "prompt": str(prompt or "").strip() or "a high quality image",
            "size": self._resolve_size(size, resolution),
            "quality": self.quality,
            "output_format": self.output_format,
            "moderation": self.moderation,
        }
        if self.output_format != "png" and self.output_compression is not None:
            payload["output_compression"] = self.output_compression

        if self.extra_body:
            payload.update(self.extra_body)
        if isinstance(extra_body, dict):
            payload.update(extra_body)

        return {k: v for k, v in payload.items() if v is not None and v != ""}

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Cache-Control": "no-store, no-cache, max-age=0",
            "Pragma": "no-cache",
        }

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        return status_code == 429 or status_code >= 500

    async def _post(self, url: str, **kwargs) -> httpx.Response:
        client = self._get_client()
        attempts = self.max_retries + 1
        last_exc: Exception | None = None

        for attempt in range(attempts):
            try:
                resp = await client.post(url, **kwargs)
                if (
                    self._is_retryable_status(resp.status_code)
                    and attempt + 1 < attempts
                ):
                    await asyncio.sleep(min(2.0, 0.4 * (2**attempt)))
                    continue
                return resp
            except Exception as exc:
                last_exc = exc
                if attempt + 1 < attempts:
                    await asyncio.sleep(min(2.0, 0.4 * (2**attempt)))
                    continue

        raise RuntimeError(
            f"请求失败（已重试 {self.max_retries} 次）: {last_exc}"
        ) from last_exc

    async def _save_response(self, resp: httpx.Response) -> Path:
        if resp.status_code != 200:
            message = ""
            try:
                data = resp.json()
                if isinstance(data, dict):
                    err = data.get("error")
                    if isinstance(err, dict):
                        message = str(err.get("message") or "")
                    message = message or str(data.get("message") or "")
            except Exception:
                message = ""
            if not message:
                message = str(getattr(resp, "text", "") or "")[:300]
            raise RuntimeError(f"请求失败 HTTP {resp.status_code}: {message}")

        content_type = str(resp.headers.get("content-type") or "").lower()
        if content_type.startswith("image/"):
            return await self.imgr.save_image(resp.content)

        try:
            payload = resp.json()
        except Exception as exc:
            raise RuntimeError(
                f"返回内容不是有效 JSON: {str(getattr(resp, 'text', '') or '')[:200]}"
            ) from exc

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or not data:
            raise RuntimeError("接口未返回图片数据")

        first = data[0]
        if isinstance(first, dict):
            b64_json = first.get("b64_json")
            if isinstance(b64_json, str) and b64_json.strip():
                image_bytes = _decode_base64_image(b64_json)
                if not image_bytes:
                    raise RuntimeError("返回图片 base64 解码失败")
                return await self.imgr.save_image(image_bytes)

            url = str(first.get("url") or "").strip()
            if url:
                return await self.imgr.download_image(url)

        raise RuntimeError(f"接口未返回可用图片数据: {str(payload)[:200]}")

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str | None = None,
        resolution: str | None = None,
        extra_body: dict | None = None,
    ) -> Path:
        payload = self._common_payload(
            prompt,
            model=model,
            size=size,
            resolution=resolution,
            extra_body=extra_body,
        )

        key = self._next_key()
        t0 = time.perf_counter()
        resp = await self._post(
            self._endpoint("images/generations"),
            headers={**self._headers(key), "Content-Type": "application/json"},
            json=payload,
        )
        out = await self._save_response(resp)
        logger.info("[OpenAIGPTImage][generate] 耗时: %.2fs", time.perf_counter() - t0)
        return out

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

        payload = self._common_payload(
            prompt or "Edit this image",
            model=model,
            size=size,
            resolution=resolution,
            extra_body=extra_body,
        )
        form_data = {str(k): str(v) for k, v in payload.items()}

        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for i, image_bytes in enumerate(images[: self.max_input_images]):
            if not image_bytes:
                continue
            mime, ext = guess_image_mime_and_ext(image_bytes)
            files.append(
                (
                    "image[]",
                    (f"input-{i + 1}.{ext}", image_bytes, mime),
                )
            )

        if not files:
            raise ValueError("没有可用的输入图片")

        key = self._next_key()
        t0 = time.perf_counter()
        resp = await self._post(
            self._endpoint("images/edits"),
            headers=self._headers(key),
            data=form_data,
            files=files,
        )
        out = await self._save_response(resp)
        logger.info("[OpenAIGPTImage][edit] 耗时: %.2fs", time.perf_counter() - t0)
        return out
