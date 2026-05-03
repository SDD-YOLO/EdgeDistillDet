from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import requests


def _normalize_base_url(api_url: str) -> str:
    return str(api_url or "").strip().rstrip("/")


def _infer_endpoint(api_url: str, endpoint: str | None = None) -> str:
    custom = str(endpoint or "").strip()
    if custom:
        return custom
    base = _normalize_base_url(api_url)
    if base.endswith("/chat/completions") or base.endswith("/responses") or base.endswith("/messages"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    if "/api/v" in base:
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _extract_text_and_reasoning(payload: dict[str, Any]) -> tuple[str, str]:
    reply = ""
    reasoning = ""
    if isinstance(payload.get("reasoning"), str):
        reasoning = payload["reasoning"].strip()
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") or {}
        if isinstance(msg.get("content"), str):
            reply = msg["content"]
        elif isinstance(msg.get("content"), list):
            parts: list[str] = []
            for item in msg["content"]:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            reply = "".join(parts)
        if not reasoning:
            rc = msg.get("reasoning_content") or msg.get("reasoning")
            if isinstance(rc, str):
                reasoning = rc.strip()
    if not reply:
        fallback = payload.get("reply") or payload.get("message") or payload.get("output")
        if isinstance(fallback, str):
            reply = fallback
        elif fallback is not None:
            reply = json.dumps(fallback, ensure_ascii=False)
    return reply, reasoning


def _build_headers(api_key: str | None, extra_headers: dict[str, Any] | None = None) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    token = str(api_key or "").strip()
    if token:
        headers["Authorization"] = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    if isinstance(extra_headers, dict):
        for k, v in extra_headers.items():
            if isinstance(k, str) and isinstance(v, str):
                headers[k] = v
    return headers


def invoke_chat_completion(
    *,
    api_url: str,
    api_key: str | None,
    model: str | None,
    messages: list[dict[str, str]],
    system_prompt: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    endpoint: str | None = None,
    timeout_sec: float = 40.0,
    extra_headers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = _infer_endpoint(api_url, endpoint)
    base_messages = list(messages or [])
    if system_prompt:
        base_messages = [{"role": "system", "content": system_prompt}, *base_messages]
    body: dict[str, Any] = {
        "model": model or "gpt-4o-mini",
        "messages": base_messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    headers = _build_headers(api_key, extra_headers)
    try:
        resp = requests.post(target, headers=headers, json=body, timeout=float(timeout_sec))
        raw = resp.text
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {raw}")
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(str(exc)) from exc
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"reply": raw}
    reply, reasoning = _extract_text_and_reasoning(payload)
    return {
        "status": "ok",
        "reply": reply,
        "reasoning": reasoning or None,
        "raw": payload,
    }


def stream_chat_completion(
    *,
    api_url: str,
    api_key: str | None,
    model: str | None,
    messages: list[dict[str, str]],
    system_prompt: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    endpoint: str | None = None,
    timeout_sec: float = 40.0,
    extra_headers: dict[str, Any] | None = None,
) -> Generator[dict[str, Any], None, dict[str, str]]:
    target = _infer_endpoint(api_url, endpoint)
    base_messages = list(messages or [])
    if system_prompt:
        base_messages = [{"role": "system", "content": system_prompt}, *base_messages]
    body: dict[str, Any] = {
        "model": model or "gpt-4o-mini",
        "messages": base_messages,
        "temperature": temperature,
        "stream": True,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    headers = _build_headers(api_key, extra_headers)
    headers.setdefault("Connection", "close")
    reply = ""
    reasoning = ""

    def _fallback_non_stream() -> dict[str, str]:
        non_stream_body = dict(body)
        non_stream_body["stream"] = False
        resp = requests.post(target, headers=headers, json=non_stream_body, timeout=float(timeout_sec))
        raw = resp.text
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {raw}")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"reply": raw}
        out_reply, out_reasoning = _extract_text_and_reasoning(payload)
        return {"reply": out_reply, "reasoning": out_reasoning}

    try:
        with requests.post(target, headers=headers, json=body, timeout=float(timeout_sec), stream=True) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
            for raw_line in resp.iter_lines(decode_unicode=False):
                if isinstance(raw_line, bytes | bytearray):
                    try:
                        line = raw_line.decode("utf-8").strip()
                    except UnicodeDecodeError:
                        line = raw_line.decode("latin-1", errors="replace").strip()
                else:
                    line = str(raw_line or "").strip()
                if not line.startswith("data:"):
                    continue
                payload_str = line[5:].strip()
                if not payload_str:
                    continue
                if payload_str == "[DONE]":
                    break
                try:
                    event = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content_delta = delta.get("content")
                reason_delta = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(content_delta, str) and content_delta:
                    reply += content_delta
                    yield {"t": "content", "d": content_delta}
                if isinstance(reason_delta, str) and reason_delta:
                    reasoning += reason_delta
                    yield {"t": "reasoning", "d": reason_delta}
    except requests.exceptions.SSLError:
        fallback = _fallback_non_stream()
        reply = fallback.get("reply") or ""
        reasoning = fallback.get("reasoning") or ""
        if reply:
            yield {"t": "content", "d": reply}
        if reasoning:
            yield {"t": "reasoning", "d": reasoning}
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(str(exc)) from exc
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc
    return {"reply": reply, "reasoning": reasoning}
