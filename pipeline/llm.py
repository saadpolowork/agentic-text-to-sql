from __future__ import annotations
"""
llm.py — Groq inference (OpenAI-compatible chat completions) with 429 backoff.
"""

import json
import os
import time

import requests

# Default: Groq free tier (any OpenAI-compatible endpoint works — override via env
# to swap providers, e.g. LLM_BASE_URL=https://api.anthropic.com/v1)
GROQ_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")

# Trial 12 model assignments (overridable via env for other providers)
STAGE1_MODEL = os.environ.get("LLM_STAGE1_MODEL", "qwen/qwen3-32b")
STAGE2_MODEL = os.environ.get("LLM_STAGE2_MODEL", "qwen/qwen3-32b")
STAGE3_MODEL = os.environ.get("LLM_STAGE3_MODEL", "llama-3.3-70b-versatile")

MAX_TOKENS = 8192
TEMPERATURE = 0.3


class LLMError(Exception):
    pass


class RateLimitedError(LLMError):
    def __init__(self, retry_after: float, message: str = ""):
        self.retry_after = retry_after
        super().__init__(message or f"Rate limited; retry after {retry_after:.0f}s")


def call_llm(system_prompt: str, user_prompt: str, model: str, api_key: str,
             max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE,
             max_retries: int = 3, request_timeout: int = 300) -> str:
    """Single chat completion. Retries on 429/5xx with backoff. Returns raw content."""
    url = f"{GROQ_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload,
                                 timeout=request_timeout)
        except requests.RequestException as e:
            last_err = LLMError(f"Network error calling Groq: {e}")
            time.sleep(2 * (attempt + 1))
            continue

        if resp.status_code == 200:
            try:
                data = resp.json()
                return data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                raise LLMError(f"Unexpected Groq response shape: {e}")

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("retry-after", 15))
            if attempt < max_retries and retry_after <= 60:
                time.sleep(retry_after + 1)
                continue
            raise RateLimitedError(retry_after)

        if resp.status_code in (500, 502, 503):
            last_err = LLMError(f"Groq server error {resp.status_code}")
            time.sleep(3 * (attempt + 1))
            continue

        # 400/401/404 etc — no point retrying
        try:
            detail = resp.json().get("error", {}).get("message", resp.text[:300])
        except Exception:
            detail = resp.text[:300]
        raise LLMError(f"Groq API error {resp.status_code}: {detail}")

    raise last_err or LLMError("Groq call failed after retries")


def list_models(api_key: str) -> list:
    """Available model ids for this key (used for a startup sanity check)."""
    resp = requests.get(f"{GROQ_BASE_URL}/models",
                        headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", [])]
