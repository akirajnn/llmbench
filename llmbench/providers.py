"""模型呼叫層：讓同一套題目可以同時送給本地（Ollama）與雲端模型（API）。

為什麼需要這層：本地模型的分數本身沒有參照點，必須把雲端模型放進
「同一套題目、同一個 prompt、同一套評分」裡當錨點（anchor），報表才能
說出「本地模型達到雲端模型的幾成」。

支援三種 provider：
- ollama：本機 Ollama（預設）
- openai：任何 OpenAI 相容的 /chat/completions 端點
  （OpenAI、Azure OpenAI、Google Gemini 的 OpenAI 相容端點、vLLM 等）
- anthropic：Anthropic Messages API（Claude）

注意：ChatGPT / Copilot 的網頁版是「產品」，帶有隱藏系統提示與聯網工具，
結果無法重現；比較時請一律用 API 呼叫底層模型，並記錄模型版本。
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass


@dataclass
class ModelSpec:
    name: str  # 報表上顯示的名稱
    provider: str = "ollama"
    model: str = ""
    base_url: str | None = None
    api_key_env: str | None = None
    max_tokens: int = 2048

    @classmethod
    def parse(cls, entry: str | dict) -> "ModelSpec":
        """config 裡的模型可以是字串（視為 Ollama 模型）或 dict。

        dict 範例：
            {name: gpt-anchor, provider: openai, model: gpt-4o-mini,
             api_key_env: OPENAI_API_KEY}
        """
        if isinstance(entry, str):
            return cls(name=entry, provider="ollama", model=entry)
        spec = cls(
            name=entry.get("name") or entry["model"],
            provider=entry.get("provider", "ollama"),
            model=entry["model"],
            base_url=entry.get("base_url"),
            api_key_env=entry.get("api_key_env"),
            max_tokens=entry.get("max_tokens", 2048),
        )
        if spec.provider not in PROVIDERS:
            raise ValueError(f"不支援的 provider: {spec.provider}（可用：{', '.join(PROVIDERS)}）")
        return spec


@dataclass
class ChatResult:
    text: str
    output_tokens: int
    tokens_per_sec: float  # 雲端 API 沒有純生成時間，這裡用總延遲估算（含網路）


def _api_key(spec: ModelSpec, default_env: str) -> str:
    env = spec.api_key_env or default_env
    key = os.environ.get(env)
    if not key:
        raise RuntimeError(f"找不到 API key，請設定環境變數 {env}")
    return key


def _post_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _chat_ollama(spec, prompt, timeout, think, temperature) -> ChatResult:
    import ollama  # 延遲載入：只跑雲端模型時不需要安裝 ollama

    client = ollama.Client(host=spec.base_url, timeout=timeout) if spec.base_url else ollama.Client(timeout=timeout)
    kwargs: dict = {}
    if think is not None:
        kwargs["think"] = think
    if temperature is not None:
        kwargs["options"] = {"temperature": temperature}
    result = client.chat(model=spec.model, messages=[{"role": "user", "content": prompt}], **kwargs)
    eval_count = result.get("eval_count", 0) or 0
    eval_ns = result.get("eval_duration", 0) or 0
    tps = eval_count / (eval_ns / 1e9) if eval_ns else 0.0
    return ChatResult(result["message"]["content"], eval_count, tps)


def _chat_openai(spec, prompt, timeout, think, temperature) -> ChatResult:
    base = (spec.base_url or "https://api.openai.com/v1").rstrip("/")
    payload: dict = {
        "model": spec.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": spec.max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    start = time.monotonic()
    data = _post_json(
        f"{base}/chat/completions",
        payload,
        {"Authorization": f"Bearer {_api_key(spec, 'OPENAI_API_KEY')}"},
        timeout,
    )
    elapsed = time.monotonic() - start
    text = data["choices"][0]["message"].get("content") or ""
    out_tokens = (data.get("usage") or {}).get("completion_tokens", 0) or 0
    return ChatResult(text, out_tokens, out_tokens / elapsed if elapsed else 0.0)


def _chat_anthropic(spec, prompt, timeout, think, temperature) -> ChatResult:
    base = (spec.base_url or "https://api.anthropic.com").rstrip("/")
    payload: dict = {
        "model": spec.model,
        "max_tokens": spec.max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        payload["temperature"] = temperature
    start = time.monotonic()
    data = _post_json(
        f"{base}/v1/messages",
        payload,
        {"x-api-key": _api_key(spec, "ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"},
        timeout,
    )
    elapsed = time.monotonic() - start
    text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
    out_tokens = (data.get("usage") or {}).get("output_tokens", 0) or 0
    return ChatResult(text, out_tokens, out_tokens / elapsed if elapsed else 0.0)


PROVIDERS = {
    "ollama": _chat_ollama,
    "openai": _chat_openai,
    "anthropic": _chat_anthropic,
}


def chat(spec: ModelSpec, prompt: str, timeout: int, think: bool | None, temperature: float | None) -> ChatResult:
    return PROVIDERS[spec.provider](spec, prompt, timeout, think, temperature)
