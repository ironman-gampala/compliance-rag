"""Call Groq, Gemini, or a local Ollama model."""

import json
import os
import time
import urllib.error
import urllib.request

from rag.config import LLM_PROVIDER


class LLMError(RuntimeError):
    pass


def complete(system: str, user: str) -> str:
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    if provider == "groq":
        return _groq(system, user)
    if provider == "gemini":
        return _gemini(system, user)
    if provider == "ollama":
        return _ollama(system, user)
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use groq, gemini, or ollama.")


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise LLMError(f"{name} is not set. Add it to .env (see .env.example).")
    return value


def _groq(system: str, user: str) -> str:
    from groq import Groq

    client = Groq(api_key=_require("GROQ_API_KEY"))
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except Exception as exc:
        raise LLMError(f"Groq request failed: {exc}") from exc
    content = response.choices[0].message.content
    if not content:
        raise LLMError("Groq returned an empty answer.")
    return content


def _gemini(system: str, user: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=_require("GEMINI_API_KEY"),
        http_options={"timeout": 60_000},
    )
    model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    response = None
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0,
                ),
            )
            break
        except Exception as exc:
            transient = any(code in str(exc) for code in ("504", "503", "DEADLINE", "UNAVAILABLE"))
            if attempt == 0 and transient:
                time.sleep(2)
                continue
            raise LLMError(f"Gemini request failed: {exc}") from exc
    text = getattr(response, "text", None)
    if not text:
        raise LLMError("Gemini returned an empty answer.")
    return text


def _ollama(system: str, user: str) -> str:
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    request = urllib.request.Request(
        host + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise LLMError(
            f"Could not reach Ollama at {host}. Is it running, and is '{model}' pulled? ({exc})"
        ) from exc
    content = (body.get("message") or {}).get("content")
    if not content:
        raise LLMError("Ollama returned an empty answer.")
    return content
