"""Provider-neutral LLM factory selected through ``LLM_PROVIDER``."""

import os
import time
from typing import Any

_SUPPORTED = ("bedrock", "anthropic", "openai", "gemini", "ollama")


def _request_timeout() -> float:
    raw = os.getenv("LLM_REQUEST_TIMEOUT", "300").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("LLM_REQUEST_TIMEOUT must be a positive number") from exc
    if value <= 0:
        raise ValueError("LLM_REQUEST_TIMEOUT must be a positive number")
    return value


def get_llm(
    temperature: float = 0.1,
    max_tokens: int = 4096,
    *,
    reasoning_mode: str = "provider_default",
    structured_output: bool = False,
):
    """
    Devuelve el LLM configurado en LLM_PROVIDER.
    Lanza ValueError si el proveedor no está soportado o faltan credenciales.
    """
    provider = os.getenv("LLM_PROVIDER", "bedrock").lower().strip()
    timeout = _request_timeout()
    reasoning_mode = reasoning_mode.lower().strip()
    if reasoning_mode not in {"provider_default", "disabled", "enabled"}:
        raise ValueError(
            "reasoning_mode must be provider_default, disabled, or enabled"
        )

    if provider == "bedrock":
        from botocore.config import Config
        from langchain_aws import ChatBedrock
        model_id = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        region   = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        _require("AWS_ACCESS_KEY_ID", provider)
        _require("AWS_SECRET_ACCESS_KEY", provider)
        return ChatBedrock(
            model_id=model_id,
            model_kwargs={"temperature": temperature, "max_tokens": max_tokens},
            region_name=region,
            config=Config(
                connect_timeout=min(10.0, timeout),
                read_timeout=timeout,
            ),
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        _require("ANTHROPIC_API_KEY", provider)
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL_ID", "claude-opus-4-8"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        _require("OPENAI_API_KEY", provider)
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_ID", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        _require("GEMINI_API_KEY", provider)
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL_ID", "gemini-2.0-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
            max_output_tokens=max_tokens,
            timeout=timeout,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        ollama_kwargs = {
            "model": os.getenv("OLLAMA_MODEL_ID", "llama3.2"),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "temperature": temperature,
            "num_predict": max_tokens,
            "client_kwargs": {"timeout": timeout},
            "async_client_kwargs": {"timeout": timeout},
        }
        if reasoning_mode != "provider_default":
            ollama_kwargs["reasoning"] = reasoning_mode == "enabled"
        if structured_output:
            ollama_kwargs["format"] = "json"
        seed = os.getenv("OLLAMA_SEED", "").strip()
        if seed:
            try:
                ollama_kwargs["seed"] = int(seed)
            except ValueError as exc:
                raise ValueError("OLLAMA_SEED must be an integer") from exc
        return ChatOllama(**ollama_kwargs)

    raise ValueError(
        f"LLM_PROVIDER='{provider}' no reconocido.\n"
        f"Opciones válidas: {', '.join(_SUPPORTED)}"
    )


def get_judge_llm(temperature: float = 0.0, max_tokens: int = 4096):
    """Create an evaluator LLM while keeping generation policy independent."""
    mode = os.getenv("LLM_JUDGE_REASONING_MODE", "disabled")
    return get_llm(
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_mode=mode,
        structured_output=True,
    )


def invoke_llm_text(
    llm: Any,
    prompt: str,
    *,
    operation: str,
    attempt: int = 1,
) -> str:
    """Invoke any configured provider with consistent attempt observability."""
    started = time.perf_counter()
    try:
        response = llm.invoke(prompt)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(
            f"      [LLM] {operation} attempt {attempt} failed after "
            f"{elapsed:.2f}s: {type(exc).__name__}",
            flush=True,
        )
        raise
    elapsed = time.perf_counter() - started
    print(
        f"      [LLM] {operation} attempt {attempt} completed in {elapsed:.2f}s",
        flush=True,
    )
    return str(getattr(response, "content", response))


def active_model_name() -> str:
    """Nombre legible del modelo activo — para logs y reportes."""
    provider = os.getenv("LLM_PROVIDER", "bedrock").lower().strip()
    if provider == "bedrock":
        return os.getenv("BEDROCK_MODEL_ID", "bedrock-default")
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL_ID", "claude-default")
    if provider == "openai":
        return os.getenv("OPENAI_MODEL_ID", "gpt-default")
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL_ID", "gemini-2.0-flash")
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL_ID", "ollama-default")
    return provider


def _require(var: str, provider: str) -> None:
    if not os.getenv(var):
        raise EnvironmentError(
            f"Falta {var} en .env para usar LLM_PROVIDER={provider}"
        )
