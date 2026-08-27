"""Provider-neutral LLM factory selected through ``LLM_PROVIDER``."""

import asyncio
import concurrent.futures
import os
import threading
import time
from typing import Any

_SUPPORTED = ("bedrock", "anthropic", "openai", "gemini", "ollama")


class LLMInvocationTimeout(TimeoutError):
    """Raised when a complete LLM operation exceeds its wall-clock deadline."""

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"LLM operation exceeded its {timeout_seconds:.2f}s deadline"
        )


async def _ainvoke_with_deadline(llm: Any, prompt: str, timeout_seconds: float):
    try:
        return await asyncio.wait_for(
            llm.ainvoke(prompt),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise LLMInvocationTimeout(timeout_seconds) from exc


class _AsyncDeadlineRunner:
    """Own one persistent event loop for cancellable provider invocations."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()
        self.ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.ready.set()
        self.loop.run_forever()

    def invoke(self, llm: Any, prompt: str, timeout_seconds: float):
        future = asyncio.run_coroutine_threadsafe(
            _ainvoke_with_deadline(llm, prompt, timeout_seconds),
            self.loop,
        )
        try:
            return future.result(timeout=timeout_seconds + 1.0)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise LLMInvocationTimeout(timeout_seconds) from exc


_DEADLINE_RUNNER: _AsyncDeadlineRunner | None = None
_DEADLINE_RUNNER_LOCK = threading.Lock()


def _deadline_runner() -> _AsyncDeadlineRunner:
    global _DEADLINE_RUNNER
    if _DEADLINE_RUNNER is None:
        with _DEADLINE_RUNNER_LOCK:
            if _DEADLINE_RUNNER is None:
                _DEADLINE_RUNNER = _AsyncDeadlineRunner()
    return _DEADLINE_RUNNER


def _invoke_with_deadline(llm: Any, prompt: str, timeout_seconds: float):
    """Run a provider's async invocation under one hard wall-clock deadline."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    return _deadline_runner().invoke(llm, prompt, timeout_seconds)


def _request_timeout(override: float | None = None) -> float:
    raw = str(override) if override is not None else os.getenv(
        "LLM_REQUEST_TIMEOUT", "210"
    ).strip()
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
    request_timeout: float | None = None,
):
    """
    Devuelve el LLM configurado en LLM_PROVIDER.
    Lanza ValueError si el proveedor no está soportado o faltan credenciales.
    """
    provider = os.getenv("LLM_PROVIDER", "bedrock").lower().strip()
    timeout = _request_timeout(request_timeout)
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


def get_generation_llm(
    temperature: float = 0.1,
    max_tokens: int = 4096,
    *,
    request_timeout: float | None = None,
):
    """Create the roadmap LLM with reasoning controlled independently from judges."""
    mode = os.getenv("LLM_GENERATION_REASONING_MODE", "provider_default")
    return get_llm(
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_mode=mode,
        request_timeout=request_timeout,
    )


def get_query_preprocessing_llm(
    temperature: float = 0.1,
    max_tokens: int = 4096,
):
    """Create a fast LLM for intent classification and query refinement."""
    return get_llm(
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_mode="disabled",
    )


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _response_text(response: Any) -> str:
    """Normalize text content without depending on a provider message class."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(getattr(block, "text", None), str):
                parts.append(block.text)
        if parts:
            return "".join(parts)
    return str(content)


def invoke_llm_with_trace(
    llm: Any,
    prompt: str,
    *,
    operation: str,
    attempt: int = 1,
    timeout_seconds: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Invoke any provider and retain portable completion diagnostics."""
    started = time.perf_counter()
    try:
        response = (
            _invoke_with_deadline(llm, prompt, timeout_seconds)
            if timeout_seconds is not None
            else llm.invoke(prompt)
        )
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
    text = _response_text(response)
    metadata = _as_mapping(getattr(response, "response_metadata", {}))
    usage = _as_mapping(getattr(response, "usage_metadata", {}))
    token_usage = _as_mapping(metadata.get("token_usage", {}))
    trace = {
        "elapsed_s": round(elapsed, 4),
        "response_characters": len(text),
        "finish_reason": (
            metadata.get("finish_reason")
            or metadata.get("stop_reason")
            or metadata.get("done_reason")
        ),
        "input_tokens": (
            usage.get("input_tokens")
            or token_usage.get("input_tokens")
            or token_usage.get("prompt_tokens")
            or metadata.get("prompt_eval_count")
        ),
        "output_tokens": (
            usage.get("output_tokens")
            or token_usage.get("output_tokens")
            or token_usage.get("completion_tokens")
            or metadata.get("eval_count")
        ),
    }
    return text, trace


def invoke_llm_text(
    llm: Any,
    prompt: str,
    *,
    operation: str,
    attempt: int = 1,
) -> str:
    """Invoke any configured provider with consistent attempt observability."""
    text, _ = invoke_llm_with_trace(
        llm,
        prompt,
        operation=operation,
        attempt=attempt,
    )
    return text


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
