"""Fábrica central de LLM. El proveedor se elige con LLM_PROVIDER en el .env."""

import os

_SUPPORTED = ("bedrock", "anthropic", "openai", "gemini", "ollama")


def get_llm(temperature: float = 0.1, max_tokens: int = 4096):
    """
    Devuelve el LLM configurado en LLM_PROVIDER.
    Lanza ValueError si el proveedor no está soportado o faltan credenciales.
    """
    provider = os.getenv("LLM_PROVIDER", "bedrock").lower().strip()

    if provider == "bedrock":
        from langchain_aws import ChatBedrock
        model_id = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        region   = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        _require("AWS_ACCESS_KEY_ID", provider)
        _require("AWS_SECRET_ACCESS_KEY", provider)
        return ChatBedrock(
            model_id=model_id,
            model_kwargs={"temperature": temperature, "max_tokens": max_tokens},
            region_name=region,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        _require("ANTHROPIC_API_KEY", provider)
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL_ID", "claude-opus-4-8"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        _require("OPENAI_API_KEY", provider)
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_ID", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        _require("GEMINI_API_KEY", provider)
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL_ID", "gemini-2.0-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL_ID", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )

    raise ValueError(
        f"LLM_PROVIDER='{provider}' no reconocido.\n"
        f"Opciones válidas: {', '.join(_SUPPORTED)}"
    )


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
