"""Fábrica central de LLM. El proveedor se elige con LLM_PROVIDER en el .env."""

import os

_SUPPORTED = ("bedrock", "anthropic", "openai", "azure", "gemini", "ollama")

# Modelos de razonamiento: solo aceptan el temperature por defecto (1) y
# consumen parte de max_tokens en "pensar" antes de escribir la salida visible.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning_model(model_name: str) -> bool:
    name = (model_name or "").lower()
    return any(name.startswith(p) for p in _REASONING_MODEL_PREFIXES)


def _supports_custom_temperature(model_name: str) -> bool:
    return not _is_reasoning_model(model_name)


def get_llm(temperature: float = 0.0, max_tokens: int = 4096, model_name: str = None,
            reasoning_effort: str = None):
    """
    Devuelve el LLM configurado en LLM_PROVIDER.
    Lanza ValueError si el proveedor no está soportado o faltan credenciales.

    reasoning_effort: solo aplica a modelos de razonamiento (gpt-5, o1, o3, o4).
    Valores válidos: "minimal" | "low" | "medium" | "high". Si no se especifica
    y el modelo es de razonamiento, se usa "minimal" por defecto para dejar el
    máximo espacio de max_tokens a la salida visible (evita respuestas vacías).
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
        # Use the ChatAnthropic class from langchain.chat_models to avoid
        # unresolved-import diagnostics for the separate package name.
        from langchain.chat_models import ChatAnthropic
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
        model_id = os.getenv("OPENAI_MODEL_ID", "gpt-5")
        kwargs = dict(
            model=model_id,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=max_tokens,
        )
        if _supports_custom_temperature(model_id):
            kwargs["temperature"] = temperature
            return ChatOpenAI(**kwargs)
        try:
            return ChatOpenAI(**kwargs, reasoning_effort=reasoning_effort or "minimal")
        except TypeError:
            # Versión de langchain-openai sin soporte para reasoning_effort todavía.
            return ChatOpenAI(**kwargs)

    if provider == "azure":
        # v1 API (GA desde ago-2025): sin api-version fecha-dependiente, con
        # ChatOpenAI apuntando a <endpoint>/openai/v1/. Ver notas de la migración
        # en el hilo del 27-jul-2026 -- el 404 anterior venía de un api-version
        # inválido ("2026-03-17" era la versión del MODELO, no de la REST API).
        from langchain_openai import ChatOpenAI
        _require("AZURE_OPENAI_API_KEY", provider)
        _require("AZURE_OPENAI_ENDPOINT", provider)
        _require("AZURE_OPENAI_DEPLOYMENT", provider)
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip("/")
        kwargs = dict(
            model=deployment,  # nombre del deployment, no del modelo base
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=f"{endpoint}/openai/v1/",
            max_tokens=max_tokens,
        )
        if _supports_custom_temperature(deployment):
            kwargs["temperature"] = temperature
            return ChatOpenAI(**kwargs)
        try:
            return ChatOpenAI(**kwargs, reasoning_effort=reasoning_effort or "minimal")
        except TypeError:
            # Versión de langchain-openai sin soporte para reasoning_effort todavía.
            return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        _require("GEMINI_API_KEY", provider)
        
        # Decide qué modelo usar: el que le pasamos o el del .env
        final_model = model_name if model_name else os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")
        
        return ChatGoogleGenerativeAI(
            model=final_model,
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
            num_predict=max_tokens,
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "2048")),
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
    if provider == "azure":
        return os.getenv("AZURE_OPENAI_DEPLOYMENT", "azure-default")
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