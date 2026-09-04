# NotionMap

## Ingesta automática de Notion

Este proceso sincroniza páginas de Notion con el vector store local de Chroma utilizado por el proyecto.

### 1. Preparar la integración

1. Crea una integración en Notion y copia su token.
2. Otorga a la integración acceso explícito a las páginas o bases de datos que deseas sincronizar.
3. Instala las dependencias del proyecto dentro de tu entorno virtual:

```bash
pip install -r requirements.txt
```

### 2. Configurar el entorno

Agrega las siguientes variables al archivo `.env`:

```env
NOTION_TOKEN=tu_token_de_integracion
NOTION_PAGE_ID=id_o_url_de_la_pagina
NOTION_DATABASE_ID=
NOTION_MAX_PAGES=
```

- `NOTION_TOKEN`: token secreto de la integración.
- `NOTION_PAGE_ID`: ID o URL de una página. Acepta varias páginas separadas por comas.
- `NOTION_DATABASE_ID`: permite sincronizar las páginas de una base de datos. Puede utilizarse en lugar de `NOTION_PAGE_ID` o junto con este.
- `NOTION_MAX_PAGES`: límite opcional de páginas, útil para pruebas rápidas.

Debes configurar al menos `NOTION_PAGE_ID` o `NOTION_DATABASE_ID`.

### 3. Ejecutar la ingesta

Desde la raíz del proyecto, ejecuta:

```bash
python -m src.ingest_notion
```

La sincronización es incremental: procesa únicamente páginas nuevas o modificadas y conserva en Chroma las páginas que no cambiaron.

### 4. Revisar los resultados

- Vector store: `vectorstore/chroma_db/`
- Exportaciones Markdown y estado de sincronización: `data/notion_exports/`
- Páginas procesadas en la última ejecución: `data/notion_exports/last_exported_pages.json`
- Páginas omitidas por errores: `data/notion_exports/skipped_pages.json`

Si una página no se procesa, verifica que la integración tenga acceso a ella y revisa el reporte de páginas omitidas.

## Generación con fuentes verificables

La interfaz permite seleccionar una política de fuentes antes de generar:

- **Base interna (`corpus`)**: utiliza exclusivamente Chroma/Notion.
- **Fuentes web (`web`)**: utiliza exclusivamente páginas recuperadas por web search.
- **Selección automática (`auto`)**: elige corpus, web o una combinación según la cobertura. Es el modo predeterminado.

Todos los modos exigen evidencia para cada paso. Si las fuentes no permiten
respaldar el roadmap, el sistema devuelve `INSUFFICIENT_EVIDENCE` en lugar de
completar la respuesta con conocimiento de entrenamiento del modelo.

Los límites end-to-end predeterminados son 120 segundos para corpus, 300 para
web y 210 para automático. Se pueden configurar con
`CORPUS_MODE_TIMEOUT`, `WEB_MODE_TIMEOUT` y `AUTO_MODE_TIMEOUT`.

### Ejecutar localmente con Ollama

Configura `.env`:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL_ID=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434
LLM_GENERATION_REASONING_MODE=disabled
LLM_JUDGE_REASONING_MODE=disabled
```

Después inicia la aplicación:

```bash
python -m uvicorn api.main:app --reload
```

Abre `http://127.0.0.1:8000` y selecciona el modo antes de enviar la pregunta.

Para un smoke de developer sin MLflow:

```bash
python -m evaluation.benchmarks.roadmap_generation_benchmark --sample-indices 15 --source-mode corpus --run-name corpus-ollama-smoke --no-mlflow
```

## Integración del clasificador de intención y las fuentes

El clasificador incorporado por el trabajo externo y los modos de fuentes del
proyecto cumplen responsabilidades distintas y se fusionan sin reemplazarse:

- El **selector de la interfaz** expresa una decisión explícita del usuario.
  `corpus` y `web` son obligatorios y omiten el clasificador.
- El **clasificador de intención de fuentes** se ejecuta únicamente en `auto` y
  detecta si la consulta pide base interna, combinación con prioridad interna o
  combinación con prioridad web.
- El **router de recuperación existente** comprueba cobertura real y vigencia.
  Si la intención es ambigua o el clasificador falla, conserva el comportamiento
  automático anterior basado en esos datos.
- El **grounding gate existente** sigue verificando que cada paso cite una de las
  evidencias recuperadas. El entrenamiento del LLM no se acepta como fuente.

| Selección | Clasificador | Recuperación resultante |
|---|---|---|
| `corpus` | Omitido | Solo corpus; falla si la evidencia es insuficiente |
| `web` | Omitido | Solo web; falla si no hay resultados utilizables |
| `auto` + KB only | Aplicado | Solo corpus |
| `auto` + KB + external | Aplicado | Híbrida, evidencia interna primero |
| `auto` + external + KB | Aplicado | Híbrida, evidencia web primero |
| `auto` + ambiguo/error | Aplicado con fallback | Router previo por cobertura/vigencia |

La traza de generación registra `source_intent`, `source_intent_plan`, la
decisión automática y el modo efectivo. No se fuerzan porcentajes artificiales:
el reparto final depende de las fuentes que realmente respaldan los pasos.

Los módulos de reranking, rechazo OOD, CRAG y judge especializados que llegaron
con el clasificador permanecen como herramientas experimentales de evaluación;
no forman parte del flujo principal hasta ser validados. Sus casos controlados
están separados en `evaluation/intent_dataset.py` para no alterar el dataset de
calidad de roadmaps.

Validación offline de la integración:

```bash
python -m evaluation.validations.source_intent_integration_validation
```
