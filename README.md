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
