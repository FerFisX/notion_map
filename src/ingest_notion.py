import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from notion_client import Client
from notion_client.errors import APIResponseError
from notion2md.config import Config
from notion2md.convertor.block import BlockConvertor

load_dotenv()

DB_PATH = "vectorstore/chroma_db"
EXPORT_PATH = Path("data/notion_exports")
SKIPPED_PAGES_REPORT = EXPORT_PATH / "skipped_pages.json"
MAX_SECTION_CHARS = 1600
SECTION_OVERLAP_CHARS = 200


def _plain_text(rich_text: list) -> str:
    return "".join(part.get("plain_text", "") for part in rich_text or []).strip()


def _page_title(page: dict) -> str:
    properties = page.get("properties", {})
    for value in properties.values():
        if value.get("type") == "title":
            title = _plain_text(value.get("title", []))
            if title:
                return title
    return page.get("id", "untitled")


def _slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "notion-page"


def _env_csv(name: str) -> List[str]:
    value = os.getenv(name, "")
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_int(name: str, default: int = 0) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    return int(value)


def _notion_id(value: str) -> str:
    compact = value.replace("-", "")
    matches = re.findall(r"[0-9a-fA-F]{32}", compact)
    if matches:
        return matches[-1]
    return value


def _compact_id(value: str) -> str:
    return _notion_id(value).replace("-", "")


def _database_id_from_page_url(page: dict) -> Optional[str]:
    page_id = page.get("id") or page.get("page_id", "")
    url = page.get("url", "")
    if not page_id or not url:
        return None

    url_id = _notion_id(url)
    if _compact_id(url_id) == _compact_id(page_id):
        return None
    return url_id


def _sanitize_block(block: dict) -> dict:
    block_type = block["type"]
    payload = block.get(block_type)
    if not isinstance(payload, dict):
        return block

    sanitized = dict(block)
    sanitized_payload = dict(payload)
    if sanitized_payload.get("icon") is None:
        sanitized_payload.pop("icon", None)
    sanitized[block_type] = sanitized_payload
    return sanitized


class SanitizedNotionClient:
    def __init__(self, notion: Client):
        self.notion = notion

    def get_children(self, parent_id: str) -> List[dict]:
        results = []
        cursor = None
        while True:
            kwargs = {"block_id": parent_id, "page_size": 100}
            if cursor:
                kwargs["start_cursor"] = cursor

            response = self.notion.blocks.children.list(**kwargs)
            results.extend(_sanitize_block(block) for block in response.get("results", []))

            if not response.get("has_more"):
                break
            cursor = response.get("next_cursor")

        return results


def _data_source_ids(notion: Client, database_id: str) -> List[str]:
    database = notion.databases.retrieve(database_id=_notion_id(database_id))
    return [source["id"] for source in database.get("data_sources", [])]


def _data_source_pages(notion: Client, data_source_id: str) -> Iterable[dict]:
    cursor = None
    while True:
        kwargs = {"data_source_id": _notion_id(data_source_id)}
        if cursor:
            kwargs["start_cursor"] = cursor

        response = notion.data_sources.query(**kwargs)
        yield from response.get("results", [])

        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")


def _database_pages(notion: Client, database_id: str) -> Iterable[dict]:
    for data_source_id in _data_source_ids(notion, database_id):
        yield from _data_source_pages(notion, data_source_id)


def _configured_pages(notion: Client) -> List[dict]:
    pages = []
    max_pages = _env_int("NOTION_MAX_PAGES")
    seen_page_ids = set()

    def add_page(page: dict) -> None:
        page_id = page.get("id")
        if page_id and page_id not in seen_page_ids:
            pages.append(page)
            seen_page_ids.add(page_id)

    for page_id in _env_csv("NOTION_PAGE_ID"):
        add_page(notion.pages.retrieve(page_id=_notion_id(page_id)))
        if max_pages and len(pages) >= max_pages:
            return pages

    database_id = os.getenv("NOTION_DATABASE_ID")
    if database_id:
        for page in _database_pages(notion, database_id):
            add_page(page)
            if max_pages and len(pages) >= max_pages:
                break

    return pages


def _skipped_page_info(page: dict, error: APIResponseError) -> dict:
    return {
        "title": _page_title(page),
        "page_id": page.get("id", ""),
        "url": page.get("url", ""),
        "last_edited_time": page.get("last_edited_time", ""),
        "archived": page.get("archived", False),
        "in_trash": page.get("in_trash", False),
        "error_code": str(error.code),
        "error_message": str(error),
    }


def _write_skipped_pages_report(skipped_pages: List[dict]) -> None:
    EXPORT_PATH.mkdir(parents=True, exist_ok=True)
    SKIPPED_PAGES_REPORT.write_text(
        json.dumps(skipped_pages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _export_page_markdown(
    notion: Client,
    page: dict,
) -> Optional[Document]:
    page_id = page["id"]
    title = _page_title(page)
    sanitized_client = SanitizedNotionClient(notion)
    converter = BlockConvertor(Config(block_id=page_id), sanitized_client)
    try:
        markdown = converter.to_string(sanitized_client.get_children(page_id))
    except APIResponseError as e:
        print(f"No se pudo exportar como pagina: {title} ({page_id}) - {e.code}")
        raise
    markdown = markdown.strip()

    if not markdown:
        markdown = f"# {title}\n\n"

    EXPORT_PATH.mkdir(parents=True, exist_ok=True)
    filename = f"{_slugify(title)}-{page_id.replace('-', '')[:8]}.md"
    export_file = EXPORT_PATH / filename
    export_file.write_text(markdown, encoding="utf-8")

    metadata = {
        "source": "notion",
        "page_id": page_id,
        "title": title,
        "url": page.get("url", ""),
        "last_edited_time": page.get("last_edited_time", ""),
        "export_file": str(export_file),
    }
    return Document(page_content=markdown, metadata=metadata)


def _export_referenced_database_pages(
    notion: Client,
    page: dict,
    skipped_pages: List[dict],
) -> List[Document]:
    referenced_database_id = _database_id_from_page_url(page)
    if not referenced_database_id:
        return []

    title = _page_title(page)
    try:
        nested_pages = list(_database_pages(notion, referenced_database_id))
    except APIResponseError as e:
        print(f"No se pudo expandir como database: {title} - {e.code}")
        return []

    print(f"Expandiendo database referenciada: {title} ({len(nested_pages)} paginas)")
    docs = []
    for nested_page in nested_pages:
        try:
            docs.append(_export_page_markdown(notion, nested_page))
        except APIResponseError as e:
            skipped_pages.append(_skipped_page_info(nested_page, e))

    return [doc for doc in docs if doc is not None]


def _split_notion_documents(docs: List[Document]) -> List[Document]:
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "heading_1"),
            ("##", "heading_2"),
            ("###", "heading_3"),
        ],
        strip_headers=False,
    )
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_SECTION_CHARS,
        chunk_overlap=SECTION_OVERLAP_CHARS,
    )

    chunks = []
    for doc in docs:
        section_docs = markdown_splitter.split_text(doc.page_content)
        if not section_docs:
            section_docs = [Document(page_content=doc.page_content, metadata={})]

        for section_doc in section_docs:
            section_metadata = {**doc.metadata, **section_doc.metadata}
            section_doc.metadata = section_metadata

            if len(section_doc.page_content) <= MAX_SECTION_CHARS:
                chunks.append(section_doc)
                continue

            subchunks = fallback_splitter.split_documents([section_doc])
            chunks.extend(subchunks)

    return chunks


def sync_notion_to_chroma() -> None:
    if not os.getenv("NOTION_TOKEN"):
        raise RuntimeError("Falta NOTION_TOKEN en .env")

    notion = Client(auth=os.environ["NOTION_TOKEN"])
    pages = _configured_pages(notion)
    if not pages:
        raise RuntimeError("Configura NOTION_PAGE_ID o NOTION_DATABASE_ID en .env")

    docs = []
    skipped_pages = []
    for page in pages:
        try:
            docs.append(_export_page_markdown(notion, page))
        except APIResponseError as e:
            database_docs = _export_referenced_database_pages(notion, page, skipped_pages)
            if database_docs:
                docs.extend(database_docs)
            else:
                skipped_pages.append(_skipped_page_info(page, e))

    if not docs:
        raise RuntimeError("No se pudo exportar ninguna pagina accesible desde Notion")
    _write_skipped_pages_report(skipped_pages)
    chunks = _split_notion_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # Replace existing chunks for these pages so the demo is rerunnable.
    for page in pages:
        vector_db.delete(where={"page_id": page["id"]})

    ids = [
        f"notion:{chunk.metadata['page_id']}:{index}"
        for index, chunk in enumerate(chunks)
    ]
    vector_db.add_documents(chunks, ids=ids)

    print(f"Paginas exportadas: {len(docs)}")
    print(f"Paginas saltadas: {len(skipped_pages)}")
    print(f"Reporte de paginas saltadas: {SKIPPED_PAGES_REPORT}")
    print(f"Chunks cargados en Chroma: {len(chunks)}")
    print("Chunking: secciones Markdown con fallback por tamano")
    print(f"Markdown generado en: {EXPORT_PATH}")


if __name__ == "__main__":
    sync_notion_to_chroma()
