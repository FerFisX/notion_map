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
from notion_client.errors import APIResponseError, RequestTimeoutError
from notion2md.config import Config
from notion2md.convertor.block import BlockConvertor

load_dotenv()

DB_PATH = "vectorstore/chroma_db"
EXPORT_PATH = Path("data/notion_exports")
SKIPPED_PAGES_REPORT = EXPORT_PATH / "skipped_pages.json"
LAST_EXPORTED_PAGES_REPORT = EXPORT_PATH / "last_exported_pages.json"
SYNC_STATE_PATH = EXPORT_PATH / "sync_state.json"
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


def _error_code(error: Exception) -> str:
    return str(getattr(error, "code", error.__class__.__name__))


def _skipped_page_info(page: dict, error: Exception) -> dict:
    return {
        "title": _page_title(page),
        "page_id": page.get("id", ""),
        "url": page.get("url", ""),
        "last_edited_time": page.get("last_edited_time", ""),
        "archived": page.get("archived", False),
        "in_trash": page.get("in_trash", False),
        "error_code": _error_code(error),
        "error_message": str(error),
    }


def _write_skipped_pages_report(skipped_pages: List[dict]) -> None:
    EXPORT_PATH.mkdir(parents=True, exist_ok=True)
    SKIPPED_PAGES_REPORT.write_text(
        json.dumps(skipped_pages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _status_label(sync_reason: str) -> str:
    if sync_reason == "updated":
        return "upd"
    if sync_reason == "unchanged":
        return "same"
    return "new"


def _exported_page_info(doc: Document, sync_reason: str) -> dict:
    status = _status_label(sync_reason)
    title = doc.metadata.get("title", "")
    return {
        "status": status,
        "display_name": f"{status} {title}",
        "title": title,
        "page_id": doc.metadata.get("page_id", ""),
        "url": doc.metadata.get("url", ""),
        "last_edited_time": doc.metadata.get("last_edited_time", ""),
        "export_file": doc.metadata.get("export_file", ""),
    }


def _referenced_database_state(page: dict, docs: List[Document]) -> dict:
    return {
        "type": "referenced_database",
        "last_edited_time": page.get("last_edited_time", ""),
        "title": _page_title(page),
        "url": page.get("url", ""),
        "referenced_database_id": _database_id_from_page_url(page),
        "children_count": len(docs),
        "child_page_ids": [doc.metadata["page_id"] for doc in docs],
    }


def _referenced_database_state_from_pages(page: dict, child_pages: List[dict]) -> dict:
    return {
        "type": "referenced_database",
        "last_edited_time": page.get("last_edited_time", ""),
        "title": _page_title(page),
        "url": page.get("url", ""),
        "referenced_database_id": _database_id_from_page_url(page),
        "children_count": len(child_pages),
        "child_page_ids": [child["id"] for child in child_pages],
    }


def _write_last_exported_pages_report(exported_pages: List[dict]) -> None:
    EXPORT_PATH.mkdir(parents=True, exist_ok=True)
    LAST_EXPORTED_PAGES_REPORT.write_text(
        json.dumps(exported_pages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_sync_state() -> dict:
    if not SYNC_STATE_PATH.exists():
        return {"pages": {}}

    return json.loads(SYNC_STATE_PATH.read_text(encoding="utf-8"))


def _write_sync_state(sync_state: dict) -> None:
    EXPORT_PATH.mkdir(parents=True, exist_ok=True)
    SYNC_STATE_PATH.write_text(
        json.dumps(sync_state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _page_needs_sync(page: dict, sync_state: dict) -> tuple[bool, str]:
    page_id = page["id"]
    previous = sync_state.get("pages", {}).get(page_id)
    if not previous:
        return True, "new"

    if previous.get("type") == "referenced_database":
        child_page_ids = previous.get("child_page_ids", [])
        missing_children = [
            child_page_id
            for child_page_id in child_page_ids
            if child_page_id not in sync_state.get("pages", {})
        ]
        if missing_children:
            return True, "updated"

    if previous.get("last_edited_time") != page.get("last_edited_time", ""):
        return True, "updated"

    return False, "unchanged"


def _update_page_state(sync_state: dict, doc: Document) -> None:
    sync_state.setdefault("pages", {})[doc.metadata["page_id"]] = {
        "last_edited_time": doc.metadata.get("last_edited_time", ""),
        "title": doc.metadata.get("title", ""),
        "url": doc.metadata.get("url", ""),
        "export_file": doc.metadata.get("export_file", ""),
    }


def _removed_page_ids(sync_state: dict, reachable_page_ids: set[str]) -> List[str]:
    if _env_int("NOTION_MAX_PAGES"):
        return []

    known_page_ids = set(sync_state.get("pages", {}))
    return sorted(known_page_ids - reachable_page_ids)


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
    except (APIResponseError, RequestTimeoutError) as e:
        print(f"No se pudo exportar como pagina: {title} ({page_id}) - {_error_code(e)}")
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
    sync_state: dict,
    exported_pages: List[dict],
    reachable_page_ids: set[str],
) -> Optional[List[Document]]:
    referenced_database_id = _database_id_from_page_url(page)
    if not referenced_database_id:
        return None

    title = _page_title(page)
    try:
        nested_pages = list(_database_pages(notion, referenced_database_id))
    except (APIResponseError, RequestTimeoutError) as e:
        print(f"No se pudo expandir como database: {title} - {_error_code(e)}")
        return None

    print(f"Expandiendo database referenciada: {title} ({len(nested_pages)} paginas)")
    for nested_page in nested_pages:
        reachable_page_ids.add(nested_page["id"])

    sync_state.setdefault("pages", {})[page["id"]] = _referenced_database_state_from_pages(
        page,
        nested_pages,
    )

    docs = []
    for nested_page in nested_pages:
        needs_sync, sync_reason = _page_needs_sync(nested_page, sync_state)
        if not needs_sync:
            continue

        try:
            doc = _export_page_markdown(notion, nested_page)
            docs.append(doc)
            exported_pages.append(_exported_page_info(doc, sync_reason))
            _update_page_state(sync_state, doc)
        except (APIResponseError, RequestTimeoutError) as e:
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

    sync_state = _load_sync_state()
    docs = []
    skipped_pages = []
    exported_pages = []
    reachable_page_ids = {page["id"] for page in pages}
    for page in pages:
        previous = sync_state.get("pages", {}).get(page["id"], {})
        if previous.get("type") == "referenced_database":
            reachable_page_ids.update(previous.get("child_page_ids", []))

    new_pages = 0
    updated_pages = 0
    unchanged_pages = 0
    for page in pages:
        needs_sync, sync_reason = _page_needs_sync(page, sync_state)
        previous = sync_state.get("pages", {}).get(page["id"], {})

        if previous.get("type") == "referenced_database":
            before_exported_count = len(exported_pages)
            database_docs = _export_referenced_database_pages(
                notion,
                page,
                skipped_pages,
                sync_state,
                exported_pages,
                reachable_page_ids,
            )
            if database_docs is not None:
                docs.extend(database_docs)
                for doc in database_docs:
                    reachable_page_ids.add(doc.metadata["page_id"])

                new_exported_pages = exported_pages[before_exported_count:]
                new_pages += sum(1 for item in new_exported_pages if item["status"] == "new")
                updated_pages += sum(1 for item in new_exported_pages if item["status"] == "upd")
                unchanged_pages += 1
                continue

        if not needs_sync:
            unchanged_pages += 1
            continue

        if sync_reason == "new":
            new_pages += 1
        else:
            updated_pages += 1

        try:
            doc = _export_page_markdown(notion, page)
            docs.append(doc)
            exported_pages.append(_exported_page_info(doc, sync_reason))
            _update_page_state(sync_state, doc)
        except (APIResponseError, RequestTimeoutError) as e:
            database_docs = _export_referenced_database_pages(
                notion,
                page,
                skipped_pages,
                sync_state,
                exported_pages,
                reachable_page_ids,
            )
            if database_docs is not None:
                docs.extend(database_docs)
                for doc in database_docs:
                    reachable_page_ids.add(doc.metadata["page_id"])
            else:
                skipped_pages.append(_skipped_page_info(page, e))

    _write_skipped_pages_report(skipped_pages)
    _write_last_exported_pages_report(exported_pages)
    removed_page_ids = _removed_page_ids(sync_state, reachable_page_ids)

    if not docs and not removed_page_ids:
        _write_sync_state(sync_state)
        print(f"Paginas detectadas: {len(pages)}")
        print(f"Paginas nuevas: {new_pages}")
        print(f"Paginas actualizadas: {updated_pages}")
        print(f"Paginas sin cambios: {unchanged_pages}")
        print(f"Paginas saltadas: {len(skipped_pages)}")
        print("Paginas eliminadas: 0")
        print(f"Reporte de paginas exportadas: {LAST_EXPORTED_PAGES_REPORT}")
        print("No hay paginas nuevas o modificadas para cargar en Chroma")
        return

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    for page_id in removed_page_ids:
        vector_db.delete(where={"page_id": page_id})
        sync_state.get("pages", {}).pop(page_id, None)

    _write_sync_state(sync_state)

    if not docs:
        print(f"Paginas detectadas: {len(pages)}")
        print(f"Paginas nuevas: {new_pages}")
        print(f"Paginas actualizadas: {updated_pages}")
        print(f"Paginas sin cambios: {unchanged_pages}")
        print(f"Paginas saltadas: {len(skipped_pages)}")
        print(f"Paginas eliminadas: {len(removed_page_ids)}")
        print(f"Reporte de paginas exportadas: {LAST_EXPORTED_PAGES_REPORT}")
        print("No hay paginas nuevas o modificadas para cargar en Chroma")
        return

    chunks = _split_notion_documents(docs)

    # Replace only changed pages; unchanged pages stay in Chroma.
    for doc in docs:
        vector_db.delete(where={"page_id": doc.metadata["page_id"]})

    ids = [
        f"notion:{chunk.metadata['page_id']}:{index}"
        for index, chunk in enumerate(chunks)
    ]
    vector_db.add_documents(chunks, ids=ids)

    print(f"Paginas detectadas: {len(pages)}")
    print(f"Paginas nuevas: {new_pages}")
    print(f"Paginas actualizadas: {updated_pages}")
    print(f"Paginas sin cambios: {unchanged_pages}")
    print(f"Paginas exportadas: {len(docs)}")
    print(f"Paginas saltadas: {len(skipped_pages)}")
    print(f"Paginas eliminadas: {len(removed_page_ids)}")
    print(f"Reporte de paginas saltadas: {SKIPPED_PAGES_REPORT}")
    print(f"Reporte de paginas exportadas: {LAST_EXPORTED_PAGES_REPORT}")
    print(f"Chunks cargados en Chroma: {len(chunks)}")
    print("Chunking: secciones Markdown con fallback por tamano")
    print(f"Markdown generado en: {EXPORT_PATH}")


if __name__ == "__main__":
    sync_notion_to_chroma()
