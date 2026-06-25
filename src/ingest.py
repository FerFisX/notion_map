"""
ingest.py — Convierte PDFs a Markdown (via markitdown), luego los indexa en ChromaDB.

Flujo:
  1. data/*.pdf  →  markitdown  →  data/markdown/*.md  (conversión limpia con estructura)
  2. data/markdown/*.md  →  chunks  →  ChromaDB (embeddings HuggingFace)
  3. Neo4j (opcional) — se añade si hay credenciales en .env
"""

import os
import sys
import glob
import shutil
from pathlib import Path
from dotenv import load_dotenv

# UTF-8 en stdout/stderr para Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
load_dotenv(os.path.join(BASE_DIR, ".env"))

DATA_PATH     = os.path.join(BASE_DIR, "data")
MARKDOWN_PATH = os.path.join(BASE_DIR, "data", "markdown")
DB_PATH       = os.path.join(BASE_DIR, "vectorstore", "chroma_db")


# ── Paso 1: PDF → Markdown ────────────────────────────────────────────────────

def convert_pdfs_to_markdown() -> list[str]:
    """
    Convierte cada PDF de data/ a Markdown usando markitdown y lo guarda en
    data/markdown/<nombre>.md. Retorna la lista de rutas .md generadas.
    """
    from markitdown import MarkItDown

    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    if not pdf_files:
        print("  [!] No hay PDFs en data/")
        return []

    os.makedirs(MARKDOWN_PATH, exist_ok=True)
    md_converter = MarkItDown()
    md_paths = []

    for pdf_path in pdf_files:
        stem     = Path(pdf_path).stem
        md_path  = os.path.join(MARKDOWN_PATH, f"{stem}.md")
        pdf_name = os.path.basename(pdf_path)

        try:
            result = md_converter.convert(pdf_path)
            text   = result.text_content.strip()

            if not text:
                print(f"  [!] {pdf_name}: markitdown no extrajo texto — omitiendo.")
                continue

            Path(md_path).write_text(text, encoding="utf-8")
            lines = text.count("\n") + 1
            print(f"  [PDF->MD] {pdf_name}")
            print(f"            -> {os.path.basename(md_path)}  ({lines} lineas, {len(text):,} chars)")
            md_paths.append(md_path)

        except Exception as e:
            print(f"  [!] Error convirtiendo {pdf_name}: {e}")

    return md_paths


# ── Paso 2: Markdown → LangChain Documents ────────────────────────────────────

def load_markdown_docs(md_paths: list[str]) -> list[Document]:
    """
    Lee los archivos .md y los convierte en LangChain Documents conservando
    el nombre del archivo original como metadata.
    """
    docs = []
    for md_path in md_paths:
        try:
            text = Path(md_path).read_text(encoding="utf-8")
            docs.append(Document(
                page_content=text,
                metadata={"source": os.path.basename(md_path), "type": "markdown"},
            ))
        except Exception as e:
            print(f"  [!] Error leyendo {md_path}: {e}")
    return docs


# ── Fuente adicional: Neo4j ───────────────────────────────────────────────────

def load_neo4j_docs() -> list:
    uri = os.getenv("NEO4J_URI", "")
    pwd = os.getenv("NEO4J_PASSWORD", "")

    if not uri or not pwd or pwd == "TU_PASSWORD_NEO4J_AQUI":
        print("  Neo4j no configurado — omitiendo.")
        return []

    # Verificar conectividad antes de intentar la carga
    try:
        import socket
        host = uri.replace("bolt://", "").replace("neo4j://", "")
        host, port = (host.split(":") + ["7687"])[:2]
        sock = socket.create_connection((host, int(port)), timeout=2)
        sock.close()
    except OSError:
        print("  Neo4j no disponible en este momento — omitiendo.")
        return []

    try:
        from src.neo4j_loader import load_from_neo4j
        return load_from_neo4j()
    except ImportError:
        print("  [!] Driver neo4j no instalado. Corre: pip install neo4j")
        return []
    except Exception as e:
        print(f"  [!] Neo4j error: {e}")
        return []


# ── ChromaDB: limpiar y recrear ───────────────────────────────────────────────

def reset_vectorstore():
    if os.path.exists(DB_PATH):
        print(f"  Limpiando vectorstore anterior...")
        try:
            shutil.rmtree(DB_PATH)
        except PermissionError:
            print("  [!] No se pudo borrar el vectorstore — el servidor API lo tiene abierto.")
            print("      Detén el servidor (Ctrl+C o cierra la terminal) y vuelve a correr el ingest.")
            sys.exit(1)
    os.makedirs(DB_PATH, exist_ok=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  NotionMap — Ingesta con Markdown (via markitdown)")
    print("=" * 55)

    # 1. Convertir PDFs a Markdown
    print("\n[1/4] Convirtiendo PDFs a Markdown...")
    md_paths = convert_pdfs_to_markdown()
    print(f"       {len(md_paths)} archivos Markdown generados en data/markdown/")

    # 2. Cargar los Markdown como Documents
    print("\n[2/4] Cargando Markdown como documentos...")
    md_docs = load_markdown_docs(md_paths)

    # 3. Cargar Neo4j (opcional)
    print("\n[3/4] Cargando desde Neo4j (opcional)...")
    neo4j_docs = load_neo4j_docs()

    all_raw = md_docs + neo4j_docs
    if not all_raw:
        print("\n[!] No hay datos para indexar.")
        print("    Agrega PDFs en data/ o configura NEO4J_* en .env")
        return

    # 4. Fragmentar y generar embeddings
    print(f"\n[4/4] Fragmentando y generando embeddings...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    md_chunks    = splitter.split_documents(md_docs) if md_docs else []
    neo4j_chunks = neo4j_docs  # ya vienen en tamaño adecuado
    all_chunks   = md_chunks + neo4j_chunks

    print(f"       {len(md_chunks)} chunks de Markdown + {len(neo4j_chunks)} de Neo4j = {len(all_chunks)} total")

    reset_vectorstore()

    print(f"       Modelo de embeddings: all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=DB_PATH,
    )

    print(f"\n  Listo — {len(all_chunks)} chunks indexados en ChromaDB")
    print(f"  Ruta: {DB_PATH}\n")


if __name__ == "__main__":
    main()
