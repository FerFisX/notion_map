"""
neo4j_loader.py
───────────────
Extrae nodos y relaciones de Neo4j y los convierte en
documentos LangChain para alimentar ChromaDB.

Uso desde ingest.py:
    from src.neo4j_loader import load_from_neo4j
    docs = load_from_neo4j()
"""

import os
from dotenv import load_dotenv
from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))


def _get_driver():
    """Crea y devuelve el driver Neo4j."""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise ImportError(
            "Instala el driver: pip install neo4j"
        )

    uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user     = os.getenv("NEO4J_USER",     "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not password:
        raise ValueError("NEO4J_PASSWORD no está definida en .env")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    print(f"  ✅ Neo4j conectado → {uri} / db:{database}")
    return driver, database


def load_from_neo4j(limit: int = 1000) -> list[Document]:
    """
    Extrae todos los nodos de Neo4j como documentos de texto.

    Cada nodo se convierte en un Document con:
      - page_content: texto con labels + propiedades
      - metadata: {source, node_id, labels}

    Returns:
        Lista de Document listos para ChromaDB.
    """
    driver, database = _get_driver()
    docs = []

    with driver.session(database=database) as session:

        # ── 1. Nodos ─────────────────────────────────────────────────
        print("  Extrayendo nodos...")
        result = session.run(f"""
            MATCH (n)
            RETURN
                elementId(n)    AS node_id,
                labels(n)       AS labels,
                properties(n)   AS props
            LIMIT {limit}
        """)

        for record in result:
            labels  = record["labels"]
            props   = record["props"]
            node_id = record["node_id"]

            # Convertir el nodo a texto legible
            lines = [f"Entidad: {', '.join(labels)}"]
            for key, value in props.items():
                if value is not None and str(value).strip():
                    lines.append(f"  {key}: {value}")

            text = "\n".join(lines)
            docs.append(Document(
                page_content=text,
                metadata={
                    "source":  "neo4j",
                    "node_id": str(node_id),
                    "labels":  ", ".join(labels),
                }
            ))

        # ── 2. Relaciones (como contexto adicional) ───────────────────
        print("  Extrayendo relaciones...")
        result = session.run(f"""
            MATCH (a)-[r]->(b)
            RETURN
                labels(a)       AS from_labels,
                properties(a)   AS from_props,
                type(r)         AS rel_type,
                properties(r)   AS rel_props,
                labels(b)       AS to_labels,
                properties(b)   AS to_props
            LIMIT {limit}
        """)

        for record in result:
            from_name = _best_name(record["from_props"], record["from_labels"])
            to_name   = _best_name(record["to_props"],   record["to_labels"])
            rel_type  = record["rel_type"]
            rel_props = record["rel_props"]

            lines = [
                f"Relación: {from_name} --[{rel_type}]--> {to_name}"
            ]
            for key, value in rel_props.items():
                if value is not None and str(value).strip():
                    lines.append(f"  {key}: {value}")

            text = "\n".join(lines)
            docs.append(Document(
                page_content=text,
                metadata={
                    "source":   "neo4j",
                    "type":     "relationship",
                    "rel_type": rel_type,
                }
            ))

    driver.close()
    print(f"  📦 Neo4j: {len(docs)} documentos extraídos")
    return docs


def _best_name(props: dict, labels: list) -> str:
    """Devuelve el mejor nombre representativo de un nodo."""
    for key in ("name", "title", "nombre", "id", "code", "label"):
        if key in props and props[key]:
            return f"{props[key]} ({', '.join(labels)})"
    return f"({', '.join(labels)})"
