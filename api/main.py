import os
import sys
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Rutas absolutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))
sys.path.append(BASE_DIR)

from src.rag_engine import RagEngine

# Estado global
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    t0 = time.time()
    print("=" * 55)
    print("  NotionMap - Iniciando...")
    print("=" * 55)
    print("\n  [1/3] Cargando modelo de embeddings HuggingFace...")
    print("        (all-MiniLM-L6-v2 via PyTorch)")
    print("        Primera vez puede tardar 2-3 min descargando ~90 MB")
    print("        Las siguientes arrancadas son inmediatas.\n")

    try:
        print("  [2/3] Conectando a ChromaDB y LLM...")
        engine = RagEngine()
        elapsed = round(time.time() - t0, 1)
        print(f"\n  [3/3] Listo en {elapsed}s - http://localhost:8000")
        print("=" * 55 + "\n")
    except Exception as e:
        print(f"\n  ERROR al iniciar: {e}")
        print("  Verifica tu .env (LLM_PROVIDER y credenciales del proveedor)\n")
        engine = None

    yield

    print("\n  NotionMap cerrado.")

app = FastAPI(title="NotionMap API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.post("/generate-roadmap")
async def generate_roadmap_endpoint(request: QueryRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Motor no iniciado. Revisa la terminal.")
    return engine.generate_roadmap(request.question)


# ── Laboratorio de evaluación paso a paso ─────────────────────────────────────

class PipelineRequest(BaseModel):
    question:       str
    rerank:         bool = True
    rerank_method:  str  = "mmr"      # mmr | crossencoder | none
    run_structure:  bool = True
    run_similarity: bool = True
    run_judge:      bool = False


def _roadmap_to_text(roadmap: dict) -> str:
    if not roadmap or "steps" not in roadmap:
        return ""
    lines = [f"# {roadmap.get('title','Roadmap')}", ""]
    for i, s in enumerate(roadmap.get("steps", []), 1):
        lines.append(f"{i}. [{s.get('type','proceso').upper()}] {s.get('label','')}")
        if s.get("description"):
            lines.append(f"   {s['description']}")
        for kp in s.get("key_points", []):
            lines.append(f"   - {kp}")
    return "\n".join(lines)


@app.post("/eval/pipeline")
async def eval_pipeline(req: PipelineRequest):
    """
    Corre el pipeline RAG paso a paso para UNA consulta, devolviendo cada etapa
    con lo que hace y su resultado — para visualización educativa.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Motor no iniciado.")

    steps = []

    # ── Paso 1: Reescritura de consulta ───────────────────────────────────────
    t0 = time.time()
    refined = engine.rewrite_query(req.question)
    steps.append({
        "key":   "rewrite",
        "title": "1. Reescritura de la consulta",
        "what":  "Un LLM expande y clarifica la consulta del usuario, anadiendo terminologia tecnica.",
        "why":   "Una consulta mas rica recupera mejores fragmentos y guia al modelo hacia un roadmap mas preciso.",
        "result": {"original": req.question, "refined": refined},
        "elapsed_s": round(time.time() - t0, 2),
    })

    # ── Paso 2: Recuperacion + Re-ranking ─────────────────────────────────────
    t0 = time.time()
    method = req.rerank_method if req.rerank else "none"
    retr = engine.retrieve_contexts_scored(refined, method=method)
    steps.append({
        "key":   "retrieval",
        "title": "2. Recuperacion + Re-ranking",
        "what":  f"Se buscan {len(retr['pool'])} fragmentos candidatos y se re-ordenan (metodo: {retr['method']}) para quedarse con los mejores {len(retr['selected'])}.",
        "why":   "El re-ranking prioriza los fragmentos mas relevantes y diversos, evitando contexto redundante o poco util.",
        "result": {
            "method":   retr["method"],
            "pool":     [{"init_rank": c.get("init_rank"), "vector_score": c.get("vector_score"),
                          "rerank_score": c.get("rerank_score"), "source": c.get("source"),
                          "preview": c["text"][:160].replace("\n", " ")} for c in retr["pool"]],
            "selected": [{"final_rank": c.get("final_rank"), "init_rank": c.get("init_rank"),
                          "final_score": c.get("final_score"), "source": c.get("source"),
                          "preview": c["text"][:160].replace("\n", " ")} for c in retr["selected"]],
        },
        "elapsed_s": round(time.time() - t0, 2),
    })

    # ── Paso 3: Generacion del roadmap ────────────────────────────────────────
    t0 = time.time()
    contexts = [c["text"] for c in retr["selected"]]
    roadmap  = engine.build_roadmap(req.question, refined, contexts)
    answer   = _roadmap_to_text(roadmap)
    steps.append({
        "key":   "generate",
        "title": "3. Generacion del roadmap",
        "what":  "El LLM construye el roadmap usando la consulta enriquecida y el contexto re-rankeado, bajo estandares estrictos de calidad.",
        "why":   "Es el resultado final que ve el usuario; las etapas previas existen para que este paso sea lo mas util posible.",
        "result": {"roadmap": roadmap, "n_steps": len(roadmap.get("steps", []))},
        "elapsed_s": round(time.time() - t0, 2),
    })

    # ── Paso 4: Validacion de estructura (deterministica) ─────────────────────
    if req.run_structure:
        t0 = time.time()
        from evaluation.structure_validator import StructureValidator
        struct = StructureValidator().validate(roadmap)
        steps.append({
            "key":   "structure",
            "title": "4. Validacion de estructura",
            "what":  "Chequeos deterministicos (sin LLM) sobre la FORMA del roadmap: nro de pasos, key_points, tipos validos, etc.",
            "why":   "Garantiza que el roadmap cumple el formato esperado antes de juzgar su contenido. Detecta fallos objetivos.",
            "result": struct,
            "elapsed_s": round(time.time() - t0, 2),
        })

    # ── Paso 5: Similitud query-respuesta (cap. 2 Rothman) ────────────────────
    if req.run_similarity:
        t0 = time.time()
        from evaluation.similarity_metrics import query_answer_relevance
        sim = query_answer_relevance(req.question, answer)
        steps.append({
            "key":   "similarity",
            "title": "5. Similitud consulta vs respuesta",
            "what":  "Mide con cosine similarity cuanto se relaciona la respuesta con la pregunta. Compara TF-IDF (superficial) vs embeddings semanticos.",
            "why":   "Metrica automatica sin ground truth. El contraste TF-IDF vs semantico muestra que el sistema entiende significado, no solo palabras.",
            "result": sim,
            "elapsed_s": round(time.time() - t0, 2),
        })

    # ── Paso 6: LLM Judge (opcional, mas lento) ───────────────────────────────
    if req.run_judge:
        t0 = time.time()
        from evaluation.llm_judge import LLMJudgeEvaluator
        judge = LLMJudgeEvaluator()
        steps_str = "\n".join(f"  {j+1}. {s.get('label','')}"
                              for j, s in enumerate(roadmap.get("steps", [])))
        raw = judge._judge_single(
            question=req.question, roadmap_text=answer, contexts=contexts,
            ground_truth="(No disponible - evaluacion ad-hoc)", steps_list=steps_str,
        )
        raw["mese"]["composite"] = judge._mese_composite(raw["mese"])
        steps.append({
            "key":   "judge",
            "title": "6. LLM como Juez (MESE)",
            "what":  "Un LLM evalua el roadmap en 4 dimensiones MESE (Mapping, Exhaustividad, Secuencia, Experiencia) + criterios clasicos.",
            "why":   "Evalua la CALIDAD del contenido (no solo la forma): precision, cobertura, orden logico y claridad.",
            "result": raw,
            "elapsed_s": round(time.time() - t0, 2),
        })

    return {"question": req.question, "steps": steps}


@app.get("/eval/lab")
async def eval_lab():
    return FileResponse(os.path.join(frontend_path, "lab.html"))

@app.get("/health")
async def health():
    from src.llm_provider import active_model_name
    return {"status": "ok" if engine else "loading", "model": active_model_name()}

frontend_path = os.path.join(BASE_DIR, "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(frontend_path, "index.html"))
