from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.rag_engine import RagEngine

app = FastAPI(title="NotionMap API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CAMBIO IMPORTANTE AQUÍ ---
# 1. Ya NO iniciamos el motor aquí. Lo dejamos vacío por ahora.
engine = None 

class QueryRequest(BaseModel):
    question: str

def get_engine():
    """Función para cargar el motor solo cuando se necesite"""
    global engine
    if engine is None:
        print("⏳ Cargando motor de IA por primera vez... (Esto puede tardar un poco)")
        engine = RagEngine()
        print("✅ Motor cargado y listo.")
    return engine

@app.post("/generate-roadmap")
async def generate_roadmap_endpoint(request: QueryRequest):
    try:
        # 2. Llamamos a la función que revisa si el motor está listo
        rag = get_engine()
        result = rag.generate_roadmap(request.question)
        return result
    except Exception as e:
        # Tip: Imprimimos el error en los logs de Render para poder depurar
        print(f"❌ Error generando roadmap: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('frontend/index.html')