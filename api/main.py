from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles # <--- NUEVO
from fastapi.responses import FileResponse # <--- NUEVO
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Ajustar path para importar src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.rag_engine import RagEngine

app = FastAPI(title="NotionMap API")

# CORS (Sigue siendo útil)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar motor
engine = RagEngine()

class QueryRequest(BaseModel):
    question: str

@app.post("/generate-roadmap") # Ruta de la API
async def generate_roadmap_endpoint(request: QueryRequest):
    try:
        result = engine.generate_roadmap(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- AQUÍ ESTÁ EL TRUCO PARA RENDER ---
# Montamos la carpeta frontend para que sea accesible
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Ruta raíz carga el index.html
@app.get("/")
async def read_index():
    return FileResponse('frontend/index.html')