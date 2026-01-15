from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Truco para que Python encuentre la carpeta 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rag_engine import RagEngine

app = FastAPI(title="NotionMap API")

# Configurar CORS (Para que el Frontend pueda hablar con el Backend sin errores de seguridad)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción esto se cambia, pero para MVP está bien
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializamos el motor UNA sola vez al arrancar (para no cargar el modelo en cada clic)
print("🚀 Cargando motor de IA...")
engine = RagEngine()
print("✅ Motor listo.")

# Definimos qué formato de datos esperamos recibir del frontend
class QueryRequest(BaseModel):
    question: str

@app.post("/generate-roadmap")
async def generate_roadmap_endpoint(request: QueryRequest):
    """
    Recibe una pregunta, consulta al RAG y devuelve el JSON del roadmap.
    """
    try:
        # Llamamos a tu cerebro Python
        result = engine.generate_roadmap(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "NotionMap API is running correctly"}