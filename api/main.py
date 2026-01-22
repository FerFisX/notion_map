import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Ajustar path para importar src (Truco para que Python encuentre tus carpetas)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.rag_engine import RagEngine

app = FastAPI(title="NotionMap API - Local")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Inicializacion
print(" Arrancando NotionMap ...")
try:
    engine = RagEngine()
    print("Conectado")
except Exception as e:
    print(f"Error cargando motor (verificar .env): {e}")
    engine = None

class QueryRequest(BaseModel):
    question: str

@app.post("/generate-roadmap")
async def generate_roadmap_endpoint(request: QueryRequest):
    if not engine:
        raise HTTPException(status_code=500, detail="El motor de IA no está listo.")
    return engine.generate_roadmap(request.question)

# ---  AQUÍ ESTÁ EL ARREGLO DE RUTAS ---

# 1. Obtenemos la ruta de ESTE archivo (api/main.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Subimos un nivel para llegar a la raíz del proyecto
root_dir = os.path.dirname(current_dir)

# 3. Buscamos la carpeta frontend en la raíz
frontend_path = os.path.join(root_dir, "frontend")

# Verificación de seguridad (te avisará en la terminal si no la encuentra)
if not os.path.exists(frontend_path):
    print(f"ERROR: No encuentro la carpeta frontend en: {frontend_path}")
    print("Asegúrate de que la carpeta 'frontend' esté en la misma carpeta que 'api' y 'src'.")

# 4. Montamos los archivos estáticos
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(frontend_path, 'index.html'))