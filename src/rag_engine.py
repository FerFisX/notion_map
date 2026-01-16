import os
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

DB_PATH = "vectorstore/chroma_db"

# --- ESTRUCTURA DE DATOS ---
class RoadmapStep(BaseModel):
    id: str = Field(description="ID corto, ej: 'step_1'")
    label: str = Field(description="Título de la acción")
    description: str = Field(description="Explicación breve")
    type: str = Field(description="Tipo: 'inicio', 'proceso', 'decision', 'fin'")
    # CAMBIO: Instrucción clara para que sea una lista opcional
    key_points: List[str] = Field(description="Lista de 1 a 3 detalles técnicos ESPECÍFICOS (versiones, comandos, herramientas). Dejar vacío si no hay detalles críticos.")

class Roadmap(BaseModel):
    title: str = Field(description="Título del proceso")
    steps: List[RoadmapStep] = Field(description="Lista de pasos secuenciales")

# --- MOTOR RAG ---
class RagEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = Chroma(persist_directory=DB_PATH, embedding_function=self.embeddings)
        # Temperature baja para precisión técnica
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        self.parser = JsonOutputParser(pydantic_object=Roadmap)

    def generate_roadmap(self, query: str):
        try:
            print(f"🔎 Buscando contexto para: '{query}'...")
            docs = self.vector_db.similarity_search(query, k=4)
            context = "\n".join([d.page_content for d in docs])
            
            if not context:
                context = "No hay contexto específico. Genera pasos lógicos estándar."

            # Prompt ajustado para bifurcaciones
            template = """
            Eres un arquitecto técnico. Crea un Roadmap para: "{query}".
            CONTEXTO: {context}
            
            REGLAS:
            1. Genera un JSON válido con una lista de pasos secuenciales.
            2. Campo 'key_points': ÚSALO SOLO SI hay un detalle técnico crucial (ej: "Usar Python 3.10", "Puerto 8000", "Cuidado con CORS").
            3. Si un paso es genérico, deja 'key_points' como lista vacía [].
            
            {format_instructions}
            """
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["query", "context"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()}
            )

            chain = prompt | self.llm | self.parser
            return chain.invoke({"query": query, "context": context})

        except Exception as e:
            print(f"❌ Error: {e}")
            return {
                "title": "Error",
                "steps": [{
                    "id": "err", "label": "Error de Sistema", "description": str(e)[:100], "type": "decision", "key_points": []
                }]
            }