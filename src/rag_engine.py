import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Importaciones del Motor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # <--- CAMBIO AQUÍ
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

DB_PATH = "vectorstore/chroma_db"

# --- ESTRUCTURAS DE DATOS ---
class RoadmapStep(BaseModel):
    id: str = Field(description="Identificador único, ej: '1'")
    label: str = Field(description="Título corto del paso")
    description: str = Field(description="Resumen breve de la acción")
    type: str = Field(description="Tipo: 'inicio', 'proceso', 'decisión', 'fin'")

class Roadmap(BaseModel):
    title: str = Field(description="Título del proceso")
    steps: list[RoadmapStep] = Field(description="Lista de pasos")

# --- MOTOR RAG ---
class RagEngine:
    def __init__(self):
        # 1. Configurar Embeddings (LOCALES - Mismo modelo que ingest.py)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. Conectar a ChromaDB
        self.vector_db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=self.embeddings
        )
        
        # 3. Configurar LLM (Gemini sigue siendo el cerebro que escribe)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )
        self.parser = JsonOutputParser(pydantic_object=Roadmap)

    def generate_roadmap(self, query: str):
        # Buscar contexto
        print(f"🔎 Buscando contexto para: '{query}'...")
        docs = self.vector_db.similarity_search(query, k=4)
        context = "\n".join([d.page_content for d in docs])
        
        if not context:
            return {"error": "No encontré información en los PDFs."}

        # Prompt
        template = """
        Eres un experto en procesos corporativos. Basado en el contexto:
        {context}
        
        Crea un roadmap secuencial para responder a: {query}
        Responde SOLO con un JSON válido.
        {format_instructions}
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["query", "context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        chain = prompt | self.llm | self.parser
        
        print("🤖 Gemini está pensando...")
        return chain.invoke({"query": query, "context": context})

if __name__ == "__main__":
    engine = RagEngine()
    # PREGUNTA DE PRUEBA
    print(engine.generate_roadmap("Como definir alcance y expectativas?"))