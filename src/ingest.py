import os
import glob
from dotenv import load_dotenv

# Importaciones
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# CAMBIO: Usamos HuggingFace en lugar de Google para los embeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DATA_PATH = "data"
DB_PATH = "vectorstore/chroma_db"

def main():
    # 1. Cargar PDF
    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    docs = []
    for f in pdf_files:
        print(f"📄 Leyendo: {f}")
        docs.extend(PyPDFLoader(f).load())

    if not docs:
        print("⚠️ No hay PDFs en la carpeta data/")
        return

    # 2. Cortar texto
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"✂️  Se generaron {len(chunks)} fragmentos.")

    # 3. Guardar en ChromaDB (Local)
    print("🧠 Creando embeddings locales (esto usa tu CPU, es gratis)...")
    
    # Este modelo se descarga una vez y vive en tu PC
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    print("🎉 ¡Listo! Base de conocimiento creada localmente.")

if __name__ == "__main__":
    main()