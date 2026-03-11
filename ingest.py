import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(BASE_DIR, "documents")
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

print("Loading PDFs...")

documents = []

for file in os.listdir(DOCS_PATH):

    if file.endswith(".pdf"):

        loader = PyPDFLoader(os.path.join(DOCS_PATH, file))
        documents.extend(loader.load())

print("Splitting documents...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

docs = splitter.split_documents(documents)

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=os.path.join(BASE_DIR, "models")
)

vector_db = FAISS.from_documents(docs, embeddings)

print("Saving FAISS index...")

vector_db.save_local(FAISS_PATH)

print("Vector DB created successfully.")