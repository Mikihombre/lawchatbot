# src/config.py
DOCS_PATH = "./documents"
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEBUG = True
SERVER_URL = "http://127.0.0.1:11434"
MODEL_NAME = "gemma3:27b-it-q4_K_M"
RETRIEVER_K = 4