# src/config.py
DOCS_PATH = "./documents"
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEBUG = True
SERVER_URL = "http://localhost:1234/v1"
MODEL_NAME = "gemma-3-4b-it-gguf"
RETRIEVER_K = 4