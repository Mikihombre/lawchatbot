import os

# --- AUTOMATYCZNE WYKRYWANIE ŚCIEŻEK ---
# 1. Gdzie jest ten plik (config.py)? -> C:\Users\mikol\...\src\config.py
CURRENT_FILE_PATH = os.path.abspath(__file__)

# 2. Folder nadrzędny (src) -> C:\Users\mikol\...\src
SRC_DIR = os.path.dirname(CURRENT_FILE_PATH)

# 3. Główny folder projektu -> C:\Users\mikol\...\lawchatbot
BASE_DIR = os.path.dirname(SRC_DIR)

# 4. Definiujemy ścieżki na sztywno względem folderu głównego
DOCS_PATH = os.path.join(BASE_DIR, "documents")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

# --- KONFIGURACJA AI ---
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 12

# --- HYBRID SEARCH CONFIG ---
USE_HYBRID = True
BM25_K = 15           # Ile dokumentów pobiera BM25 (musi być więcej, żeby filtracja nie wycięła wszystkiego)
HYBRID_ALPHA = 0.5    # Waga: 0.5 to równowaga. 0.3 to przewaga BM25, 0.7 przewaga Wektorów.

# --- RERANKING CONFIG ---
USE_RERANKER = True
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3" # Świetny model, wspiera wielojęzyczność (w tym PL)
INITIAL_RETRIEVAL_K = 60  # Pobieramy szeroko (lejek wlotowy)
FINAL_K = 10              # Tyle oddajemy do LLM (lejek wylotowy)

# --- KONFIGURACJA OLLAMA ---
SERVER_URL = "http://127.0.0.1:11434"
MODEL_NAME = "gemma3:12b"
DEBUG = True

# --- DEBUG (Wypisze w konsoli przy starcie) ---
print(f"🔧 [CONFIG] Projekt: {BASE_DIR}")
print(f"🔧 [CONFIG] Dokumenty: {DOCS_PATH}")
print(f"🔧 [CONFIG] Baza danych: {DB_PATH}")