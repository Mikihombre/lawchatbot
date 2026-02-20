# src/vectorstore.py
import os
import json
import re
from typing import List, Set, Tuple, Any, Optional

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import (
    DOCS_PATH,
    DB_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVER_K
)

# ============================================================
#  HELPERS
# ============================================================

def _batched(seq, batch_size: int):
    """Dzieli listę na batch'e."""
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]

MAX_BATCH = 4000 

def _is_allowed_metadata_value(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None

def _sanitize_metadata(meta: dict) -> dict:
    clean = {}
    for k, v in meta.items():
        if _is_allowed_metadata_value(v):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean

def _list_existing_sources(db: Chroma) -> Set[str]:
    """Zwraca zbiór nazw plików (source), które są już w bazie."""
    try:
        # Pobieramy tylko metadane
        raw = db.get(include=["metadatas"])
        metas = raw.get("metadatas") or []
        
        sources = set()
        for m in metas:
            if isinstance(m, dict) and m.get("source"):
                # Normalizujemy nazwę źródła (bierzemy samą nazwę pliku, bez ścieżki)
                src = os.path.basename(m.get("source"))
                sources.add(src)
        
        return sources
    except Exception as e:
        print(f"⚠️ Błąd odczytu źródeł z bazy: {e}")
        return set()

def _clean_act_name(raw_source: str) -> str:
    """
    Zamienia 'kodeks_postępowania_administracyjnego.txt' -> 'Kodeks postępowania administracyjnego'
    Zamienia 'udip.txt' -> 'Udip'
    """
    # Usuwamy rozszerzenie
    name = raw_source.replace(".txt", "").replace(".json", "")
    # Zamieniamy podkreślenia na spacje
    name = name.replace("_", " ")
    # Usuwamy dziwne znaki
    name = name.strip()
    # Wielka litera na początku
    if name:
        return name[0].upper() + name[1:]
    return "Ustawy"

# ============================================================
#  LOADER (Poprawiona wersja)
# ============================================================

def _load_json_files(docs_path: str, filenames: List[str]) -> List[Document]:
    documents: List[Document] = []

    for filename in filenames:
        file_path = os.path.join(docs_path, filename)
        print(f"   📖 [LOADER] Otwieram plik: {filename}...")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines_processed = 0
                
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line: continue
                    
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # TREŚĆ
                    content = item.get("rag_text") or item.get("content") or item.get("text")
                    if not content or len(str(content)) < 5:
                        continue

                    # ---------------------------------------------------------
                    # FIX: LOGIKA NAZEWNICTWA AKTÓW (Kluczowe dla Routingu!)
                    # ---------------------------------------------------------
                    
                    # 1. Ustal źródło (nazwę pliku wewnątrz metadanych)
                    inner_source = item.get("source", filename)

                    # 2. Ustal nazwę aktu (act_name)
                    # JEŚLI ETL zapisał "act_name" (np. "PPSA"), bierzemy to w ciemno.
                    # JEŚLI NIE, dopiero wtedy używamy funkcji czyszczącej.
                    act_name_from_json = item.get("act_name")
                    
                    if act_name_from_json:
                        real_act_name = act_name_from_json
                    else:
                        real_act_name = _clean_act_name(inner_source)

                    # ---------------------------------------------------------

                    # METADANE
                    meta = {
                        "source": inner_source,      # np. PPSA (ważne dla deduplikacji)
                        "act_name": real_act_name,   # np. PPSA (ważne dla filtrowania w Chroma)
                        "article": item.get("article", ""),
                        "hierarchy": item.get("hierarchy", ""),
                    }
                    
                    # Opcjonalne czyszczenie metadanych (jeśli masz taką funkcję)
                    if "_sanitize_metadata" in globals():
                        meta = _sanitize_metadata(meta)

                    documents.append(
                        Document(
                            page_content=content,
                            metadata=meta,
                        )
                    )
                    lines_processed += 1

                print(f"      -> Przetworzono rekordów: {lines_processed}")

        except Exception as e:
            print(f"   ❌ Błąd krytyczny przy pliku {filename}: {e}")

    return documents

# ============================================================
#  MAIN BUILDER (Poprawiona logika inkrementalna)
# ============================================================

def build_vector_store(embeddings) -> Tuple[Chroma, Any]:
    print(f"\n📂 Inicjalizacja ChromaDB w: '{DB_PATH}'")
    
    db: Optional[Chroma] = None
    existing_sources: Set[str] = set()

    # 1. Sprawdź czy baza istnieje i pobierz listę źródeł
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        try:
            db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
            existing_sources = _list_existing_sources(db)
            print(f"✅ Baza załadowana. Znaleziono {len(existing_sources)} źródeł.")
        except Exception as e:
            print(f"❌ Błąd ładowania bazy: {e}. Tworzę nową.")
            db = None
    else:
        print("⚡ Tworzę nową, pustą bazę.")

    if db is None:
        db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # 2. Skanowanie plików na dysku
    all_files_on_disk = [f for f in os.listdir(DOCS_PATH) if f.lower().endswith((".json", ".jsonl"))]
    
    # 3. Wykrywanie nowych plików
    # Porównujemy nazwy plików. Jeśli 'ppsa.jsonl' jest w existing_sources, to go pomijamy.
    new_files = [f for f in all_files_on_disk if f not in existing_sources]

    print("\n📊 STATUS:")
    print(f" - Pliki na dysku: {len(all_files_on_disk)}")
    print(f" - Pliki w bazie:  {len(existing_sources)}")
    print(f" - Do dodania:     {len(new_files)} -> {new_files}")

    # 4. Jeśli brak nowości -> Koniec
    if not new_files:
        print("✅ Baza jest aktualna. Pomijam indeksowanie.")
        return db, db.as_retriever(search_kwargs={"k": RETRIEVER_K})

    # 5. Ładowanie TYLKO nowych plików
    print(f"\n🚀 Przetwarzanie {len(new_files)} nowych plików...")
    new_docs = _load_json_files(DOCS_PATH, new_files)
    
    if not new_docs:
        print("⚠️ Pliki są puste lub uszkodzone.")
        return db, db.as_retriever(search_kwargs={"k": RETRIEVER_K})

    # 6. Chunking (Tylko dla nowych)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    final_chunks = splitter.split_documents(new_docs)
    print(f"✂️  Wygenerowano {len(final_chunks)} nowych chunków.")

    # 7. Zapis (Tylko nowe chunki)
    total_added = 0
    print("💾 Dopisuję do bazy wektorowej...")
    
    for batch in _batched(final_chunks, MAX_BATCH):
        db.add_documents(batch)
        total_added += len(batch)
        print(f"   → Zapisano partię {total_added}/{len(final_chunks)}")

    print("✅ Aktualizacja zakończona pomyślnie.")
    return db, db.as_retriever(search_kwargs={"k": RETRIEVER_K})