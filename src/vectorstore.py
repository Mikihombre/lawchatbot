# src/vectorstore.py
import os
import json
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


MAX_BATCH = 4000  # bezpieczny limit dla Chroma


def _is_allowed_metadata_value(v: Any) -> bool:
    """
    Chroma akceptuje tylko:
    str, int, float, bool, None
    """
    return isinstance(v, (str, int, float, bool)) or v is None


def _sanitize_metadata(meta: dict) -> dict:
    """
    Usuwa z metadanych wszystko, czego Chroma nie przyjmie
    (listy, dict, itp.)
    """
    clean = {}
    for k, v in meta.items():
        if _is_allowed_metadata_value(v):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def _list_existing_sources(db: Chroma) -> Set[str]:
    """Zwraca nazwy plików JSON już obecnych w bazie."""
    try:
        raw = db.get(include=["metadatas"])
        metas = raw.get("metadatas") or []
        return {
            m.get("source")
            for m in metas
            if isinstance(m, dict) and m.get("source")
        }
    except Exception:
        return set()


def _fallback_act_name_from_filename(filename: str) -> str:
    """
    Fallback, gdy JSON nie ma metadata['act_name'].
    Chcemy format spójny z routing.py, np.:
    'kodeks_postępowania_karnego.json' -> 'Kodeks postępowania karnego'
    Bez .title(), bo to rozwala dopasowanie.
    """
    base = filename.replace(".json", "").replace("_", " ").strip()
    if not base:
        return "Nieznany akt prawny"
    return base[0].upper() + base[1:]


# ============================================================
#  JSON LOADER
# ============================================================

def _load_json_files(docs_path: str, new_files: List[str]) -> List[Document]:
    """
    Wczytuje JSON-y wygenerowane przez parsery.
    """
    documents: List[Document] = []

    for filename in new_files:
        file_path = os.path.join(docs_path, filename)
        print(f"   📖 Czytam plik: {filename}...")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                print(f"   ⚠️ Pomijam {filename} – JSON nie jest listą.")
                continue

            fallback_act_name = _fallback_act_name_from_filename(filename)

            for item in data:
                if not isinstance(item, dict):
                    continue

                content = (
                    item.get("text_content")
                    or item.get("text")
                    or item.get("content")
                )

                if not isinstance(content, str) or len(content.strip()) < 5:
                    continue

                raw_meta = item.get("metadata", {})
                if not isinstance(raw_meta, dict):
                    raw_meta = {}

                meta = _sanitize_metadata(raw_meta)

                # Wymuszone pola
                meta["source"] = filename

                # Kluczowa zmiana: act_name bierzemy z metadanych jeśli jest,
                # w przeciwnym razie fallback z nazwy pliku (bez title-case).
                meta_act_name = meta.get("act_name")
                if not isinstance(meta_act_name, str) or not meta_act_name.strip():
                    meta_act_name = fallback_act_name
                meta["act_name"] = meta_act_name

                # Page default
                meta["page"] = meta.get("page", 1)

                full_content = (
                    f"USTAWA: {meta_act_name}\n"
                    f"TREŚĆ PRZEPISU:\n{content}"
                )

                documents.append(
                    Document(
                        page_content=full_content,
                        metadata=meta,
                    )
                )

        except Exception as e:
            print(f"   ❌ Błąd przy wczytywaniu {filename}: {e}")

    return documents


# ============================================================
#  MAIN
# ============================================================

def build_vector_store(embeddings) -> Tuple[Chroma, Any]:
    """
    Buduje lub aktualizuje bazę Chroma.
    """


    db: Optional[Chroma] = None
    existing_sources: Set[str] = set()

    # 1) Czy baza już istnieje?
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        print(f"✅ Wykryto istniejącą bazę w '{DB_PATH}'.")
        db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings,
        )
        existing_sources = _list_existing_sources(db)
    else:
        print("⚡ Tworzę nową, pustą bazę Chroma.")

    # 2) JSON-y w folderze
    all_files = [
        f for f in os.listdir(DOCS_PATH)
        if f.lower().endswith(".json")
    ]
    print("[DEBUG] DOCS_PATH =", DOCS_PATH)
    print("[DEBUG] JSON files in DOCS_PATH:")
    for f in sorted(all_files):
        print(" -", f)

    new_files = [f for f in all_files if f not in existing_sources]

    print("\n📊 STATUS BAZY:")
    print(f" - Pliki w bazie: {len(existing_sources)}")
    print(f" - Pliki w folderze: {len(all_files)}")
    print(f" - Do dodania: {len(new_files)}")

    # 3) Jeśli nic nowego, tylko zwróć DB+retriever
    if not new_files:
        print("✅ Baza jest aktualna.")
        if db is None:
            db = Chroma(
                persist_directory=DB_PATH,
                embedding_function=embeddings,
            )

        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={RETRIEVER_K},
        )
        return db, retriever

    # 4) Wczytanie JSON-ów
    print(f"\n🚀 Rozpoczynam procesowanie {len(new_files)} nowych plików...")
    raw_docs = _load_json_files(DOCS_PATH, new_files)

    if not raw_docs:
        raise RuntimeError("❌ Nie udało się wczytać żadnych dokumentów.")

    # 5) Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    final_chunks = splitter.split_documents(raw_docs)
    print(f"✂️  Dokumenty pocięte na {len(final_chunks)} chunków.")

    # 6) Zapis do bazy
    if db is None:
        db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings,
        )

    total_added = 0
    print("💾 Zapisywanie do bazy wektorowej...")

    for batch in _batched(final_chunks, MAX_BATCH):
        db.add_documents(batch)
        total_added += len(batch)
        print(f"   → Zapisano {total_added}/{len(final_chunks)}")

    print("✅ Baza zaktualizowana.")

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )
    return db, retriever
