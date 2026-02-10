import os
from langchain_chroma import Chroma
from src.embeddings import build_embeddings
from src.config import DB_PATH

def debug_article_3():
    print(f"📂 Otwieram bazę w: {DB_PATH}")
    embeddings = build_embeddings()
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    # 1. Sprawdźmy wszystkie unikalne numery artykułów, które wyglądają jak "3"
    print("\n🔍 Szukam dokumentów z metadata['article'] == '3'...")
    
    results = db.get(where={"article": "3"})
    ids = results['ids']
    metadatas = results['metadatas']
    documents = results['documents']
    
    print(f"👉 Znaleziono {len(ids)} chunków dla artykułu 3.")
    
    if len(ids) == 0:
        print("❌ BŁĄD: Baza nie widzi artykułu '3'. Sprawdźmy sąsiednie...")
        # Pobierzmy próbkę, żeby zobaczyć jak wyglądają metadane
        sample = db.get(limit=5)
        print("Przykładowe metadane w bazie:", sample['metadatas'])
    else:
        print("✅ Artykuł 3 istnieje. Oto podgląd pierwszego chunka:")
        print(f"--- Metadata: {metadatas[0]}")
        print(f"--- Treść (pierwsze 200 znaków): {documents[0][:200]}...")

if __name__ == "__main__":
    debug_article_3()