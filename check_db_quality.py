import os
from langchain_chroma import Chroma
from src.embeddings import build_embeddings
from src.config import DB_PATH

def inspect_database():
    print(f"📂 Otwieram bazę w: {DB_PATH}")
    embeddings = build_embeddings()
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    # 1. SPRAWDŹMY SŁOWNICZEK (ART. 3)
    # Powinien być pocięty na wiele kawałków (np. >15)
    print("\n🔍 --- TEST 1: Artykuł 3 (Słowniczek) ---")
    results_art3 = db.get(where={"article": "3"})
    count_3 = len(results_art3['ids'])
    print(f"👉 Liczba chunków dla Art. 3: {count_3}")
    
    if count_3 > 0:
        first_content = results_art3['documents'][0]
        print("📝 Przykładowa treść (sprawdź czy jest Context Injection):")
        print("-" * 40)
        print(first_content[:300] + "...") # Pokaż początek
        print("-" * 40)
        if "Ilekroć w ustawie" in first_content or "PRZEPIS:" in first_content:
            print("✅ SUKCES: Context Injection działa (widać nagłówek w treści punktu).")
        else:
            print("⚠️ OSTRZEŻENIE: Nie widzę wstrzykniętego kontekstu.")

    # 2. SPRAWDŹMY TABELĘ OPŁAT
    # Powinna być jako "TABELA" lub w Załączniku
    print("\n🔍 --- TEST 2: Tabela Opłat (Załącznik) ---")
    # Szukamy po metadanych article="TABELA" (jeśli parser tak zapisał) 
    # lub po prostu szukamy tekstu "Kategoria XVII"
    results_table = db.get(where={"article": "TABELA"})
    
    if not results_table['ids']:
        # Fallback: szukamy w treści
        print("   (Szukam po treści 'Kategoria XVII', bo nie znalazłem po metadanych TABELA...)")
        all_docs = db.get() # To może chwilę potrwać, pobiera wszystko
        found = False
        for doc in all_docs['documents']:
            if "Kategoria XVII" in doc and "500 zł" in doc: # Szukamy fragmentu tabeli
                print("✅ SUKCES: Tabela z opłatami jest w bazie!")
                print("📝 Fragment tabeli:")
                print(doc[:300] + "...")
                found = True
                break
        if not found:
             print("❌ BŁĄD: Nie widzę tabeli opłat w bazie.")
    else:
        print(f"✅ SUKCES: Znaleziono {len(results_table['ids'])} chunków oznaczonych jako TABELA.")
        print("📝 Fragment:")
        print(results_table['documents'][0][:300] + "...")

    # 3. SPRAWDŹMY INSTALACJE GAZOWE (ART. 29)
    print("\n🔍 --- TEST 3: Artykuł 29 (Instalacje) ---")
    results_art29 = db.get(where={"article": "29"})
    print(f"👉 Liczba chunków dla Art. 29: {len(results_art29['ids'])}")

if __name__ == "__main__":
    inspect_database()