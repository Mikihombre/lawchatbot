import os
from src.config import DEBUG

def debug_retrieved_documents(docs, query, max_chars=400):
    print("\n" + "=" * 90)
    print("[DEBUG] QUERY:", query)
    print(f"[DEBUG] ZWRÓCONO {len(docs)} DOKUMENTÓW")
    print("=" * 90)

    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}

        print(f"\n--- DOKUMENT #{i} ---")
        print(f"ŹRÓDŁO     : {meta.get('source', 'brak')}")
        print(f"AKT        : {meta.get('act_name', 'brak')}")
        print(f"ARTYKUŁ    : {meta.get('article', 'brak')}")
        print(f"STRONA     : {meta.get('page', 'brak')}")
        print("-" * 90)

        text = doc.page_content.strip().replace("\n", " ")
        print(text[:max_chars] + ("..." if len(text) > max_chars else ""))

    print("\n" + "=" * 90)

def display_answer(result):
    docs = result.get("context") or result.get("documents") or []
    answer = result.get("answer") or result.get("output") or "Brak odpowiedzi."

    if DEBUG and docs:
        print("\n[DEBUG] Podgląd pierwszego dokumentu:")
        try:
            print(f"Źródło: {docs[0].metadata.get('source')}, strona {docs[0].metadata.get('page')}")
            print(docs[0].page_content[:500])
        except Exception:
            print("[DEBUG] Nie udało się odczytać podglądu dokumentu.")

    print("\nOdpowiedź:")
    print(answer.strip())

    print("\nŹródła:")
    if not docs:
        print("- Brak źródeł.")
    else:
        for doc in docs:
            try:
                src = doc.metadata.get("source", "Nieznane źródło")
                page = doc.metadata.get("page", "N/A")
                print(f"- {os.path.basename(str(src))}, strona {page}")
            except Exception:
                print("- [Nie udało się odczytać metadanych źródła]")
