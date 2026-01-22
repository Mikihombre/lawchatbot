# src/rerank.py
from langchain_core.documents import Document
from typing import List, Set
# Upewnij się, że importujesz poprawną funkcję z utils!
from src.utils import extract_ner_metadata 

# -----------------------------------------------------------------
# Ta prosta funkcja rerankera jest OK
# (Upewnij się, że masz tę "kuloodporną" wersję z poprzedniej rozmowy)
# -----------------------------------------------------------------
def rerank_neural(model, query: str, docs: List[Document], top_n: int = 4) -> List[Document]:
    if not docs or not query:
        return []
        
    print(f"Reranker (Prosty): Otrzymano {len(docs)} dokumentów.")
    
    valid_docs_with_content = []
    pairs = []
    for doc in docs:
        if doc.page_content and doc.page_content.strip():
            valid_docs_with_content.append(doc)
            pairs.append([query, doc.page_content])
        else:
            print(f"Reranker (Prosty): POMIJANIE (pusta treść): {doc.metadata}")
    
    if not pairs:
        print("Reranker (Prosty): Brak poprawnych dokumentów do oceny.")
        return []
    
    try:
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception as e:
        print(f"Reranker (Prosty): KRYTYCZNY BŁĄD model.predict: {e}")
        return []

    doc_score_pairs = list(zip(valid_docs_with_content, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    sorted_docs = [doc for doc, score in doc_score_pairs[:top_n]]
    print(f"Reranker (Prosty): Zwracam {len(sorted_docs)} najlepszych dokumentów.")
    return sorted_docs

# -----------------------------------------------------------------
# NOWA, POPRAWIONA funkcja rerankera z encjami
# -----------------------------------------------------------------
def _get_all_entity_values(metadata: dict) -> Set[str]:
    """
    Helper do wyciągania WSZYSTKICH kluczowych danych (NER + Regex) ze "spłaszczonych" metadanych.
    """
    entity_values = set()
    for key, value_str in metadata.items():
        # TERAZ UWZGLĘDNIAMY TEŻ ARTYKUŁY I PARAGRAFY!
        if key.startswith("ner_") or key in ["articles", "paragraphs"]: 
            if isinstance(value_str, str): # Zabezpieczenie
                # Dzieli string "Jan Kowalski, 278" na listę
                entities = [e.strip() for e in value_str.split(',')]
                entity_values.update(entities)
    return entity_values

def rerank_neural_with_entities(model, query: str, docs: List[Document], top_n: int = 4) -> List[Document]:
    """
    Reranking, który "boostuje" dokumenty pasujące do encji z zapytania.
    Działa na "płaskich" metadanych (np. ner_pers: "Jan Kowalski").
    """
    
    # 1. Pobierz encje z zapytania (w nowym, płaskim formacie)
    query_ner_map = extract_ner_metadata(query)
    query_entity_values = _get_all_entity_values(query_ner_map)
    
    # 2. Wykonaj normalny reranking, ale poproś o WIĘCEJ wyników
    # Dajemy sobie większe pole do filtrowania (np. top 20)
    print(f"Reranker (NER): Wykonuję reranking bazowy top 20...")
    sorted_docs = rerank_neural(model, query, docs, top_n=20) 
    
    # 3. Jeśli zapytanie NIE zawiera żadnych encji, po prostu zwróć top_n
    if not query_entity_values:
        print("Reranker (NER): Zapytanie nie zawiera encji, zwracam proste top_n.")
        return sorted_docs[:top_n]
        
    print(f"Reranker (NER): Znaleziono encje w zapytaniu: {query_entity_values}")
    
    # 4. Jeśli zapytanie MA encje, przefiltruj posortowane wyniki
    filtered_docs = []
    for doc in sorted_docs:
        doc_entity_values = _get_all_entity_values(doc.metadata)
        
        # SPRAWDŹ, CZY JEST CZĘŚĆ WSPÓLNA
        if query_entity_values.intersection(doc_entity_values):
            print(f"Reranker (NER): ZNALEZIONO DOPASOWANIE encji w {doc.metadata.get('source')}")
            filtered_docs.append(doc)
            
        if len(filtered_docs) >= top_n:
            break # Znaleźliśmy wystarczająco dużo pasujących dokumentów
    
    # 5. Zabezpieczenie (Fallback)
    # Jeśli filtracja nic nie dała (bo np. żaden dokument nie pasował),
    # zwróć po prostu 4 najlepsze wyniki z normalnego rerankingu.
    if not filtered_docs:
        print("Reranker (NER): Filtracja nie znalazła nic, zwracam proste top_n.")
        return sorted_docs[:top_n]
        
    print(f"Reranker (NER): Zwracam {len(filtered_docs)} przefiltrowanych dokumentów.")
    return filtered_docs