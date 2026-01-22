import re
import spacy
from typing import List, Dict, Any
from langchain_core.documents import Document
from collections import defaultdict # Zaimportuj defaultdict

# --- REGEX (bez zmian) ---
FIND_ARTICLES_REGEX = re.compile(r"(?:Art\.|Artykuł)\s*(\d+[a-z]*)", re.IGNORECASE)
FIND_PARAGRAPHS_REGEX = re.compile(r"(?:§|Par\.|Paragraf)\s*(\d+[a-z]*)", re.IGNORECASE)

def extract_regex_metadata(text: str) -> Dict[str, Any]:
    articles = list(set(FIND_ARTICLES_REGEX.findall(text)))
    paragraphs = list(set(FIND_PARAGRAPHS_REGEX.findall(text)))
    
    metadata = {}
    if articles:
        metadata["articles"] = ", ".join(articles) # To już jest poprawne (string)
    if paragraphs:
        metadata["paragraphs"] = ", ".join(paragraphs) # To też jest poprawne (string)
    return metadata

# --- SPACY NER (ZMIANY TUTAJ) ---
print("Ładowanie modelu spaCy (pl_core_news_lg)...")
nlp = spacy.load("pl_core_news_lg")
print("Model spaCy załadowany.")

def extract_ner_metadata(text: str) -> Dict[str, str]:
    """
    Ekstrahuje encje NER i "spłaszcza" je do słownika stringów
    (np. {"pers": "Jan, Adam", "loc": "Warszawa"})
    """
    doc = nlp(text)
    entities_by_label = defaultdict(list)
    
    # Pogrupuj encje według ich etykiet
    for ent in doc.ents:
        # Używamy set() aby uniknąć duplikatów (np. "Jan Kowalski", "Jan Kowalski")
        entities_by_label[ent.label_].append(ent.text)

    # Stwórz finalny słownik metadanych, łącząc listy w stringi
    flat_metadata = {}
    for label, texts in entities_by_label.items():
        # Usuń duplikaty i połącz przecinkiem
        unique_texts = list(set(texts))
        # Klucz będzie wyglądał tak: "ner_pers", "ner_org"
        flat_metadata[f"ner_{label.lower()}"] = ", ".join(unique_texts) 
        
    return flat_metadata

# --- GŁÓWNA FUNKCJA WZBOGACAJĄCA (ZMIANY TUTAJ) ---
def enrich_chunks(chunks: List[Document]) -> List[Document]:
    print("Rozpoczynam wzbogacanie fragmentów (Regex + NER)...")
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        if (i + 1) % (total_chunks // 10 or 1) == 0:
            print(f"Przetworzono {i + 1}/{total_chunks} fragmentów...")

        # 1. Wzbogać o Regex (Art. i §)
        regex_md = extract_regex_metadata(chunk.page_content)
        chunk.metadata.update(regex_md)
        
        # 2. Wzbogać o NER (Encje)
        ner_md = extract_ner_metadata(chunk.page_content)
        chunk.metadata.update(ner_md)

    print("✅ Zakończono wzbogacanie fragmentów.")
    return chunks


def extract_legal_refs(query: str):
    act = None
    article = None
    paragraph = None
    query = query.lower()

    if "kodeks cywilny" in query:
        act = "Kodeks Cywilny"
    elif "kodeks karny" in query:
        act = "Kodeks Karny"
    elif "kodeks rodzinny" in query:
        act = "Kodeks Rodzinny i Opiekuńczy"
    # Dodaj inne akty

    article_match = re.search(r"art\.?\s?(\d+[a-z]*)", query)
    if article_match:
        article = article_match.group(1)

    paragraph_match = re.search(r"§\s?(\d+[a-z]*)", query)
    if paragraph_match:
        paragraph = paragraph_match.group(1)

    return act, article, paragraph

def fallback_by_metadata(query, vectorstore, top_k=20):
    act, article, paragraph = extract_legal_refs(query)
    if not act or not article:
        return None

    docs = vectorstore.similarity_search(f"Art. {article}", k=top_k)

    filtered = []
    for d in docs:
        md = d.metadata
        if md.get("act_name") == act and (
            md.get("article") == article or article in md.get("articles", "")
        ) and (
            paragraph is None or paragraph in md.get("paragraphs", "") or md.get("paragraph") in [paragraph, "all"]
        ):
            filtered.append(d)

    if filtered:
        print("✅ Fallback zadziałał przez metadane.")
        return filtered
    return None