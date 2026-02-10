import json
import os
from datasets import load_dataset
from tqdm import tqdm

# --- KONFIGURACJA ---
OUTPUT_DIR = "datasets"
OUTPUT_FILENAME = "dataset_udip.jsonl"
MAX_SAMPLES = 2000  # Limit orzeczeń

OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# Słowa kluczowe (Dostęp do Informacji Publicznej)
KEYWORDS = [
    "dostęp do informacji publicznej",
    "informacja publiczna",
    "wniosek o udostępnienie",
    "ustawa o dostępie",
    "bezczynność organu"
]

def is_relevant(text):
    if not text:
        return False
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in KEYWORDS)

def clean_text(text):
    return " ".join(text.split())

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Rozpoczynam pobieranie do pliku: {OUTPUT_PATH}")
    
    # Ładowanie datasetu
    dataset = load_dataset("JuDDGES/pl-nsa-enriched", split="train", streaming=True)
    
    collected_count = 0
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, record in tqdm(enumerate(dataset)):
            
            # Priorytet ma "reasons_for_judgment" (Uzasadnienie)
            content = record.get("reasons_for_judgment")
            
            # Zapasowo "full_text"
            if not content:
                content = record.get("full_text")
            
            if not content:
                continue
                
            # Filtrowanie
            if is_relevant(content):
                
                cleaned_content = clean_text(content)
                
                if len(cleaned_content) < 500:
                    continue

                doc_data = {
                    "id": record.get("judgment_id"),
                    "sygnatura": record.get("docket_number"),
                    "data_orzeczenia": record.get("judgment_date"), # Tutaj był błąd, naprawiamy to niżej
                    "sad": record.get("court_name"),
                    "tresc": cleaned_content,
                    "teza": record.get("thesis"),
                    "source": "JuDDGES"
                }
                
                # --- NAPRAWA BŁĘDU ---
                # Dodano: default=str 
                # To automatycznie zamienia daty (i inne obiekty) na tekst przy zapisie
                f.write(json.dumps(doc_data, ensure_ascii=False, default=str) + "\n")
                
                collected_count += 1
                
                if collected_count >= MAX_SAMPLES:
                    print(f"\nOsiągnięto limit {MAX_SAMPLES} orzeczeń.")
                    break
    
    print(f"Gotowe! Sprawdź plik w: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()