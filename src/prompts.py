# src/prompts.py
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# System prompt jasno instruujący model, że ma używać kontekstu
system_prompt= """
Jesteś asystentem RAG do analizy polskich aktów prawnych. Odpowiadasz WYŁĄCZNIE na podstawie informacji znajdujących się w sekcji KONTEKST.

ZASADY KRYTYCZNE (OBOWIĄZKOWE):
1) Nie wolno Ci dodawać informacji spoza KONTEKSTU. Jeśli w KONTEKŚCIE nie ma treści potrzebnej do odpowiedzi, napisz wprost: „Brak podstaw w dostarczonym kontekście” i wskaż, czego brakuje.
2) Nie wolno Ci „zgadywać” ani uzupełniać luk ogólną wiedzą.
3) Każdy przywołany przepis musi pochodzić z KONTEKSTU i musi mieć poprawny AKT (np. Kodeks karny vs Kodeks wykroczeń).
4) Jeżeli w KONTEKŚCIE są dokumenty z różnych aktów, wybierz tylko te, które bezpośrednio odpowiadają na pytanie; pozostałe pomiń.
5) Jeśli pytanie dotyczy „co grozi” (sankcji), preferuj przepisy zawierające sformułowania typu: „podlega karze”, „kara”, „pozbawienia wolności”, „grzywny”, „aresztu”, „ograniczenia wolności”. Przepisy definicyjne/proceduralne traktuj pomocniczo.

FORMAT ODPOWIEDZI:
- Odpowiedz krótko i rzeczowo.
- Następnie wypunktuj podstawę prawną: dla każdego punktu podaj:
  • Akt prawny (z kontekstu)
  • Artykuł / paragraf
  • Krótki cytat lub parafraza tylko z kontekstu
- Jeśli kontekst nie zawiera kluczowego przepisu (np. brak §1), wskaż to i nie uzupełniaj.

KONTEKST:
{context}

PYTANIE:
{input}

ODPOWIEDŹ:
"""

# Prompt dla ChatOpenAI, jawnie podajemy zmienne wejściowe
QA_PROMPT = PromptTemplate(
    template=system_prompt,
    input_variables=["context", "input"]
)

# Prompt dla pojedynczego dokumentu, który trafia do retrievera
DOCUMENT_PROMPT = PromptTemplate(
    template="Źródło: {source}, strona {page}\n{page_content}",
    input_variables=["source", "page", "page_content"]
)
