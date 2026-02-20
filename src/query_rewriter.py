# src/query_rewriter.py
import json
import re
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate


REWRITER_SYSTEM_PROMPT = """Jesteś modułem normalizacji pytań do prawniczego systemu RAG.
Twoje zadanie: przekształcić pytanie użytkownika na wersję precyzyjną, formalną i jednoznaczną, bez zmiany sensu.

Zasady:
- NIE dodawaj nowych faktów, których nie ma w pytaniu.
- NIE odpowiadaj na pytanie.
- Jeśli pytanie jest niejednoznaczne, nie zgaduj: zachowaj treść i dodaj tylko neutralne doprecyzowanie językowe.
- Wypisz słowa kluczowe, które pomogą wyszukiwaniu (BM25 + wektory).
- Wypisz maks. 2 akty prawne jako "act_hints" tylko wtedy, gdy są bardzo prawdopodobne.
- Dozwolone akty w act_hints: ["Udip","Kodeks postępowania administracyjnego","Rodo ue","Rodo","PPSA","Prawo budowlane"].

Zwróć WYŁĄCZNIE czysty JSON w formacie:
{{
  "rewritten_query": "...",
  "keywords": ["..."],
  "act_hints": ["..."],
  "notes": "..."
}}
"""

_rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", REWRITER_SYSTEM_PROMPT),
    ("human", "Pytanie użytkownika:\n{query}")
])


def _parse_json_only(text: str) -> Dict[str, Any]:
    clean = text.replace("```json", "").replace("```", "").strip()
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if not m:
        return {"rewritten_query": "", "keywords": [], "act_hints": [], "notes": f"FORMAT_ERROR: {clean[:120]}"}
    try:
        return json.loads(m.group(0))
    except Exception as e:
        return {"rewritten_query": "", "keywords": [], "act_hints": [], "notes": f"PARSE_ERROR: {e}"}


def rewrite_query(rewriter_llm: Any, query: str) -> Dict[str, Any]:
    raw = (_rewriter_prompt | rewriter_llm).invoke({"query": query}).content
    data = _parse_json_only(raw)

    rq = (data.get("rewritten_query") or "").strip()
    if not rq:
        rq = query.strip()

    kw = data.get("keywords") or []
    if not isinstance(kw, list):
        kw = []

    hints = data.get("act_hints") or []
    if not isinstance(hints, list):
        hints = []

    kw = [str(x).strip() for x in kw if str(x).strip()][:12]
    hints = [str(x).strip() for x in hints if str(x).strip()][:2]
    notes = (data.get("notes") or "").strip()

    return {"rewritten_query": rq, "keywords": kw, "act_hints": hints, "notes": notes}
