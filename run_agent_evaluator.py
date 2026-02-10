import json
import re

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from src.routing_retriever import ActRoutingRetriever
from src.embeddings import build_embeddings
from src.vectorstore import build_vector_store
from src.rag_chain import build_rag_chain
from src.prompts import QA_PROMPT, DOCUMENT_PROMPT


# =========================
# KONFIG (na sztywno tutaj)
# =========================
SERVER_URL = "http://127.0.0.1:11434"

LAWYER_MODEL = "SpeakLeash/bielik-11b-v3.0-instruct:bf16"
JUDGE_MODEL  = "gemma3:12b"

TEMPERATURE = 0.0


JUDGE_SYSTEM_PROMPT = """Jesteś surowym egzaminatorem prawnym. Twoim zadaniem jest ocena odpowiedzi innego bota.

Otrzymasz:
1) Pytanie użytkownika
2) Odpowiedź bota
3) Źródła (fragmenty aktów)

Oceń w skali 1-5 (1=źle, 5=idealnie) wg:
- zgodność ze źródłami
- podanie podstawy prawnej (art.)
- trafność i kompletność

Zasady:
- Jeśli źródła są puste lub nie zawierają podstawy: score maks. 2.
- Jeśli odpowiedź wykracza poza źródła lub halucynuje przepisy: score maks. 2.
- "reasoning" max 240 znaków.

Zwróć TYLKO czysty JSON dokładnie w formacie:
{{
  "score": <liczba całkowita 1-5>,
  "reasoning": "<krótkie uzasadnienie>"
}}
"""


def parse_judge_json(text: str) -> dict:
    clean = text.replace("```json", "").replace("```", "").strip()
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if not m:
        return {"score": 1, "reasoning": f"Błąd formatu JSON: {clean[:120]}..."}

    try:
        data = json.loads(m.group(0))
    except Exception as e:
        return {"score": 1, "reasoning": f"Nie da się zparsować JSON: {e}"}

    score = data.get("score", 1)
    reasoning = data.get("reasoning", "")

    if not isinstance(score, int) or not (1 <= score <= 5):
        score = 1
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    reasoning = reasoning.strip()
    if len(reasoning) > 240:
        reasoning = reasoning[:237] + "..."

    return {"score": score, "reasoning": reasoning}


def main():
    print("👨‍⚖️ Uruchamiam Agenta Egzaminatora (LLM-as-a-Judge)...")

    # 1) Prawnik (Bielik)
    llm = ChatOllama(base_url=SERVER_URL, model=LAWYER_MODEL, temperature=TEMPERATURE)

    # 2) Sędzia (Gemma)
    judge_llm = ChatOllama(base_url=SERVER_URL, model=JUDGE_MODEL, temperature=TEMPERATURE)

    embeddings = build_embeddings()
    db, _ = build_vector_store(embeddings)
    retriever = ActRoutingRetriever(vectorstore=db, max_acts=2)
    rag_chain = build_rag_chain(llm, retriever, QA_PROMPT, DOCUMENT_PROMPT)

    test_questions = [
    {"id": "q1", "query": "Ile dni ma urząd na udostępnienie informacji publicznej?"},
    {"id": "q2", "query": "Jaki jest termin na wniesienie odwołania od decyzji administracyjnej?"},
    {"id": "q3", "query": "Kto jest stroną w postępowaniu administracyjnym?"},
    {"id": "q4", "query": "Co grozi za nieudostępnienie informacji publicznej wbrew obowiązkowi?"},
    {"id": "q5", "query": "Czy można żądać informacji przetworzonej?"},
    {"id": "q6", "query": "Kiedy organ może załatwić sprawę milcząco?"},
    {"id": "q7", "query": "O czym mówi art. 13 ustawy o dostępie do informacji publicznej?"},
    {"id": "q8", "query": "Jaka jest opłata za udostępnienie informacji?"},
    {"id": "q9", "query": "Co to jest ponowne wykorzystywanie informacji sektora publicznego?"},
    {"id": "q10", "query": "Kiedy można zawiesić postępowanie administracyjne?"},
    {"id": "q11", "query": "W jakim terminie należy zgłosić naruszenie ochrony danych osobowych organowi nadzorczemu?"},
    {"id": "q12", "query": "Co może być dowodem w postępowaniu administracyjnym?"},
    {"id": "q13", "query": "Czy prywatność osoby fizycznej zawsze ogranicza dostęp do informacji publicznej?"},
    {"id": "q14", "query": "Kiedy pracownik organu administracji podlega wyłączeniu od udziału w sprawie?"},
    {"id": "q15", "query": "Jaka jest maksymalna kara pieniężna za naruszenie RODO dla przedsiębiorstwa?"},
    {"id": "q16", "query": "W jakiej formie wnosi się podanie do organu administracji?"},
    {"id": "q17", "query": "Czy organ może sprostować błędy pisarskie w wydanej decyzji?"},
    {"id": "q18", "query": "Jaka jest stawka podatku VAT na usługi prawne?"},
    {"id": "q19", "query": "Co to jest profilowanie zgodnie z RODO?"},
    {"id": "q20", "query": "Kiedy decyzja administracyjna staje się ostateczna?"},
    {"id": "q21", "query": "Co się dzieje, gdy ostatni dzień terminu na wniesienie odwołania przypada w niedzielę?"},
    {"id": "q22", "query": "Czy jawność informacji publicznej jest ograniczona ze względu na prywatność osoby fizycznej pełniącej funkcję publiczną?"},
    {"id": "q23", "query": "Czy organ administracji może aresztować stronę za niewykonanie decyzji?"},
    {"id": "q24", "query": "Co musi zawierać metryka sprawy?"},
    {"id": "q25", "query": "Jaki jest termin na załatwienie sprawy w postępowaniu uproszczonym?"},
    {"id": "q26", "query": "Jaka jest kara za naruszenie RODO dla firmy?"},
    {"id": "q27", "query": "Ile dni ma urząd na odpowiedź na wniosek o informację publiczną?"},
    {"id": "q28", "query": "Kiedy pracownik organu podlega wyłączeniu?"},
    {"id": "q29", "query": "Co to jest profilowanie?"},
]

    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human", "PYTANIE:\n{question}\n\nODPOWIEDŹ BOTA:\n{answer}\n\nŹRÓDŁA:\n{sources}")
    ])
    judge_chain = judge_prompt | judge_llm

    results = []
    print(f"📝 Rozpoczynam egzamin na {len(test_questions)} pytaniach...\n")

    for i, item in enumerate(test_questions, start=1):
        qid = item["id"]
        question = item["query"]

        print(f"🔹 {qid} | Pytanie {i}: {question}")

        response = rag_chain.invoke({"input": question, "chat_history": []})
        answer = response.get("answer", "")
        docs = response.get("context", [])

        if not docs:
            sources_text = "(brak źródeł z retrievera)"
        else:
            sources_text = "\n".join([
                f"- {d.metadata.get('act_name','Akt')} art. {d.metadata.get('article','?')}: "
                f"{(d.page_content or '')[:220].replace('\\n',' ')}..."
                for d in docs
            ])

        evaluation_raw = judge_chain.invoke({
            "question": question,
            "answer": answer,
            "sources": sources_text
        }).content

        evaluation = parse_judge_json(evaluation_raw)

        print(f"   🤖 Odpowiedź: {answer[:120]}...")
        print(f"   👨‍⚖️ Ocena: {evaluation['score']}/5 | {evaluation['reasoning']}")
        print("-" * 50)

        results.append({
            "id": qid,
            "question": question,
            "answer": answer,
            "score": evaluation["score"],
            "reasoning": evaluation["reasoning"]
        })


    with open("raport_egzaminatora.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n✅ Egzamin zakończony. Wyniki w 'raport_egzaminatora.json'.")


if __name__ == "__main__":
    main()
