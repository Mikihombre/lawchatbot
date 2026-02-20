import json
import re

from src.config import MODEL_NAME
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from src.routing_retriever import ActRoutingRetriever
from src.embeddings import build_embeddings
from src.vectorstore import build_vector_store
from src.rag_chain import build_rag_chain
from src.prompts import QA_PROMPT, DOCUMENT_PROMPT


# =========================
# KONFIG (na sztywno tutaj)
# =========================
SERVER_URL = "http://127.0.0.1:11434"

LAWYER_MODEL = MODEL_NAME
JUDGE_MODEL = MODEL_NAME
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


# =========================================================
# HELPERY DO TESTÓW (single-shot + multi-turn)
# =========================================================
def run_turn(rag_chain, question: str, history: list):
    """
    history: lista wiadomości LangChain (HumanMessage/AIMessage)
    Zwraca: answer, docs, updated_history
    """
    resp = rag_chain.invoke({"input": question, "chat_history": history})
    answer = resp.get("answer") or resp.get("output_text") or ""
    docs = resp.get("context") or resp.get("documents") or []
    history = history + [HumanMessage(content=question), AIMessage(content=answer)]
    return answer, docs, history


def docs_to_sources_text(docs, max_docs=6, max_chars=260):
    if not docs:
        return "(brak źródeł z retrievera)"
    lines = []
    for d in docs[:max_docs]:
        act = d.metadata.get("act_name", "Akt")
        art = d.metadata.get("article", "?")
        txt = (d.page_content or "").replace("\n", " ").strip()
        lines.append(f"- {act} art. {art}: {txt[:max_chars]}...")
    return "\n".join(lines)


def main():
    print("👨‍⚖️ Uruchamiam Agenta Egzaminatora (LLM-as-a-Judge)...")

    # 1) Prawnik (gemma3:12b)
    llm = ChatOllama(base_url=SERVER_URL, model=LAWYER_MODEL, temperature=TEMPERATURE)

    # 2) Sędzia (gemma3:12b)
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
        # --- Prawo budowlane (PB) ---
        {"id": "pb1", "query": "Kiedy wymagane jest pozwolenie na budowę, a kiedy wystarczy zgłoszenie budowy?"},
        {"id": "pb2", "query": "Co grozi za samowolę budowlaną i jak wygląda legalizacja samowoli?"},
        {"id": "pb3", "query": "Czym jest opłata legalizacyjna i od czego zależy jej wysokość?"},
        {"id": "pb4", "query": "Kiedy potrzebne jest pozwolenie na użytkowanie i czym różni się od zawiadomienia o zakończeniu budowy?"},
        {"id": "pb5", "query": "Na czym polega zmiana sposobu użytkowania obiektu i kiedy trzeba ją zgłosić?"},
        {"id": "pb6", "query": "Jakie obowiązki ma kierownik budowy i kto prowadzi dziennik budowy?"},
        {"id": "pb7", "query": "Jakie uprawnienia ma nadzór budowlany (PINB/WINB) w razie nieprawidłowości?"},
        {"id": "pb8", "query": "Co to jest katastrofa budowlana i jakie są obowiązki po jej wystąpieniu?"},
        {"id": "pb9", "query": "Kiedy organ może nakazać rozbiórkę obiektu budowlanego?"},
        {"id": "pb10", "query": "Do jakiej kategorii obiektu (załącznik) zalicza się budynek mieszkalny jednorodzinny i jakie ma współczynniki?"},
        # --- PPSA ---
        {"id": "ppsa1", "query": "Jaki jest termin na wniesienie skargi do WSA od decyzji administracyjnej?"},
        {"id": "ppsa2", "query": "Co to jest skarga kasacyjna do NSA i w jakim terminie się ją wnosi?"},
        {"id": "ppsa3", "query": "Kiedy sąd administracyjny może wstrzymać wykonanie decyzji (wstrzymanie wykonania)?"},
        {"id": "ppsa4", "query": "Na czym polega skarga na bezczynność organu i czego można żądać przed WSA?"},
        {"id": "ppsa5", "query": "Na czym polega skarga na przewlekłość postępowania i jakie są jej skutki?"},
        {"id": "ppsa6", "query": "Kiedy WSA odrzuca skargę i jakie są typowe przyczyny odrzucenia?"},
        {"id": "ppsa7", "query": "Czy WSA może wymierzyć grzywnę organowi za bezczynność?"},
        {"id": "ppsa8", "query": "Jakie są zasady kosztów postępowania sądowoadministracyjnego i kiedy zwraca się wpis?"},
        {"id": "ppsa9", "query": "Czym różni się oddalenie skargi od odrzucenia skargi w PPSA?"},
        {"id": "ppsa10", "query": "Jak wnieść skargę do sądu administracyjnego: jakie elementy musi zawierać?"},
        # --- Graniczne / mylące ---
        {"id": "mix1", "query": "Organ nie odpowiada na mój wniosek o informację publiczną — co mogę zrobić?"},
        {"id": "mix2", "query": "Organ nie załatwia sprawy w terminie — jakie mam środki (ponaglenie)?"},
        {"id": "mix3", "query": "Czy mogę zaskarżyć bezczynność organu do WSA?"},
    ]

    test_dialogs = [
        {
            "id": "dlg_kpa_1",
            "turns": [
                "Złożyłem wniosek i urząd milczy od dawna.",
                "A co mogę z tym zrobić?",
                "A jakie mam terminy?",
                "To gdzie to składam?"
            ]
        },
        {
            "id": "dlg_udip_1",
            "turns": [
                "Chcę od gminy umowę na remont drogi — muszą mi to dać?",
                "A jak mi odmówią?",
                "A jak nic nie odpowiedzą?",
                "To jakie są terminy?"
            ]
        },
        {
            "id": "dlg_ppsa_1",
            "turns": [
                "Urząd nie załatwia mojej sprawy w terminie.",
                "Czy mogę zaskarżyć bezczynność do WSA?",
                "A co muszę zrobić wcześniej?",
                "Jakie mam terminy na skargę?"
            ]
        },
    ]

    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human", "PYTANIE:\n{question}\n\nODPOWIEDŹ BOTA:\n{answer}\n\nŹRÓDŁA:\n{sources}")
    ])
    judge_chain = judge_prompt | judge_llm

    results = []
    print(f"📝 Rozpoczynam egzamin (single-shot) na {len(test_questions)} pytaniach...\n")

    # =========================================================
    # SINGLE-SHOT
    # =========================================================
    for i, item in enumerate(test_questions, start=1):
        qid = item["id"]
        question = item["query"]

        print(f"🔹 {qid} | Pytanie {i}: {question}")

        response = rag_chain.invoke({"input": question, "chat_history": []})
        answer = response.get("answer") or response.get("output_text") or ""
        docs = response.get("context") or response.get("documents") or []
        sources_text = docs_to_sources_text(docs)

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
            "reasoning": evaluation["reasoning"],
        })

    # =========================================================
    # MULTI-TURN (history-aware + rewriter)
    # =========================================================
    print("\n💬 Testy wieloturowe (history-aware + rewriter)...\n")

    for suite in test_dialogs:
        suite_id = suite["id"]
        history = []
        print(f"🧪 Dialog: {suite_id}")

        for t, q in enumerate(suite["turns"], start=1):
            print(f"   ▶ Turn {t}: {q}")

            answer, docs, history = run_turn(rag_chain, q, history)
            sources_text = docs_to_sources_text(docs)

            evaluation_raw = judge_chain.invoke({
                "question": q,
                "answer": answer,
                "sources": sources_text
            }).content
            evaluation = parse_judge_json(evaluation_raw)

            print(f"      🤖 Odp: {answer[:120]}...")
            print(f"      👨‍⚖️ Ocena: {evaluation['score']}/5 | {evaluation['reasoning']}")

            results.append({
                "id": f"{suite_id}_t{t}",
                "question": q,
                "answer": answer,
                "score": evaluation["score"],
                "reasoning": evaluation["reasoning"],
            })

        print("-" * 50)

    with open("raport_egzaminatora.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n✅ Egzamin zakończony. Wyniki w 'raport_egzaminatora.json'.")


if __name__ == "__main__":
    main()
