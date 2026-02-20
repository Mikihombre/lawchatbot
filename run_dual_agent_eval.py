import json
import re
from typing import List, Dict, Any, Tuple

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from src.config import SERVER_URL, MODEL_NAME
from src.routing_retriever import ActRoutingRetriever
from src.embeddings import build_embeddings
from src.vectorstore import build_vector_store
from src.rag_chain import build_rag_chain
from src.prompts import QA_PROMPT, DOCUMENT_PROMPT


TEMPERATURE = 0.0
LAWYER_MODEL = MODEL_NAME
EXAMINER_MODEL = MODEL_NAME   # może być też inny model, jeśli masz


# -------------------------
# PROMPT: Examiner tworzy kolejne pytanie
# -------------------------
EXAMINER_ASK_SYSTEM = """Jesteś surowym egzaminatorem systemu RAG prawniczego.
Twoje zadanie: prowadzić dialog testowy i generować kolejne pytanie (po polsku),
żeby sprawdzić:
- routing między ustawami (UDIP/KPA/PPSA/PB/RODO),
- zachowanie w dialogu (doprecyzowania typu "A jakie terminy?"),
- pytania potoczne i skrótowe,
- pytania warunkowe ("jeżeli... to...").

Dostaniesz:
- dotychczasowy przebieg dialogu (pytania i odpowiedzi),
- ostatnią ocenę 1-5 i powód.

Zasady generowania:
1) Zwróć WYŁĄCZNIE jedno pytanie użytkownika (jedna linia, zakończone '?').
2) Nie komentuj, nie oceniaj, nie dawaj porad.
3) Czasem specjalnie użyj potocznego języka (np. "urząd milczy", "muszą mi dać umowę?").
4) Co 2-3 tury dodaj dopytanie o: termin / właściwość / tryb wniesienia / środek zaskarżenia.
"""


EXAMINER_ASK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", EXAMINER_ASK_SYSTEM),
    ("human", "DIALOG DOTYCHCZAS:\n{dialog}\n\nOSTATNIA OCENA:\nscore={score}\nreasoning={reasoning}\n\nWygeneruj kolejne pytanie:")
])


# -------------------------
# PROMPT: Examiner ocenia odpowiedź (jak u Ciebie, ale z escapowanymi klamrami)
# -------------------------
JUDGE_SYSTEM_PROMPT = """Jesteś surowym egzaminatorem prawnym. Twoim zadaniem jest ocena odpowiedzi bota RAG.

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
- "reasoning" max 800 znaków.
W polu "reasoning" (max 800 znaków) podaj konkretnie:
- Problem (co jest nie tak),
- Dowód (czy źródła to potwierdzają / brak podstaw),
- Poprawka (co zmienić: routing, retrieval, prompt, keywords).

Zwróć TYLKO czysty JSON dokładnie w formacie:
{{
  "score": <liczba całkowita 1-5>,
  "reasoning": "<uzasadnienie>"
}}

"""


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", JUDGE_SYSTEM_PROMPT),
    ("human", "PYTANIE:\n{question}\n\nODPOWIEDŹ BOTA:\n{answer}\n\nŹRÓDŁA:\n{sources}")
])


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


def run_lawyer_turn(rag_chain, question: str, history: List) -> Tuple[str, List, List]:
    resp = rag_chain.invoke({"input": question, "chat_history": history})
    answer = resp.get("answer") or resp.get("output_text") or ""
    docs = resp.get("context") or resp.get("documents") or []
    history = history + [HumanMessage(content=question), AIMessage(content=answer)]
    return answer, docs, history


def format_dialog_for_examiner(turns: List[Dict[str, str]], max_chars=3500) -> str:
    # turns: [{"q": "...", "a":"..."}, ...]
    parts = []
    for i, t in enumerate(turns, start=1):
        q = t["q"].strip()
        a = t["a"].strip()
        parts.append(f"[{i}] U: {q}\n[{i}] A: {a}")
    dialog = "\n\n".join(parts)
    # utnij, żeby nie zapchać prompta
    if len(dialog) > max_chars:
        dialog = dialog[-max_chars:]
    return dialog


def main():
    print("🤝 Dual-agent eval: Examiner (pyta) ↔ Lawyer (RAG)")

    # LLMs
    lawyer_llm = ChatOllama(base_url=SERVER_URL, model=LAWYER_MODEL, temperature=TEMPERATURE)
    examiner_llm = ChatOllama(base_url=SERVER_URL, model=EXAMINER_MODEL, temperature=TEMPERATURE)

    # Build RAG
    embeddings = build_embeddings()
    db, _ = build_vector_store(embeddings)
    retriever = ActRoutingRetriever(vectorstore=db, max_acts=2, debug=True)
    rag_chain = build_rag_chain(lawyer_llm, retriever, QA_PROMPT, DOCUMENT_PROMPT)

    # Chains
    judge_chain = JUDGE_PROMPT | examiner_llm
    ask_chain = EXAMINER_ASK_PROMPT | examiner_llm

    # Seed question (możesz rotować domeny)
    current_question = "Organ nie odpowiada na mój wniosek o informację publiczną — co mogę zrobić?"
    lawyer_history: List = []
    dialog_turns: List[Dict[str, str]] = []
    transcript: List[Dict[str, Any]] = []

    last_eval = {"score": 3, "reasoning": "start"}

    N_TURNS = 10
    for turn in range(1, N_TURNS + 1):
        print(f"\n=== TURN {turn} ===")
        print("👤 Examiner asks:", current_question)

        # Lawyer answers
        answer, docs, lawyer_history = run_lawyer_turn(rag_chain, current_question, lawyer_history)
        sources_text = docs_to_sources_text(docs)

        print("⚖️ Lawyer answer (preview):", (answer[:200] + "...") if len(answer) > 200 else answer)

        # Judge evaluation
        evaluation_raw = judge_chain.invoke({
            "question": current_question,
            "answer": answer,
            "sources": sources_text
        }).content
        eval_parsed = parse_judge_json(evaluation_raw)
        print(f"👨‍⚖️ Eval: {eval_parsed['score']}/5 | {eval_parsed['reasoning']}")

        # Save
        dialog_turns.append({"q": current_question, "a": answer})
        transcript.append({
            "turn": turn,
            "question": current_question,
            "answer": answer,
            "score": eval_parsed["score"],
            "reasoning": eval_parsed["reasoning"],
            "sources": sources_text
        })

        # Examiner picks next question
        dialog_text = format_dialog_for_examiner(dialog_turns)
        next_q = ask_chain.invoke({
            "dialog": dialog_text,
            "score": eval_parsed["score"],
            "reasoning": eval_parsed["reasoning"]
        }).content.strip()

        # Sanity: wymuś jedno pytanie
        next_q = next_q.splitlines()[0].strip()
        if not next_q.endswith("?"):
            next_q = next_q + "?"

        current_question = next_q
        last_eval = eval_parsed

    # Save transcript
    out_path = "raport_dual_agent.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    # Summary stats
    scores = [x["score"] for x in transcript]
    avg = sum(scores) / len(scores) if scores else 0
    low = sum(1 for s in scores if s <= 2)
    print("\n✅ Zapisano:", out_path)
    print(f"📊 Podsumowanie: turns={len(scores)} avg={avg:.2f} <=2={low} ({(low/len(scores)*100):.1f}%)")


if __name__ == "__main__":
    main()
