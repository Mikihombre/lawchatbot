import json
import re
import time
from pathlib import Path
from typing import Dict, Any, Iterable

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from src.config import SERVER_URL

JUDGE_MODEL = "gemma3:4b"
TEMPERATURE = 0.0

# Rubryka: ocena 0-2 + flagi. Trzymamy krótko, deterministycznie.
JUDGE_SYSTEM = """Jesteś surowym egzaminatorem prawnym.
Oceniasz odpowiedź modelu na pytanie prawnicze, korzystając z:
- pytania,
- odpowiedzi modelu,
- odpowiedzi referencyjnej (ground truth),
- źródeł (akty i artykuły z retrievera).

Zwróć WYŁĄCZNIE JSON w formacie:
{{
  "legal_accuracy": 0|1|2,
  "citation_accuracy": 0|1,
  "hallucination": 0|1,
  "out_of_scope": 0|1,
  "reasoning": "max 600 znaków, konkretnie: co jest nie tak i dlaczego"
}}

Zasady:
- legal_accuracy=2 gdy konkluzja i główne przesłanki są zgodne z ground_truth.
- legal_accuracy=1 gdy konkluzja poprawna, ale braki/nieścisłości w przesłankach lub reżimie.
- legal_accuracy=0 gdy konkluzja błędna lub pominięto warunek kluczowy (np. ponaglenie, termin, dopuszczalność).
- citation_accuracy=1 tylko jeśli podano właściwe artykuły/ustawy zgodne z ground_truth (nie muszą być wszystkie).
- hallucination=1 jeśli podano przepisy nieadekwatne lub twierdzenia bez podstawy w źródłach/ground_truth.
- out_of_scope=1 jeśli pytanie wykracza poza UDIP/KPA/PPSA/RODO i model powinien odmówić lub ograniczyć się.
"""

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", JUDGE_SYSTEM),
    ("human",
     "PYTANIE:\n{question}\n\n"
     "ODPOWIEDŹ MODELU:\n{model_answer}\n\n"
     "GROUND TRUTH:\n{ground_truth}\n\n"
     "ŹRÓDŁA:\n{sources_text}\n")
])


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def sources_to_text(sources) -> str:
    # sources w Twoim wyniku to lista {"act": "...", "article": "..."}
    if not sources:
        return "(brak źródeł)"
    return "\n".join([f"- {s.get('act')} {s.get('article')}" for s in sources])


def parse_json_strict(text: str) -> Dict[str, Any]:
    clean = text.replace("```json", "").replace("```", "").strip()
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if not m:
        return {"legal_accuracy": 0, "citation_accuracy": 0, "hallucination": 1, "out_of_scope": 0,
                "reasoning": f"Błąd formatu JSON: {clean[:200]}..."}

    try:
        data = json.loads(m.group(0))
    except Exception as e:
        return {"legal_accuracy": 0, "citation_accuracy": 0, "hallucination": 1, "out_of_scope": 0,
                "reasoning": f"Nie da się zparsować JSON: {e}"}

    # Normalizacja
    def clamp_int(v, lo, hi, default):
        return v if isinstance(v, int) and lo <= v <= hi else default

    out = {
        "legal_accuracy": clamp_int(data.get("legal_accuracy"), 0, 2, 0),
        "citation_accuracy": clamp_int(data.get("citation_accuracy"), 0, 1, 0),
        "hallucination": clamp_int(data.get("hallucination"), 0, 1, 0),
        "out_of_scope": clamp_int(data.get("out_of_scope"), 0, 1, 0),
        "reasoning": str(data.get("reasoning", "")).strip()
    }
    if len(out["reasoning"]) > 600:
        out["reasoning"] = out["reasoning"][:597] + "..."
    return out


def main():
    project_root = Path(__file__).resolve().parent
    in_path = project_root / "tests" / "benchmark_results.jsonl"
    out_path = project_root / "tests" / "benchmark_judged.jsonl"
    out_summary = project_root / "tests" / "benchmark_summary.json"

    if not in_path.exists():
        raise FileNotFoundError(f"Brak pliku wejściowego: {in_path}")

    judge_llm = ChatOllama(
        base_url=SERVER_URL,
        model=JUDGE_MODEL,
        temperature=TEMPERATURE
    )
    judge_chain = JUDGE_PROMPT | judge_llm

    totals = {"n": 0, "legal_accuracy_sum": 0, "citation_sum": 0, "halluc_sum": 0, "oos_sum": 0}
    start_all = time.time()

    with out_path.open("w", encoding="utf-8") as f_out:
        for row in iter_jsonl(in_path):
            q = row.get("question", "")
            a = row.get("model_answer", "")
            gt = row.get("ground_truth", "")
            sources_text = sources_to_text(row.get("sources", []))

            judged_raw = judge_chain.invoke({
                "question": q,
                "model_answer": a,
                "ground_truth": gt,
                "sources_text": sources_text
            }).content

            judged = parse_json_strict(judged_raw)

            out_row = dict(row)
            out_row["judge_model"] = JUDGE_MODEL
            out_row["judge"] = judged

            f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f_out.flush()

            totals["n"] += 1
            totals["legal_accuracy_sum"] += judged["legal_accuracy"]
            totals["citation_sum"] += judged["citation_accuracy"]
            totals["halluc_sum"] += judged["hallucination"]
            totals["oos_sum"] += judged["out_of_scope"]

            print(f"[{totals['n']}] legal={judged['legal_accuracy']} cite={judged['citation_accuracy']} hall={judged['hallucination']} oos={judged['out_of_scope']}")

    # Podsumowanie
    n = max(totals["n"], 1)
    summary = {
        "n": totals["n"],
        "judge_model": JUDGE_MODEL,
        "legal_accuracy_avg": totals["legal_accuracy_sum"] / (2 * n),   # normalizacja do 0..1
        "citation_accuracy": totals["citation_sum"] / n,
        "hallucination_rate": totals["halluc_sum"] / n,
        "out_of_scope_rate": totals["oos_sum"] / n,
        "elapsed_s": round(time.time() - start_all, 2)
    }

    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✅ Zapisano:", out_path.resolve())
    print("✅ Summary:", out_summary.resolve())
    print(summary)


if __name__ == "__main__":
    main()
