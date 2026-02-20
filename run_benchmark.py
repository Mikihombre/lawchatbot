# src/run_benchmark.py
import json
import time
from pathlib import Path

from langchain_community.chat_models import ChatOllama

from src.config import SERVER_URL, MODEL_NAME
from src.routing_retriever import ActRoutingRetriever
from src.embeddings import build_embeddings
from src.vectorstore import build_vector_store
from src.rag_chain import build_rag_chain
from src.prompts import QA_PROMPT, DOCUMENT_PROMPT

TEMPERATURE = 0.0


def load_dataset(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_question(record: dict) -> str:
    instruction = (record.get("instruction") or "").strip()
    input_text = (record.get("input") or "").strip()

    if input_text:
        return f"{instruction}\n\nStan faktyczny:\n{input_text}"
    return instruction


def main():
    # --- ścieżki liczone od root projektu (nie od cwd) ---
    project_root = Path(__file__).resolve().parent
    dataset_path = project_root / "datasets" / "dataset_clean.jsonl"
    tests_dir = project_root / "tests"
    out_path = tests_dir / "benchmark_results.jsonl"
    tests_dir.mkdir(parents=True, exist_ok=True)

    # JSONL = zapis przyrostowy (po każdym rekordzie), więc nawet po przerwaniu masz wyniki
    out_path = tests_dir / "benchmark_results.jsonl"

    print("CWD:", Path.cwd())
    print("Dataset:", dataset_path.resolve())
    print("Wyniki:", out_path.resolve())

    dataset = load_dataset(dataset_path)

    llm = ChatOllama(
        base_url=SERVER_URL,
        model=MODEL_NAME,
        temperature=TEMPERATURE
    )

    embeddings = build_embeddings()
    db, _ = build_vector_store(embeddings)
    retriever = ActRoutingRetriever(vectorstore=db, max_acts=2, debug=False)
    rag_chain = build_rag_chain(llm, retriever, QA_PROMPT, DOCUMENT_PROMPT)

    # --- benchmark ---
    with out_path.open("w", encoding="utf-8") as f_out:
        for idx, record in enumerate(dataset):
            question = build_question(record)
            ground_truth = record.get("output", "")

            start = time.time()
            response = rag_chain.invoke({
                "input": question,
                "chat_history": []
            })
            latency_ms = round((time.time() - start) * 1000)

            answer = response.get("answer") or response.get("output_text") or ""
            docs = response.get("context") or response.get("documents") or []

            sources = []
            for d in docs:
                # d może być Documentem z metadanymi, ale zachowujemy ostrożność
                meta = getattr(d, "metadata", {}) or {}
                sources.append({
                    "act": meta.get("act_name"),
                    "article": meta.get("article")
                })

            row = {
                "id": idx + 1,
                "model": MODEL_NAME,
                "question": question,
                "ground_truth": ground_truth,
                "model_answer": answer,
                "sources": sources,
                "latency_ms": latency_ms,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            # zapis 1-linia = 1 przypadek testowy
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            f_out.flush()

            print(f"[{idx + 1}/{len(dataset)}] done")

    print("✅ Benchmark zapisany:", out_path.resolve())


if __name__ == "__main__":
    main()
