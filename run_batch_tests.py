import argparse
import json
import time
from pathlib import Path
from datetime import datetime

from langchain_community.chat_models import ChatOllama

from src.config import MODEL_NAME, SERVER_URL, DEBUG, RETRIEVER_K
from src.embeddings import build_embeddings
from src.vectorstore import build_vector_store
from src.prompts import QA_PROMPT, DOCUMENT_PROMPT
from src.rag_chain import build_rag_chain
from src.routing_retriever import ActRoutingRetriever
from src.routing import route_act_names


def _doc_to_dict(doc, max_preview_chars: int = 500):
    meta = doc.metadata or {}
    text = (doc.page_content or "").strip().replace("\n", " ")

    return {
        "source": meta.get("source"),
        "act_name": meta.get("act_name"),
        "article": meta.get("article"),
        "paragraph": meta.get("paragraph"),
        "page": meta.get("page"),
        "preview": text[:max_preview_chars] + ("..." if len(text) > max_preview_chars else ""),
    }


def init_rag():
    llm = ChatOllama(
        base_url=SERVER_URL,
        model=MODEL_NAME,
        temperature=0.2,
    )

    embeddings = build_embeddings()
    db, _base_retriever = build_vector_store(embeddings)

    # Zamiast db.as_retriever() używamy Twojego routingu po aktach (metadata["act_name"])
    routed_retriever = ActRoutingRetriever(
        vectorstore=db,
        k=RETRIEVER_K,
        max_acts=2,
        debug=DEBUG,
    )

    rag_chain = build_rag_chain(
        llm=llm,
        retriever=routed_retriever,
        qa_prompt=QA_PROMPT,
        document_prompt=DOCUMENT_PROMPT,
    )

    return rag_chain, routed_retriever


def run_one(rag_chain, query: str):
    t0 = time.time()
    result = rag_chain.invoke({"input": query})
    elapsed_ms = int((time.time() - t0) * 1000)

    answer = (result.get("answer") or "").strip()
    docs = result.get("context") or []

    return answer, docs, elapsed_ms


def main():
    parser = argparse.ArgumentParser(description="Batch test runner for RAG (JSONL -> JSONL).")
    parser.add_argument("--in", dest="in_path", default="tests/questions.jsonl", help="Input questions JSONL path")
    parser.add_argument("--out", dest="out_path", default="tests/results.jsonl", help="Output results JSONL path")
    parser.add_argument("--limit", dest="limit", type=int, default=0, help="Limit number of questions (0 = no limit)")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Brak pliku wejściowego: {in_path}")

    rag_chain, _retriever = init_rag()

    run_meta = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model": MODEL_NAME,
        "server_url": SERVER_URL,
        "retriever_k": RETRIEVER_K,
    }

    processed = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("a", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            qid = item.get("id")
            query = item.get("query")

            # Dodatkowo zapisujemy routing (jakie akty zostały wybrane)
            routed_acts = route_act_names(query, max_acts=2)

            answer, docs, elapsed_ms = run_one(rag_chain, query)

            out = {
                **run_meta,
                "id": qid,
                "query": query,
                "routing": routed_acts if routed_acts else "ALL (fallback)",
                "elapsed_ms": elapsed_ms,
                "answer": answer,
                "docs": [_doc_to_dict(d) for d in docs],
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

            processed += 1
            print(f"[OK] {qid} | {elapsed_ms} ms | docs={len(docs)}")

            if args.limit and processed >= args.limit:
                break

    print(f"\nZapisano wyniki do: {out_path.resolve()}")


if __name__ == "__main__":
    main()
