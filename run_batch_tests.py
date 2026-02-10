import argparse
import json
import time
import re
import hashlib
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

# Używamy nowej biblioteki zgodnie z warningiem
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

# Importy z Twojego projektu
# Upewnij się, że te pliki istnieją w folderze src/
from src.config import MODEL_NAME, SERVER_URL, DEBUG, RETRIEVER_K
from src.vectorstore import build_vector_store
# Zakładam, że poniższe moduły masz lub dopiero tworzysz (bazując na Twoim kodzie):
try:
    from src.embeddings import build_embeddings
    from src.prompts import QA_PROMPT, DOCUMENT_PROMPT
    from src.rag_chain import build_rag_chain
    from src.routing_retriever import ActRoutingRetriever
    from src.routing import route_act_names
except ImportError as e:
    print(f"❌ BŁĄD IMPORTU: Brakuje jednego z modułów w folderze 'src'.")
    print(f"Szczegóły: {e}")
    print("Upewnij się, że masz pliki: routing.py, routing_retriever.py, embeddings.py, prompts.py, rag_chain.py")
    sys.exit(1)


# -----------------------------
# Helpers
# -----------------------------

def _doc_to_dict(doc, max_preview_chars: int = 500):
    meta = doc.metadata or {}
    text = (doc.page_content or "").strip().replace("\n", " ")

    return {
        "source": meta.get("source"),
        "act_name": meta.get("act_name"),
        "article": meta.get("article"),
        "hierarchy": meta.get("hierarchy"), # Zmieniłem paragraph na hierarchy (zgodnie z Twoim parserem)
        "page": meta.get("page"),
        "preview": text[:max_preview_chars] + ("..." if len(text) > max_preview_chars else ""),
    }


def file_sha256(path: Path) -> str:
    """Krótki hash pliku do identyfikacji wersji kodu."""
    try:
        if not path.exists(): return "missing"
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()[:12]
    except Exception:
        return "error"


def git_sha() -> str:
    """Jeśli projekt jest w git, dołącz short SHA."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"


def docs_stats(docs):
    acts = [d.metadata.get("act_name") for d in docs if getattr(d, "metadata", None)]
    arts = [d.metadata.get("article") for d in docs if getattr(d, "metadata", None)]
    return {
        "docs_count": len(docs),
        "unique_acts": sorted({str(a) for a in acts if a}),
        "unique_articles": sorted({str(a) for a in arts if a}),
    }


def query_flags(query: str, retriever):
    """Szybkie flagi diagnostyczne dot. zapytania."""
    # Uwaga: Ta funkcja zadziała tylko jeśli w RoutingRetriever masz metodę _extract_refs
    # Jeśli nie, ustawiamy wartości domyślne
    article, paragraph = None, None
    is_sanction = False
    
    if hasattr(retriever, "_extract_refs"):
        article, paragraph = retriever._extract_refs(query)
    
    if hasattr(retriever, "_is_sanction_question"):
        is_sanction = retriever._is_sanction_question(query)
        
    q = (query or "").lower()
    return {
        "has_article": bool(article),
        "article": article,
        "paragraph": paragraph,
        "is_sanction_q": is_sanction,
        "has_pln_amount": bool(re.search(r"\b\d[\d\s]{0,10}\s*z[łl]\b", q)),
    }


# -----------------------------
# Init RAG
# -----------------------------

def init_rag():
    print(f"🔌 Łączenie z modelem: {MODEL_NAME} na {SERVER_URL}...")
    llm = ChatOllama(
        base_url=SERVER_URL,
        model=MODEL_NAME,
        temperature=0.2,
    )

    print("📚 Ładowanie bazy wektorowej...")
    # Tutaj używamy Twojego build_embeddings (musisz go mieć w src/embeddings.py)
    # Jeśli nie masz, możesz użyć: HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    embeddings = build_embeddings()
    
    db, _base_retriever = build_vector_store(embeddings)

    # Parametry retrievera
    retriever_params = {
        "vectorstore": db,
        "k": RETRIEVER_K,
        "max_acts": 2,
        "debug": DEBUG,
        "search_type": "mmr",
        "fetch_k": 60,
        "lambda_mult": 0.6,
        "enable_sanction_filter": True,
        "sanction_k": 6,
    }

    # Inicjalizacja Twojego zaawansowanego Retrievera
    routed_retriever = ActRoutingRetriever(**retriever_params)

    rag_chain = build_rag_chain(
        llm=llm,
        retriever=routed_retriever,
        qa_prompt=QA_PROMPT,
        document_prompt=DOCUMENT_PROMPT,
    )

    return rag_chain, routed_retriever, retriever_params


def run_one(rag_chain, query: str):
    t0 = time.time()
    result = rag_chain.invoke({"input": query})
    elapsed_ms = int((time.time() - t0) * 1000)

    # Obsługa różnych formatów wyjścia łańcucha
    if isinstance(result, dict):
        answer = (result.get("answer") or "").strip()
        docs = result.get("context") or []
    else:
        # Fallback jeśli chain zwraca stringa
        answer = str(result)
        docs = []

    return answer, docs, elapsed_ms


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch test runner for RAG.")
    parser.add_argument("--in", dest="in_path", default="tests/questions.jsonl", help="Input questions JSONL path")
    parser.add_argument("--out", dest="out_path", default="tests/results.jsonl", help="Output results JSONL path")
    parser.add_argument("--limit", dest="limit", type=int, default=0, help="Limit number of questions (0 = no limit)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file instead of append")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"❌ BŁĄD: Nie znaleziono pliku wejściowego: {in_path}")
        print("Utwórz folder 'tests' i plik 'questions.jsonl'.")
        return

    rag_chain, retriever, retriever_params = init_rag()

    # ---- Metadata ----
    run_meta = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "model": MODEL_NAME,
        "retriever_config": retriever_params.get("k"), # uproszczone do logu
    }

    processed = 0
    fallback_count = 0
    routing_counter = Counter()
    elapsed_list = []
    sanction_total = 0

    mode = "w" if args.overwrite else "a"
    
    print(f"\n🚀 Rozpoczynam testy. Wyniki trafią do: {out_path}")
    
    with in_path.open("r", encoding="utf-8") as fin, out_path.open(mode, encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line: continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️ Pomijam błędną linię JSON")
                continue
                
            qid = item.get("id", f"q{processed}")
            query = item.get("query") or ""

            # Routing
            routed_acts = route_act_names(query, max_acts=2)
            if routed_acts:
                for a in routed_acts:
                    routing_counter[a] += 1
            else:
                fallback_count += 1

            # Flags
            flags = query_flags(query, retriever)
            if flags.get("is_sanction_q"):
                sanction_total += 1

            # Run RAG
            print(f"   🔹 Pytanie [{qid}]: {query[:50]}...")
            answer, docs, elapsed_ms = run_one(rag_chain, query)
            elapsed_list.append(elapsed_ms)

            d_stats = docs_stats(docs)

            out = {
                **run_meta,
                "id": qid,
                "query": query,
                "routing": routed_acts if routed_acts else "ALL",
                "flags": flags,
                "elapsed_ms": elapsed_ms,
                "docs_stats": d_stats,
                "answer": answer,
                "docs": [_doc_to_dict(d) for d in docs],
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

            processed += 1
            
            if args.limit and processed >= args.limit:
                break

    # ---- Summary ----
    print("\n--- 📊 PODSUMOWANIE ---")
    print(f"Przetworzono pytań: {processed}")
    print(f"Routing Fallback (szukanie we wszystkim): {fallback_count}")

    if elapsed_list:
        avg_ms = sum(elapsed_list) // len(elapsed_list)
        print(f"Czas odpowiedzi: średni={avg_ms}ms | min={min(elapsed_list)}ms | max={max(elapsed_list)}ms")

    top = routing_counter.most_common(5)
    if top:
        print("Najczęściej wybierane akty prawne:")
        for act, cnt in top:
            print(f"  - {act}: {cnt}")

if __name__ == "__main__":
    main()