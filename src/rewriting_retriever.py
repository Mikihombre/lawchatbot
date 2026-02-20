# src/rewriting_retriever.py
from typing import Any, List, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from src.query_rewriter import rewrite_query


class RewritingRetriever(BaseRetriever):
    """
    Wrapper: przepuszcza query przez rewriter, potem deleguje do bazowego retrievera.
    """
    base_retriever: Any
    rewriter_llm: Any
    debug: bool = False

    _cache: dict = {}

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:

        if query in self._cache:
            rw = self._cache[query]
        else:
            rw = rewrite_query(self.rewriter_llm, query)
            self._cache[query] = rw

        rewritten = (rw.get("rewritten_query") or query).strip()

        # --- BRIDGE: hinty zaczynają wpływać na retrieval ---
        augmented = rewritten

        act_hints = rw.get("act_hints") or []
        keywords = rw.get("keywords") or []

        if act_hints:
            augmented += "\n\nKontekst prawny (akty): " + ", ".join(act_hints)

        if keywords:
            augmented += "\n\nSłowa kluczowe: " + ", ".join(keywords[:8])

        if self.debug:
            print(f"[REWRITE] {query} -> {rewritten}")
            if act_hints:
                print(f"[REWRITE] act_hints={act_hints}")
            if keywords:
                print(f"[REWRITE] keywords={keywords[:8]}")
            print(f"[REWRITE] final_query_for_router:\n{augmented}\n")

        # Delegacja do Twojego ActRoutingRetriever (routing, BM25, rerank itd. już wewnątrz)
        base = self.base_retriever
        if hasattr(base, "invoke"):
            return base.invoke(augmented)
        elif hasattr(base, "get_relevant_documents"):
            return base.get_relevant_documents(augmented)
        else:
            raise TypeError(f"base_retriever nie wspiera invoke ani get_relevant_documents: {type(base)}")
