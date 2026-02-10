# src/routing_retriever.py
import re
import torch
from typing import List, Optional, Any, Tuple, Dict
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

from src.routing import route_act_names
from src.config import RERANKER_MODEL, INITIAL_RETRIEVAL_K, FINAL_K, BM25_K

class ActRoutingRetriever(BaseRetriever):
    vectorstore: Any
    max_acts: int = 2
    debug: bool = True
    
    # Komponenty inicjalizowane leniwie
    bm25_retriever: Optional[BM25Retriever] = None
    cross_encoder: Optional[CrossEncoder] = None

    def _initialize_components(self):
        """Inicjalizacja BM25 oraz Modelu Rerankera (na GPU jeśli dostępne)."""
        # 1. BM25
        if self.bm25_retriever is None:
            if self.debug: print("   ⚙️ [INIT] Budowanie indeksu BM25...")
            try:
                data = self.vectorstore.get()
                texts = data["documents"]
                metadatas = data["metadatas"]
                if texts:
                    docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
                    self.bm25_retriever = BM25Retriever.from_documents(docs)
                    self.bm25_retriever.k = BM25_K
                    if self.debug: print(f"   ✅ BM25 gotowy ({len(docs)} dok).")
            except Exception as e:
                print(f"   ❌ Błąd BM25: {e}")

        # 2. Reranker (Cross Encoder)
        if self.cross_encoder is None:
            if self.debug: print(f"   ⚙️ [INIT] Ładowanie Rerankera: {RERANKER_MODEL}...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self.cross_encoder = CrossEncoder(RERANKER_MODEL, device=device)
                if self.debug: print(f"   ✅ Reranker gotowy na: {device.upper()}")
            except Exception as e:
                print(f"   ❌ Błąd ładowania Rerankera: {e}")

    def _extract_refs(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        match = re.search(r"(?:art\.?|artykuł)\s*(\d+[a-z]*)", query, re.IGNORECASE)
        if match: return match.group(1), None
        return None, None

    def _rrf_fusion(self, vector_docs: List[Document], bm25_docs: List[Document]) -> List[Document]:
        """Szybka fuzja list przed rerankingiem."""
        scores = {}
        doc_map = {}
        
        # Unikalny klucz
        def get_key(doc):
            # Używamy contentu jako klucza, bo metadane mogą być identyczne dla chunków z tej samej strony
            return str(hash(doc.page_content))

        # Punktacja (RRF)
        for rank, doc in enumerate(vector_docs):
            k = get_key(doc)
            doc_map[k] = doc
            scores[k] = scores.get(k, 0.0) + (1 / (60 + rank + 1))

        for rank, doc in enumerate(bm25_docs):
            k = get_key(doc)
            if k not in doc_map: doc_map[k] = doc
            scores[k] = scores.get(k, 0.0) + (1 / (60 + rank + 1))

        # Sortujemy i zwracamy szeroką listę (INITIAL_K) do rerankingu
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[k] for k in sorted_keys[:INITIAL_RETRIEVAL_K]]

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """Ostateczne sortowanie modelem AI."""
        if not self.cross_encoder or not docs:
            return docs[:FINAL_K]

        # Przygotowanie par [pytanie, dokument]
        pairs = [[query, d.page_content] for d in docs]
        
        # Inferencja (obliczanie punktów relewancji)
        scores = self.cross_encoder.predict(pairs)

        # Sortowanie po wyniku modelu
        results = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
        if self.debug:
            print(f"      [RERANKER] Top score: {results[0][1]:.4f} | Low score: {results[-1][1]:.4f}")

        # Zwracamy TOP N
        return [doc for doc, score in results[:FINAL_K]]

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        self._initialize_components()

        # 1. Routing
        target_acts = route_act_names(query, max_acts=self.max_acts)
        target_art, _ = self._extract_refs(query)
        
        if self.debug: 
            print(f"\n[DEBUG] Pytanie: {query}")
            print(f"[DEBUG] Routing: {target_acts}")

        # 2. Wektory (Chroma)
        filter_dict = None
        conditions = []
        if target_acts:
            conditions.append({"act_name": target_acts[0]} if len(target_acts) == 1 else {"$or": [{"act_name": a} for a in target_acts]})
        if target_art:
            conditions.append({"article": f"Art. {target_art}."})
        
        if conditions:
            filter_dict = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        # Pobieramy szeroko (INITIAL_K)
        vector_docs = self.vectorstore.similarity_search(query, k=INITIAL_RETRIEVAL_K, filter=filter_dict)
        
        # Fallback wektorowy
        if not vector_docs and target_art and filter_dict:
             fallback_cond = [c for c in conditions if "article" not in str(c)]
             fallback_filter = fallback_cond[0] if fallback_cond else None
             vector_docs = self.vectorstore.similarity_search(query, k=INITIAL_RETRIEVAL_K, filter=fallback_filter)

        # 3. BM25
        bm25_docs = []
        if self.bm25_retriever:
            raw_bm25 = self.bm25_retriever.invoke(query)
            # Ręczny filtr po aktach
            if target_acts:
                bm25_docs = [d for d in raw_bm25 if d.metadata.get("act_name") in target_acts]
            else:
                bm25_docs = raw_bm25

        if self.debug:
            print(f"[DEBUG] Candidates -> Vector: {len(vector_docs)}, BM25: {len(bm25_docs)}")

        # 4. Fuzja (połączenie list)
        merged_docs = self._rrf_fusion(vector_docs, bm25_docs)
        
        # 5. RERANKING (AI sortowanie)
        final_docs = self._rerank(query, merged_docs)
        
        if self.debug: print(f"[DEBUG] Final Docs (Reranked): {len(final_docs)}")
        
        return final_docs