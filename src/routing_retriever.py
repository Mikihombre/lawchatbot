import re
from typing import List, Optional, Any, Tuple

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.routing import route_act_names


class ActRoutingRetriever(BaseRetriever):
    """
    Wrapper retrievera: wybiera akt(y) na podstawie pytania i filtruje Chroma po metadata['act_name'].
    Rozszerzenia:
      - MMR (max_marginal_relevance_search) dla lepszej różnorodności wyników
      - filtr sankcyjny dla pytań "co grozi / jaka kara"
      - jeśli pytanie sankcyjne i brak przepisów sankcyjnych -> zwróć pustą listę (wymusi "Brak podstaw...")
    """
    vectorstore: Any
    k: int = 12
    max_acts: int = 2
    debug: bool = True

    # NOWE ustawienia:
    search_type: str = "mmr"       # "mmr" albo "similarity"
    fetch_k: int = 60              # kandydaci dla MMR
    lambda_mult: float = 0.6       # 0.5-0.7; niżej = większa różnorodność

    enable_sanction_filter: bool = True
    sanction_k: int = 6            # ile doców sankcyjnych ostatecznie przepuścić

    _SANCTION_Q = ("co grozi", "jaka kara", "jaką karę", "kara", "sankcj", "odpowiedzialnosc")
    _SANCTION_T = ("podlega karze", "pozbawienia wolności", "grzywn", "areszt", "ograniczenia wolności", "kara")

    def _extract_refs(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        article_match = re.search(r"(?:art\.?|artykuł)\s*(\d+[a-z]*)", query, re.IGNORECASE)
        paragraph_match = re.search(r"(?:§|par\.?|paragraf)\s*(\d+[a-z]*)", query, re.IGNORECASE)
        article = article_match.group(1).lower() if article_match else None
        paragraph = paragraph_match.group(1).lower() if paragraph_match else None
        return article, paragraph

    def _is_sanction_question(self, query: str) -> bool:
        q = query.lower()
        return any(x in q for x in self._SANCTION_Q)

    def _where(self, act_names: List[str]) -> Optional[dict]:
        if not act_names:
            return None
        if len(act_names) == 1:
            return {"act_name": act_names[0]}
        return {"$or": [{"act_name": n} for n in act_names]}

    def _where_article(self, act_names: List[str], article: str, paragraph: Optional[str]) -> dict:
        if len(act_names) == 1:
            act_filter = {"act_name": act_names[0]}
        else:
            act_filter = {"$or": [{"act_name": n} for n in act_names]}

        filters = [act_filter, {"article": article}]
        if paragraph:
            filters.append({"$or": [{"paragraph": paragraph}, {"paragraph": "all"}]})
        return {"$and": filters}

    def _search(self, query: str, where: Optional[dict]) -> List[Document]:
        """
        Chroma wspiera:
          - similarity_search(query, k=..., filter=...)
          - max_marginal_relevance_search(query, k=..., fetch_k=..., lambda_mult=..., filter=...)
        """
        if self.search_type == "mmr":
            if where:
                return self.vectorstore.max_marginal_relevance_search(
                    query,
                    k=self.k,
                    fetch_k=self.fetch_k,
                    lambda_mult=self.lambda_mult,
                    filter=where,
                )
            return self.vectorstore.max_marginal_relevance_search(
                query,
                k=self.k,
                fetch_k=self.fetch_k,
                lambda_mult=self.lambda_mult,
            )

        # similarity fallback
        if where:
            return self.vectorstore.similarity_search(query, k=self.k, filter=where)
        return self.vectorstore.similarity_search(query, k=self.k)

    def _filter_sanctions(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Jeśli pytanie dotyczy sankcji, zostaw tylko fragmenty mające język sankcyjny.
        Jeśli po filtrze nie ma nic -> zwróć [] (wymusi "Brak podstaw..." na promptcie).
        """
        if not self.enable_sanction_filter or not self._is_sanction_question(query):
            return docs

        scored = []
        for d in docs:
            t = (d.page_content or "").lower()
            score = 0

            if any(x in t for x in self._SANCTION_T):
                score += 3

            # Spychanie typowych definicji/śmieci w pytaniu o karę
            if "jest" in t and "to" in t and "podlega" not in t:
                score -= 1
            if "definicj" in t:
                score -= 1

            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        best = [d for s, d in scored if s > 0]

        # Najważniejsze: brak sankcji -> pusty kontekst
        if not best:
            return []

        return best[: self.sanction_k]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        act_names = route_act_names(query, max_acts=self.max_acts)
        article, paragraph = self._extract_refs(query)

        if self.debug:
            print(f"[DEBUG] ROUTING: {act_names if act_names else 'ALL (fallback)'}")
            if article:
                print(f"[DEBUG] ARTICLE FILTER: art. {article}" + (f" § {paragraph}" if paragraph else ""))

        # 1) Jeśli user podał art./§ to próbujemy twardy filtr
        if act_names and article:
            where = self._where_article(act_names, article, paragraph)
            docs = self._search(query, where)
            docs = self._filter_sanctions(query, docs)
            if docs:
                return docs

        # 2) Normalnie: filtr po akcie (albo ALL)
        where = self._where(act_names)
        docs = self._search(query, where)
        docs = self._filter_sanctions(query, docs)
        return docs
