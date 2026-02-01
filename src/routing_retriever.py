import re
from typing import List, Optional, Any, Tuple
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.routing import route_act_names


class ActRoutingRetriever(BaseRetriever):
    """
    Wrapper retrievera: wybiera akt(y) na podstawie pytania i filtruje Chroma po metadata['act_name'].
    """
    vectorstore: Any
    k: int = 12
    max_acts: int = 2
    debug: bool = True

    def _extract_refs(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        article_match = re.search(r"(?:art\\.?|artykuł)\\s*(\\d+[a-z]*)", query, re.IGNORECASE)
        paragraph_match = re.search(r"(?:§|par\\.?|paragraf)\\s*(\\d+[a-z]*)", query, re.IGNORECASE)
        article = article_match.group(1).lower() if article_match else None
        paragraph = paragraph_match.group(1).lower() if paragraph_match else None
        return article, paragraph


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
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        act_names = route_act_names(query, max_acts=self.max_acts)
        article, paragraph = self._extract_refs(query)

        if self.debug:
            print(f"[DEBUG] ROUTING: {act_names if act_names else 'ALL (fallback)'}")
            if article:
                print(f"[DEBUG] ARTICLE FILTER: art. {article}" + (f" § {paragraph}" if paragraph else ""))
        
        if act_names and article:
            where = self._where_article(act_names, article, paragraph)
            docs = self.vectorstore.similarity_search(query, k=self.k, filter=where)
            if docs:
                return docs

        where = self._where(act_names)

        # UWAGA: dla LangChain+Chroma używamy parametru "filter"
        if where:
            return self.vectorstore.similarity_search(query, k=self.k, filter=where)
        return self.vectorstore.similarity_search(query, k=self.k)
