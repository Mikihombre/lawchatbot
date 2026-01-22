from typing import List, Optional, Any
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

    def _where(self, act_names: List[str]) -> Optional[dict]:
        if not act_names:
            return None
        if len(act_names) == 1:
            return {"act_name": act_names[0]}
        return {"$or": [{"act_name": n} for n in act_names]}
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        act_names = route_act_names(query, max_acts=self.max_acts)
        where = self._where(act_names)

        if self.debug:
            print(f"[DEBUG] ROUTING: {act_names if act_names else 'ALL (fallback)'}")

        # UWAGA: dla LangChain+Chroma używamy parametru "filter"
        if where:
            return self.vectorstore.similarity_search(query, k=self.k, filter=where)
        return self.vectorstore.similarity_search(query, k=self.k)
