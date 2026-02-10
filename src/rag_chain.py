# src/rag_chain.py
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from src.prompts import CONDENSE_QUESTION_PROMPT


def build_rag_chain(llm, retriever, qa_prompt, document_prompt):
    """
    Tworzy zaawansowany łańcuch RAG z obsługą historii (History Aware).
    """
    
    # ---------------------------------------------------------
    # KROK 1: AGENT REFORMUŁUJĄCY (History Aware Retriever)
    # ---------------------------------------------------------
    # Ten łańcuch bierze historię rozmowy i nowe pytanie, a następnie
    # "przepisuje" pytanie tak, aby było zrozumiałe dla wyszukiwarki (Retrievera).
    # Np. User: "A dla kogo?" -> Agent: "Jaka jest kara RODO dla urzędów?"
    
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=CONDENSE_QUESTION_PROMPT
    )

    # ---------------------------------------------------------
    # KROK 2: AGENT ODPOWIADAJĄCY (Stuff Documents Chain)
    # ---------------------------------------------------------
    # To jest standardowy łańcuch, który bierze dokumenty i generuje odpowiedź prawną.
    
    stuff_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt,
        document_variable_name="context",
        document_prompt=document_prompt
    )

    # ---------------------------------------------------------
    # KROK 3: POŁĄCZENIE (Final Pipeline)
    # ---------------------------------------------------------
    # Łączymy oba kroki w jeden główny łańcuch.
    # Input -> [History Retriever] -> Dokumenty -> [Stuff Chain] -> Odpowiedź
    
    rag_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=stuff_chain
    )

    return rag_chain