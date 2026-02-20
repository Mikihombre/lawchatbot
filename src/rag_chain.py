# src/rag_chain.py
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever

from src.prompts import CONDENSE_QUESTION_PROMPT
from src.rewriting_retriever import RewritingRetriever


def build_rag_chain(llm, retriever, qa_prompt, document_prompt):
    """
    Tworzy wielowarstwowy łańcuch RAG z obsługą historii + rewriterem zapytań.

    Finalnie pipeline wygląda tak:
    Input -> [History Aware] -> [Query Rewriter] -> [Retriever/Router] -> Dokumenty -> [Generator] -> Odpowiedź
    """

    # ---------------------------------------------------------
    # KROK 1: AGENT NORMALIZUJĄCY ZAPYTANIE (Query Rewriter)
    # ---------------------------------------------------------
    # Ten wrapper stoi "przed" Twoim właściwym retrieverem.
    # Bierze pytanie i przepisuje je na wersję prawniczą / formalną, np.:
    # "urząd milczy" -> "bezczynność organu administracji"
    #
    # Ważne: on NIE wyszukuje dokumentów sam — tylko przygotowuje lepsze query,
    # a potem deleguje wyszukiwanie do retrievera bazowego (np. routera aktów).
    #
    # RewritingRetriever używa funkcji rewrite_query(...) (LLM + JSON),
    # bierze rewritten_query i woła base_retriever z tym przepisanym pytaniem.

    smart_retriever = RewritingRetriever(
        base_retriever=retriever,  # <- Twój dotychczasowy retriever/router
        rewriter_llm=llm,          # <- Ten sam LLM (np. Bielik)
        debug=True                 # <- Logi w konsoli: [REWRITE] ...
    )

    # ---------------------------------------------------------
    # KROK 2: AGENT REFORMUŁUJĄCY Z HISTORIĄ (History Aware Retriever)
    # ---------------------------------------------------------
    # Ten etap pilnuje ciągłości rozmowy.
    # Jeśli użytkownik odnosi się do poprzedniego kontekstu:
    # "A jakie są kary?" -> "Jakie są kary za [temat z poprzedniego pytania]?"
    #
    # I dopiero to "pełne" pytanie przekazuje dalej do retrievera.
    #
    # Uwaga: tutaj jako retriever wpinamy smart_retriever, czyli:
    # History-aware -> Rewriter -> Bazowy retriever/router -> dokumenty

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=smart_retriever,     # <- ZAMIANA: wpinamy rewriter
        prompt=CONDENSE_QUESTION_PROMPT
    )

    # ---------------------------------------------------------
    # KROK 3: AGENT ODPOWIADAJĄCY (Stuff Documents Chain / Generator)
    # ---------------------------------------------------------
    # Ten łańcuch bierze znalezione dokumenty ("context") i generuje odpowiedź.
    # To jest warstwa "prawnika": ma instrukcję w qa_prompt jak odpowiadać,
    # oraz document_prompt jak formatować pojedyncze fragmenty w kontekście.

    stuff_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt,
        document_variable_name="context",
        document_prompt=document_prompt
    )

    # ---------------------------------------------------------
    # KROK 4: POŁĄCZENIE (Final Pipeline)
    # ---------------------------------------------------------
    # Składamy wszystko w jeden główny chain:
    # Input (question + chat_history)
    #   -> history_aware_retriever (kondensuje pytanie z historią)
    #       -> smart_retriever (przepisuje na język prawniczy)
    #           -> retriever/router (wyszukuje dokumenty)
    #   -> stuff_chain (LLM generuje odpowiedź na podstawie dokumentów)

    rag_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=stuff_chain
    )

    return rag_chain
