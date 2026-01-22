# src/rag_chain.py
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain



def build_rag_chain(llm, retriever, qa_prompt, document_prompt):
    """
    Tworzy Retrieval-Augmented Generation chain
    - llm: model LLM
    - retriever: instancja retrievera Chroma
    - qa_prompt: prompt z zmiennymi ['context', 'input']
    - document_prompt: formatowanie pojedynczego dokumentu
    """
    # Łańcuch łączący dokumenty z promptem
    stuff_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt,
        document_variable_name="context",
        document_prompt=document_prompt
    )

    # Pełny łańcuch Retrieval-Augmented Generation
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=stuff_chain
    )

    # Zwracamy oba obiekty:
    # - rag_chain (do automatycznego wyszukiwania)
    # - stuff_chain (do ręcznego podania dokumentów, np. po rerankingu BM25)
    return rag_chain