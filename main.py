import sys
from src.embeddings import build_embeddings
from src.vectorstore import build_vector_store
from src.prompts import QA_PROMPT, DOCUMENT_PROMPT
from src.rag_chain import build_rag_chain
from src.chat import display_answer
from langchain_community.chat_models import ChatOllama
from src.config import SERVER_URL, MODEL_NAME, RETRIEVER_K
from src.chat import debug_retrieved_documents
from src.routing_retriever import ActRoutingRetriever

def main():
    # LLM
    llm = ChatOllama(
    base_url=SERVER_URL,   # http://127.0.0.1:11434
    model=MODEL_NAME,      # gemma3:27b-it-q4_K_M
    temperature=0.2,
    )
    
    # Embeddings
    embeddings = build_embeddings()
    
    # Vectorstore
    db, retriever = build_vector_store(embeddings)

    retriever = ActRoutingRetriever(vectorstore=db, k=RETRIEVER_K, max_acts=2, debug=True)
    # RAG chain
    rag_chain = build_rag_chain(llm, retriever, QA_PROMPT, DOCUMENT_PROMPT)
    
    # CLI loop
    while True:
        query = input("\nTy: ")
        if query.lower() in ["wyjdz", "exit", "quit"]:
            print("👋 Do widzenia!")
            break
        if not query.strip():
            continue
        print("\nChatbot (analizuję dokumenty...)")
        docs = retriever.invoke(query)
        debug_retrieved_documents(docs, query)
        result = rag_chain.invoke({"input": query})
        display_answer(result)

if __name__ == "__main__":
    main()
