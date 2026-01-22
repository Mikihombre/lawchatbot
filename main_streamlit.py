import streamlit as st
from PIL import Image
from langchain_openai import ChatOpenAI

from src.config import MODEL_NAME, SERVER_URL
from src.embeddings import build_embeddings
from src.vectorstore import build_vector_store
from src.prompts import QA_PROMPT, DOCUMENT_PROMPT
from src.rag_chain import build_rag_chain



# ---------- Ustawienia strony ----------
st.set_page_config(page_title="Chatbot Prawniczy", layout="wide")
st.title("🤖 Chatbot Prawniczy")


# ---------- Stan sesji (historia chatu + RAG) ----------
if "messages" not in st.session_state:
    # [{"role": "user"/"assistant", "content": str}]
    st.session_state.messages = []

if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False


# ---------- Inicjalizacja LLM + RAG + reranker ----------
@st.cache_resource(show_spinner=True)
def init_rag():
    llm = ChatOpenAI(
        base_url=SERVER_URL,
        api_key="not-needed",
        model=MODEL_NAME,
        temperature=0.2,
        request_timeout=120,
    )

    embeddings = build_embeddings()
    db, retriever = build_vector_store()

    # Tworzy: rag_chain (retriever+LLM) oraz combine_docs_chain (LLM na podanych docach)
    rag_chain = build_rag_chain(
        llm, retriever, QA_PROMPT, DOCUMENT_PROMPT
    )

    #st.text("System RAG jest gotowy.")

    return rag_chain,retriever


if not st.session_state.rag_ready:
    rag_chain,retriever = init_rag()
    st.session_state.rag_chain = rag_chain
    st.session_state.retriever = retriever
    st.session_state.rag_ready = True
else:
    rag_chain = st.session_state.rag_chain
    retriever = st.session_state.retriever


# ---------- Placeholder na tekst z załączonych plików ----------
def extract_text_from_files(files) -> str:
    """
    TODO:
      - dla PDF: dodać wyciąganie tekstu (PyMuPDF / pdfplumber)
      - dla obrazów: dodać OCR (pytesseract, lang='pol')
    Teraz tylko wypisujemy nazwy plików jako 'treść wniosku'.
    """
    if not files:
        return ""
    lines = [f"[plik] {f.name}" for f in files]
    return "\n".join(lines)


# ---------- Pipeline RAG dla jednego pytania ----------
def run_rag_pipeline(user_query: str):
    # Pobieramy gotowy łańcuch z sesji
    rag_chain = st.session_state.rag_chain
    
    # Uruchamiamy łańcuch. 
    # Dzięki 'create_retrieval_chain', on sam pobierze dokumenty (context) 
    # na podstawie pytania (input).
    result = rag_chain.invoke({"input": user_query})

    # Wyciągamy odpowiedź
    answer = result.get("answer", "Brak odpowiedzi")

    # Wyciągamy dokumenty, które znalazł retriever
    # (W 'create_retrieval_chain' są one zwracane pod kluczem "context")
    retrieved_docs = result.get("context", [])

    # Zwracamy wynik.
    # Uwaga: raw_docs i final_docs to teraz to samo, 
    # bo usunęliśmy etap filtrowania (rerankera).
    return answer.strip(), retrieved_docs, retrieved_docs


# ---------- Wyświetlanie historii chatu ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------- Nowa wiadomość użytkownika + załączniki przy input ----------
# WYMAGA Streamlit >= 1.43 (accept_file/file_type w chat_input)

# chat_input_value to teraz dict-like obiekt z .text i .files (zgodnie z docs)
chat_value = st.chat_input(
    "Zadaj pytanie lub napisz polecenie...",
    accept_file="multiple",
    file_type=["pdf", "png", "jpg", "jpeg"],
)

if chat_value is not None:
    # chat_value to obiekt ChatInputValue: ma .text i .files
    user_text = chat_value.text or ""
    user_files = chat_value.files or []

    # jeśli jest jakikolwiek tekst albo pliki, to działamy
    if user_text.strip() or user_files:
        # 1) dopisz wiadomość użytkownika do historii
        st.session_state.messages.append(
            {"role": "user", "content": user_text}
        )
        with st.chat_message("user"):
            st.markdown(user_text if user_text.strip() else "[wiadomość z załącznikami]")

        # 2) prosty „wniosek” z plików (na razie tylko nazwy)
        wniosek_text = extract_text_from_files(user_files)
        if wniosek_text:
            with st.expander("Załączone pliki (do analizy wniosku)"):
                st.text(wniosek_text)

        # 3) RAG + debug + odpowiedź
        with st.chat_message("assistant"):
            with st.spinner("Analizuję dokumenty..."):
                answer_text, raw_docs, final_docs = run_rag_pipeline(user_text)

                with st.expander("Informacje debugowe (retriever + reranker)", expanded=False):
                    st.subheader("Krok 1: Surowe wyniki z bazy wektorowej (raw_docs)")
                    for i, doc in enumerate(raw_docs):
                        st.write(
                            f"**Wynik [RAW] #{i}** "
                            f"(Source: {doc.metadata.get('source')}, "
                            f"Page: {doc.metadata.get('page')})"
                        )
                        st.text(f"{doc.page_content[:500]}...")

                    st.subheader("Krok 2: Wyniki po Rerankingu Neuronowym (final_docs)")
                    for i, doc in enumerate(final_docs):
                        st.write(
                            f"**Wynik [FINAL] #{i}** "
                            f"(Source: {doc.metadata.get('source')}, "
                            f"Page: {doc.metadata.get('page')})"
                        )
                        st.text(f"{doc.page_content[:500]}...")

                st.markdown(answer_text)

                st.markdown("**Źródła:**")
                if not final_docs:
                    st.write("- Brak źródeł.")
                else:
                    for doc in final_docs:
                        src = doc.metadata.get("source", "Nieznane źródło")
                        page = doc.metadata.get("page", "N/A")
                        st.write(f"- {src}, strona {page}")

        # 4) zapisz odpowiedź w historii
        st.session_state.messages.append(
            {"role": "assistant", "content": answer_text}
        )