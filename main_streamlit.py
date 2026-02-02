import streamlit as st
from PIL import Image
from langchain_community.chat_models import ChatOllama

from src.config import MODEL_NAME, SERVER_URL
from src.embeddings import build_embeddings
from src.vectorstore import build_vector_store
from src.prompts import QA_PROMPT, DOCUMENT_PROMPT
from src.rag_chain import build_rag_chain
from src.routing_retriever import ActRoutingRetriever
from src.config import RETRIEVER_K


# ---------- Ustawienia strony ----------
st.set_page_config(page_title="Chatbot Prawniczy", layout="wide")

# ==========================================
# STYLE CSS - FINALNA WERSJA (SYMETRIA)
# ==========================================
st.markdown(
    """
    <style>
    /* 1. KONTENER GŁÓWNY */
    [data-testid="stChatInput"] {
        max-width: 800px;           /* Szerokość jak w ChatGPT */
        margin-left: auto;          /* Centrowanie na ekranie */
        margin-right: auto;
        margin-bottom: 40px;        /* Podniesienie nad dolną krawędź */
        
        /* KLUCZOWE DLA SYMETRII: */
        align-items: center !important; /* Wymusza, by wszystko w środku (ikony i tekst) było w jednej linii poziomej */
        border-radius: 20px;        /* Zaokrąglenie całego paska */
    }

    /* 2. POLE TEKSTOWE (ŚRODEK) */
    [data-testid="stChatInput"] textarea {
        min-height: 55px !important;    /* Wysokość fizyczna paska */
        padding-top: 16px !important;   /* Wypychanie tekstu, żeby był na środku wysokości */
        padding-bottom: 16px !important;
    }

    /* 3. PRZYCISKI (IKONA PLIKU + IKONA WYSYŁANIA) */
    /* Ten selektor łapie każdy guzik wewnątrz paska inputu */
    [data-testid="stChatInput"] button {
        align-self: center !important;  /* Centruje ikonę w pionie względem wysokiego paska */
        margin-top: 0px !important;     /* Kasuje ewentualne domyślne przesunięcia Streamlit */
        height: auto !important;
    }
    
    /* Opcjonalnie: Jeśli ikona pliku jest zbyt blisko krawędzi, dodaj jej margines */
    [data-testid="stChatInputFileUploader"] {
        margin-left: 5px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("🤖 Chatbot Prawniczy")


# ---------- Stan sesji (historia chatu + RAG) ----------
if "messages" not in st.session_state:
    # [{"role": "user"/"assistant", "content": str}]
    st.session_state.messages = []

if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False


# ---------- Inicjalizacja LLM + RAG  ----------
@st.cache_resource(show_spinner=True)
def init_rag():
    llm = ChatOllama(
        base_url=SERVER_URL,   # http://127.0.0.1:11434
        model=MODEL_NAME,      # gemma3:27b-it-q4_K_M
        temperature=0.2,
    )

    embeddings = build_embeddings()
    db, retriever = build_vector_store(embeddings)
    retriever = ActRoutingRetriever(vectorstore=db, k=RETRIEVER_K, max_acts=2, debug=True)

    # Tworzy: rag_chain (retriever+LLM) oraz combine_docs_chain (LLM na podanych docach)
    rag_chain = build_rag_chain(
        llm, retriever, QA_PROMPT, DOCUMENT_PROMPT
    )

    return rag_chain, retriever


if not st.session_state.rag_ready:
    rag_chain, retriever = init_rag()
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
    result = rag_chain.invoke({"input": user_query})

    # Wyciągamy odpowiedź
    answer = result.get("answer", "Brak odpowiedzi")

    # Wyciągamy dokumenty, które znalazł retriever
    retrieved_docs = result.get("context", [])

    return answer.strip(), retrieved_docs, retrieved_docs


# ---------- Wyświetlanie historii chatu ----------
# Jeśli chcesz, aby wiadomości też były węższe i na środku (jak w ChatGPT),
# możesz odkomentować styl .stChatMessage w sekcji CSS powyżej.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------- Nowa wiadomość użytkownika + załączniki przy input ----------
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

                st.markdown(answer_text)

                st.markdown("**Źródła:**")
                if not final_docs:
                    st.write("- Brak źródeł.")
                else:
                    for doc in final_docs:
                        src = doc.metadata.get("source", "Nieznane źródło")
                        page = doc.metadata.get("page", "N/A")
                        st.write(f"- {src}, strona {page}")

                with st.expander("Informacje debugowe (retriever)", expanded=False):
                    st.subheader("Krok 1: Surowe wyniki z bazy wektorowej (raw_docs)")
                    for i, doc in enumerate(raw_docs):
                        st.write(
                            f"**Wynik [RAW] #{i}** "
                            f"(Source: {doc.metadata.get('source')}, "
                            f"Page: {doc.metadata.get('page')})"
                        )
                        st.text(f"{doc.page_content[:500]}...")

                    st.subheader("Krok 2: Wyniki końcowe(final_docs)")
                    for i, doc in enumerate(final_docs):
                        st.write(
                            f"**Wynik [FINAL] #{i}** "
                            f"(Source: {doc.metadata.get('source')}, "
                            f"Page: {doc.metadata.get('page')})"
                        )
                        st.text(f"{doc.page_content[:500]}...")

        # 4) zapisz odpowiedź w historii
        st.session_state.messages.append(
            {"role": "assistant", "content": answer_text}
        )