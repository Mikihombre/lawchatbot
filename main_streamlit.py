import streamlit as st
from PIL import Image
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from src.routing_retriever import ActRoutingRetriever
from src.config import MODEL_NAME, SERVER_URL, RETRIEVER_K
from src.embeddings import build_embeddings
from src.vectorstore import build_vector_store
from src.prompts import QA_PROMPT, DOCUMENT_PROMPT
from src.rag_chain import build_rag_chain

# ---------- Ustawienia strony ----------
st.set_page_config(
    page_title="Asystent Prawny AI", 
    page_icon="⚖️", 
    layout="wide"
)

# ---------- Zaawansowany CSS (Efekty Hover i Layout) ----------
st.markdown(
    """
    <style>
    /* Kontener wejściowy chat_input */
    [data-testid="stChatInput"] {
        max-width: 850px;
        margin-left: auto;
        margin-right: auto;
        margin-bottom: 40px;
        border-radius: 20px;
        border: 1px solid #e0e0e0;
    }

    /* --- ANIMACJA IKONEK I KURSOR --- */
    /* Celujemy w przycisk wysyłania i ikonę dodawania plików */
    [data-testid="stChatInput"] button, 
    [data-testid="stChatInput"] label[data-testid="stWidgetLabel"] {
        transition: transform 0.2s ease-in-out !important;
        cursor: pointer !important; /* <--- TUTAJ DODANO EFEKT POINTERA */
    }

    /* Powiększenie przycisku wyślij po najechaniu */
    [data-testid="stChatInput"] button:hover {
        transform: scale(1.18) !important;
    }

    /* Powiększenie ikony plusa (upload) po najechaniu */
    [data-testid="stChatInput"] label:hover {
        transform: scale(1.18) !important;
    }

    /* Styl dla źródeł prawnych */
    .source-box {
        background-color: rgba(255, 255, 255, 0.06) !important;   
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #ff4b4b;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Marginesy głównego kontenera */
    .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Stan sesji ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

# ---------- Inicjalizacja RAG (Bez zmian w logice) ----------
@st.cache_resource(show_spinner=True)
def init_rag():
    llm = ChatOllama(
        base_url=SERVER_URL,
        model=MODEL_NAME,
        temperature=0.2,
    )
    embeddings = build_embeddings()
    db, _ = build_vector_store(embeddings)

    retriever = ActRoutingRetriever(
    vectorstore=db,
    max_acts=2,
    debug=True,
    )

    rag_chain = build_rag_chain(llm, retriever, QA_PROMPT, DOCUMENT_PROMPT)
    return rag_chain, retriever

if not st.session_state.rag_ready:
    with st.spinner("🚀 Inicjalizacja bazy przepisów..."):
        rag_chain, retriever = init_rag()
        st.session_state.rag_chain = rag_chain
        st.session_state.retriever = retriever
        st.session_state.rag_ready = True

# ---------- Ekran Główny ----------

st.title("⚖️ Asystent Prawny AI")
st.markdown("Skonsultuj problem prawny w oparciu o aktualne kodeksy.")

# ---------- Wyświetlanie Historii ----------
if not st.session_state.messages:
    st.write("")
    st.info("Zadaj pytanie, aby rozpocząć analizę. Możesz przeciągnąć dokumenty bezpośrednio do pola tekstowego.")

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "⚖️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

def build_chat_history(messages, max_pairs=3):
    """
    Buduje historię do HistoryAwareRetriever tylko z kompletnych par:
    user -> assistant. Dzięki temu condense nie dostaje "śmieci"
    ani nieparzystych elementów.

    max_pairs=3 => maks. 6 wiadomości w historii (3 pytania + 3 odpowiedzi)
    """
    history = []
    pairs = []
    current_user = None

    for msg in messages:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue

        if role == "user":
            current_user = content

        elif role == "assistant":
            if current_user is not None:
                pairs.append((current_user, content))
                current_user = None

    # bierzemy ostatnie max_pairs par
    for u, a in pairs[-max_pairs:]:
        history.append(HumanMessage(content=u))
        history.append(AIMessage(content=a))

    return history


def run_rag_pipeline(user_query: str):
    rag_chain = st.session_state.rag_chain

    # ✅ Historia tylko z kompletnych par user->assistant
    chat_history_objs = build_chat_history(st.session_state.messages, max_pairs=3)

    result = rag_chain.invoke({
        "input": user_query,
        "chat_history": chat_history_objs
    })

    answer = result.get("answer") or result.get("output_text") or ""
    docs = result.get("context") or result.get("documents") or []

    return answer, docs

# ---------- INPUT ----------
chat_value = st.chat_input(
    "Napisz pytanie lub załącz pliki...",
    accept_file="multiple",
    file_type=["pdf", "png", "jpg"]
)

if chat_value:
    user_text = chat_value.text or ""
    user_files = chat_value.files or []

    if user_text.strip() or user_files:
        # 1. User Message
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_text)

        # 2. AI Response
        with st.chat_message("assistant", avatar="⚖️"):
            message_placeholder = st.empty()
            with st.spinner("⚖️ Analizuję treść aktów prawnych..."):
                answer_text, final_docs = run_rag_pipeline(user_text)
                message_placeholder.markdown(answer_text)
                
                if final_docs:
                    with st.expander("📚 Wykorzystane źródła"):
                        for doc in final_docs:
                            src = doc.metadata.get("source", "Dokument").split("/")[-1]
                            act = doc.metadata.get("act_name", "Przepis")
                            st.markdown(
                                f"""
                                <div class="source-box">
                                    <strong>{act}</strong> <small>({src})</small><br>
                                    <p style="font-size: 0.85rem; color: #444; margin-top: 8px;">
                                    "{doc.page_content[:350]}..."
                                    </p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )

        # 3. Save History
        st.session_state.messages.append({"role": "assistant", "content": answer_text})