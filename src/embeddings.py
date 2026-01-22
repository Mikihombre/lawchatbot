from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL

def build_embeddings():
    try:
        emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cuda"})
        _ = emb.embed_query("test")
        print("✅ Embeddings na GPU (CUDA).")
        return emb
    except Exception as e:
        print(f"⚠️ CUDA niedostępna: {e}. Przełączam na CPU...")
        emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
        _ = emb.embed_query("test")
        print("✅ Embeddings na CPU.")
        return emb
