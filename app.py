import os
import io
import json
import asyncio
import aiohttp
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from pypdf import PdfReader

# --- Model / API config ---
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# ------------------------------
# API Key helper
# ------------------------------
def get_api_key() -> str:
    # Priority: Streamlit secrets → ENV
    key = None
    if hasattr(st, "secrets"):
        key = st.secrets.get("GEMINI_API_KEY", None)
    if not key:
        key = os.getenv("GEMINI_API_KEY")
    return key or ""

# ------------------------------
# Utilities
# ------------------------------
def chunk_text(text: str, max_chars: int = 1200, overlap: int = 100) -> List[str]:
    text = text.replace("\x00", "")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def try_run(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    else:
        return asyncio.run(coro)

# ------------------------------
# Minimal RAG (FAISS + sentence-transformers)
# ------------------------------
class SimpleRAG:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.emb_model = None
        self.index = None
        self.docs: List[Dict[str, Any]] = []

    def _ensure_models(self):
        if self.emb_model is None:
            from sentence_transformers import SentenceTransformer
            self.emb_model = SentenceTransformer(self.model_name)

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        if metadatas is None:
            metadatas = [{} for _ in texts]
        for t, m in zip(texts, metadatas):
            self.docs.append({"text": t, "metadata": m})

    def build(self):
        if not self.docs:
            return
        self._ensure_models()
        import numpy as np
        import faiss
        corpus = [d["text"] for d in self.docs]
        embeddings = self.emb_model.encode(corpus, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype("float32"))
        self._embeddings = embeddings

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.docs or self.index is None:
            return []
        import numpy as np
        self._ensure_models()
        q = self.emb_model.encode([query], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = self.index.search(q, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            d = self.docs[idx]
            results.append({"text": d["text"], "metadata": d.get("metadata", {}), "score": float(score)})
        return results

# ------------------------------
# Gemini REST (async)
# ------------------------------
async def gemini_generate_async(
    api_key: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
) -> str:
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}

    parts = []
    if system_prompt:
        parts.append({"text": f"[SYSTEM]\n{system_prompt}".strip()})
    parts.append({"text": user_prompt})

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        }
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=headers, params=params, json=payload, timeout=120) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Gemini API error {resp.status}: {body}")
            data = await resp.json()
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                return json.dumps(data, indent=2)

def gemini_generate(*args, **kwargs) -> str:
    return try_run(gemini_generate_async(*args, **kwargs))

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="MANISH SINGH - Data Center RCA (Gemini)", layout="wide")
st.title("🧭 MANISH SINGH -  Intelligent Data Center RCA")

with st.sidebar:
    st.subheader("🔐 API Key")
    manual_key = st.text_input("Enter GEMINI_API_KEY (optional)", type="password")
    api_key = manual_key or get_api_key()
    if not api_key:
        st.error("❌ GEMINI_API_KEY is missing. Please add it in `.streamlit/secrets.toml` or enter manually above.")

    st.subheader("⚙️ Model Settings")
    temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

incident_text = st.text_area(
    "🧩 Describe the Incident",
    value="At 10:15 IST, multiple microservices in AZ-2 reported elevated 5xx rates and latency spikes. Ingress shows connection resets. DB connection pool saturation observed. A deployment occurred 5 minutes before the spike.",
    height=160
)

if st.button("🔍 Analyze RCA"):
    if not api_key:
        st.error("API key required to analyze.")
    else:
        with st.spinner("Analyzing incident..."):
            final_report = gemini_generate(
                api_key=api_key,
                user_prompt=f"Perform a root cause analysis and remediation plan for this incident:\n\n{incident_text}",
                system_prompt="You are a senior data center SRE. Provide RCA with evidence, impact, remediation, and prevention steps.",
                temperature=temp,
                max_output_tokens=1200
            )
        st.subheader("📄 RCA Report")
        st.markdown(final_report)
