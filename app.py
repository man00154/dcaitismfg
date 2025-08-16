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
        embeddings = self.emb_model.encode(
            corpus, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
        )
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype("float32"))
        self._embeddings = embeddings

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.docs or self.index is None:
            return []
        import numpy as np
        self._ensure_models()
        q = self.emb_model.encode(
            [query], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
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
        },
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
# Agentic AI Layer
# ------------------------------
class AgenticAI:
    def __init__(self, rag: SimpleRAG, api_key: str):
        self.rag = rag
        self.api_key = api_key

    def analyze(self, query: str) -> str:
        context_docs = self.rag.similarity_search(query, k=3)
        context_texts = "\n\n".join([doc["text"] for doc in context_docs])
        prompt = f"""
        Incident Description:
        {query}

        Retrieved Knowledge:
        {context_texts}

        Task: Perform an RCA (root cause analysis), including:
        - Symptoms & Impact
        - Root Cause
        - Evidence
        - Immediate Solution
        - Preventive Measures
        """
        return gemini_generate(
            api_key=self.api_key,
            user_prompt=prompt,
            system_prompt="You are an expert Data Center SRE performing RCA with actionable solutions.",
            temperature=0.2,
            max_output_tokens=1000,
        )

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Intelligent Data Center RCA", layout="wide")
st.title("🧭 MANISH SINGH - Intelligent Data Center RCA (with LLM + RAG + Agentic AI)")

with st.sidebar:
    st.subheader("🔐 API Key")
    manual_key = st.text_input("Enter GEMINI_API_KEY", type="password", key="api_key_input")
    api_key = manual_key or get_api_key()
    if not api_key:
        st.error("❌ GEMINI_API_KEY is missing. Please set it.")

    st.subheader("📄 Knowledge Base Upload")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# RAG setup
rag = SimpleRAG()
if uploaded_files:
    for pdf in uploaded_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                chunks = chunk_text(text)
                rag.add_documents(chunks, [{"source": pdf.name}] * len(chunks))
    rag.build()

incident_text = st.text_area(
    "🧩 Describe the Incident",
    value="At 10:15 IST, multiple microservices in AZ-2 reported elevated 5xx errors. Deployment occurred 5 minutes before.",
    height=160,
    key="incident_input"
)

if st.button("🔍 Analyze RCA", key="analyze_button"):
    if not api_key:
        st.error("API key required.")
    else:
        agent = AgenticAI(rag, api_key)
        with st.spinner("Analyzing incident..."):
            final_report = agent.analyze(incident_text)
        st.subheader("📄 RCA Report")
        st.markdown(final_report)
