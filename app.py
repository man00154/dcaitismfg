import os
import io
import json
import asyncio
import aiohttp
import streamlit as st
from typing import List, Dict, Any, Optional
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

def safe_trim(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]

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
        import numpy as np  # noqa: F401 (import used by FAISS)
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
        import numpy as np  # noqa: F401
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
# Gemini REST (async) + robust decode + auto-retry
# ------------------------------
async def gemini_generate_async(
    api_key: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1200,
    retries: int = 2,
) -> str:
    """
    Robust Gemini call:
    - Prepends a concise instruction to reduce MAX_TOKENS risk.
    - Retries with smaller maxOutputTokens if finishReason hits a boundary.
    - Safely decodes candidate text; if missing, returns JSON for debugging.
    """
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}

    concise_guardrail = (
        "Respond concisely and within the token budget. Use the exact sections:"
        "\nRoot Cause\nEvidence\nImpact\nImmediate Solution\nPreventive Measures"
    )

    for attempt in range(retries + 1):
        effective_max = max(512, int(max_output_tokens * (0.7 ** attempt)))

        parts = []
        if system_prompt:
            parts.append({"text": f"[SYSTEM]\n{system_prompt}\n\n{concise_guardrail}".strip()})
        else:
            parts.append({"text": concise_guardrail})
        parts.append({"text": user_prompt})

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": effective_max,
                "responseMimeType": "text/plain",
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL, headers=headers, params=params, json=payload, timeout=120) as resp:
                body_text = await resp.text()
                if resp.status != 200:
                    # On rate limits / server errors: back off then retry.
                    if attempt < retries and resp.status in (429, 500, 502, 503, 504):
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise RuntimeError(f"Gemini API error {resp.status}: {body_text}")

                data = json.loads(body_text)

                # Try to read the candidate text normally
                try:
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    if text and text.strip():
                        return text
                except Exception:
                    # If finishReason like MAX_TOKENS prevented parts, retry with smaller budget
                    finish_reason = (
                        data.get("candidates", [{}])[0].get("finishReason")
                        if data.get("candidates") else None
                    )
                    if attempt < retries and finish_reason == "MAX_TOKENS":
                        # back off and retry with lower token cap
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    # As a last resort, return the JSON so user can see what came back
                    return json.dumps(data, indent=2)

    # Should not reach here; return a safe fallback
    return "Unable to generate a response at this time."

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
        # Retrieve top docs and enforce a context size cap to avoid hitting token limits
        context_docs = self.rag.similarity_search(query, k=3)
        context_texts = "\n\n".join([doc["text"] for doc in context_docs])
        context_texts = safe_trim(context_texts, max_chars=3500)  # cap context

        prompt = f"""
Incident Description:
{query}

Retrieved Knowledge (summarize & cite minimally):
{context_texts}

Task: Perform an RCA (root cause analysis) with the following exact sections:
- Root Cause
- Evidence
- Impact
- Immediate Solution
- Preventive Measures

Be concise and action-oriented. Keep total output under ~350 words.
"""
        return gemini_generate(
            api_key=self.api_key,
            user_prompt=prompt,
            system_prompt="You are an expert Data Center SRE performing RCA with actionable solutions.",
            temperature=0.2,
            max_output_tokens=1200,
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
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    chunks = chunk_text(text)
                    rag.add_documents(chunks, [{"source": pdf.name}] * len(chunks))
        except Exception as e:
            st.warning(f"Failed to parse {getattr(pdf, 'name', 'PDF')}: {e}")
    rag.build()

incident_text = st.text_area(
    "🧩 Describe the Incident",
    value="At 10:15 IST, multiple microservices in AZ-2 reported elevated 5xx errors and latency spikes. "
          "Ingress shows connection resets. DB connection pool saturation observed. "
          "A deployment occurred 5 minutes before the spike.",
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
