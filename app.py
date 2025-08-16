import os
import io
import json
import asyncio
import aiohttp
import streamlit as st
from typing import List, Dict, Any, Optional
from pypdf import PdfReader

# =========================================================
# Model / API config (Gemini REST)
# =========================================================
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# =========================================================
# API Key helper
# =========================================================
def get_api_key() -> str:
    # Priority: Streamlit secrets → ENV
    key = None
    if hasattr(st, "secrets"):
        key = st.secrets.get("GEMINI_API_KEY", None)
    if not key:
        key = os.getenv("GEMINI_API_KEY")
    return key or ""

# =========================================================
# Async runner helper (Streamlit-safe)
# =========================================================
def try_run(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    else:
        return asyncio.run(coro)

# =========================================================
# Gemini REST (async)
# =========================================================
async def gemini_generate_async(
    api_key: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 800,  # keep outputs concise to avoid MAX_TOKENS
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
            # Safe extraction: fall back to json string if no text is available
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                return json.dumps(data, indent=2)

def gemini_generate(*args, **kwargs) -> str:
    return try_run(gemini_generate_async(*args, **kwargs))

# =========================================================
# Utilities
# =========================================================
def chunk_text(text: str, max_chars: int = 1200, overlap: int = 100) -> List[str]:
    """Simple character chunker for large files."""
    text = text.replace("\x00", "")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def extract_text_from_pdf(file: io.BytesIO) -> str:
    reader = PdfReader(file)
    out = []
    for page in reader.pages:
        out.append(page.extract_text() or "")
    return "\n".join(out)

def trim_for_token_budget(text: str, max_chars: int = 6000) -> str:
    """Hard-limit context to help avoid MAX_TOKENS on the API."""
    if len(text) <= max_chars:
        return text
    # keep beginning and end (often contains headings + errors)
    head = text[: int(max_chars * 0.6)]
    tail = text[-int(max_chars * 0.4):]
    return head + "\n...\n" + tail

# =========================================================
# Minimal RAG (FAISS + sentence-transformers)
# =========================================================
class SimpleRAG:
    """
    A tiny RAG helper:
    - add_documents(texts)
    - build() to create FAISS index
    - similarity_search(query, k)
    """
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

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.docs or self.index is None:
            return []
        self._ensure_models()
        import numpy as np
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

# =========================================================
# Agentic loop
# =========================================================
def agentic_rca_pipeline(api_key: str, incident: str, kb_texts: List[str], temperature: float = 0.2) -> str:
    """
    A simple agent loop:
    1) Generate top hypotheses.
    2) Generate search queries from hypotheses.
    3) RAG retrieve top context chunks.
    4) Ask LLM for structured RCA incl. Solution using retrieved evidence.
    """
    # 1) Hypothesis generation
    hypo_prompt = f"""You are a senior data center SRE.
Given the incident, propose 3 concise hypotheses with 1-2 sentence rationale each.

Incident:
{incident}

Respond strictly in this format:
- Hypothesis 1: <one line> | Rationale: <one line>
- Hypothesis 2: <one line> | Rationale: <one line>
- Hypothesis 3: <one line> | Rationale: <one line>
Keep it brief.
"""
    hypotheses = gemini_generate(
        api_key=api_key,
        user_prompt=hypo_prompt,
        system_prompt="Think like a pragmatic SRE focused on evidence.",
        temperature=temperature,
        max_output_tokens=300
    )

    # 2) Query generation
    qry_prompt = f"""From these hypotheses, produce 5 short keyword queries to search a log/KB.
Be specific: include components, error codes, and metrics if relevant.
Return ONLY a bullet list, one query per line.

Hypotheses:
{hypotheses}"""
    queries_text = gemini_generate(
        api_key=api_key,
        user_prompt=qry_prompt,
        system_prompt="Output queries only.",
        temperature=0.2,
        max_output_tokens=200
    )
    queries = [q.strip("- •\n ") for q in queries_text.splitlines() if q.strip()]

    # 3) Build RAG and retrieve
    ctx = ""
    if kb_texts:
        try:
            rag = SimpleRAG()
            rag.add_documents(kb_texts)
            rag.build()
            collected = []
            for q in queries[:5]:
                hits = rag.similarity_search(q, k=3)
                for h in hits:
                    collected.append(h["text"])
            # Dedup + trim
            unique = []
            seen = set()
            for t in collected:
                t = t.strip()
                if t and t not in seen:
                    unique.append(t)
                    seen.add(t)
            ctx = "\n---\n".join(unique[:12])  # cap retrieved context
            ctx = trim_for_token_budget(ctx, max_chars=6000)
        except Exception as e:
            # If FAISS / sentence-transformers not available, gracefully degrade with no RAG context
            ctx = ""
            st.warning(f"RAG unavailable or failed ({e}). Proceeding without retrieved context.")
    else:
        ctx = ""

    # 4) Final RCA (with Solution)
    final_prompt = f"""
You are a senior Data Center SRE. Perform a Root Cause Analysis (RCA) and produce an actionable Solution.

Incident:
{incident}

Hypotheses (from prior reasoning):
{hypotheses}

Retrieved Evidence (top relevant snippets from logs/KB):
{ctx if ctx else "[No retrieved context available]"}

Write a concise, professional report using EXACTLY the following sections and headings:

1. **Root Cause** – single paragraph describing the technical cause.
2. **Impact** – who/what was affected, scope, duration, SLO/SLA impact.
3. **Evidence** – bullet list with log lines, metrics, error codes, or config snippets that support the RCA.
4. **Solution / Remediation** – numbered, step-by-step commands/actions to resolve the issue now.
5. **Prevention** – 3–5 specific, durable actions (runbooks, alerts, test, capacity, config guardrails).
6. **Rollback/Recovery Notes** – if relevant, detail how to roll back or recover safely.

Constraints:
- Be direct and brief.
- Prefer concrete commands (linux, k8s, db, network) where applicable.
- If evidence is missing, state the assumptions and the minimum extra data needed.
"""
    final_report = gemini_generate(
        api_key=api_key,
        user_prompt=final_prompt,
        system_prompt="Deliver a crisp RCA. Do not exceed a few short paragraphs per section.",
        temperature=temperature,
        max_output_tokens=800
    )
    return final_report

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="Intelligent Data Center RCA (Gemini + RAG + Agent)", layout="wide")
st.title("🧭 MANISH SINGH -  Intelligent Data Center RCA (Gemini + RAG + Agentic)")

with st.sidebar:
    st.subheader("🔐 API Key")
    manual_key = st.text_input("Enter GEMINI_API_KEY (optional)", type="password")
    api_key = manual_key or get_api_key()
    if not api_key:
        st.error("❌ GEMINI_API_KEY is missing. Please add it in `.streamlit/secrets.toml` or enter manually above.")

    st.subheader("⚙️ Model Settings")
    temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

st.markdown("Paste logs/notes or upload files to build the knowledge base (used by RAG).")

col1, col2 = st.columns(2)
with col1:
    kb_text_area = st.text_area("📓 Knowledge Base (paste logs, runbooks, configs)", height=240, placeholder="Paste any relevant logs or KB text here...")
with col2:
    uploaded_files = st.file_uploader("📎 Upload files (.txt, .log, .pdf)", type=["txt", "log", "pdf"], accept_multiple_files=True)

# Build KB texts
kb_texts: List[str] = []
if kb_text_area.strip():
    kb_texts.extend(chunk_text(kb_text_area.strip(), max_chars=1200, overlap=100))

if uploaded_files:
    for uf in uploaded_files:
        try:
            if uf.type in ("text/plain",) or uf.name.lower().endswith((".txt", ".log")):
                txt = uf.read().decode("utf-8", errors="ignore")
                kb_texts.extend(chunk_text(txt, max_chars=1200, overlap=100))
            elif uf.name.lower().endswith(".pdf"):
                buf = io.BytesIO(uf.read())
                txt = extract_text_from_pdf(buf)
                kb_texts.extend(chunk_text(txt, max_chars=1200, overlap=100))
        except Exception as e:
            st.warning(f"Could not read {uf.name}: {e}")

incident_text = st.text_area(
    "🧩 Describe the Incident",
    value="At 10:15 IST, multiple microservices in AZ-2 reported elevated 5xx rates and latency spikes. Ingress shows connection resets. DB connection pool saturation observed. A deployment occurred 5 minutes before the spike.",
    height=160
)

# Buttons
build_kb = st.button("🧠 Build/Refresh Knowledge Base (local)")
analyze_btn = st.button("🔍 Run Agentic RCA (uses RAG + LLM)")

# We don't need to persist FAISS index object because we rebuild on each analyze call
if build_kb:
    st.success(f"Knowledge Base prepared with ~{len(kb_texts)} chunks.")

if analyze_btn:
    if not api_key:
        st.error("API key required to analyze.")
    elif not incident_text.strip():
        st.warning("Please describe the incident.")
    else:
        with st.spinner("Analyzing incident with agentic loop (hypotheses → queries → retrieval → RCA)..."):
            report = agentic_rca_pipeline(
                api_key=api_key,
                incident=incident_text.strip(),
                kb_texts=kb_texts,
                temperature=temp
            )
        st.subheader("📄 RCA Report")
        st.markdown(report)
        st.download_button(
            "⬇️ Download RCA",
            data=report,
            file_name="rca_report.md",
            mime="text/markdown",
            use_container_width=True
        )
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
