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
# Utilities
# ------------------------------
def get_api_key() -> str:
    # Priority: Streamlit Secrets -> ENV
    key = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
    if not key:
        key = os.getenv("GEMINI_API_KEY")
    if not key:
        st.warning("API key not found. Set GEMINI_API_KEY in Streamlit secrets or environment variables.")
    return key or ""

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 100) -> List[str]:
    text = text.replace("\x00", "")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def try_run(coro):
    """Safely run an async function from Streamlit."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # Running inside an event loop (rare in Streamlit), create a task and wait
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    else:
        return asyncio.run(coro)

# ------------------------------
# Minimal RAG (FAISS + sentence-transformers)
# ------------------------------
# We keep the import local so importing app.py is fast; the models load only when needed.
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
        # Keep the matrix to map indices -> docs quickly
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
    headers = {
        "Content-Type": "application/json",
    }
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
# Agentic tools (operate on logs + RAG)
# ------------------------------
def tool_search_kb(query: str, rag: SimpleRAG, k: int = 5) -> Dict[str, Any]:
    hits = rag.similarity_search(query, k=k) if rag else []
    return {
        "tool": "search_kb",
        "query": query,
        "hits": [{"text": h["text"][:1000], "score": h["score"], "metadata": h.get("metadata", {})} for h in hits]
    }

def tool_grep_logs(pattern: str, logs_text: str, window: int = 2) -> Dict[str, Any]:
    if not logs_text:
        return {"tool": "grep_logs", "pattern": pattern, "matches": []}
    lines = logs_text.splitlines()
    matches = []
    for i, line in enumerate(lines):
        if pattern.lower() in line.lower():
            start = max(0, i - window)
            end = min(len(lines), i + window + 1)
            snippet = "\n".join(lines[start:end])
            matches.append(snippet)
    return {"tool": "grep_logs", "pattern": pattern, "matches": matches[:10]}

def tool_propose_runbook(incident: str, evidence: str) -> Dict[str, Any]:
    # Lightweight, deterministic checklist generator (no LLM)
    checklist = [
        "Validate alert: confirm severity, time window, impacted services.",
        "Correlate metrics: CPU, memory, I/O, network, error rates.",
        "Check recent changes: deploys, config flags, feature rollouts.",
        "Inspect service health: restarts, OOM, throttling, crash loops.",
        "Check dependencies: DB, cache, message brokers, external APIs.",
        "Rollback or scale: rollback the last change or scale replicas.",
        "Create RCA doc: root cause, impact, timeline, remediation."
    ]
    return {
        "tool": "propose_runbook",
        "incident": incident[:500],
        "evidence_excerpt": evidence[:500],
        "checklist": checklist
    }

# Agent policy: Ask the LLM for the next tool call (JSON); loop a few times; then ask for final RCA.
AGENT_SYSTEM = """You are an SRE/Infra expert Agent. You have 3 tools:
1) search_kb{query} -> returns top KB chunks from vector store.
2) grep_logs{pattern} -> searches uploaded logs for a substring and returns nearby lines.
3) propose_runbook{incident, evidence} -> returns a neutral checklist (no LLM).

When asked, respond ONLY with a JSON dict like:
{"action":"tool","name":"search_kb","args":{"query":"..."}}
or
{"action":"tool","name":"grep_logs","args":{"pattern":"..."}}
or
{"action":"tool","name":"propose_runbook","args":{"incident":"...","evidence":"..."}}
or
{"action":"final"}

DO NOT include explanations in tool steps. Keep args concise.
"""

FINAL_SYSTEM = """Now craft the FINAL output.
Return sections with clear headings:
- Executive Summary
- Probable Root Cause (with reasoning)
- Blast Radius / Impact
- Evidence (citations to KB chunks or log snippets)
- Remediation Steps (immediate)
- Preventive Actions (long term)
- Verification / Rollback Plan
- Incident Timeline (bulleted, approximate ok)
Be specific and pragmatic for data center environments (compute/network/storage/DB)."""

def run_agentic_rca(
    api_key: str,
    incident_text: str,
    rag: Optional[SimpleRAG],
    logs_text: str,
    max_steps: int = 4
) -> Tuple[List[Dict[str, Any]], str]:
    """Returns (trace, final_report)"""
    trace: List[Dict[str, Any]] = []
    context_summary = ""

    # Tool loop
    for step in range(max_steps):
        user_prompt = json.dumps({
            "incident": incident_text[:4000],
            "so_far": context_summary[-4000:]
        }, ensure_ascii=False)

        tool_cmd_raw = gemini_generate(
            api_key=api_key,
            user_prompt=user_prompt,
            system_prompt=AGENT_SYSTEM,
            temperature=0.1,
            max_output_tokens=256
        )

        # Try to parse JSON dict
        tool_cmd: Dict[str, Any]
        try:
            tool_cmd = json.loads(tool_cmd_raw.strip())
        except Exception:
            # If parsing fails, nudge with a recovery prompt
            tool_cmd = {"action": "final"}

        if tool_cmd.get("action") != "tool":
            break

        name = tool_cmd.get("name")
        args = tool_cmd.get("args", {})
        observation: Dict[str, Any] = {"name": name, "args": args, "result": None}

        if name == "search_kb":
            q = str(args.get("query", "")).strip() or incident_text[:200]
            observation["result"] = tool_search_kb(q, rag)
        elif name == "grep_logs":
            p = str(args.get("pattern", "")).strip() or "error"
            observation["result"] = tool_grep_logs(p, logs_text)
        elif name == "propose_runbook":
            observation["result"] = tool_propose_runbook(incident_text, context_summary)
        else:
            observation["result"] = {"error": f"unknown tool {name}"}

        trace.append(observation)

        # Append a brief observation summary to context for next step
        obs_str = json.dumps(observation["result"], ensure_ascii=False)[:1200]
        context_summary += f"\n[OBS:{name}] {obs_str}"

    # Final report using all gathered observations
    final_prompt = f"""Incident:\n{incident_text[:4000]}

Collected Observations:\n{context_summary[-6000:]}

If evidence includes KB hits, you may quote short snippets. If logs were matched, include those snippets.
"""
    final_report = gemini_generate(
        api_key=api_key,
        user_prompt=final_prompt,
        system_prompt=FINAL_SYSTEM,
        temperature=0.2,
        max_output_tokens=1600
    )

    return trace, final_report

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Data Center Incident RCA (RAG + Agent)", layout="wide")
st.title("🧭 Intelligent Data Center RCA (RAG + Agentic AI)")

with st.sidebar:
    st.subheader("🔐 API & Model")
    st.caption(f"Model: `{MODEL_NAME}` (Gemini via REST)")
    manual_key = st.text_input("GEMINI_API_KEY (optional if set in secrets/env)", type="password")
    use_key = manual_key or get_api_key()
    temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.subheader("📚 Knowledge Base (RAG)")
    kb_text = st.text_area("Paste KB/Runbook text (optional)", height=120, placeholder="e.g., DC network failover SOP, storage throttling guidance, DB connection limits...")
    kb_pdfs = st.file_uploader("Upload KB PDFs (optional)", type=["pdf"], accept_multiple_files=True)

    st.subheader("🧾 Logs")
    logs_file = st.file_uploader("Upload log file (txt, optional)", type=["txt", "log"])
    logs_text = ""
    if logs_file:
        try:
            logs_text = logs_file.read().decode("utf-8", errors="ignore")
        except Exception:
            logs_text = logs_file.read().decode("latin-1", errors="ignore")
    default_incident = (
        "At 10:15 IST, multiple microservices in AZ-2 reported elevated 5xx rates and latency spikes. "
        "Ingress shows connection resets. DB connection pool saturation observed. A deployment occurred 5 minutes "
        "before the spike. Node autoscaler events show churn. Network errors include TCP retransmissions."
    )
    st.subheader("🧩 Incident")
    incident_text = st.text_area("Describe the incident", value=default_incident, height=140)

    build_btn = st.button("⚙️ Build / Refresh Vector Store")

# Build RAG store (cached by session)
if "rag" not in st.session_state:
    st.session_state["rag"] = None

rag_store: Optional[SimpleRAG] = st.session_state["rag"]

if build_btn:
    texts = []
    metas = []

    if kb_text.strip():
        for ch in chunk_text(kb_text, max_chars=1200):
            texts.append(ch)
            metas.append({"source": "KB:text"})

    if kb_pdfs:
        for f in kb_pdfs:
            try:
                reader = PdfReader(io.BytesIO(f.read()))
                content = ""
                for page in reader.pages:
                    try:
                        content += page.extract_text() or ""
                    except Exception:
                        continue
                for ch in chunk_text(content, max_chars=1200):
                    texts.append(ch)
                    metas.append({"source": f"PDF:{f.name}"})
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}")

    if texts:
        rag = SimpleRAG()
        rag.add_documents(texts, metas)
        with st.spinner("Building vector index..."):
            rag.build()
        st.session_state["rag"] = rag
        st.success(f"Vector store ready with {len(texts)} chunks.")
    else:
        st.session_state["rag"] = None
        st.info("No KB content provided. RAG disabled.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🛠️ Agentic RCA")
    if st.button("🔍 Analyze Incident", use_container_width=True):
        if not use_key:
            st.error("GEMINI_API_KEY is required.")
        else:
            with st.spinner("Reasoning over KB & logs..."):
                trace, report = run_agentic_rca(
                    api_key=use_key,
                    incident_text=incident_text,
                    rag=st.session_state.get("rag"),
                    logs_text=logs_text or ""
                )
            st.session_state["trace"] = trace
            st.session_state["report"] = report

    if "report" in st.session_state:
        st.subheader("📄 Final RCA & Solution")
        st.markdown(st.session_state["report"])

        # Download JSON package
        bundle = {
            "model": MODEL_NAME,
            "incident": incident_text,
            "report_markdown": st.session_state["report"],
            "trace": st.session_state.get("trace", []),
        }
        st.download_button(
            "⬇️ Download RCA JSON",
            data=json.dumps(bundle, indent=2),
            file_name="rca_report.json",
            mime="application/json",
            use_container_width=True
        )

with col2:
    st.subheader("🧪 Agent Trace (Tools)")
    trace = st.session_state.get("trace", [])
    if not trace:
        st.info("Click **Analyze Incident** to view the agent’s tool calls and observations.")
    else:
        for i, step in enumerate(trace, 1):
            with st.expander(f"Step {i}: {step['name']}"):
                st.code(json.dumps(step, indent=2)[:5000], language="json")

    st.subheader("🧾 Log Preview")
    if logs_text:
        st.text_area("Logs (first 5000 chars)", logs_text[:5000], height=200)
    else:
        st.caption("No logs uploaded.")

st.caption("Tip: load your runbooks/PDFs into the vector store for better grounded answers. The agent will combine KB hits, log snippets, and its reasoning to produce an RCA with remediation.")
