import os
import json
import time
import re
import numpy as np
from annoy import AnnoyIndex
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & INITIAL LOAD
# ──────────────────────────────────────────────────────────────────────────────

HERE = Path(__file__).parent
NORMAL_FILE = HERE / "normalized_docs2.json"
EMBED_FILE = HERE / "embeddings.npz"

EMBED_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-1.5-flash"
TOP_K = 5

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Set the GOOGLE_API_KEY environment variable")
genai.configure(api_key=API_KEY)

# Load normalized docs + metadata
docs = json.loads((HERE / NORMAL_FILE.name).read_text(encoding="utf-8"))
texts = [d["page_content"] for d in docs]
metas = [d["metadata"] for d in docs]

# Build Annoy index
arr = np.load(EMBED_FILE)["arr_0"].astype("float32")
dim = arr.shape[1]
index = AnnoyIndex(dim, metric="euclidean")
for i, vec in enumerate(arr):
    index.add_item(i, vec.tolist())
index.build(10)

# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def parse_json_string(raw_string: str) -> Optional[dict]:
    # Strip markdown code blocks and parse
    cleaned = re.sub(r"^```(?:json)?\n?", "", raw_string.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# EMBEDDING & RETRIEVAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def embed_query(text: str) -> np.ndarray:
    resp = genai.embed_content(model=EMBED_MODEL, content=[text])
    if "embeddings" in resp:
        vec = resp["embeddings"][0]
    elif "data" in resp:
        vec = resp["data"][0]["embedding"]
    elif "embedding" in resp:
        vec = resp["embedding"]
    else:
        raise KeyError(f"No embedding found in response: {resp.keys()}")
    return np.array(vec, dtype="float32").reshape(1, -1)

def retrieve(question: str, k: int = TOP_K):
    # Annoy returns (idxs, dists)
    q_vec = embed_query(question)[0].tolist()
    idxs, dists = index.get_nns_by_vector(q_vec, k, include_distances=True)
    hits = []
    for idx, dist in zip(idxs, dists):
        hits.append({
            "chunk_text": texts[idx],
            **metas[idx],
            "score": float(dist)
        })
    return hits

# ──────────────────────────────────────────────────────────────────────────────
# RAG PROMPT + GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def generate_answer(question: str) -> str:
    hits = retrieve(question)
    # Build context block
    lines = []
    for i, h in enumerate(hits, 1):
        url = h.get("source_url", "") or h.get("url", "")
        snippet = h["chunk_text"].replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."
        lines.append(f"[{i}] \"{snippet}\" (Source: {url})")
    context_block = "\n".join(lines)

    system = (
        "You are an amazing professor of applications of data science tools with experience of 20+ years. You are replying to students on the Discourse forum to solve their problems using the context. "
        "You have also been provided with context containing FAQs and course content to use as reference. "
        "add all the links provided in context as it is return all the links you can make the content short but mention all links and you can leave text blank"
        "You must respond **only** with a valid JSON object, NO EXTRA TEXT — only JSON format, nothing extra (not even ```json or any prefix/suffix).\n"
        "Schema:\n"
        "{\n"
        '  "answer": string,\n'
        '  "links": [\n'
        '    {\n'
        '      "url": "https://........",\n'
        '      "text": "description for link."\n'
        '    },\n'
        '    {\n'
        '      "url": "https://discourse......." ,\n'
        '      "text": "description for link"\n'
        '    }\n'
        '  ]\n'
        "}"
    )
    user = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Always output exactly one JSON object following the schema above."
        
    )

    gen_model = genai.GenerativeModel(CHAT_MODEL, system_instruction=system)
    try:
        response = gen_model.generate_content(user)
    except ResourceExhausted:
        time.sleep(60)
        response = gen_model.generate_content(user)
    return response.text.strip()

# ──────────────────────────────────────────────────────────────────────────────
# FASTAPI SETUP
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "It works!"}

class Query(BaseModel):
    question: str
    image: Optional[str] = None  # ignored

@app.post("/api/")
def api_endpoint(q: Query):
    raw = generate_answer(q.question)
    parsed = parse_json_string(raw)
    if parsed is None:
        raise HTTPException(status_code=500, detail="Failed to parse model output as JSON")
    return parsed
