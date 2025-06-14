import os
import json
import time
import numpy as np
import faiss
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & INITIAL LOAD
# ──────────────────────────────────────────────────────────────────────────────

HERE        = Path(__file__).parent
NORMAL_FILE = HERE / "normalized_docs2.json"
EMBED_FILE  = HERE / "embeddings.npz"
META_FILE   = HERE / "metadata.json"

EMBED_MODEL = "models/embedding-001"
CHAT_MODEL  = "gemini-1.5-flash"
TOP_K       = 5

API_KEY     = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise RuntimeError("Set the GOOGLE_API_KEY environment variable")
genai.configure(api_key=API_KEY)

# Load normalized docs + metadata
with open(NORMAL_FILE, "r", encoding="utf-8") as f:
    docs = json.load(f)
texts = [d["page_content"] for d in docs]
metas = [d["metadata"]     for d in docs]

# Load embeddings & build FAISS
arr   = np.load(EMBED_FILE)["arr_0"].astype("float32")
dim   = arr.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(arr)


import json
import re

def parse_json_string(raw_string):
    """
    Attempts to extract and parse a JSON object from a string.
    Handles cases with markdown-style backticks, bad formatting, or stray characters.
    """

    try:
        # 1. Strip markdown code block wrappers like ```json ... ```
        cleaned = re.sub(r"^```(?:json)?\n?", "", raw_string.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\n?```$", "", cleaned.strip())

        # 2. Remove trailing/leading whitespace
        cleaned = cleaned.strip()

        # 3. Try to parse the cleaned string
        parsed = json.loads(cleaned)
        return parsed

    except json.JSONDecodeError as e:
        print("⚠️ Failed to parse JSON string:")
        print("Error:", e)
        print("Raw input:", raw_string)
        return None

    except Exception as e:
        print("❌ Unexpected error while parsing JSON:")
        print("Error:", e)
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
    q_vec = embed_query(question)
    dists, idxs = index.search(q_vec, k)
    hits = []
    for dist, idx in zip(dists[0], idxs[0]):
        hit = {
            "chunk_text": texts[idx],
            **metas[idx],
            "score": float(dist)
        }
        hits.append(hit)
    return hits

# ──────────────────────────────────────────────────────────────────────────────
# RAG PROMPT + GENERATION
# ──────────────────────────────────────────────────────────────────────────────
def generate_answer(question: str) -> dict:
    hits = retrieve(question)

    # Build context block
    lines = []
    for i, h in enumerate(hits, 1):
        url = h.get("source_url") or h.get("url", "")
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

    gen_model = genai.GenerativeModel( CHAT_MODEL,system_instruction=(system))
    
    try:
        response = gen_model.generate_content(user)
    except ResourceExhausted:
        time.sleep(60)
        response = gen_model.generate_content(user)

    text = response.text.strip()
    return (text)



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
    result = generate_answer(q.question)
    x=parse_json_string(result)
    return x
