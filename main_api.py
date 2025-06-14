import os
import json
import time
import zipfile
import struct
import ast

from annoy import AnnoyIndex
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
metas  = [d["metadata"]     for d in docs]

# ──────────────────────────────────────────────────────────────────────────────
# PURE-PYTHON .npz → Python list loader
# ──────────────────────────────────────────────────────────────────────────────
def load_npz_as_list(npz_path: Path):
    """
    Load the 'arr_0.npy' array from a .npz archive and return
    it as a list of lists of floats (dtype '<f4' only).
    """
    # 1) Extract the .npy bytes
    with zipfile.ZipFile(npz_path, "r") as zf:
        name = next(n for n in zf.namelist() if n.endswith("arr_0.npy"))
        data = zf.read(name)

    # 2) Parse header
    if not data.startswith(b'\x93NUMPY'):
        raise RuntimeError("Unexpected .npy format")
    header_len = struct.unpack('<H', data[8:10])[0]
    header_str = data[10:10+header_len].decode("latin1")
    header = ast.literal_eval(header_str)
    shape = header["shape"]           # e.g. (N, D)
    descr = header["descr"]           # e.g. '<f4'
    offset = 10 + header_len

    # 3) Unpack raw floats
    if descr != '<f4':
        raise RuntimeError(f"Unsupported dtype {descr}")
    count = shape[0] * (shape[1] if len(shape) > 1 else 1)
    fmt = "<" + "f" * count
    raw = data[offset:offset + 4*count]
    flat = struct.unpack(fmt, raw)

    # 4) Reshape
    if len(shape) == 2:
        rows, cols = shape
        return [ list(flat[i*cols:(i+1)*cols]) for i in range(rows) ]
    else:
        return list(flat)


# Load embeddings into a pure-Python list and build Annoy index
arr = load_npz_as_list(EMBED_FILE)
dim = len(arr[0])
index = AnnoyIndex(dim, metric="euclidean")
for i, vec in enumerate(arr):
    index.add_item(i, vec)
index.build(10)   # tweak tree count (10) for your speed/accuracy trade-off


# ──────────────────────────────────────────────────────────────────────────────
# JSON PARSER (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
import re
def parse_json_string(raw_string):
    try:
        cleaned = re.sub(r"^```(?:json)?\n?", "", raw_string.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\n?```$", "", cleaned.strip())
        return json.loads(cleaned)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# EMBEDDING & RETRIEVAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def embed_query(text: str):
    resp = genai.embed_content(model=EMBED_MODEL, content=[text])
    if "embeddings" in resp:
        return resp["embeddings"][0]
    if "data" in resp:
        return resp["data"][0]["embedding"]
    if "embedding" in resp:
        return resp["embedding"]
    raise KeyError(f"No embedding in response: {resp.keys()}")


def retrieve(question: str, k: int = TOP_K):
    q_vec = embed_query(question)
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
    context_block = "\n".join(
        f"[{i+1}] \"{h['chunk_text'][:197]}...\" (Source: {h.get('source_url','')})"
        for i, h in enumerate(hits)
    )

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
    gen = genai.GenerativeModel(CHAT_MODEL, system_instruction=system)
    try:
        resp = gen.generate_content(user)
    except ResourceExhausted:
        time.sleep(60)
        resp = gen.generate_content(user)
    return resp.text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# FASTAPI SETUP
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI()

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG API Online</title>
    </head>
    <body>
        <h1>✅ Your FastAPI RAG app is running on Vercel!</h1>
        <p>Try posting to <code>/api/</code> with a JSON question payload.</p>
        <p><strong>Example:</strong></p>
        <pre>{
  "question": "What model should I use?",
  "image": ""
}</pre>
    </body>
    </html>
    """

class Query(BaseModel):
    question: str
    image: Optional[str] = None  # ignored

@app.post("/api/")
def api_endpoint(q: Query):
    raw = generate_answer(q.question)
    parsed = parse_json_string(raw)
    if parsed is None:
        raise HTTPException(500, "Failed to parse LLM response as JSON")
    return parsed
