#!/usr/bin/env python3
"""
pdf_extractor_with_json_update.py

Same functionality as the previous pdf_extractor, but adds Option A:
- After extracting & chunking PDFs, update existing JSON files (same base name)
  by adding/updating a top-level key "extracted_passages" containing:
    [{ "doc":..., "page":..., "start_char":..., "end_char":..., "text":..., "embedding": [...]? }, ...]
- Backs up any JSON before modifying to filename.json.bak.TIMESTAMP

Usage examples:
    # Index PDFs and update JSONs (no embeddings saved in JSON)
    python pdf_extractor_with_json_update.py --index --update_json

    # Index PDFs, update JSONs and also store embeddings in JSON (may be large)
    OPENAI_API_KEY=... python pdf_extractor_with_json_update.py --index --update_json --save_embeddings_in_json --use_openai

    # Query (unchanged)
    python pdf_extractor_with_json_update.py --query "What is the punishment for murder?" --top_k 5
"""

import os
import re
import json
import time
import argparse
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import fitz  # PyMuPDF
import numpy as np
from itertools import islice

# Embedding libs (optional)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

import faiss

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

# ---------- CONFIG ----------
DEFAULT_FILES = [
    "/mnt/data/bns.pdf",
    "/mnt/data/bnss.pdf",
    "/mnt/data/bsa.pdf",
]
DB_DIR = Path("./pdf_index_db")
VECTOR_INDEX_PATH = DB_DIR / "faiss.index"
SQLITE_PATH = DB_DIR / "metadata.sqlite"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBED_DIM = 384
DB_DIR.mkdir(parents=True, exist_ok=True)
# ----------------------------

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def extract_pdf_text_with_structure(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    results = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        raw = page.get_text("blocks")
        raw_sorted = sorted(raw, key=lambda b: (b[1], b[0]))
        text_lines = []
        headings = []
        for block in raw_sorted:
            block_text = block[4].strip()
            if not block_text:
                continue
            for line in block_text.splitlines():
                line_s = line.strip()
                if not line_s:
                    continue
                if re.match(r'^(CHAPTER|CLAUSES|CLAUSE|SECTION)\b', line_s, re.I):
                    headings.append(line_s)
                elif len(line_s) > 5 and line_s == line_s.upper() and re.search(r'[A-Z]', line_s):
                    headings.append(line_s)
                text_lines.append(line_s)
        full_text = normalize_whitespace("\n".join(text_lines))
        results.append({
            "doc": os.path.basename(pdf_path),
            "page": i + 1,
            "text": full_text,
            "headings": headings,
        })
    doc.close()
    return results

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        yield chunk, start, min(end, text_len)
        start = max(start + chunk_size - overlap, end)

class EmbeddingProvider:
    def __init__(self, use_openai: bool = False, openai_model: str = "text-embedding-3-small", hf_model: str = EMBEDDING_MODEL_NAME):
        self.use_openai = use_openai and (OPENAI_KEY is not None)
        self.openai_model = openai_model
        self.hf_model = hf_model
        self._init_model()

    def _init_model(self):
        if self.use_openai:
            import openai
            openai.api_key = OPENAI_KEY
            self.model = openai
        else:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not installed. Install it or set OPENAI_API_KEY and use --use_openai.")
            self.model = SentenceTransformer(self.hf_model)

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.use_openai:
            import openai
            embeds = []
            for t in texts:
                resp = openai.Embedding.create(input=t, model=self.openai_model)
                v = resp["data"][0]["embedding"]
                embeds.append(np.array(v, dtype=np.float32))
            return np.vstack(embeds)
        else:
            embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return embs.astype(np.float32)

class RetrieverIndex:
    def __init__(self, dim: int = EMBED_DIM, index_path: Path = VECTOR_INDEX_PATH, meta_db: Path = SQLITE_PATH):
        self.dim = dim
        self.index_path = index_path
        self.meta_db = meta_db
        self._connect_sqlite()
        self._init_faiss()

    def _connect_sqlite(self):
        self.conn = sqlite3.connect(str(self.meta_db))
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS passages (
                id INTEGER PRIMARY KEY,
                doc TEXT,
                page INTEGER,
                start_char INTEGER,
                end_char INTEGER,
                text TEXT
            )
        """)
        self.conn.commit()

    def _init_faiss(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexFlatIP(self.dim)

    def save(self):
        faiss.write_index(self.index, str(self.index_path))
        self.conn.commit()

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Dict]):
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        cur = self.conn.cursor()
        for md in metadatas:
            cur.execute(
                "INSERT INTO passages (doc, page, start_char, end_char, text) VALUES (?, ?, ?, ?, ?)",
                (md["doc"], md["page"], md["start_char"], md["end_char"], md["text"])
            )
        self.conn.commit()

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict]]:
        if query_emb.ndim == 1:
            query_emb = query_emb[np.newaxis, :]
        faiss.normalize_L2(query_emb)
        D, I = self.index.search(query_emb.astype(np.float32), top_k)
        results = []
        cur = self.conn.cursor()
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            cur.execute("SELECT id, doc, page, start_char, end_char, text FROM passages WHERE id = ?", (idx + 1,))
            row = cur.fetchone()
            if row:
                pid, doc, page, start_char, end_char, text = row
                md = {
                    "id": pid,
                    "doc": doc,
                    "page": page,
                    "start_char": start_char,
                    "end_char": end_char,
                    "text": text
                }
                results.append((float(score), md))
        return results

def backup_file(path: Path):
    if not path.exists():
        return
    ts = int(time.time())
    bak = path.with_suffix(path.suffix + f".bak.{ts}")
    path.replace(bak)  # atomic move
    # restore original name to new backup file name by copying back a fresh original if needed
    # We'll write a fresh file later when updating, so leave backup here.
    print(f"[BACKUP] {path} -> {bak}")

def write_json_backup_and_restore(orig_path: Path, new_data: dict):
    """
    Backup original JSON to .bak.TIMESTAMP and write new_data to original path.
    """
    if orig_path.exists():
        ts = int(time.time())
        bak_path = orig_path.with_suffix(orig_path.suffix + f".bak.{ts}")
        orig_path.replace(bak_path)
        print(f"[BACKUP] {orig_path} -> {bak_path}")
    # write new
    with open(orig_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"[WRITE] Updated JSON written to {orig_path}")

def update_json_files_for_passages(passages: List[Dict], save_embeddings_in_json: bool = False, embeddings: Optional[np.ndarray] = None):
    """
    passages: list of dicts: { doc, page, start_char, end_char, text }
    embeddings: optional numpy array (n, dim) matching passages order
    For each unique doc, find a JSON file at /mnt/data/<doc_basename_no_ext>.json.
    If found, backup it, then insert/update key "extracted_passages" with the list of passages
    that belong to that doc. If not found, create a new JSON file with basic structure.
    """
    # group by doc basename (e.g., 'bns.pdf' -> 'bns')
    grouped = {}
    for idx, p in enumerate(passages):
        doc = p["doc"]
        base = Path(doc).stem  # e.g., bns
        grouped.setdefault(base, []).append((idx, p))

    for base, items in grouped.items():
        json_path = Path("/mnt/data") / f"{base}.json"
        # load existing or create new
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = {}
            # backup original
            ts = int(time.time())
            bak_path = json_path.with_suffix(json_path.suffix + f".bak.{ts}")
            json_path.replace(bak_path)
            print(f"[BACKUP] {json_path} -> {bak_path}")
        else:
            data = {}
        # prepare extracted_passages list
        extracted_list = data.get("extracted_passages", [])
        # Append new passages (do not deduplicate; you can enhance to dedupe by start_char)
        for idx, p in items:
            entry = {
                "page": p["page"],
                "start_char": p["start_char"],
                "end_char": p["end_char"],
                "text": p["text"]
            }
            if save_embeddings_in_json:
                if embeddings is None:
                    entry["embedding"] = None
                else:
                    # convert vector to python list (float)
                    entry["embedding"] = embeddings[idx].astype(float).tolist()
            extracted_list.append(entry)
        data["extracted_passages"] = extracted_list
        # write updated JSON to original path (overwrite)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[UPDATE] {json_path} updated with {len(items)} passages (total now {len(extracted_list)})")

def index_pdfs(pdf_paths: List[str], embed_provider: EmbeddingProvider, reindex: bool = False,
               update_json: bool = False, save_embeddings_in_json: bool = False):
    retriever = RetrieverIndex(dim=EMBED_DIM)
    if reindex and retriever.index_path.exists():
        retriever.index_path.unlink()
        retriever.conn.close()
        retriever.meta_db.unlink(missing_ok=True)
        retriever = RetrieverIndex(dim=EMBED_DIM)

    all_chunks = []
    all_meta = []
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"[WARN] file not found, skipping: {pdf_path}")
            continue
        pages = extract_pdf_text_with_structure(pdf_path)
        for p in pages:
            text = p["text"]
            if not text:
                continue
            for chunk, s, e in chunk_text(text):
                chunk_norm = normalize_whitespace(chunk)
                if len(chunk_norm) < 40:
                    continue
                all_chunks.append(chunk_norm)
                all_meta.append({
                    "doc": p["doc"],
                    "page": p["page"],
                    "start_char": s,
                    "end_char": e,
                    "text": chunk_norm
                })

    # Optionally compute embeddings in batches and index to FAISS + store metadata in SQLite
    BATCH = 64
    all_embeddings = []
    for i in range(0, len(all_chunks), BATCH):
        batch_texts = all_chunks[i:i+BATCH]
        batch_emb = embed_provider.embed(batch_texts)
        all_embeddings.append(batch_emb)
        retriever.add_embeddings(batch_emb, all_meta[i:i+BATCH])
        print(f"[INDEXED] added batch {i}-{i+len(batch_texts)}")
    if all_embeddings:
        all_embeddings_arr = np.vstack(all_embeddings)
    else:
        all_embeddings_arr = None

    retriever.save()
    print("[DONE] Indexing complete. FAISS index saved to", retriever.index_path, "SQLite at", retriever.meta_db)

    # Update JSON files with extracted passages if requested
    if update_json:
        # if user requested saving embeddings in JSON but we didn't compute embeddings (shouldn't happen),
        # warn and skip embeddings save.
        if save_embeddings_in_json and all_embeddings_arr is None:
            print("[WARN] save_embeddings_in_json requested but no embeddings computed; continuing without embeddings.")
        update_json_files_for_passages(all_meta, save_embeddings_in_json, all_embeddings_arr)

def query_index(question: str, embed_provider: EmbeddingProvider, top_k: int = 5):
    retriever = RetrieverIndex(dim=EMBED_DIM)
    q_emb = embed_provider.embed([question])[0]
    results = retriever.search(q_emb, top_k=top_k)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="Index default files")
    parser.add_argument("--files", nargs="+", help="List of pdf files to index")
    parser.add_argument("--query", type=str, help="Run query against index")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--use_openai", action="store_true", help="Use OpenAI embeddings (requires OPENAI_API_KEY)")
    parser.add_argument("--reindex", action="store_true", help="Force reindex (delete existing DB)")
    parser.add_argument("--update_json", action="store_true", help="Update corresponding JSON files with extracted passages")
    parser.add_argument("--save_embeddings_in_json", action="store_true", help="Store numeric embeddings in JSON (can be very large)")
    args = parser.parse_args()

    use_openai = bool(args.use_openai and OPENAI_KEY)
    embedder = EmbeddingProvider(use_openai=use_openai)

    if args.index:
        files = args.files if args.files else DEFAULT_FILES
        print("[INFO] Indexing files:", files)
        index_pdfs(files, embedder, reindex=args.reindex,
                   update_json=args.update_json, save_embeddings_in_json=args.save_embeddings_in_json)
    elif args.query:
        print("[INFO] Querying index for:", args.query)
        results = query_index(args.query, embedder, top_k=args.top_k)
        print("Top results:")
        for score, md in results:
            print(f"--- score={score:.4f} | doc={md['doc']} | page={md['page']} ---")
            print(md['text'][:800].replace("\n", " "))
            print()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
