"""
stage5_extract.py
─────────────────
Stage 6 (Plan §23): Policy extraction using Qwen 2.5 via Ollama.

GPU acceleration
────────────────
- Ollama is started with num_gpu_layers = 35 (all layers of Qwen 2.5 7B on GPU).
- Context window is set to 16 000 tokens — safe for 8 GB VRAM.
- Each Ollama request passes options.num_gpu so it is enforced per call.

Retrieval strategy
──────────────────
For each query category in config.yaml:
    1. Query ChromaDB with the embedding model (GPU).
    2. Collect the top-N most relevant chunks.
Deduplicate across categories (by chunk_id).
Batch deduplicated chunks into context windows ≤ llm.context_length tokens.
Send each batch to Qwen with the extraction prompt.
Collect candidate policy sections.

Extraction prompt
─────────────────
Instructs the model to:
  - Preserve all Markdown (headings, tables, bullets)
  - Keep verbatim rule/policy language
  - Remove non-policy narrative prose
  - NOT summarise — return the content as-is or lightly reformatted

Usage
─────
    python scripts/stage5_extract.py
"""

from __future__ import annotations

import json
import sys
import hashlib
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, resolve, setup_logger, timer, vram_snapshot, PROJECT_ROOT

cfg    = load_config()
logger = setup_logger("stage5_extract", cfg)

# ── Prompt templates ──────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """You are a precise policy extraction assistant.
Your task is to identify and return ONLY the policy-relevant content from the provided document chunks.

RULES:
- Return content in clean Markdown format.
- Preserve ALL tables, bullet lists, numbered lists, and headings exactly as they appear.
- Preserve verbatim rule language, obligations, prohibitions, and requirements.
- Remove narrative prose, introductions, examples, and non-policy filler text.
- Do NOT summarise, paraphrase, or rewrite the policy content.
- Do NOT add commentary, explanations, or your own words.
- If a section contains no policy content, output nothing for that section.
- Output only the extracted policy Markdown. No preamble. No closing remarks.
"""

EXTRACTION_USER = """Below are document chunks. Extract all policy content.

--- CHUNKS START ---
{chunks}
--- CHUNKS END ---

Extracted policy content (Markdown only):"""


def build_ollama_options(llm_cfg: dict) -> dict:
    return {
        "num_gpu":       llm_cfg.get("num_gpu_layers", 35),
        "num_thread":    llm_cfg.get("num_thread", 8),
        "num_ctx":       llm_cfg.get("context_length", 16000),
        "temperature":   llm_cfg.get("temperature", 0.1),
        "top_p":         llm_cfg.get("top_p", 0.9),
    }


def retrieve_chunks(embed_model, collection, categories: list[str], n_results: int) -> list[dict]:
    """Query ChromaDB for each category; deduplicate by chunk_id."""
    from sentence_transformers import SentenceTransformer

    seen_ids: set[str] = set()
    all_chunks: list[dict] = []

    for category in tqdm(categories, desc="Querying ChromaDB", unit="cat"):
        q_embed = embed_model.encode(
            [category],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        results = collection.query(
            query_embeddings=q_embed,
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            cid = f"{meta['source']}::{meta['chunk_index']}"
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_chunks.append({
                    "id":       cid,
                    "text":     doc,
                    "source":   meta.get("source"),
                    "distance": dist,
                })

    logger.info(f"Retrieved {len(all_chunks)} unique chunks across {len(categories)} categories.")
    return all_chunks


def batch_chunks_by_tokens(chunks: list[dict], max_tokens: int, tokens_per_char: float = 0.30) -> list[list[dict]]:
    """
    Split chunks into batches that fit within max_tokens context.
    Approximation: 1 token ≈ 3.3 chars  →  chars_per_token = 1/0.30 ≈ 3.3
    """
    max_chars = int(max_tokens / tokens_per_char)
    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_chars = 0

    for chunk in chunks:
        n = len(chunk["text"])
        if current_chars + n > max_chars and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
        current_batch.append(chunk)
        current_chars += n

    if current_batch:
        batches.append(current_batch)

    return batches


def call_qwen(prompt_text: str, llm_cfg: dict) -> str:
    import ollama as ol_client

    response = ol_client.chat(
        model=llm_cfg.get("model", "qwen2.5:7b"),
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user",   "content": prompt_text},
        ],
        options=build_ollama_options(llm_cfg),
    )
    return response["message"]["content"]


@timer(logger)
def run():
    import chromadb
    from sentence_transformers import SentenceTransformer

    llm_cfg      = cfg["llm"]
    embed_cfg    = cfg["embeddings"]
    retrieval_cfg = cfg["retrieval"]
    chroma_cfg   = cfg["chroma"]
    chroma_path  = PROJECT_ROOT / cfg["paths"]["chroma_db"]

    # ── Load models ────────────────────────────────────────────────────────────
    device = embed_cfg.get("device", "cuda")
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA unavailable — using CPU for embeddings.")
        device = "cpu"

    logger.info(f"Loading embedding model on {device.upper()}…")
    embed_model = SentenceTransformer(embed_cfg["model"], device=device)
    if device == "cuda":
        embed_model = embed_model.half()

    client     = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(chroma_cfg["collection_name"])
    logger.info(f"ChromaDB collection: {collection.count()} chunks.")

    # ── Retrieve ───────────────────────────────────────────────────────────────
    vram_snapshot(logger, "before retrieval")
    chunks = retrieve_chunks(
        embed_model,
        collection,
        retrieval_cfg["query_categories"],
        retrieval_cfg.get("n_results", 20),
    )
    vram_snapshot(logger, "after retrieval")

    # Free embedding model VRAM before LLM inference
    del embed_model
    torch.cuda.empty_cache()
    vram_snapshot(logger, "after embed model unload")

    # ── Batch and extract ──────────────────────────────────────────────────────
    # Reserve ~30% of context for system prompt + response
    usable_ctx = int(llm_cfg.get("context_length", 16000) * 0.70)
    batches    = batch_chunks_by_tokens(chunks, usable_ctx)
    logger.info(f"Extraction: {len(batches)} batch(es) → Qwen")

    extracted_sections: list[str] = []

    for i, batch in enumerate(tqdm(batches, desc="Extracting policy", unit="batch")):
        chunk_text = "\n\n---\n\n".join(c["text"] for c in batch)
        prompt     = EXTRACTION_USER.format(chunks=chunk_text)

        logger.debug(f"  Batch {i+1}/{len(batches)} — {len(batch)} chunks, "
                     f"~{len(chunk_text)//4} tokens")

        try:
            result = call_qwen(prompt, llm_cfg)
            if result.strip():
                extracted_sections.append(result.strip())
        except Exception as e:
            logger.error(f"  LLM call failed on batch {i+1}: {e}")

    # ── Save candidate output ──────────────────────────────────────────────────
    candidate_dir = resolve(cfg["paths"]["final"])
    candidate_path = candidate_dir / "candidate_policies.md"
    combined = "\n\n---\n\n".join(extracted_sections)
    candidate_path.write_text(combined, encoding="utf-8")
    logger.info(f"Stage 5 complete — candidate output: {candidate_path}  "
                f"({len(combined):,} chars)")


if __name__ == "__main__":
    run()
