"""
stage4_embed_store.py
─────────────────────
Stage 4 & 5: GPU-accelerated embedding + ChromaDB ingestion.

GPU acceleration
────────────────
- SentenceTransformer is loaded directly onto CUDA.
- encode() runs in configurable GPU batches (default 128).
- torch.cuda.empty_cache() is called after each document to prevent VRAM creep.
- Embeddings are computed in float16 (half precision) to halve VRAM usage
  without meaningful accuracy loss for retrieval.

ChromaDB
────────
- Uses a persistent local client (no server required).
- Checks for existing chunk IDs and skips re-embedding already-stored chunks
  (idempotent — safe to re-run after partial failures).

Usage
─────
    python scripts/stage4_embed_store.py
    python scripts/stage4_embed_store.py --reset   # drop and rebuild collection
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, resolve, setup_logger, timer, vram_snapshot

cfg    = load_config()
logger = setup_logger("stage4_embed_store", cfg)


def load_embedder(embed_cfg: dict):
    """Load SentenceTransformer onto GPU with float16 precision."""
    from sentence_transformers import SentenceTransformer

    device     = embed_cfg.get("device", "cuda")
    model_name = embed_cfg.get("model", "all-MiniLM-L6-v2")

    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA not available — falling back to CPU.")
        device = "cpu"

    logger.info(f"Loading embedding model '{model_name}' on {device.upper()}…")
    model = SentenceTransformer(model_name, device=device)

    # Cast model weights to float16 to halve VRAM usage
    if device == "cuda":
        model = model.half()

    logger.info(f"Embedding model ready on {model.device}.")
    return model


def get_or_create_collection(chroma_cfg: dict, chroma_path: Path, reset: bool):
    """Open (or recreate) the ChromaDB persistent collection."""
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_path))

    name = chroma_cfg.get("collection_name", "policy_chunks")

    if reset:
        try:
            client.delete_collection(name)
            logger.info(f"Deleted existing collection '{name}'.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": chroma_cfg.get("distance_metric", "cosine")},
    )
    logger.info(f"ChromaDB collection '{name}' — {collection.count()} existing chunk(s).")
    return collection


@timer(logger)
def run(reset: bool = False):
    chunks_dir  = resolve(cfg["paths"]["chunks"])
    chroma_path = PROJECT_ROOT / cfg["paths"]["chroma_db"]
    chroma_path.mkdir(parents=True, exist_ok=True)

    embed_cfg  = cfg["embeddings"]
    chroma_cfg = cfg["chroma"]
    batch_size = embed_cfg.get("embedding_batch_size", 128)

    chunk_files = sorted(chunks_dir.glob("*_chunks.json"))
    if not chunk_files:
        logger.warning(f"No chunk files in {chunks_dir}. Run Stage 3 first.")
        return

    model      = load_embedder(embed_cfg)
    collection = get_or_create_collection(chroma_cfg, chroma_path, reset)

    # Fetch IDs already in the collection to skip re-embedding
    existing_ids: set[str] = set()
    try:
        existing_ids = set(collection.get(include=[])["ids"])
        logger.info(f"Skipping {len(existing_ids)} already-embedded chunk(s).")
    except Exception:
        pass

    total_added = 0

    for chunk_file in chunk_files:
        chunks = json.loads(chunk_file.read_text(encoding="utf-8"))

        # Filter out already-stored chunks
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
        if not new_chunks:
            logger.info(f"  {chunk_file.name}: all chunks already stored. Skipping.")
            continue

        texts  = [c["text"] for c in new_chunks]
        ids    = [c["chunk_id"] for c in new_chunks]
        metas  = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in new_chunks]

        logger.info(f"  Embedding {len(texts)} chunk(s) from {chunk_file.name}…")
        vram_snapshot(logger, f"before {chunk_file.stem}")

        # GPU batched encoding
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Required for cosine similarity in ChromaDB
        ).tolist()

        vram_snapshot(logger, f"after {chunk_file.stem}")

        # Ingest into ChromaDB in sub-batches (ChromaDB has a 5461-item upsert limit)
        CHROMA_BATCH = 500
        for i in range(0, len(ids), CHROMA_BATCH):
            collection.add(
                ids=ids[i:i+CHROMA_BATCH],
                documents=texts[i:i+CHROMA_BATCH],
                embeddings=embeddings[i:i+CHROMA_BATCH],
                metadatas=metas[i:i+CHROMA_BATCH],
            )

        total_added += len(ids)
        torch.cuda.empty_cache()

    logger.info(f"Stage 4 complete — {total_added} new chunk(s) added. "
                f"Collection total: {collection.count()}")


# ── Import guard ──────────────────────────────────────────────────────────────
from utils import PROJECT_ROOT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Embed + store in ChromaDB")
    parser.add_argument("--reset", action="store_true",
                        help="Drop and rebuild the ChromaDB collection from scratch")
    args = parser.parse_args()
    run(args.reset)
