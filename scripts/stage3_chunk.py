"""
stage3_chunk.py
───────────────
Stage 3: Split cleaned Markdown into semantic chunks.

Uses LangChain's RecursiveCharacterTextSplitter with Markdown-aware separators.
Chunks are saved to data/chunks/ as JSON files (one per source document).

Each chunk record:
    {
        "chunk_id":   "docname_0042",
        "source":     "policy_doc.md",
        "text":       "...",
        "char_start": 4200,
        "char_end":   5700,
        "chunk_index": 42
    }

Usage
─────
    python scripts/stage3_chunk.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, resolve, setup_logger, timer

cfg    = load_config()
logger = setup_logger("stage3_chunk", cfg)


def chunk_document(text: str, source_name: str, chunk_cfg: dict) -> list[dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_cfg.get("chunk_size", 1500),
        chunk_overlap=chunk_cfg.get("chunk_overlap", 300),
        separators=chunk_cfg.get("separators", ["\n\n", "\n", ". ", " ", ""]),
        length_function=len,
        is_separator_regex=False,
    )

    raw_chunks = splitter.create_documents([text])
    stem       = Path(source_name).stem

    records: list[dict] = []
    cursor = 0
    for i, doc in enumerate(raw_chunks):
        content = doc.page_content
        start   = text.find(content, cursor)
        end     = start + len(content) if start != -1 else -1
        cursor  = end if end != -1 else cursor

        records.append({
            "chunk_id":    f"{stem}_{i:04d}",
            "source":      source_name,
            "text":        content,
            "char_start":  start,
            "char_end":    end,
            "chunk_index": i,
        })

    return records


@timer(logger)
def run():
    in_dir   = resolve(cfg["paths"]["cleaned"])
    out_dir  = resolve(cfg["paths"]["chunks"])
    chunk_cfg = cfg["chunking"]

    md_files = sorted(in_dir.glob("*.md"))
    if not md_files:
        logger.warning(f"No .md files in {in_dir}. Run Stage 2 first.")
        return

    total_chunks = 0
    for md_path in md_files:
        text   = md_path.read_text(encoding="utf-8")
        chunks = chunk_document(text, md_path.name, chunk_cfg)

        out_path = out_dir / (md_path.stem + "_chunks.json")
        out_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

        total_chunks += len(chunks)
        logger.info(f"  {md_path.name}: {len(chunks)} chunks → {out_path.name}")

    logger.info(f"Stage 3 complete — {total_chunks} total chunks from {len(md_files)} document(s).")


if __name__ == "__main__":
    run()
