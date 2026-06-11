# PolicyPipeline

A fully local, GPU-accelerated pipeline that ingests PDF policy documents (50–100+ pages) and produces a clean, structured `Policy_Only.md` containing only the policy-relevant content. No cloud APIs are used — PDF parsing, embeddings, and LLM inference all run on your GPU via CUDA.

**Hardware Target:** Windows 11 · RTX 5060 8 GB · 32 GB RAM

---

## Overview

The pipeline runs in 7 sequential, disk-checkpointed stages:

```
PDF(s) ──Stage 1 (GPU/Marker)──► Raw .md
       ──Stage 2 (CPU)─────────► Cleaned .md
       ──Stage 3 (CPU)─────────► JSON chunks
       ──Stage 4 (GPU)─────────► ChromaDB vectors
       ──Stage 5 (GPU/Ollama)──► candidate_policies.md
       ──Stage 6 (GPU/Ollama)──► validated_policies.md
       ──Stage 7 (CPU)─────────► Policy_Only.md ✅
```

| Stage | Script | Compute | Description |
|---|---|---|---|
| 1 | `stage1_pdf_parse.py` | GPU | PDF → Markdown via Marker (Surya layout + OCR) |
| 2 | `stage2_clean.py` | CPU | Strip headers/footers, page numbers, artifacts |
| 3 | `stage3_chunk.py` | CPU | Markdown-aware semantic chunking (LangChain) |
| 4 | `stage4_embed_store.py` | GPU | Embeddings (float16) → ChromaDB |
| 5 | `stage5_extract.py` | GPU | RAG-based policy extraction via Qwen 2.5 7B |
| 6 | `stage6_validate.py` | GPU | Validation/repair pass via Qwen 2.5 7B |
| 7 | `stage7_assemble.py` | CPU | Dedup, header, final assembly |

Every stage writes its output to disk before the next begins, providing crash recovery at any boundary.

---

## Component Stack

| Layer | Component | Version | GPU |
|---|---|---|---|
| Runtime | Python | 3.11 | — |
| PDF Parsing | Marker | ≥0.3.0 | ✅ |
| Embeddings | sentence-transformers | ≥3.0.0 | ✅ |
| Vector Store | ChromaDB | ≥0.5.0 | — |
| LLM | Qwen 2.5 7B (Ollama) | ≥0.2.0 | ✅ |
| Chunking | LangChain Text Splitters | ≥0.2.0 | — |
| Orchestration | run_pipeline.py | — | — |

---

## Setup

```powershell
cd D:\PolicyPipeline
venv\Scripts\activate
```

Run a GPU pre-flight check before first use (verifies driver, CUDA, Ollama, and `qwen2.5:7b`):

```powershell
python scripts/gpu_check.py
```

Place your input PDFs in `data/pdfs/`.

---

## Usage

### Process all PDFs
```powershell
python run_pipeline.py
```
Merges every PDF in `data/pdfs/` into one `Policy_Only.md`.

### Process specific PDFs
```powershell
python run_pipeline.py --pdfs document_one document_two
```
- Names are case-insensitive and exclude `.pdf`
- Unmatched names are skipped with a warning

### Process a single PDF
```powershell
python run_pipeline.py --pdfs klarna_governance
```

### Switch to a new document set
```powershell
python run_pipeline.py --pdfs doc1 doc2 --reset-chroma
```
> ⚠️ `--reset-chroma` deletes the entire ChromaDB collection and re-embeds everything from scratch.

### Resume after a failure
```powershell
python run_pipeline.py --from-stage N
```

### Run a single stage only
```powershell
python run_pipeline.py --only-stage N
```

### Dry run (preview only)
```powershell
python run_pipeline.py --dry-run
```

### Quick Reference

| Goal | Command |
|---|---|
| Process all PDFs | `python run_pipeline.py` |
| Process two specific PDFs | `python run_pipeline.py --pdfs doc1 doc2` |
| Process one PDF | `python run_pipeline.py --pdfs doc1` |
| Switch document set (fresh output) | `python run_pipeline.py --pdfs doc1 doc2 --reset-chroma` |
| Resume after crash at stage N | `python run_pipeline.py --from-stage N` |
| Re-run one stage only | `python run_pipeline.py --only-stage N` |
| Preview without running | `python run_pipeline.py --dry-run` |
| Check GPU before first run | `python scripts/gpu_check.py` |

---

## Configuration

All tunable parameters live in `config.yaml`.

| Parameter | Path | Default | Impact |
|---|---|---|---|
| `batch_multiplier` | `marker.batch_multiplier` | 4 | Stage 1 throughput (raise to 6–8 for faster parsing) |
| `embedding_batch_size` | `gpu.embedding_batch_size` | 128 | Stage 4 throughput (up to 256 on 8 GB VRAM) |
| `num_gpu_layers` | `llm.num_gpu_layers` | 35 | All Qwen 7B layers on GPU; lower if OOM |
| `context_length` | `llm.context_length` | 16000 | LLM context window (max safe ~24000) |
| `chunk_size` | `chunking.chunk_size` | 1500 | Larger = fewer, more context-rich chunks |
| `n_results` | `retrieval.n_results` | 20 | Chunks retrieved per query category |

---

## GPU VRAM Budget (RTX 5060 8 GB)

| Workload | Estimated VRAM | Notes |
|---|---|---|
| Marker (layout + OCR) | ~3.5–4.5 GB | Stage 1 only |
| SentenceTransformer (fp16) | ~0.08 GB | 384-dim model |
| Qwen 2.5 7B (fp16) | ~4.5–5.5 GB | Stages 5–6 |
| ChromaDB | CPU RAM only | — |
| CUDA overhead + OS | ~0.5 GB | Always reserved |

Stages run sequentially so no two large models compete for VRAM at once: Marker loads/unloads in Stage 1, and the embedding model is unloaded before Ollama inference begins in Stage 5.

---

## Testing

```powershell
# GPU diagnostics
python scripts/gpu_check.py

# Smoke tests (no PDF required)
pytest tests/test_pipeline.py -v -k "not integration"

# Full integration tests (synthetic PDFs)
pytest tests/test_pipeline.py -v -m integration
```

| Test | Pages | Verifies |
|---|---|---|
| `test_10_page_pdf` | 10 | End-to-end produces output |
| `test_50_page_pdf` | 50 | Tables preserved |
| `test_100_page_pdf` | 100 | No truncation, no crash |

---

## Directory Structure

```
PolicyPipeline/
├── config.yaml
├── requirements.txt
├── run_pipeline.py
├── scripts/
│   ├── utils.py
│   ├── gpu_check.py
│   ├── stage1_pdf_parse.py
│   ├── stage2_clean.py
│   ├── stage3_chunk.py
│   ├── stage4_embed_store.py
│   ├── stage5_extract.py
│   ├── stage6_validate.py
│   └── stage7_assemble.py
├── tests/
│   └── test_pipeline.py
├── data/
│   ├── pdfs/        ← place input PDFs here
│   ├── markdown/    ← Stage 1 output
│   ├── cleaned/      ← Stage 2 output
│   ├── chunks/       ← Stage 3 output
│   └── final/        ← Stages 5-7 output
├── chroma_db/        ← persistent vector store
├── logs/             ← logs + run_summary.json
└── models/           ← reserved for local model storage
```

---

## Future Upgrade Path

| Upgrade | Benefit | Prerequisites |
|---|---|---|
| Qwen 2.5 14B | Higher extraction quality | 16+ GB VRAM |
| `all-mpnet-base-v2` embeddings | Better retrieval quality | ~2× slower encode |
| PostgreSQL + pgvector | Production-scale vector store | PostgreSQL server |
| Batch processing queue | Parallel PDF processing | Multi-GPU or A100 |
| Human review dashboard | Manual approval before final output | FastAPI + React |
| Hybrid OCR | Better scanned document handling | EasyOCR / PaddleOCR fallback |
| Context length 32000 | Larger LLM batches | ≥10 GB VRAM available to Ollama |