# PolicyPipeline — Architecture Report

**Version:** 1.0  
**Hardware Target:** Windows 11 · RTX 5060 8 GB · 32 GB RAM  
**Date:** June 2026

---

## 1. Executive Summary

PolicyPipeline is a fully local, GPU-accelerated pipeline that ingests PDF policy documents of 50–100+ pages and produces a clean, structured `Policy_Only.md` file containing only the policy-relevant content. No cloud APIs are used. Every compute-intensive step — PDF layout detection, OCR, embedding generation, and LLM inference — is pushed to the RTX 5060 GPU via CUDA.

The pipeline is divided into 7 sequential stages, each writing its output to disk before the next stage begins. This design provides crash recovery at any stage boundary.

---

## 2. High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         POLICY PIPELINE                             │
│                                                                     │
│  data/pdfs/           data/markdown/     data/cleaned/              │
│  ┌──────────┐  GPU    ┌────────────┐     ┌──────────────┐           │
│  │  PDF(s)  │──────►  │  Raw .md   │────►│  Cleaned .md │           │
│  └──────────┘ Marker  └────────────┘     └──────┬───────┘           │
│   Stage 1             Stage 2 (CPU)             │                   │
│                                          Stage 3 │ LangChain         │
│                                                  ▼                   │
│                                         data/chunks/                 │
│                                         ┌───────────────┐           │
│                                         │  JSON chunks  │           │
│                                         └───────┬───────┘           │
│                                                 │                   │
│                                          Stage 4│ GPU (Sentence-    │
│                                                 │ Transformers)     │
│                                                 ▼                   │
│                                         chroma_db/                  │
│                                         ┌───────────────┐           │
│                                         │  ChromaDB     │           │
│                                         │  (vectors)    │           │
│                                         └───────┬───────┘           │
│                                                 │                   │
│                                          Stage 5│ GPU (Ollama)      │
│                                                 ▼                   │
│                                   data/final/candidate_policies.md  │
│                                                 │                   │
│                                          Stage 6│ GPU (Ollama)      │
│                                                 ▼                   │
│                                   data/final/validated_policies.md  │
│                                                 │                   │
│                                          Stage 7│ CPU               │
│                                                 ▼                   │
│                                   data/final/Policy_Only.md ✅       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Stack

| Layer | Component | Version | GPU Used? | Notes |
|---|---|---|---|---|
| Runtime | Python | 3.11 | — | Virtual environment in `venv/` |
| PDF Parsing | Marker | ≥0.3.0 | ✅ Yes | Surya layout + OCR models on CUDA |
| Embeddings | sentence-transformers | ≥3.0.0 | ✅ Yes | float16, batch_size=128 |
| Vector Store | ChromaDB | ≥0.5.0 | — | Local persistent, cosine distance |
| LLM | Qwen 2.5 7B via Ollama | ≥0.2.0 | ✅ Yes | 35 layers offloaded, ctx=16 000 |
| Chunking | LangChain Text Splitters | ≥0.2.0 | — | Recursive, Markdown-aware |
| Orchestration | run_pipeline.py | — | — | Subprocess-based, resumable |
| Logging | colorlog + logging | — | — | Console + file |
| Config | config.yaml | — | — | Single source of truth |

---

## 4. Stage-by-Stage Architecture

### Stage 1 — PDF Parsing (GPU)

**Script:** `scripts/stage1_pdf_parse.py`  
**Input:** `data/pdfs/*.pdf`  
**Output:** `data/markdown/*.md` + `*_meta.json`

Marker is used to convert PDFs to Markdown. Marker runs two neural models internally: a **layout detection model** (Surya) that identifies headings, paragraphs, tables, lists, and figures, and an **OCR model** for scanned pages. Both models are loaded onto the CUDA device by setting `TORCH_DEVICE=cuda` before any import. A configurable `batch_multiplier` (default 4) increases the number of pages processed per GPU batch, significantly reducing processing time on an RTX 5060.

Models are loaded once and reused across all PDFs in the session to avoid repeated GPU allocation overhead.

**GPU levers:**
- `TORCH_DEVICE=cuda` environment variable forces all Marker internals to CUDA.
- `batch_multiplier: 4` in config — scales pages-per-GPU-batch; increase to 6–8 if VRAM permits.

---

### Stage 2 — Markdown Cleaning (CPU)

**Script:** `scripts/stage2_clean.py`  
**Input:** `data/markdown/*.md`  
**Output:** `data/cleaned/*.md`

Pure CPU regex pass. Removes page numbers, running headers/footers (detected by frequency ≥ 5 occurrences), Marker artifact tokens, and collapses excessive blank lines. Tables, headings, bullets, and code blocks are fully preserved. This stage is fast (milliseconds per document) and does not benefit from GPU.

---

### Stage 3 — Semantic Chunking (CPU)

**Script:** `scripts/stage3_chunk.py`  
**Input:** `data/cleaned/*.md`  
**Output:** `data/chunks/*_chunks.json`

Uses LangChain's `RecursiveCharacterTextSplitter` with Markdown-aware separators. Chunk size is 1 500 characters with 300-character overlap, which is empirically good for policy documents: large enough to contain a complete rule clause, small enough to be specific when queried. Each chunk is stored as a JSON record with source, index, and character range metadata.

---

### Stage 4 — GPU Embedding + ChromaDB Ingestion (GPU)

**Script:** `scripts/stage4_embed_store.py`  
**Input:** `data/chunks/*_chunks.json`  
**Output:** `chroma_db/` (persistent)

This stage is the primary GPU compute bottleneck outside of LLM inference. `all-MiniLM-L6-v2` is loaded directly onto CUDA and the model weights are cast to **float16** (half precision), halving VRAM consumption with negligible quality loss for retrieval tasks.

Embeddings are generated in batches of 128 (configurable). ChromaDB uses cosine similarity and stores the normalized float32 embeddings, document text, and source metadata. The stage is **idempotent**: it checks for existing chunk IDs and skips re-embedding already-stored chunks, making it safe to restart after a partial failure.

**GPU levers:**
- `device: cuda` in config forces SentenceTransformer to CUDA.
- `embedding_batch_size: 128` — tune this based on VRAM; 256 is feasible on 8 GB for this model.
- `model.half()` casts to float16 before any encode call.
- `torch.cuda.empty_cache()` is called after each document to reclaim VRAM.

---

### Stage 5 — Policy Extraction via LLM (GPU)

**Script:** `scripts/stage5_extract.py`  
**Input:** ChromaDB collection  
**Output:** `data/final/candidate_policies.md`

This stage performs retrieval-augmented generation (RAG) for policy extraction:

1. The embedding model (GPU) encodes each of the 6 query categories.
2. ChromaDB returns the top-20 most semantically relevant chunks per category.
3. Chunks are deduplicated across categories.
4. Deduplicated chunks are batched to fit within the LLM's safe context window (70% of 16 000 tokens).
5. Each batch is sent to Qwen 2.5 7B via Ollama with an extraction prompt that instructs the model to preserve all Markdown, keep verbatim rule language, and remove non-policy prose.

The embedding model is explicitly unloaded from VRAM (`del embed_model; torch.cuda.empty_cache()`) before Ollama inference begins, freeing all embedding model VRAM for the LLM.

**GPU levers:**
- `num_gpu_layers: 35` in Ollama options offloads all 35 transformer layers of Qwen 7B to GPU.
- `num_ctx: 16000` — safe context ceiling for 8 GB VRAM. Setting this to 32 000 is possible but may cause OOM with other processes running.
- `temperature: 0.1` for deterministic, conservative extraction.

---

### Stage 6 — Validation Pass (GPU)

**Script:** `scripts/stage6_validate.py`  
**Input:** `data/final/candidate_policies.md`  
**Output:** `data/final/validated_policies.md`

A second LLM pass sends the candidate output back through Qwen with a repair-focused system prompt. The model is instructed to fix broken Markdown tables, complete truncated lists, remove empty sections, and deduplicate repeated content. Up to 2 retry passes are run (configurable). If the candidate document exceeds the safe context window, it is split on top-level headings (`#`) before being sent in parts, which preserves section boundaries.

`temperature: 0.05` — even lower than extraction to maximise determinism in repair.

---

### Stage 7 — Final Assembly (CPU)

**Script:** `scripts/stage7_assemble.py`  
**Input:** `data/final/validated_policies.md`  
**Output:** `data/final/Policy_Only.md`

A CPU-only pass that performs a final hash-based deduplication of sections (keyed on the first 200 characters of each section), prepends a document header with generation timestamp and source document list, and writes the final output. A JSON run summary is written to `logs/run_summary.json`.

---

## 5. GPU VRAM Budget (RTX 5060 8 GB)

| Workload | Estimated VRAM | Notes |
|---|---|---|
| Marker (layout + OCR models) | ~3.5–4.5 GB | Loaded during Stage 1 only |
| SentenceTransformer (float16) | ~0.08 GB | Very small; 384-dim model |
| Qwen 2.5 7B (all layers, float16) | ~4.5–5.5 GB | Active during Stages 5–6 |
| ChromaDB | CPU RAM only | No GPU memory |
| CUDA overhead + OS | ~0.5 GB | Always reserved |

**Key design decision:** Stages 1 and 4–6 are the GPU-heavy stages. Marker runs in Stage 1 and then releases its models. The embedding model is loaded and released before Ollama inference starts. This sequential GPU usage pattern ensures no two large models compete for VRAM simultaneously.

---

## 6. Configuration Reference

All tunable parameters live in `config.yaml`. The most impactful performance levers:

| Parameter | Path | Default | Impact |
|---|---|---|---|
| `batch_multiplier` | `marker.batch_multiplier` | 4 | Stage 1 throughput. Raise to 6–8 for faster parsing. |
| `embedding_batch_size` | `gpu.embedding_batch_size` | 128 | Stage 4 throughput. Raise to 256 if VRAM allows. |
| `num_gpu_layers` | `llm.num_gpu_layers` | 35 | All layers of Qwen 7B on GPU. Lower if OOM. |
| `context_length` | `llm.context_length` | 16000 | Higher = more context but more VRAM. Max safe: ~24 000. |
| `chunk_size` | `chunking.chunk_size` | 1500 | Larger chunks = fewer retrieval hits but more context per hit. |
| `n_results` | `retrieval.n_results` | 20 | More results = more coverage but larger LLM context needed. |

---

## 7. Failure Recovery Model

Every stage writes output to disk before the next stage begins. If a stage fails:

```
python run_pipeline.py --from-stage N
```

This resumes from stage N without reprocessing earlier stages. Stage 4 is additionally idempotent: it skips chunks already in ChromaDB.

---

## 8. Verification & Acceptance Tests

Run GPU diagnostics before first use:
```
python scripts/gpu_check.py
```

Run smoke tests (no PDF required):
```
pytest tests/test_pipeline.py -v -k "not integration"
```

Run full integration tests (generates synthetic PDFs):
```
pytest tests/test_pipeline.py -v -m integration
```

Three acceptance test tiers (per Plan §30):

| Test | Pages | Verified |
|---|---|---|
| `test_10_page_pdf` | 10 | End-to-end produces output |
| `test_50_page_pdf` | 50 | Tables preserved in output |
| `test_100_page_pdf` | 100 | No truncation, no crash |

---

## 9. Future Upgrade Path

| Upgrade | Benefit | Prerequisites |
|---|---|---|
| Qwen 2.5 14B | Higher extraction quality | 16+ GB VRAM (RTX 4090 / dual-GPU) |
| `all-mpnet-base-v2` embedding | Better retrieval quality | Same VRAM; ~2× slower encode |
| PostgreSQL + pgvector | Production-scale vector store | PostgreSQL server |
| Batch processing queue | Multiple PDFs in parallel | Multi-GPU or A100 |
| Human review dashboard | Manual approval before Policy_Only.md | Web framework (FastAPI + React) |
| Hybrid OCR | Better scanned document handling | Add EasyOCR or PaddleOCR as fallback |
| Context length 32 000 | Larger batches to LLM | ≥10 GB VRAM available to Ollama |

---

## 10. Directory Structure Reference

```
PolicyPipeline/
├── config.yaml                  ← All pipeline configuration
├── requirements.txt             ← Python dependencies
├── run_pipeline.py              ← Master orchestrator
│
├── scripts/
│   ├── utils.py                 ← Logger, config loader, helpers
│   ├── gpu_check.py             ← Pre-flight GPU diagnostics
│   ├── stage1_pdf_parse.py      ← PDF → Markdown (GPU: Marker)
│   ├── stage2_clean.py          ← Markdown cleaning (CPU)
│   ├── stage3_chunk.py          ← Semantic chunking (CPU)
│   ├── stage4_embed_store.py    ← Embeddings + ChromaDB (GPU)
│   ├── stage5_extract.py        ← Policy extraction via Qwen (GPU)
│   ├── stage6_validate.py       ← Validation & repair pass (GPU)
│   └── stage7_assemble.py       ← Final assembly (CPU)
│
├── tests/
│   └── test_pipeline.py         ← Acceptance tests (pytest)
│
├── data/
│   ├── pdfs/                    ← INPUT: place your PDFs here
│   ├── markdown/                ← Stage 1 output
│   ├── cleaned/                 ← Stage 2 output
│   ├── chunks/                  ← Stage 3 output
│   └── final/                   ← Stages 5–7 output
│
├── chroma_db/                   ← ChromaDB persistent storage
├── logs/                        ← Pipeline logs + run_summary.json
└── models/                      ← Reserved for future local model storage
```
