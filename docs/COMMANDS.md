# PolicyPipeline — PDF Processing Command Reference

All commands are run from `D:\PolicyPipeline` in PowerShell with the virtual environment active:

```powershell
venv\Scripts\activate
```

---

# 1. Process All PDFs

Processes every `.pdf` file found in `data/pdfs/` and merges them all into a single `Policy_Only.md` output.

```powershell
python run_pipeline.py
```

## When to Use

* First time setup with all your documents loaded
* You want one unified policy output from your entire document library
* You have added new PDFs and want to rebuild everything

---

# 2. Process Two Specific PDFs and Merge

Processes exactly the PDFs you name. Output is a single `Policy_Only.md` that merges content from both documents.

```powershell
python run_pipeline.py --pdfs document_one document_two
```

### Example

```powershell
python run_pipeline.py --pdfs klarna_governance annual_report
```

## Rules

* Names are the filename **without** the `.pdf` extension
* Names are case-insensitive
* Separate multiple names with spaces
* Any name that does not match a file in `data/pdfs/` is skipped with a warning (the pipeline does not crash)

## When to Use

* You want a merged output from a specific subset of your documents
* You are comparing or combining two related policies

---

# 3. Process a Single PDF

Processes exactly one PDF. Output is `Policy_Only.md` containing only content from that document.

```powershell
python run_pipeline.py --pdfs document_name
```

### Example

```powershell
python run_pipeline.py --pdfs klarna_governance
```

## When to Use

* You want to extract policy from one document in isolation
* You are testing the pipeline on a new document
* You want a clean output without mixing other documents

---

# 4. Switch to a Different Set of PDFs

Processes a new set of PDFs and wipes the previous ChromaDB so old document content does not bleed into the new output.

```powershell
python run_pipeline.py --pdfs doc1 doc2 --reset-chroma
```

### Example

```powershell
python run_pipeline.py --pdfs adobe_policy firefly_terms --reset-chroma
```

## When to Use

* You previously processed a different set of documents and now want a fresh output from a new set
* Your `Policy_Only.md` is showing content from documents you did not intend to include
* Any time you change which PDFs you are working with

> **Warning**
>
> `--reset-chroma` deletes the entire ChromaDB collection. All previous embeddings are lost. Stage 4 will re-embed everything from scratch, which takes extra time.

---

# 5. Resume After a Failure

If the pipeline stops at a stage, fix the error and resume from that stage without reprocessing earlier stages.

```powershell
python run_pipeline.py --from-stage N
```

### Example — Resume from Stage 4

```powershell
python run_pipeline.py --from-stage 4
```

### Example — Resume from Stage 4 with Specific PDFs

```powershell
python run_pipeline.py --pdfs doc1 doc2 --from-stage 4
```

## Stage Numbers

| Stage | Description                      |
| ----- | -------------------------------- |
| 1     | PDF to Markdown (Marker, GPU)    |
| 2     | Markdown cleaning                |
| 3     | Semantic chunking                |
| 4     | Embeddings + ChromaDB (GPU)      |
| 5     | Policy extraction via Qwen (GPU) |
| 6     | Validation pass (GPU)            |
| 7     | Final assembly                   |

## When to Use

* Power cut or crash mid-run
* You adjusted `config.yaml` and only want to re-run from a specific stage onward
* Stage 1 succeeded but Stage 4 ran out of VRAM — close other apps and resume from `--from-stage 4`

---

# 6. Run a Single Stage Only

Runs exactly one stage and exits. Useful for debugging or re-running a specific step after making changes.

```powershell
python run_pipeline.py --only-stage N
```

### Example — Re-run Only the Validation Pass

```powershell
python run_pipeline.py --only-stage 6
```

### Example — Re-run Only Final Assembly

```powershell
python run_pipeline.py --only-stage 7
```

## When to Use

* You edited the extraction prompt and want to re-run Stage 5 without touching anything else
* You want to regenerate `Policy_Only.md` after manually editing `validated_policies.md`
* Debugging a specific stage in isolation

---

# 7. Dry Run — Preview Without Executing

Prints the full execution plan and which stages would run without actually processing anything.

```powershell
python run_pipeline.py --dry-run
```

### Example with PDF Selection

```powershell
python run_pipeline.py --pdfs doc1 doc2 --dry-run
```

## When to Use

* Verify your command is correct before committing to a long run
* Check which stages will be skipped or executed

---

# 8. Pre-Flight GPU Check

Diagnoses GPU, CUDA, Ollama, and embedding model before running the pipeline.

Run this first on a new machine or after a driver update.

```powershell
python scripts/gpu_check.py
```

## What It Checks

* NVIDIA driver and available VRAM
* PyTorch CUDA availability
* Ollama is running and `qwen2.5:7b` is pulled
* SentenceTransformer loads onto GPU correctly

---

# Quick Reference Cheat Sheet

| Goal                               | Command                                                  |
| ---------------------------------- | -------------------------------------------------------- |
| Process all PDFs                   | `python run_pipeline.py`                                 |
| Process two specific PDFs          | `python run_pipeline.py --pdfs doc1 doc2`                |
| Process one PDF                    | `python run_pipeline.py --pdfs doc1`                     |
| Switch document set (fresh output) | `python run_pipeline.py --pdfs doc1 doc2 --reset-chroma` |
| Resume after crash at stage N      | `python run_pipeline.py --from-stage N`                  |
| Re-run one stage only              | `python run_pipeline.py --only-stage N`                  |
| Preview without running            | `python run_pipeline.py --dry-run`                       |
| Check GPU before first run         | `python scripts/gpu_check.py`                            |
