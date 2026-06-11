# Option B Implementation Plan (Production Grade)
## Local PDF → Markdown → Policy Extraction Pipeline
### Windows 11 + RTX 5060 8GB + Ollama + Marker + ChromaDB + Qwen 2.5

---

# 1. Project Goal

Build a fully local pipeline that:

1. Ingests PDFs (50–100+ pages)
2. Extracts text, headings, tables, and layout
3. Converts content into Markdown
4. Splits content into semantic chunks
5. Stores chunks in a vector database
6. Uses a local LLM to identify policy content
7. Runs a validation pass
8. Produces a clean Policy_Only.md output

No cloud APIs.
No OpenAI API costs.
Everything runs locally.

---

# 2. Hardware Requirements

## Minimum

- Windows 10/11
- RTX 5060 8GB
- 16 GB RAM
- 30 GB free storage

## Recommended

- Windows 11
- RTX 5060 8GB
- 32 GB RAM
- 100 GB free SSD storage

---

# 3. Software Overview

| Component | Purpose |
|------------|----------|
| Python 3.11 | Main runtime |
| Git | Source control |
| Visual Studio Build Tools | Native compilation |
| Tesseract | OCR |
| Poppler | PDF processing |
| Marker | PDF → Markdown |
| Ollama | Local inference |
| Qwen 2.5 7B | Policy extraction |
| ChromaDB | Vector database |
| LangChain | Chunking and retrieval |
| PyTorch | GPU acceleration |

---

# 4. Windows Preparation

## Update Windows

Settings → Windows Update

Install all updates.

Reboot.

---

## Install NVIDIA Driver

Install latest Game Ready or Studio Driver.

Verify:

```powershell
nvidia-smi
```

Expected:

```text
RTX 5060
CUDA Version: XX.X
```

---

# 5. Install Python

Download:

https://www.python.org/downloads/

Install Python 3.11

CHECK:

Add Python to PATH

Verify:

```powershell
python --version
```

Expected:

```text
Python 3.11.x
```

---

# 6. Install Git

Install Git for Windows.

Verify:

```powershell
git --version
```

---

# 7. Install Visual Studio Build Tools

Install:

Desktop Development with C++

Required components:

- MSVC Compiler
- Windows SDK
- CMake Tools

Reboot after installation.

---

# 8. Install Tesseract OCR

Install Tesseract.

Default location:

```text
C:\Program Files\Tesseract-OCR
```

Add folder to PATH.

Verify:

```powershell
tesseract --version
```

---

# 9. Install Poppler

Extract to:

```text
C:\poppler
```

Add:

```text
C:\poppler\Library\bin
```

to PATH.

Verify:

```powershell
pdftotext -v
```

---

# 10. Create Project Structure

```text
D:\PolicyPipeline
│
├── config.yaml                        ← root (next to run_pipeline.py)
├── requirements.txt                   ← root
├── run_pipeline.py                    ← root
├── ARCHITECTURE.md                    ← root
│
├── data
│   ├── pdfs                           ← YOUR INPUT: drop PDFs here
│   ├── markdown                       ← Stage 1 writes here (*.md + *_meta.json)
│   ├── cleaned                        ← Stage 2 writes here (*.md)
│   ├── chunks                         ← Stage 3 writes here (*_chunks.json)
│   └── final                          ← Stages 5/6/7 write here:
│                                           candidate_policies.md
│                                           validated_policies.md
│                                           Policy_Only.md  ✅
│
├── chroma_db                          ← Stage 4 writes here (vector DB files)
│
├── logs                               ← utils.py writes pipeline.log here
│                                        Stage 7 writes run_summary.json here
│
├── models                             ← reserved (empty for now; future use)
│
├── scripts
│   ├── utils.py
│   ├── gpu_check.py
│   ├── stage1_pdf_parse.py
│   ├── stage2_clean.py
│   ├── stage3_chunk.py
│   ├── stage4_embed_store.py
│   ├── stage5_extract.py
│   ├── stage6_validate.py
│   └── stage7_assemble.py
│
├── tests
│   └── test_pipeline.py
│
└── venv                               ← created by you: python -m venv venv
```

---

# 11. Create Virtual Environment

```powershell
cd D:\PolicyPipeline

python -m venv venv
```

Activate:

```powershell
venv\Scripts\activate
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

---

# 12. Install PyTorch

Install:

```powershell
pip install torch torchvision torchaudio
```

Verify:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Expected:

```text
True
RTX 5060
```

---

# 13. Install Marker

```powershell
pip install marker-pdf
```

Verify:

```powershell
marker_single --help
```

Run test:

```powershell
marker_single sample.pdf output
```

---

# 14. Install Ollama

Install Ollama for Windows.

Verify:

```powershell
ollama --version
```

---

# 15. Install Model

Recommended:

```powershell
ollama pull qwen2.5:7b
```

Test:

```powershell
ollama run qwen2.5:7b
```

Prompt:

```text
Hello
```

Confirm response is generated.

---

# 16. Install Pipeline Libraries

```powershell
pip install chromadb
```

```powershell
pip install langchain
```

```powershell
pip install langchain-community
```

```powershell
pip install langchain-text-splitters
```

```powershell
pip install sentence-transformers
```

```powershell
pip install pypdf
```

```powershell
pip install ollama
```

```powershell
pip install pandas
```

```powershell
pip install tqdm
```

---

# 17. Verification Phase

Verify imports:

```python
import chromadb
import langchain
import torch
import ollama
```

No errors should occur.

---

# 18. Pipeline Stage 1: PDF Parsing

Input:

```text
data/pdfs
```

Output:

```text
data/markdown
```

Process:

PDF
→ Marker
→ Markdown

Verification:

Check:

- Headings preserved
- Tables preserved
- Lists preserved

---

# 19. Pipeline Stage 2: Markdown Cleaning

Remove:

- Page numbers
- Repeated headers
- Repeated footers
- Empty sections

Output:

```text
data/cleaned
```

Verification:

Markdown remains readable.

---

# 20. Pipeline Stage 3: Semantic Chunking

Use:

RecursiveCharacterTextSplitter

Recommended:

```text
Chunk Size: 1500
Chunk Overlap: 300
```

Output:

```text
data/chunks
```

---

# 21. Pipeline Stage 4: Embeddings

Recommended embedding model:

```text
all-MiniLM-L6-v2
```

Download automatically using sentence-transformers.

Each chunk becomes:

```text
Chunk
→ Embedding Vector
```

---

# 22. Pipeline Stage 5: ChromaDB

Store:

- Chunk text
- Source document
- Chunk ID
- Embedding

Database location:

```text
chroma_db
```

Verification:

Query database.

Confirm results are returned.

---

# 23. Pipeline Stage 6: Policy Extraction

Query categories:

- Policies
- Rules
- Compliance
- Violations
- Penalties
- Requirements

Send retrieved chunks to Qwen.

Prompt should instruct:

- Keep markdown
- Preserve tables
- Remove non-policy content
- Do not summarize

Output:

```text
Candidate Policies
```

---

# 24. Pipeline Stage 7: Validation Pass

Run extracted content through Qwen again.

Validation prompt:

Check:

- Missing policies
- Broken tables
- Truncated sections
- Missing bullet points

Output:

```text
Validated Policies
```

---

# 25. Pipeline Stage 8: Final Assembly

Merge validated sections.

Save:

```text
data/final/Policy_Only.md
```

---

# 26. Logging

Create:

```text
logs
```

Store:

- Processing time
- Errors
- Token counts
- Chunk counts

Recommended:

```python
logging
```

module.

---

# 27. RTX 5060 Optimization

Use:

```text
Qwen 2.5 7B
```

Recommended context:

```text
16000–32000
```

Avoid:

```text
64000
128000
```

unless extensively tested.

---

# 28. Memory Optimization

Close:

- Chrome
- Games
- Unreal Engine
- Blender

Keep:

```text
10+ GB RAM free
```

during processing.

---

# 29. Failure Recovery

Every stage writes output to disk.

If extraction fails:

Restart from:

- Markdown
- Chunks
- ChromaDB

instead of reprocessing the PDF.

---

# 30. Acceptance Tests

Test 1:

- 10-page PDF

Test 2:

- 50-page PDF

Test 3:

- 100-page PDF

Verify:

- Tables preserved
- Policies preserved
- Output markdown valid

---

# 31. Future Upgrades

Possible upgrades:

- Qwen 14B
- Hybrid OCR
- PostgreSQL + pgvector
- Multi-document search
- Web interface
- Batch processing queue
- Human review dashboard

---

# Final Recommended Configuration

Hardware:
- RTX 5060 8GB
- 32 GB RAM

Model:
- Qwen 2.5 7B

Parser:
- Marker

Vector DB:
- ChromaDB

Framework:
- LangChain

Output:
- Policy_Only.md

This configuration provides the best balance between accuracy, cost, speed, and maintainability for large policy-document extraction workflows on consumer hardware.
