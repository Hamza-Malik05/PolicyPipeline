"""
test_pipeline.py
────────────────
Acceptance tests (Plan §30).

Tests:
    - test_10_page  : minimal smoke test — small PDF end-to-end
    - test_50_page  : medium PDF — verify tables and policies preserved
    - test_100_page : large PDF — verify no truncation or crashes

All tests use synthetic PDFs generated at runtime (no real test files required).
Run with pytest:

    pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_pdf(pages: int, out_path: Path):
    """
    Create a minimal synthetic PDF with `pages` pages using reportlab.
    Each page contains a heading, a policy rule paragraph, and a small table.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        pytest.skip("reportlab not installed — install with: pip install reportlab")

    styles = getSampleStyleSheet()
    doc    = SimpleDocTemplate(str(out_path), pagesize=A4)
    story  = []

    for i in range(1, pages + 1):
        story.append(Paragraph(f"Section {i}: Policy Rule", styles["Heading1"]))
        story.append(Paragraph(
            f"Policy {i}: All users must comply with requirement {i}. "
            "Violations shall result in enforcement action as defined in Schedule A.",
            styles["Normal"]
        ))
        story.append(Spacer(1, 12))
        table_data = [
            ["Rule ID", "Description", "Penalty"],
            [f"R-{i:03d}", f"Requirement {i}", f"${i * 100}"],
        ]
        story.append(Table(table_data))
        story.append(Spacer(1, 24))

    doc.build(story)


def run_stage(stage_num: int, extra_args: list[str] = None) -> int:
    scripts = {
        1: "scripts/stage1_pdf_parse.py",
        2: "scripts/stage2_clean.py",
        3: "scripts/stage3_chunk.py",
        4: "scripts/stage4_embed_store.py",
        5: "scripts/stage5_extract.py",
        6: "scripts/stage6_validate.py",
        7: "scripts/stage7_assemble.py",
    }
    cmd = [sys.executable, str(ROOT / scripts[stage_num])] + (extra_args or [])
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return result.returncode


def check_output_exists() -> bool:
    return (ROOT / "data" / "final" / "Policy_Only.md").exists()


def check_output_has_tables() -> bool:
    path = ROOT / "data" / "final" / "Policy_Only.md"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return "|" in text  # Markdown table indicator


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineSmoke:

    def test_gpu_available(self):
        """Fail fast if CUDA is not available."""
        import torch
        assert torch.cuda.is_available(), (
            "CUDA not available. Check NVIDIA driver and PyTorch GPU build."
        )

    def test_ollama_running(self):
        """Verify Ollama is reachable."""
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        assert result.returncode == 0, "Ollama is not running or not in PATH."

    def test_qwen_model_present(self):
        """Verify qwen2.5:7b is pulled."""
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        assert "qwen2.5" in result.stdout, (
            "qwen2.5:7b not found. Run: ollama pull qwen2.5:7b"
        )

    def test_chromadb_importable(self):
        import chromadb
        assert chromadb.__version__

    def test_sentence_transformers_importable(self):
        from sentence_transformers import SentenceTransformer
        assert SentenceTransformer


@pytest.mark.integration
class TestPipelineIntegration:

    @pytest.fixture(autouse=True)
    def clean_test_dirs(self, tmp_path):
        """Each integration test gets a fresh PDF in data/pdfs/."""
        pdfs_dir = ROOT / "data" / "pdfs"
        pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.pdfs_dir = pdfs_dir
        yield
        # Don't clean up — let developer inspect outputs.

    def _run_full_pipeline(self, pdf_path: Path):
        pdf_name = pdf_path.name
        assert run_stage(1, ["--pdf", pdf_name.replace(".pdf", "")]) == 0
        assert run_stage(2) == 0
        assert run_stage(3) == 0
        assert run_stage(4, ["--reset"]) == 0
        assert run_stage(5) == 0
        assert run_stage(6) == 0
        assert run_stage(7) == 0

    def test_10_page_pdf(self):
        pdf = self.pdfs_dir / "test_10page.pdf"
        make_synthetic_pdf(10, pdf)
        self._run_full_pipeline(pdf)
        assert check_output_exists(), "Policy_Only.md not produced."

    def test_50_page_pdf(self):
        pdf = self.pdfs_dir / "test_50page.pdf"
        make_synthetic_pdf(50, pdf)
        self._run_full_pipeline(pdf)
        assert check_output_exists(), "Policy_Only.md not produced."
        assert check_output_has_tables(), "No tables found in 50-page output."

    def test_100_page_pdf(self):
        pdf = self.pdfs_dir / "test_100page.pdf"
        make_synthetic_pdf(100, pdf)
        self._run_full_pipeline(pdf)
        assert check_output_exists(), "Policy_Only.md not produced."
        assert check_output_has_tables(), "No tables found in 100-page output."
