"""
stage1_pdf_parse.py
───────────────────
Stage 1: PDF → Markdown using Marker with full GPU acceleration.

GPU acceleration notes
──────────────────────
- Marker uses a layout-detection model (Surya) + OCR model internally.
- Both run on CUDA when TORCH_DEVICE=cuda and the correct PyTorch build is present.
- batch_multiplier scales internal page batching — higher = faster on ≥8 GB VRAM.
- We pin the CUDA device via environment variables before importing marker so it
  picks up the GPU from the very first import.

Usage
─────
    # Process all PDFs in data/pdfs/
    python scripts/stage1_pdf_parse.py

    # Process exactly two specific PDFs (names without .pdf extension)
    python scripts/stage1_pdf_parse.py --pdfs gdpr_policy annual_report

    # Process a single PDF (legacy single filter)
    python scripts/stage1_pdf_parse.py --pdf gdpr_policy
"""

from __future__ import annotations

import argparse
import os
import sys
import json
from pathlib import Path

# ── Force GPU BEFORE any marker import ────────────────────────────────────────
os.environ["TORCH_DEVICE"] = "cuda"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, resolve, setup_logger, timer, vram_snapshot

cfg    = load_config()
logger = setup_logger("stage1_pdf_parse", cfg)


def parse_single_pdf(pdf_path: Path, out_dir: Path, marker_cfg: dict) -> Path | None:
    """
    Convert one PDF to Markdown using Marker on GPU.
    Returns the path of the produced .md file, or None on failure.
    """
    import torch
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    if not torch.cuda.is_available():
        logger.error("CUDA not available. Aborting.")
        sys.exit(1)

    logger.info(f"Parsing: {pdf_path.name}  (GPU: {torch.cuda.get_device_name(0)})")
    vram_snapshot(logger, "before parse")

    # Load all Marker models onto GPU once and reuse
    if not hasattr(parse_single_pdf, "_artifact_dict"):
        logger.info("Loading Marker models onto GPU (first call only)…")
        parse_single_pdf._artifact_dict = create_model_dict()
        logger.info("Marker models loaded.")

    artifact_dict = parse_single_pdf._artifact_dict

    langs = marker_cfg.get("langs", "English")
    if isinstance(langs, str):
        langs = [langs]

    marker_options = {
        "output_format": "markdown",
        "languages": langs,
        "force_ocr": marker_cfg.get("ocr_all_pages", False),
    }

    max_pages = marker_cfg.get("max_pages")
    if max_pages is not None:
        marker_options["page_range"] = f"0-{max_pages - 1}"

    try:
        config_parser = ConfigParser(marker_options)

        converter_kwargs = {
            "config": config_parser.generate_config_dict(),
            "artifact_dict": artifact_dict,
            "processor_list": config_parser.get_processors(),
            "renderer": config_parser.get_renderer(),
        }

        try:
            converter_kwargs["llm_service"] = config_parser.get_llm_service()
        except AttributeError:
            pass

        converter = PdfConverter(**converter_kwargs)
        rendered = converter(str(pdf_path))

        full_text, _, images = text_from_rendered(rendered)

        out_meta = {
            "source_pdf": pdf_path.name,
            "output_format": "markdown",
            "marker_api": "PdfConverter",
            "image_count": len(images) if images else 0,
            "options": marker_options,
        }

    except Exception as e:
        logger.error(f"Marker failed on {pdf_path.name}: {e}")
        return None

    vram_snapshot(logger, "after parse")

    # Write markdown
    md_path = out_dir / (pdf_path.stem + ".md")
    md_path.write_text(full_text, encoding="utf-8")

    # Write metadata sidecar
    meta_path = out_dir / (pdf_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(out_meta, indent=2), encoding="utf-8")

    logger.info(f"  → {md_path.name}  ({len(full_text):,} chars)")
    return md_path

@timer(logger)
def run(pdf_names: list[str] | None = None):
    """
    pdf_names: optional list of stem names (without .pdf) to process.
               If None or empty, all PDFs in data/pdfs/ are processed.
    """
    in_dir  = resolve(cfg["paths"]["pdfs"])
    out_dir = resolve(cfg["paths"]["markdown"])

    all_pdfs = sorted(in_dir.glob("*.pdf"))

    if pdf_names:
        # Match exact stems (case-insensitive) — user passes names without .pdf
        targets = [n.lower().replace(".pdf", "") for n in pdf_names]
        pdfs    = [p for p in all_pdfs if p.stem.lower() in targets]

        # Warn about any names that didn't match a file
        found_stems = {p.stem.lower() for p in pdfs}
        for t in targets:
            if t not in found_stems:
                logger.warning(f"  ⚠  No PDF found matching '{t}' in {in_dir}")
    else:
        pdfs = all_pdfs

    if not pdfs:
        logger.warning(f"No PDFs found to process in {in_dir}.")
        return

    logger.info(f"Processing {len(pdfs)} PDF(s): {[p.name for p in pdfs]}")

    results = {"ok": [], "failed": []}
    for pdf in pdfs:
        out = parse_single_pdf(pdf, out_dir, cfg["marker"])
        if out:
            results["ok"].append(str(out))
        else:
            results["failed"].append(str(pdf))

    logger.info(f"Stage 1 complete — ✅ {len(results['ok'])}  ❌ {len(results['failed'])}")
    if results["failed"]:
        logger.warning(f"Failed: {results['failed']}")

    # Write manifest
    manifest = out_dir / "stage1_manifest.json"
    manifest.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: PDF → Markdown")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pdfs",
        nargs="+",
        metavar="NAME",
        help="Space-separated list of PDF stems to process (without .pdf). "
             "Example: --pdfs gdpr_policy annual_report"
    )
    group.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="(Legacy) Single PDF stem filter string."
    )

    args = parser.parse_args()

    if args.pdfs:
        run(args.pdfs)
    elif args.pdf:
        run([args.pdf])
    else:
        run(None)  # process everything