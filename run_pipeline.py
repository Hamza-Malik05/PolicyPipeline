"""
run_pipeline.py
───────────────
Master pipeline orchestrator.
Runs all stages in sequence with failure recovery — each stage writes to disk
so a restart can resume from any completed stage.

Usage
─────
    # Process ALL PDFs in data/pdfs/
    python run_pipeline.py

    # Process exactly two PDFs and merge them
    python run_pipeline.py --pdfs gdpr_policy annual_report

    # Process a single PDF
    python run_pipeline.py --pdfs gdpr_policy

    # Start from a specific stage (1–7)
    python run_pipeline.py --from-stage 3

    # Run only one stage
    python run_pipeline.py --only-stage 4

    # Reset ChromaDB and re-embed (use when switching document sets)
    python run_pipeline.py --pdfs doc1 doc2 --reset-chroma

    # Dry-run: print stage plan and exit
    python run_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from utils import load_config, setup_logger

cfg    = load_config(ROOT / "config.yaml")
logger = setup_logger("pipeline", cfg)

STAGES = [
    (1, "PDF → Markdown",        "scripts/stage1_pdf_parse.py",     []),
    (2, "Markdown Cleaning",     "scripts/stage2_clean.py",         []),
    (3, "Semantic Chunking",     "scripts/stage3_chunk.py",         []),
    (4, "Embed + Store",         "scripts/stage4_embed_store.py",   []),
    (5, "Policy Extraction",     "scripts/stage5_extract.py",       []),
    (6, "Validation Pass",       "scripts/stage6_validate.py",      []),
    (7, "Final Assembly",        "scripts/stage7_assemble.py",      []),
]


def run_stage(num: int, label: str, script: str, extra_args: list[str], dry_run: bool) -> bool:
    SEP = "─" * 60
    logger.info(f"\n{SEP}\n  Stage {num}: {label}\n{SEP}")

    if dry_run:
        logger.info(f"  [DRY RUN] Would execute: python {script} {' '.join(extra_args)}")
        return True

    cmd = [sys.executable, str(ROOT / script)] + extra_args
    start = time.perf_counter()

    result = subprocess.run(cmd, cwd=str(ROOT))

    elapsed = time.perf_counter() - start
    if result.returncode == 0:
        logger.info(f"  ✅  Stage {num} OK  ({elapsed:.1f}s)")
        return True
    else:
        logger.error(f"  ❌  Stage {num} FAILED (exit {result.returncode}, {elapsed:.1f}s)")
        return False


def main():
    parser = argparse.ArgumentParser(description="PolicyPipeline orchestrator")
    parser.add_argument("--from-stage",   type=int, default=1,
                        help="Start from this stage number (1–7). Default: 1")
    parser.add_argument("--only-stage",   type=int, default=None,
                        help="Run only this stage and exit.")
    parser.add_argument("--reset-chroma", action="store_true",
                        help="Drop and rebuild ChromaDB (use when switching document sets)")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print the execution plan without running anything.")
    parser.add_argument(
        "--pdfs",
        nargs="+",
        metavar="NAME",
        help="PDF stem names to process (without .pdf). "
             "Example: --pdfs gdpr_policy annual_report. "
             "Omit to process ALL PDFs in data/pdfs/."
    )
    args = parser.parse_args()

    # Log which PDFs are selected
    if args.pdfs:
        logger.info(f"  Selected PDFs: {args.pdfs}")
    else:
        logger.info("  Selected PDFs: ALL in data/pdfs/")

    logger.info("=" * 60)
    logger.info("  PolicyPipeline — Starting")
    logger.info("=" * 60)

    total_start = time.perf_counter()
    failed_stage = None

    for num, label, script, _ in STAGES:
        # Stage selection
        if args.only_stage is not None and num != args.only_stage:
            continue
        if num < args.from_stage:
            logger.info(f"  Skipping Stage {num}: {label}")
            continue

        # Extra args per stage
        extra: list[str] = []
        if num == 1 and args.pdfs:
            extra += ["--pdfs"] + args.pdfs
        if num == 4 and args.reset_chroma:
            extra += ["--reset"]

        ok = run_stage(num, label, script, extra, args.dry_run)
        if not ok:
            failed_stage = num
            break

    total = time.perf_counter() - total_start
    logger.info("=" * 60)
    if failed_stage:
        logger.error(f"  Pipeline STOPPED at Stage {failed_stage}.")
        logger.error(f"  Fix the error then resume with: "
                     f"python run_pipeline.py --from-stage {failed_stage}")
        sys.exit(1)
    else:
        logger.info(f"  ✅  Pipeline complete!  Total time: {total:.1f}s")
        logger.info(f"  Output → data/final/Policy_Only.md")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()