"""
stage2_clean.py
───────────────
Stage 2: Clean raw Markdown output from Marker.

Removes:
  - Page number lines  (e.g.  "3", "Page 3", "- 3 -")
  - Repeated running headers/footers (lines appearing >N times)
  - Consecutive blank lines (collapsed to single blank)
  - Marker artifact tokens  (e.g.  "<!-- image -->")
  - Zero-width and non-printable characters

Preserves:
  - All heading levels (#, ##, ###…)
  - Tables (pipe-delimited Markdown)
  - Bullet lists and numbered lists
  - Code blocks

Usage
─────
    python scripts/stage2_clean.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, resolve, setup_logger, timer

cfg    = load_config()
logger = setup_logger("stage2_clean", cfg)


# ── Regex patterns ─────────────────────────────────────────────────────────────
_RE_PAGE_NUM   = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")
_RE_PAGE_LABEL = re.compile(r"^\s*[Pp]age\s+\d+\s*(of\s+\d+)?\s*$")
_RE_IMG_MARKER = re.compile(r"<!--\s*image\s*-->", re.IGNORECASE)
_RE_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")
_RE_MULTI_BLANK= re.compile(r"\n{3,}")


def find_repeated_lines(lines: list[str], threshold: int = 5) -> set[str]:
    """
    Lines that appear verbatim ≥ threshold times are likely running
    headers/footers and should be stripped.
    Only considers short lines (≤ 120 chars) to avoid stripping real content.
    """
    counts = Counter(
        l.strip() for l in lines
        if 1 <= len(l.strip()) <= 120
    )
    return {text for text, n in counts.items() if n >= threshold}


def clean_markdown(raw: str) -> str:
    lines = raw.splitlines()

    # Pass 1: identify repeated boilerplate lines
    repeated = find_repeated_lines(lines, threshold=5)
    if repeated:
        logger.debug(f"  Stripping {len(repeated)} repeated line(s): {list(repeated)[:5]}")

    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()

        # Drop image markers
        if _RE_IMG_MARKER.match(stripped):
            continue

        # Drop page numbers
        if _RE_PAGE_NUM.match(stripped) or _RE_PAGE_LABEL.match(stripped):
            continue

        # Drop repeated boilerplate
        if stripped in repeated:
            continue

        # Remove zero-width characters
        line = _RE_ZERO_WIDTH.sub("", line)

        cleaned.append(line)

    text = "\n".join(cleaned)

    # Collapse 3+ blank lines → 1 blank line
    text = _RE_MULTI_BLANK.sub("\n\n", text)

    return text.strip()


@timer(logger)
def run():
    in_dir  = resolve(cfg["paths"]["markdown"])
    out_dir = resolve(cfg["paths"]["cleaned"])

    md_files = sorted(in_dir.glob("*.md"))
    if not md_files:
        logger.warning(f"No .md files found in {in_dir}. Run Stage 1 first.")
        return

    logger.info(f"Cleaning {len(md_files)} Markdown file(s).")

    for md_path in md_files:
        raw = md_path.read_text(encoding="utf-8")
        cleaned = clean_markdown(raw)

        out_path = out_dir / md_path.name
        out_path.write_text(cleaned, encoding="utf-8")

        reduction = (1 - len(cleaned) / max(len(raw), 1)) * 100
        logger.info(f"  {md_path.name}: {len(raw):,} → {len(cleaned):,} chars  ({reduction:.1f}% removed)")

    logger.info(f"Stage 2 complete — {len(md_files)} file(s) cleaned.")


if __name__ == "__main__":
    run()
