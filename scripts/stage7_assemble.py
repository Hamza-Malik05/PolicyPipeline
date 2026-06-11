"""
stage7_assemble.py
──────────────────
Stage 8 (Plan §25): Final assembly — merge validated sections, deduplicate,
add document header, and write data/final/Policy_Only.md.

Also writes a run summary JSON to logs/.

Usage
─────
    python scripts/stage7_assemble.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, resolve, setup_logger, timer, PROJECT_ROOT

cfg    = load_config()
logger = setup_logger("stage7_assemble", cfg)


def deduplicate_sections(text: str) -> str:
    """
    Split on section boundaries and remove duplicate sections (keep first copy).
    A section is identified by its heading + first 120 chars of content hash.
    """
    lines    = text.splitlines()
    sections: list[str] = []
    current: list[str] = []
    seen_hashes: set[str] = set()

    def flush():
        block = "\n".join(current).strip()
        if not block:
            return
        h = hashlib.md5(block[:200].encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            sections.append(block)
        else:
            logger.debug("Removed duplicate section.")

    for line in lines:
        if line.startswith("#") and current:
            flush()
            current = [line]
        else:
            current.append(line)

    flush()
    return "\n\n".join(sections)


def build_header(source_names: list[str]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sources = "\n".join(f"- {n}" for n in source_names)
    return f"""# Policy Extraction — Final Output

**Generated:** {ts}  
**Pipeline:** PolicyPipeline v1.0 (Local GPU | Qwen 2.5 7B | ChromaDB)  
**Source documents:**

{sources}

---

"""


@timer(logger)
def run():
    final_dir = resolve(cfg["paths"]["final"])

    # Prefer validated; fall back to candidate
    validated = final_dir / "validated_policies.md"
    candidate = final_dir / "candidate_policies.md"

    if validated.exists():
        source_file = validated
        logger.info("Using validated_policies.md as assembly source.")
    elif candidate.exists():
        source_file = candidate
        logger.warning("validated_policies.md not found — using candidate_policies.md.")
    else:
        logger.error("No extracted policy file found. Run Stage 5 (and optionally 6) first.")
        return

    text = source_file.read_text(encoding="utf-8")

    # Deduplication
    before = len(text.splitlines())
    text   = deduplicate_sections(text)
    after  = len(text.splitlines())
    logger.info(f"Deduplication: {before} → {after} lines.")

    # Collect source names from chunks manifest
    chunks_dir   = resolve(cfg["paths"]["chunks"])
    source_names = sorted(set(
        p.stem.replace("_chunks", "") for p in chunks_dir.glob("*_chunks.json")
    ))

    header = build_header(source_names)
    final  = header + text

    out_path = final_dir / "Policy_Only.md"
    out_path.write_text(final, encoding="utf-8")
    logger.info(f"✅  Final output written: {out_path}  ({len(final):,} chars)")

    # Run summary
    summary = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "source_file":     str(source_file),
        "output_file":     str(out_path),
        "source_docs":     source_names,
        "final_chars":     len(final),
        "final_lines":     len(final.splitlines()),
    }
    log_path = PROJECT_ROOT / cfg["paths"]["logs"] / "run_summary.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Run summary → {log_path}")


if __name__ == "__main__":
    run()
