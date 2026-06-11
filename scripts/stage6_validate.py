"""
stage6_validate.py
──────────────────
Stage 7 (Plan §24): Validation pass — send candidate_policies.md back through
Qwen to detect missing policies, broken tables, truncated sections, and
missing bullet points.

On each retry, the model is asked to fix issues found in the previous pass.
Max retries configurable in config.yaml (validation.max_retries, default 2).

GPU notes
─────────
Same Ollama GPU settings as Stage 5 — all 35 layers on RTX 5060.

Usage
─────
    python scripts/stage6_validate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, resolve, setup_logger, timer, PROJECT_ROOT

cfg    = load_config()
logger = setup_logger("stage6_validate", cfg)


VALIDATION_SYSTEM = """You are a strict Markdown policy document validator and repair agent.
You will receive a draft policy extraction. Your job is to:

1. CHECK for these issues:
   - Incomplete or broken Markdown tables (missing pipes, misaligned headers).
   - Truncated bullet points or numbered lists (list that ends mid-way).
   - Section headings with no following content.
   - Duplicate policy sections (same rule appearing twice).
   - Non-policy content that was not removed (introductions, disclaimers, filler prose).

2. FIX all issues you find IN-PLACE.
   - Repair broken tables.
   - Complete truncated lists where context allows, or remove the incomplete stub.
   - Remove empty-headed sections.
   - Deduplicate repeated sections (keep the more complete copy).
   - Remove non-policy filler.

3. OUTPUT the repaired Markdown. Nothing else — no commentary, no issue list,
   no preamble. Just the fixed Markdown document.
"""

VALIDATION_USER = """Validate and repair this policy extraction:

--- DRAFT START ---
{draft}
--- DRAFT END ---

Repaired policy document (Markdown only):"""


def call_qwen_validate(draft: str, llm_cfg: dict) -> str:
    import ollama as ol_client

    options = {
        "num_gpu":    llm_cfg.get("num_gpu_layers", 35),
        "num_thread": llm_cfg.get("num_thread", 8),
        "num_ctx":    llm_cfg.get("context_length", 16000),
        "temperature": 0.05,   # Even lower for repair — deterministic
        "top_p":      0.9,
    }

    response = ol_client.chat(
        model=llm_cfg.get("model", "qwen2.5:7b"),
        messages=[
            {"role": "system", "content": VALIDATION_SYSTEM},
            {"role": "user",   "content": VALIDATION_USER.format(draft=draft)},
        ],
        options=options,
    )
    return response["message"]["content"]


def split_for_validation(text: str, max_chars: int) -> list[str]:
    """
    If the candidate document is too large for one validation call,
    split it on top-level headings (# ) to keep sections intact.
    """
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if line.startswith("# ") and len(current) > 200:
            parts.append(current)
            current = line
        else:
            current += line
    if current:
        parts.append(current)

    # Re-merge parts that are under half max_chars
    merged: list[str] = []
    buf = ""
    for part in parts:
        if len(buf) + len(part) <= max_chars:
            buf += part
        else:
            if buf:
                merged.append(buf)
            buf = part
    if buf:
        merged.append(buf)

    return merged


@timer(logger)
def run():
    if not cfg.get("validation", {}).get("enabled", True):
        logger.info("Validation disabled in config. Skipping Stage 6.")
        return

    final_dir    = resolve(cfg["paths"]["final"])
    candidate    = final_dir / "candidate_policies.md"
    llm_cfg      = cfg["llm"]
    max_retries  = cfg.get("validation", {}).get("max_retries", 2)

    if not candidate.exists():
        logger.error(f"{candidate} not found. Run Stage 5 first.")
        return

    current_text = candidate.read_text(encoding="utf-8")
    logger.info(f"Validating candidate ({len(current_text):,} chars), "
                f"up to {max_retries} pass(es).")

    # Context budget: 70% for input, leaving 30% for output
    max_input_chars = int(llm_cfg.get("context_length", 16000) * 0.70 * 3.3)

    for attempt in range(1, max_retries + 1):
        logger.info(f"  Validation pass {attempt}/{max_retries}…")
        parts  = split_for_validation(current_text, max_input_chars)
        repaired_parts: list[str] = []

        for i, part in enumerate(parts):
            try:
                fixed = call_qwen_validate(part, llm_cfg)
                repaired_parts.append(fixed.strip())
                logger.debug(f"    Part {i+1}/{len(parts)} validated.")
            except Exception as e:
                logger.error(f"    Validation call failed on part {i+1}: {e}")
                repaired_parts.append(part)  # Keep original on failure

        current_text = "\n\n".join(repaired_parts)
        logger.info(f"  Pass {attempt} complete — {len(current_text):,} chars.")

    # Save validated output
    validated_path = final_dir / "validated_policies.md"
    validated_path.write_text(current_text, encoding="utf-8")
    logger.info(f"Stage 6 complete — validated output: {validated_path}")


if __name__ == "__main__":
    run()
