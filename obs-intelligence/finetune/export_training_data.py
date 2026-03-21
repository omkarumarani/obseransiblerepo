#!/usr/bin/env python3
"""
export_training_data.py
─────────────────────────────────────────────────────────────────────────────
Export LoRA fine-tuning training data from the obs-intelligence SQLite stores.

Sources
───────
  /data/learning.db  — knowledge_entries table
    Each row is a real incident that went through the full pipeline:
      external LLM analysis ➜ local LLM validation ➜ final outcome.
    Rows where both validation_status and outcome are confirmed (not pending /
    unavailable) make the highest-quality training samples.

  /data/outcomes.db  — scenario_outcomes table
    Used to cross-reference training rows: if the outcome for a run_id is
    "resolved" or "success" the external→local analysis pair was likely correct,
    boosting its quality weight.

Output format
─────────────
  OpenAI / HuggingFace chat-format JSONL:
    { "messages": [
        {"role": "system",    "content": "<system prompt>"},
        {"role": "user",      "content": "<incident context + external analysis>"},
        {"role": "assistant", "content": "<json validation verdict>"}
    ]}

  This format is directly consumable by HuggingFace TRL SFTTrainer with the
  ``dataset_text_field`` set to the tokenizer's chat template.

Usage
─────
  python export_training_data.py \\
      [--db-dir    /data]    \\   # directory containing learning.db + outcomes.db
      [--output-dir /output] \\   # destination for train.jsonl + eval.jsonl
      [--eval-split 0.10]    \\   # fraction held out for evaluation (default 10%)
      [--min-quality 0]           # include rows with any validation_status (0)
                                  # or only confirmed ones (1, default)

Exit codes
──────────
  0 — success, files written
  1 — fewer than MIN_TRAIN_ROWS useful rows found (not enough data to train)
"""

import argparse
import json
import os
import random
import sqlite3
import sys
from pathlib import Path

MIN_TRAIN_ROWS = 5  # Abort if we can't produce at least this many training rows.

_SYSTEM_PROMPT = (
    "You are an expert SRE validation assistant embedded in an AIOps platform. "
    "You will be presented with an incident description, an external LLM's root-cause "
    "analysis and recommended remediation action, and optionally a set of similar "
    "historical incidents retrieved from a knowledge store.\n\n"
    "Your task is to evaluate the external analysis and return a JSON object with the "
    "following fields ONLY — no extra text, no markdown fences:\n"
    "  verdict           — one of: corroborated | weak_support | divergent | "
    "insufficient_context\n"
    "  confidence        — float 0.0–1.0\n"
    "  rca_alignment     — one of: aligned | partial | misaligned\n"
    "  action_alignment  — one of: aligned | partial | misaligned\n"
    "  reasoning_summary — one sentence explaining your verdict\n"
    "  suggested_adjustment — brief suggestion for the operator, or 'none'\n\n"
    "Ground your verdict in the historical incidents when available.  "
    "Return insufficient_context only when there is genuinely nothing to compare against."
)


def _build_user_message(row: sqlite3.Row) -> str:
    parts = [
        f"## Incident",
        f"Service:    {row['service_name']}",
        f"Alert:      {row['alert_name']}",
        f"Domain:     {row['domain']}",
        f"Scenario:   {row['scenario_id']}",
        f"Risk score: {row['top_similarity']:.2f}",
        "",
        f"## Incident narrative",
        row["document"] or "(no narrative)",
        "",
        f"## External LLM analysis",
        f"Root cause:          {row['root_cause'] or '(not provided)'}",
        f"Recommended action:  {row['recommended_action'] or '(not provided)'}",
    ]

    # Add similar incidents from the stored JSON if present
    similar_raw = row["similar_entries_json"] or "[]"
    try:
        similar = json.loads(similar_raw)
    except (json.JSONDecodeError, TypeError):
        similar = []

    if similar:
        parts += ["", "## Historical similar incidents (from knowledge store)"]
        for i, entry in enumerate(similar[:3], 1):
            doc = entry.get("document") or entry.get("description", "(no text)")
            sim = entry.get("similarity", 0.0)
            outcome = entry.get("metadata", {}).get("outcome", "unknown")
            parts.append(f"  [{i}] similarity={sim:.2f} outcome={outcome}")
            parts.append(f"       {doc[:200]}")

    return "\n".join(parts)


def _build_assistant_message(row: sqlite3.Row) -> str:
    verdict_map = {
        "corroborated":        "corroborated",
        "weak_support":        "weak_support",
        "divergent":           "divergent",
        "insufficient_context": "insufficient_context",
    }
    verdict = verdict_map.get(row["validation_status"], "insufficient_context")

    response = {
        "verdict":              verdict,
        "confidence":           float(row["validation_confidence"] or 0.7),
        "rca_alignment":        "aligned" if verdict == "corroborated" else "partial",
        "action_alignment":     "aligned" if verdict == "corroborated" else "partial",
        "reasoning_summary":    row["validation_reason"] or f"Analysis {verdict} by local validator.",
        "suggested_adjustment": "none",
    }
    return json.dumps(response)


def _quality_score(row: sqlite3.Row) -> int:
    """
    Return a quality weight for a training row.
    Higher is better; rows with 0 are excluded when --min-quality=1.
    """
    status = row["validation_status"] or ""
    outcome = row["outcome"] or ""

    if status in ("corroborated", "divergent") and outcome in ("resolved", "success", "auto_resolved"):
        return 2   # Best: we know the verdict AND the final outcome
    if status in ("corroborated", "divergent", "weak_support"):
        return 1   # Good: confirmed verdict, outcome unknown
    if status == "insufficient_context" and outcome in ("resolved", "success"):
        return 1   # Useful negative: model should have said something useful
    return 0       # Pending / unavailable — not useful for training


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LoRA training data from SQLite")
    parser.add_argument("--db-dir",      default=os.getenv("DB_DIR", "/data"))
    parser.add_argument("--output-dir",  default=os.getenv("OUTPUT_DIR", "/output"))
    parser.add_argument("--eval-split",  type=float, default=0.10)
    parser.add_argument("--min-quality", type=int,   default=1,
                        help="0=all rows, 1=confirmed verdicts only")
    args = parser.parse_args()

    learning_db = Path(args.db_dir) / "learning.db"
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not learning_db.exists():
        print(f"ERROR: {learning_db} not found.  Run obs-intelligence first to populate it.")
        sys.exit(1)

    con = sqlite3.connect(str(learning_db))
    con.row_factory = sqlite3.Row

    rows = con.execute("""
        SELECT
            entry_id, domain, service_name, alert_name, scenario_id,
            document, evidence_summary, root_cause, recommended_action,
            validation_status, validation_confidence, validation_reason,
            top_similarity, similar_entries_json, outcome, resolution_time_seconds
        FROM knowledge_entries
        WHERE validation_status NOT IN ('', 'pending')
          AND local_validation_completed = 1
        ORDER BY created_at DESC
    """).fetchall()
    con.close()

    examples = []
    for row in rows:
        quality = _quality_score(row)
        if args.min_quality and quality == 0:
            continue
        examples.append({
            "messages": [
                {"role": "system",    "content": _SYSTEM_PROMPT},
                {"role": "user",      "content": _build_user_message(row)},
                {"role": "assistant", "content": _build_assistant_message(row)},
            ],
            "_quality": quality,
            "_entry_id": row["entry_id"],
        })

    if len(examples) < MIN_TRAIN_ROWS:
        print(
            f"ERROR: Only {len(examples)} usable training rows found "
            f"(minimum required: {MIN_TRAIN_ROWS}).  "
            "Record more real incidents through the pipeline first."
        )
        sys.exit(1)

    # Sort by quality (best first) then shuffle deterministically for reproducibility
    examples.sort(key=lambda x: x["_quality"], reverse=True)
    random.seed(42)
    random.shuffle(examples)

    split_idx = max(1, int(len(examples) * (1 - args.eval_split)))
    train_examples = examples[:split_idx]
    eval_examples  = examples[split_idx:]

    for fname, data in [("train.jsonl", train_examples), ("eval.jsonl", eval_examples)]:
        fpath = output_dir / fname
        with open(fpath, "w") as f:
            for ex in data:
                row_out = {k: v for k, v in ex.items() if not k.startswith("_")}
                f.write(json.dumps(row_out) + "\n")
        print(f"Wrote {len(data):>4} rows → {fpath}")

    print(f"\nTotal: {len(train_examples)} train + {len(eval_examples)} eval rows")
    print("Next step: python train_lora.py [options]")


if __name__ == "__main__":
    main()
