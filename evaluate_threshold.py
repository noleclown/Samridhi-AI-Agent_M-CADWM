"""
evaluate_threshold.py — Confidence Threshold Calibration Utility
=================================================================
Samridhi v4.0

PURPOSE
-------
The FAISS confidence threshold (default 0.35 in config.yaml) determines
whether a retrieved document set is "good enough" to answer from, or
whether the pipeline should fall through to the live web scrape / fallback.

Setting it too LOW  → poor/irrelevant answers served from FAISS.
Setting it too HIGH → good answers discarded, unnecessary live scrapes triggered.

This script runs your own test questions against the FAISS index and
reports precision, recall, and F1 at every threshold from 0.10 to 0.90,
so you can pick the value that best fits your data.

USAGE
-----
1. Edit TEST_QUESTIONS below — add 20-50 questions you KNOW your FAISS
   index should answer well (label them "relevant") and a few it should NOT
   answer (label them "irrelevant").

2. Run from your project root:
       python evaluate_threshold.py

3. Read the output table and CSV, pick the threshold with the best F1,
   update config.yaml:
       retrieval:
         faiss_confidence_threshold: <your chosen value>

OUTPUT
------
  - Printed table: threshold | precision | recall | f1 | avg_confidence
  - threshold_eval.csv: same data, importable into Excel / Sheets
  - Best threshold recommendation printed at the end

REQUIREMENTS
------------
  pip install tabulate   (optional — falls back to plain text if missing)
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

# ── make sure samridhi package is importable ──────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from samridhi.config import BASE_DIR, FAISS_PATH, cfg
from samridhi.retrieval import _keywords, _keyword_score, expand_query

# ══════════════════════════════════════════════════════════════
# TEST SET
# ══════════════════════════════════════════════════════════════
# Fill this list with (question, is_relevant) tuples.
#
#   is_relevant = True  → your FAISS index SHOULD return confident results
#   is_relevant = False → your FAISS index should NOT be confident
#                         (out-of-scope queries, unrelated topics)
#
# The more questions you add, the more reliable the calibration.
# Aim for at least 20 relevant + 10 irrelevant.
# ──────────────────────────────────────────────────────────────

TEST_QUESTIONS: list[tuple[str, bool]] = [
    # ── M-CADWM / SMIS related (relevant = True) ─────────────
    ("What is M-CADWM?",                                        True),
    ("What is the full form of SMIS?",                          True),
    ("How do I register on the SMIS portal?",                   True),
    ("What are the objectives of M-CADWM scheme?",              True),
    ("What is a Water Users Association?",                      True),
    ("What documents are required for WUA formation?",          True),
    ("What is the eligibility criteria for M-CADWM benefits?",  True),
    ("How does M-CADWM funding work?",                          True),
    ("What is the role of WUA in command area development?",    True),
    ("What is PMKSY and how is M-CADWM related to it?",         True),
    ("How to submit a report on SMIS portal?",                  True),
    ("What is the process for O&M under M-CADWM?",              True),
    ("What is CAD and how is it implemented?",                  True),
    ("What are the guidelines for minor irrigation under M-CADWM?", True),
    ("How to login to SMIS portal?",                            True),
    ("What is the fund allocation for M-CADWM?",                True),
    ("What is the definition of command area?",                 True),
    ("How many beneficiaries are covered under M-CADWM?",       True),
    ("What is the procedure for field channel construction?",   True),
    ("What is the role of IMTI in M-CADWM?",                    True),
    # ── Out of scope (relevant = False) ──────────────────────
    ("What is the capital of France?",                          False),
    ("Who won the cricket World Cup?",                          False),
    ("What is the best smartphone to buy in 2024?",             False),
    ("How do I cook biryani?",                                  False),
    ("What is the GDP of India?",                               False),
    ("How does machine learning work?",                         False),
    ("What are the best tourist places in Rajasthan?",          False),
    ("Who is the Prime Minister of UK?",                        False),
    ("What is the stock price of Infosys?",                     False),
    ("How to apply for a passport?",                            False),
]

# ══════════════════════════════════════════════════════════════
# THRESHOLDS TO EVALUATE
# ══════════════════════════════════════════════════════════════

THRESHOLDS = [round(t * 0.05, 2) for t in range(2, 19)]  # 0.10 to 0.90 step 0.05
OUTPUT_CSV = ROOT / "threshold_eval.csv"

# ══════════════════════════════════════════════════════════════
# LOAD RESOURCES
# ══════════════════════════════════════════════════════════════

def load_index():
    print("Loading FAISS index and embedding model...")
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=str(BASE_DIR / "models"),
    )
    db = FAISS.load_local(FAISS_PATH, emb, allow_dangerous_deserialization=True)
    print(f"Index loaded.\n")
    return db


# ══════════════════════════════════════════════════════════════
# SCORING — mirrors retrieve_docs hybrid logic exactly
# ══════════════════════════════════════════════════════════════

def get_confidence(question: str, vector_db) -> float:
    """
    Compute the hybrid confidence score for a question against the FAISS index.
    Mirrors the exact scoring logic in retrieve_docs() so results are comparable.
    """
    max_dist = cfg["retrieval"]["faiss_max_dist"]
    sem_w    = cfg["retrieval"]["faiss_semantic_weight"]
    kw_w     = cfg["retrieval"]["faiss_keyword_weight"]
    fetch_k  = cfg["retrieval"]["mmr_fetch_k"]

    expanded   = expand_query(question)   # no expansion_store in eval — that's fine
    candidates = vector_db.similarity_search_with_score(expanded, k=fetch_k)
    kws        = _keywords(question)

    scored = []
    for doc, dist in candidates:
        if dist >= max_dist:
            continue
        sem   = 1.0 / (1.0 + dist)
        kw    = _keyword_score(doc.page_content, kws)
        score = sem_w * sem + kw_w * kw
        scored.append(score)

    return max(scored) if scored else 0.0


# ══════════════════════════════════════════════════════════════
# EVALUATION LOOP
# ══════════════════════════════════════════════════════════════

def evaluate(vector_db) -> list[dict]:
    """
    For each threshold, compute precision / recall / F1 over the test set.

    Definitions:
      TP — relevant question, confidence >= threshold  (correctly served from FAISS)
      FP — irrelevant question, confidence >= threshold (incorrectly served from FAISS)
      FN — relevant question, confidence < threshold   (incorrectly sent to fallback)
      TN — irrelevant question, confidence < threshold (correctly sent to fallback)

      precision = TP / (TP + FP)   how many FAISS answers were actually good?
      recall    = TP / (TP + FN)   how many good questions were answered by FAISS?
      f1        = harmonic mean of precision and recall
    """
    print(f"Evaluating {len(TEST_QUESTIONS)} questions...")
    confidences: list[tuple[float, bool]] = []
    for i, (q, is_rel) in enumerate(TEST_QUESTIONS, 1):
        conf = get_confidence(q, vector_db)
        confidences.append((conf, is_rel))
        print(f"  [{i:2d}/{len(TEST_QUESTIONS)}]  conf={conf:.4f}  relevant={is_rel}  {q[:55]}")

    print()
    results = []
    for thresh in THRESHOLDS:
        tp = sum(1 for c, r in confidences if c >= thresh and r)
        fp = sum(1 for c, r in confidences if c >= thresh and not r)
        fn = sum(1 for c, r in confidences if c < thresh  and r)
        tn = sum(1 for c, r in confidences if c < thresh  and not r)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        avg_conf  = (sum(c for c, _ in confidences) / len(confidences))

        results.append({
            "threshold": thresh,
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "avg_confidence": round(avg_conf, 4),
        })
    return results


# ══════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════

def print_table(results: list[dict]):
    headers = ["Threshold", "Precision", "Recall", "F1", "TP", "FP", "FN", "TN", "Avg Conf"]
    rows    = [
        [r["threshold"], r["precision"], r["recall"], r["f1"],
         r["tp"], r["fp"], r["fn"], r["tn"], r["avg_confidence"]]
        for r in results
    ]
    try:
        from tabulate import tabulate
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    except ImportError:
        # Plain text fallback
        print("  ".join(f"{h:>10}" for h in headers))
        print("  ".join("-" * 10 for _ in headers))
        for row in rows:
            print("  ".join(f"{v:>10}" for v in row))


def save_csv(results: list[dict]):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {OUTPUT_CSV}")


def recommend(results: list[dict]):
    """Print the threshold with the highest F1 score."""
    best = max(results, key=lambda r: (r["f1"], r["recall"]))
    current = cfg["retrieval"]["faiss_confidence_threshold"]
    print("\n" + "═" * 55)
    print(f"  Current threshold in config.yaml : {current}")
    print(f"  Recommended threshold (best F1)  : {best['threshold']}")
    print(f"  F1={best['f1']}  Precision={best['precision']}  Recall={best['recall']}")
    print("═" * 55)
    if best["threshold"] != current:
        print(f"\n  To apply, update config.yaml:")
        print(f"      retrieval:")
        print(f"        faiss_confidence_threshold: {best['threshold']}")
    else:
        print(f"\n  Your current threshold is already optimal.")
    print()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not Path(FAISS_PATH).exists():
        print(f"ERROR: FAISS index not found at {FAISS_PATH}")
        print("Run this script from the Samridhi_AI project root directory.")
        sys.exit(1)

    if len(TEST_QUESTIONS) < 10:
        print("WARNING: fewer than 10 test questions — results may not be reliable.")
        print("         Add more questions to the TEST_QUESTIONS list.\n")

    vector_db = load_index()
    results   = evaluate(vector_db)

    print("\n" + "═" * 55)
    print("  THRESHOLD EVALUATION RESULTS")
    print("═" * 55 + "\n")
    print_table(results)
    save_csv(results)
    recommend(results)
