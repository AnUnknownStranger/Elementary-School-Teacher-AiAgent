# evaluation.py
import json
from typing import List, Dict, Any, Optional, Tuple
from retriever_logic import retrieve


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load evaluation data from a JSONL file (one JSON object per line)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def hit(doc_md: Dict[str, Any],
        gold_sources: Optional[List[str]],
        gold_pages: Optional[List[int]]) -> bool:
    """Check whether a retrieved document matches gold reference."""
    src = str(doc_md.get("source", ""))
    page = doc_md.get("page", doc_md.get("page_number", None))

    try:
        page = int(page) if page is not None else None
    except Exception:
        page = None

    source_match = True if not gold_sources else any(gs in src for gs in gold_sources)
    # If gold_pages not provided, do not enforce page matching
    page_match = True if (not gold_pages or page is None) else (page in gold_pages)

    return source_match and page_match


def _eval_once(
    data: List[Dict[str, Any]],
    k: int,
    mode: str,
    keep_subject_when_unconditioned: bool = True,
    debug: bool = False,
    max_debug_cases: int = 5
) -> Dict[str, float]:
    """
    mode:
      - "conditioned": use grade/subject from eval_set
      - "unconditioned": force grade=None (optionally keep subject)
    """
    total = len(data)
    recall = 0
    mrr_sum = 0.0

    debug_printed = 0

    for idx, ex in enumerate(data):
        question = ex["question"]
        grade = ex.get("grade", None)
        subject = ex.get("subject", None)
        gold_sources = ex.get("gold_sources", None)
        gold_pages = ex.get("gold_pages", None)

        if mode == "unconditioned":
            grade = None
            if not keep_subject_when_unconditioned:
                subject = None

        docs = retrieve(question, grade=grade, subject=subject, k=k)

        found_rank = None
        for i, d in enumerate(docs, start=1):
            if hit(d.metadata or {}, gold_sources, gold_pages):
                found_rank = i
                break

        if found_rank is not None:
            recall += 1
            mrr_sum += 1.0 / found_rank

        # Optional debug logging (print a few examples only)
        if debug and debug_printed < max_debug_cases:
            print("\n" + "=" * 80)
            print(f"[{mode}] case={idx+1}")
            print("Q:", question)
            print("grade:", ex.get("grade", None), "subject:", ex.get("subject", None))
            print("used_grade:", grade, "used_subject:", subject)
            print("gold_sources:", gold_sources, "gold_pages:", gold_pages)
            print(f"top-{k} retrieved sources:")
            for j, d in enumerate(docs, start=1):
                md = d.metadata or {}
                print(f"  {j}. source={md.get('source')} page={md.get('page', md.get('page_number'))}")
            print("HIT rank:", found_rank)
            debug_printed += 1

    return {
        "n": total,
        f"recall@{k}": recall / total if total else 0.0,
        f"mrr@{k}": mrr_sum / total if total else 0.0,
    }


def evaluate(
    eval_path: str,
    ks: List[int],
    keep_subject_when_unconditioned: bool = True,
    debug: bool = False,
    max_debug_cases: int = 5
) -> Dict[str, Any]:
    """
    Run evaluation for multiple k values and return results for:
      - conditioned
      - unconditioned (grade=None)
    """
    data = load_jsonl(eval_path)

    results = {
        "conditioned": {},
        "unconditioned": {},
    }

    for k in ks:
        results["conditioned"][str(k)] = _eval_once(
            data, k=k, mode="conditioned",
            keep_subject_when_unconditioned=keep_subject_when_unconditioned,
            debug=debug, max_debug_cases=max_debug_cases
        )
        results["unconditioned"][str(k)] = _eval_once(
            data, k=k, mode="unconditioned",
            keep_subject_when_unconditioned=keep_subject_when_unconditioned,
            debug=debug, max_debug_cases=max_debug_cases
        )

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True, help="Path to eval_set.jsonl")
    parser.add_argument("--k", type=int, nargs="+", default=[5], help="One or more k values, e.g. --k 3 5 8")
    parser.add_argument("--debug", action="store_true", help="Print debug info for a few cases")
    parser.add_argument("--max_debug_cases", type=int, default=5)
    parser.add_argument(
        "--drop_subject_in_unconditioned",
        action="store_true",
        help="If set, unconditioned mode will also set subject=None"
    )
    args = parser.parse_args()

    results = evaluate(
        eval_path=args.eval,
        ks=args.k,
        keep_subject_when_unconditioned=(not args.drop_subject_in_unconditioned),
        debug=args.debug,
        max_debug_cases=args.max_debug_cases
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))