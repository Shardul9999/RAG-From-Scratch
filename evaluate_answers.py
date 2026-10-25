"""
Phase 5D: Answer Quality and Citation Evaluation
Executes live RAG generation pipeline (pgvector retrieval -> cross-encoder reranking -> Gemini generation),
bypassing Redis cache. Programmatically verifies citation validity and computes lexical overlap metrics.
"""
import os
import re
import time
import json
import string
import argparse
from typing import List, Dict, Any, Tuple
from rag import embed, generate_answer
from db import search_chunks, engine, text
from reranker import rerank

def normalize_text(s: str) -> str:
    """Lowercases, removes punctuation and excess whitespace for fair lexical comparison."""
    if not s:
        return ""
    s = s.lower()
    # Remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # Remove extra spaces
    return " ".join(s.split())

def compute_token_metrics(prediction: str, ground_truth: str) -> Dict[str, float]:
    """
    Computes exact match, token precision, recall, F1, and Jaccard similarity.
    Limitations: Measures surface lexical overlap, not semantic entailment or factual correctness.
    """
    norm_pred = normalize_text(prediction)
    norm_gt = normalize_text(ground_truth)

    exact_match = 1.0 if norm_pred == norm_gt else 0.0

    pred_tokens = norm_pred.split()
    gt_tokens = norm_gt.split()

    if not pred_tokens and not gt_tokens:
        return {"exact_match": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "jaccard": 1.0}
    if not pred_tokens or not gt_tokens:
        return {"exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0}

    pred_counter = set(pred_tokens)
    gt_counter = set(gt_tokens)
    common_tokens = pred_counter & gt_counter

    precision = len(common_tokens) / len(pred_counter) if pred_counter else 0.0
    recall = len(common_tokens) / len(gt_counter) if gt_counter else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    union_tokens = pred_counter | gt_counter
    jaccard = len(common_tokens) / len(union_tokens) if union_tokens else 0.0

    return {
        "exact_match": exact_match,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "jaccard": round(jaccard, 4)
    }

def verify_citations(
    citations_indices: List[int],
    reranked_chunks: list,
    tenant: str = "default"
) -> Dict[str, Any]:
    """
    Verifies that citations:
    1. Are valid 1-based indices into the supplied reranked candidates.
    2. Correspond to real chunks in the database for the given tenant.
    3. Match the chunk's stored title and source.
    """
    valid = True
    issues = []
    verified_citations = []

    if not citations_indices:
        # Check if empty citations is an issue (flagged for review)
        issues.append("Warning: No citations provided by LLM response.")

    # Fetch DB records for verification
    with engine.begin() as conn:
        db_rows = conn.execute(
            text("SELECT id, source, title FROM chunks WHERE tenant = :tenant"),
            {"tenant": tenant}
        ).fetchall()
        db_chunks = {r[0]: {"source": r[1], "title": r[2]} for r in db_rows}

    for cit in citations_indices:
        cit_record = {
            "citation_number": cit,
            "is_valid": True,
            "issues": []
        }

        if not isinstance(cit, int):
            cit_record["is_valid"] = False
            cit_record["issues"].append(f"Citation index is not an integer: {cit}")
            valid = False
            verified_citations.append(cit_record)
            continue

        idx = cit - 1
        if idx < 0 or idx >= len(reranked_chunks):
            cit_record["is_valid"] = False
            cit_record["issues"].append(f"Citation index [{cit}] is out of bounds (1..{len(reranked_chunks)}).")
            valid = False
            verified_citations.append(cit_record)
            continue

        chunk = reranked_chunks[idx]
        chunk_id = getattr(chunk, "id", None)
        cit_record["chunk_id"] = chunk_id
        cit_record["chunk_source"] = getattr(chunk, "source", None)
        cit_record["chunk_title"] = getattr(chunk, "title", None)

        if chunk_id not in db_chunks:
            cit_record["is_valid"] = False
            cit_record["issues"].append(f"Referenced chunk id={chunk_id} does not exist in database.")
            valid = False
        else:
            db_info = db_chunks[chunk_id]
            if db_info["source"] != cit_record["chunk_source"]:
                cit_record["is_valid"] = False
                cit_record["issues"].append(f"Source mismatch: expected '{db_info['source']}', got '{cit_record['chunk_source']}'.")
                valid = False
            if db_info["title"] != cit_record["chunk_title"]:
                cit_record["is_valid"] = False
                cit_record["issues"].append(f"Title mismatch: expected '{db_info['title']}', got '{cit_record['chunk_title']}'.")
                valid = False

        verified_citations.append(cit_record)

    return {
        "all_valid": valid and len(citations_indices) > 0,
        "issues": issues,
        "verified_citations": verified_citations
    }

def run_answer_evaluation(
    dataset_path: str = "eval.json",
    tenant: str = "default",
    similarity_threshold: float = 0.50,
    top_k_rerank: int = 5,
    enable_llm_judge: bool = False,
    output_path: str = "evaluation_results/answer_results.json"
) -> Dict[str, Any]:
    if not os.path.exists(dataset_path):
        if os.path.exists("evaluation.json"):
            dataset_path = "evaluation.json"
        else:
            raise FileNotFoundError(f"Evaluation dataset {dataset_path} not found.")

    with open(dataset_path, "r", encoding="utf-8") as f:
        evaluation_data = json.load(f)

    print(f"Starting Answer & Citation Evaluation | Dataset: {dataset_path} | Tenant: {tenant}")
    print(f"Questions: {len(evaluation_data)} | LLM Judge: {'ENABLED (external API)' if enable_llm_judge else 'DISABLED (by design)'}")
    print(f"Redis Semantic Cache: BYPASS (testing live retrieval & generation)")
    print("-" * 70)

    cases = []
    total_valid_citations = 0
    total_with_citations = 0
    f1_scores = []
    jaccard_scores = []
    em_scores = []

    for idx, item in enumerate(evaluation_data, start=1):
        question = item["question"]
        expected_chunks = item.get("Expected chunk") or item.get("expected_chunk") or item.get("expected_chunks")
        if isinstance(expected_chunks, int):
            expected_chunks = [expected_chunks]
        expected_answer = item.get("answer") or item.get("expected_answer")

        try:
            # 1. Query vector embedding
            vector = embed(question)

            # 2. pgvector candidate retrieval
            raw_results = search_chunks(
                tenant=tenant,
                embedding=vector,
                limit=20
            )

            # 3. Similarity threshold filter
            filtered_results = [r for r in raw_results if r.score >= similarity_threshold]

            # 4. Neural Cross-Encoder Reranking
            reranked_results = rerank(question, filtered_results, top_k=top_k_rerank)

            # 5. LLM Answer Generation with Rate-Limit Handling (Gemini 2.5 Flash Free Tier limit is 5 RPM)
            max_retries = 3
            rag_resp = None
            raw_llm_resp = None
            for attempt in range(max_retries):
                try:
                    rag_resp, raw_llm_resp = generate_answer(question, reranked_results)
                    break
                except Exception as ex:
                    err_msg = str(ex).lower()
                    if "429" in err_msg or "rate" in err_msg or "quota" in err_msg:
                        wait_sec = 35 + attempt * 10
                        print(f"     [RateLimit/Quota] Pausing {wait_sec}s for Gemini Free Tier window (attempt {attempt+1}/{max_retries})...")
                        time.sleep(wait_sec)
                    else:
                        raise ex

            if rag_resp is None:
                raise RuntimeError("Failed to generate answer after rate-limit retries.")

            # Pace between queries to respect Free Tier 5 RPM limit
            time.sleep(12)
            generated_text = rag_resp.answer
            raw_citations = rag_resp.citations

            # 6. Citation Verification
            cit_eval = verify_citations(raw_citations, reranked_results, tenant=tenant)
            if cit_eval["all_valid"]:
                total_valid_citations += 1
            if raw_citations:
                total_with_citations += 1

            # 7. Lexical overlap metrics
            metrics = compute_token_metrics(generated_text, expected_answer) if expected_answer else {}
            if metrics:
                em_scores.append(metrics["exact_match"])
                f1_scores.append(metrics["f1"])
                jaccard_scores.append(metrics["jaccard"])

            case_result = {
                "query_index": idx,
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_text,
                "raw_citations": raw_citations,
                "citation_evaluation": cit_eval,
                "retrieved_chunk_ids": [r.id for r in raw_results],
                "reranked_chunk_ids": [r.id for r in reranked_results],
                "lexical_metrics": metrics,
                "error": None
            }

            print(f"Q{idx:02d}: {question[:45]}...")
            print(f"     Generated: {generated_text[:65]}...")
            print(f"     Citations: {raw_citations} | Valid: {cit_eval['all_valid']} | F1: {metrics.get('f1', 0):.2f}")

        except Exception as e:
            case_result = {
                "query_index": idx,
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": None,
                "raw_citations": [],
                "citation_evaluation": {"all_valid": False, "issues": [str(e)], "verified_citations": []},
                "retrieved_chunk_ids": [],
                "reranked_chunk_ids": [],
                "lexical_metrics": {},
                "error": str(e)
            }
            print(f"Q{idx:02d}: ERROR: {e}")

        cases.append(case_result)

    total_q = len(evaluation_data)
    summary = {
        "tenant": tenant,
        "total_questions": total_q,
        "citation_metrics": {
            "questions_with_citations": total_with_citations,
            "questions_with_valid_citations": total_valid_citations,
            "citation_validity_rate": (total_valid_citations / total_q) if total_q else 0.0
        },
        "lexical_metrics_summary": {
            "exact_match_rate": sum(em_scores) / len(em_scores) if em_scores else 0.0,
            "mean_token_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            "mean_jaccard_similarity": sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0,
            "limitations_note": (
                "Lexical metrics (Exact Match, Token F1, Jaccard) reflect surface text overlap, not semantic "
                "or factual correctness. A generated answer can be 100% factually accurate while using different "
                "syntactic wording than the reference answer."
            )
        },
        "llm_judge_enabled": enable_llm_judge,
        "details": cases
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved answer evaluation results to {output_path}")

    print("\n" + "=" * 70)
    print("PHASE 5D: ANSWER QUALITY & CITATION EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Questions Evaluated    : {total_q}")
    print(f"Citations Present Rate       : {total_with_citations}/{total_q} ({total_with_citations / total_q * 100:.1f}%)")
    print(f"Citations 100% Valid Rate    : {total_valid_citations}/{total_q} ({total_valid_citations / total_q * 100:.1f}%)")
    print(f"Mean Token F1 (Overlap)      : {summary['lexical_metrics_summary']['mean_token_f1']:.4f}")
    print(f"Mean Jaccard Similarity      : {summary['lexical_metrics_summary']['mean_jaccard_similarity']:.4f}")
    print(f"Exact Match Rate             : {summary['lexical_metrics_summary']['exact_match_rate'] * 100:.2f}%")
    print("=" * 70)

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Answer Quality and Citations")
    parser.add_argument("--dataset", default="eval.json", help="Path to evaluation dataset")
    parser.add_argument("--tenant", default="default", help="Tenant ID")
    parser.add_argument("--threshold", type=float, default=0.50, help="Similarity threshold")
    parser.add_argument("--top-k", type=int, default=5, help="Top K rerank")
    parser.add_argument("--enable-llm-judge", action="store_true", default=False, help="Enable optional LLM judge")
    parser.add_argument("--output", default="evaluation_results/answer_results.json", help="Output path")
    args = parser.parse_args()

    run_answer_evaluation(
        dataset_path=args.dataset,
        tenant=args.tenant,
        similarity_threshold=args.threshold,
        top_k_rerank=args.top_k,
        enable_llm_judge=args.enable_llm_judge,
        output_path=args.output
    )
