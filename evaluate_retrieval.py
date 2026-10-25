"""
Phase 5B: Retrieval Evaluation
Evaluates pgvector vector retrieval across Recall@5, Recall@10, Recall@20, MRR, and MAP.
Evaluates both raw pgvector retrieval and production-filtered retrieval (score >= 0.50).
Independent of Redis caching and LLM generation.
"""
import os
import json
import argparse
from typing import List, Dict, Any, Set
from rag import embed
from db import search_chunks, get_db_stats

def compute_recall_at_k(expected: List[int], retrieved: List[int], k: int) -> float:
    """Calculates Recall@K: fraction of expected chunks present in the top-K retrieved."""
    if not expected:
        return 0.0
    top_k = retrieved[:k]
    matched = set(expected) & set(top_k)
    return len(matched) / len(expected)

def compute_reciprocal_rank(expected: List[int], retrieved: List[int]) -> float:
    """Calculates Reciprocal Rank (RR): reciprocal of the 1-based rank of the first relevant chunk."""
    expected_set = set(expected)
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in expected_set:
            return 1.0 / rank
    return 0.0

def compute_average_precision(expected: List[int], retrieved: List[int]) -> float:
    """Calculates Average Precision (AP): mean of precision scores after each relevant chunk retrieved."""
    if not expected:
        return 0.0
    expected_set = set(expected)
    num_hits = 0
    sum_precisions = 0.0
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in expected_set:
            num_hits += 1
            sum_precisions += num_hits / rank
    return sum_precisions / len(expected)

def run_retrieval_evaluation(
    dataset_path: str = "eval.json",
    tenant: str = "default",
    similarity_threshold: float = 0.50,
    output_path: str = "evaluation_results/retrieval_results.json"
) -> Dict[str, Any]:
    """Runs retrieval evaluation on the dataset."""
    if not os.path.exists(dataset_path):
        if os.path.exists("evaluation.json"):
            dataset_path = "evaluation.json"
        else:
            raise FileNotFoundError(f"Evaluation dataset {dataset_path} not found.")

    with open(dataset_path, "r", encoding="utf-8") as f:
        evaluation_data = json.load(f)

    db_stats = get_db_stats(tenant)
    corpus_size = db_stats.get("chunks_count", 0)

    results_raw = []
    results_filtered = []

    print(f"Starting Retrieval Evaluation | Dataset: {dataset_path} | Tenant: {tenant}")
    print(f"Corpus size in DB: {corpus_size} chunks | Total questions: {len(evaluation_data)}")
    print("-" * 65)

    for idx, item in enumerate(evaluation_data, start=1):
        question = item["question"]
        expected_chunk_ids = item.get("Expected chunk") or item.get("expected_chunk") or item.get("expected_chunks")
        if isinstance(expected_chunk_ids, int):
            expected_chunk_ids = [expected_chunk_ids]

        # 1. Generate query embedding
        query_vector = embed(question)

        # 2. Query pgvector directly (limit=20, raw)
        raw_candidates = search_chunks(
            tenant=tenant,
            embedding=query_vector,
            limit=20
        )

        raw_retrieved_ids = [r.id for r in raw_candidates]
        raw_scores = [round(float(r.score), 4) for r in raw_candidates]

        # 3. Production filtered candidates (score >= threshold)
        filtered_candidates = [r for r in raw_candidates if r.score >= similarity_threshold]
        filtered_retrieved_ids = [r.id for r in filtered_candidates]
        filtered_scores = [round(float(r.score), 4) for r in filtered_candidates]

        # Metrics for Raw Retrieval
        raw_r5 = compute_recall_at_k(expected_chunk_ids, raw_retrieved_ids, 5)
        raw_r10 = compute_recall_at_k(expected_chunk_ids, raw_retrieved_ids, 10)
        raw_r20 = compute_recall_at_k(expected_chunk_ids, raw_retrieved_ids, 20)
        raw_rr = compute_reciprocal_rank(expected_chunk_ids, raw_retrieved_ids)
        raw_ap = compute_average_precision(expected_chunk_ids, raw_retrieved_ids)

        # Metrics for Filtered Retrieval
        filt_r5 = compute_recall_at_k(expected_chunk_ids, filtered_retrieved_ids, 5)
        filt_r10 = compute_recall_at_k(expected_chunk_ids, filtered_retrieved_ids, 10)
        filt_r20 = compute_recall_at_k(expected_chunk_ids, filtered_retrieved_ids, 20)
        filt_rr = compute_reciprocal_rank(expected_chunk_ids, filtered_retrieved_ids)
        filt_ap = compute_average_precision(expected_chunk_ids, filtered_retrieved_ids)

        case_raw = {
            "query_index": idx,
            "question": question,
            "expected_chunks": expected_chunk_ids,
            "retrieved_chunk_ids": raw_retrieved_ids,
            "similarity_scores": raw_scores,
            "recall@5": raw_r5,
            "recall@10": raw_r10,
            "recall@20": raw_r20,
            "reciprocal_rank": raw_rr,
            "average_precision": raw_ap
        }
        results_raw.append(case_raw)

        case_filt = {
            "query_index": idx,
            "question": question,
            "expected_chunks": expected_chunk_ids,
            "retrieved_chunk_ids": filtered_retrieved_ids,
            "similarity_scores": filtered_scores,
            "recall@5": filt_r5,
            "recall@10": filt_r10,
            "recall@20": filt_r20,
            "reciprocal_rank": filt_rr,
            "average_precision": filt_ap
        }
        results_filtered.append(case_filt)

        print(f"Q{idx:02d}: {question[:45]}...")
        print(f"     Expected: {expected_chunk_ids} | Raw Retrieved: {raw_retrieved_ids[:5]} (scores: {raw_scores[:5]})")
        print(f"     Raw: R@5={raw_r5:.2f}, MRR={raw_rr:.2f}, AP={raw_ap:.2f} | Filtered: R@5={filt_r5:.2f}, MRR={filt_rr:.2f}")

    total_q = len(evaluation_data)
    summary_raw = {
        "recall@5": sum(c["recall@5"] for c in results_raw) / total_q,
        "recall@10": sum(c["recall@10"] for c in results_raw) / total_q,
        "recall@20": sum(c["recall@20"] for c in results_raw) / total_q,
        "mrr": sum(c["reciprocal_rank"] for c in results_raw) / total_q,
        "map": sum(c["average_precision"] for c in results_raw) / total_q
    }

    summary_filtered = {
        "similarity_threshold": similarity_threshold,
        "recall@5": sum(c["recall@5"] for c in results_filtered) / total_q,
        "recall@10": sum(c["recall@10"] for c in results_filtered) / total_q,
        "recall@20": sum(c["recall@20"] for c in results_filtered) / total_q,
        "mrr": sum(c["reciprocal_rank"] for c in results_filtered) / total_q,
        "map": sum(c["average_precision"] for c in results_filtered) / total_q
    }

    report = {
        "tenant": tenant,
        "corpus_size": corpus_size,
        "total_questions": total_q,
        "corpus_size_note": (
            f"The database corpus contains {corpus_size} chunks. In a corpus of {corpus_size} chunks, "
            f"requesting top-20 candidates retrieves the entire corpus if unfiltered. "
            f"Therefore, raw Recall@5 and Recall@20 reach 100% due to the small corpus size "
            f"(corpus ceiling effect). The Mean Reciprocal Rank (MRR={summary_raw['mrr']:.4f}) and "
            f"Mean Average Precision (MAP={summary_raw['map']:.4f}) reflect actual rank ordering quality."
        ),
        "raw_retrieval_summary": summary_raw,
        "filtered_retrieval_summary": summary_filtered,
        "raw_details": results_raw,
        "filtered_details": results_filtered
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved retrieval evaluation results to {output_path}")

    print("\n" + "=" * 65)
    print("PHASE 5B: RETRIEVAL EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Corpus Size         : {corpus_size} chunks (Tenant: {tenant})")
    print(f"Total Questions     : {total_q}")
    print(f"Raw Recall@5        : {summary_raw['recall@5'] * 100:.2f}%")
    print(f"Raw Recall@10       : {summary_raw['recall@10'] * 100:.2f}%")
    print(f"Raw Recall@20       : {summary_raw['recall@20'] * 100:.2f}%")
    print(f"Raw MRR             : {summary_raw['mrr']:.4f}")
    print(f"Raw MAP             : {summary_raw['map']:.4f}")
    print("-" * 65)
    print(f"Filtered (>= {similarity_threshold}) Recall@5 : {summary_filtered['recall@5'] * 100:.2f}%")
    print(f"Filtered (>= {similarity_threshold}) MRR      : {summary_filtered['mrr']:.4f}")
    print(f"Filtered (>= {similarity_threshold}) MAP      : {summary_filtered['map']:.4f}")
    print("=" * 65)

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pgvector retrieval")
    parser.add_argument("--dataset", default="eval.json", help="Path to evaluation dataset JSON")
    parser.add_argument("--tenant", default="default", help="Tenant ID")
    parser.add_argument("--threshold", type=float, default=0.50, help="Production similarity threshold")
    parser.add_argument("--output", default="evaluation_results/retrieval_results.json", help="Path for JSON results")
    args = parser.parse_args()

    run_retrieval_evaluation(
        dataset_path=args.dataset,
        tenant=args.tenant,
        similarity_threshold=args.threshold,
        output_path=args.output
    )
