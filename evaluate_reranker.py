"""
Phase 5C: Reranker Evaluation
Compares baseline pgvector retrieval ranking against cross-encoder neural reranking
(BAAI/bge-reranker-base) on the exact same candidate sets.
Computes Recall@5, MRR, MAP, rank deltas, and candidate set completeness.
"""
import os
import json
import argparse
from typing import List, Dict, Any
import torch
from rag import embed
from db import search_chunks, get_db_stats
from reranker import reranker

def compute_recall_at_k(expected: List[int], retrieved: List[int], k: int) -> float:
    if not expected:
        return 0.0
    matched = set(expected) & set(retrieved[:k])
    return len(matched) / len(expected)

def compute_reciprocal_rank(expected: List[int], retrieved: List[int]) -> float:
    expected_set = set(expected)
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in expected_set:
            return 1.0 / rank
    return 0.0

def compute_average_precision(expected: List[int], retrieved: List[int]) -> float:
    if not expected:
        return 0.0
    expected_set = set(expected)
    hits = 0
    sum_precisions = 0.0
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in expected_set:
            hits += 1
            sum_precisions += hits / rank
    return sum_precisions / len(expected)

def rerank_candidates(query: str, chunks: list) -> List[tuple]:
    """Runs cross-encoder reranker and returns list of (chunk, score) sorted descending."""
    if not chunks:
        return []
    pairs = [(query, getattr(c, "content", str(c))) for c in chunks]
    scores = reranker.predict(pairs, batch_size=8, show_progress_bar=False)
    ranked = list(zip(chunks, scores))
    ranked.sort(key=lambda x: float(x[1]), reverse=True)
    return ranked

def run_reranker_evaluation(
    dataset_path: str = "eval.json",
    tenant: str = "default",
    top_k: int = 5,
    output_path: str = "evaluation_results/reranker_results.json"
) -> Dict[str, Any]:
    if not os.path.exists(dataset_path):
        if os.path.exists("evaluation.json"):
            dataset_path = "evaluation.json"
        else:
            raise FileNotFoundError(f"Evaluation dataset {dataset_path} not found.")

    with open(dataset_path, "r", encoding="utf-8") as f:
        evaluation_data = json.load(f)

    db_stats = get_db_stats(tenant)
    corpus_size = db_stats.get("chunks_count", 0)

    results = []

    print(f"Starting Reranker Evaluation | Dataset: {dataset_path} | Tenant: {tenant}")
    print(f"Corpus size in DB: {corpus_size} chunks | Evaluation questions: {len(evaluation_data)}")
    print(f"Reranker Model: BAAI/bge-reranker-base (device: {reranker.model.device})")
    print("-" * 70)

    baseline_r5_list = []
    reranked_r5_list = []
    baseline_mrr_list = []
    reranked_mrr_list = []
    baseline_map_list = []
    reranked_map_list = []

    for idx, item in enumerate(evaluation_data, start=1):
        question = item["question"]
        expected_chunk_ids = item.get("Expected chunk") or item.get("expected_chunk") or item.get("expected_chunks")
        if isinstance(expected_chunk_ids, int):
            expected_chunk_ids = [expected_chunk_ids]

        # 1. Retrieve candidates from pgvector (baseline)
        vector = embed(question)
        candidates = search_chunks(
            tenant=tenant,
            embedding=vector,
            limit=20
        )

        baseline_ids = [c.id for c in candidates]
        baseline_scores = [round(float(c.score), 4) for c in candidates]

        # Verify candidate completeness
        missing_from_candidates = [cid for cid in expected_chunk_ids if cid not in baseline_ids]

        # 2. Run neural cross-encoder reranking
        ranked_pairs = rerank_candidates(question, candidates)
        reranked_ids = [c.id for c, _ in ranked_pairs]
        reranker_scores = [round(float(s), 4) for _, s in ranked_pairs]

        # Baseline metrics (evaluated on top_k)
        b_r5 = compute_recall_at_k(expected_chunk_ids, baseline_ids, top_k)
        b_rr = compute_reciprocal_rank(expected_chunk_ids, baseline_ids)
        b_ap = compute_average_precision(expected_chunk_ids, baseline_ids)

        # Reranked metrics
        r_r5 = compute_recall_at_k(expected_chunk_ids, reranked_ids, top_k)
        r_rr = compute_reciprocal_rank(expected_chunk_ids, reranked_ids)
        r_ap = compute_average_precision(expected_chunk_ids, reranked_ids)

        baseline_r5_list.append(b_r5)
        reranked_r5_list.append(r_r5)
        baseline_mrr_list.append(b_rr)
        reranked_mrr_list.append(r_rr)
        baseline_map_list.append(b_ap)
        reranked_map_list.append(r_ap)

        # Compute rank movements for expected chunks
        chunk_movements = []
        for cid in expected_chunk_ids:
            b_rank = baseline_ids.index(cid) + 1 if cid in baseline_ids else None
            r_rank = reranked_ids.index(cid) + 1 if cid in reranked_ids else None
            movement = "not_in_candidates"
            shift = 0
            if b_rank and r_rank:
                shift = b_rank - r_rank  # positive means improved rank (moved up)
                if shift > 0:
                    movement = "improved"
                elif shift < 0:
                    movement = "declined"
                else:
                    movement = "unchanged"
            chunk_movements.append({
                "chunk_id": cid,
                "baseline_rank": b_rank,
                "reranked_rank": r_rank,
                "rank_shift": shift,
                "movement": movement
            })

        mrr_delta = r_rr - b_rr
        map_delta = r_ap - b_ap

        case_result = {
            "query_index": idx,
            "question": question,
            "expected_chunks": expected_chunk_ids,
            "missing_from_candidates": missing_from_candidates,
            "baseline": {
                "chunk_ids": baseline_ids,
                "scores": baseline_scores,
                "recall@5": b_r5,
                "reciprocal_rank": b_rr,
                "average_precision": b_ap
            },
            "reranked": {
                "chunk_ids": reranked_ids,
                "scores": reranker_scores,
                "recall@5": r_r5,
                "reciprocal_rank": r_rr,
                "average_precision": r_ap
            },
            "deltas": {
                "mrr_delta": mrr_delta,
                "map_delta": map_delta,
                "chunk_movements": chunk_movements
            }
        }
        results.append(case_result)

        delta_symbol = "=" if mrr_delta == 0 and map_delta == 0 else ("+" if map_delta > 0 or mrr_delta > 0 else "-")
        print(f"Q{idx:02d}: {question[:45]}...")
        print(f"     Baseline: {baseline_ids[:top_k]} (MRR={b_rr:.2f}, MAP={b_ap:.2f})")
        print(f"     Reranked: {reranked_ids[:top_k]} (MRR={r_rr:.2f}, MAP={r_ap:.2f}) [{delta_symbol} MRR delta: {mrr_delta:+.2f}, MAP delta: {map_delta:+.2f}]")

    total_q = len(evaluation_data)
    mean_baseline_r5 = sum(baseline_r5_list) / total_q
    mean_reranked_r5 = sum(reranked_r5_list) / total_q
    mean_baseline_mrr = sum(baseline_mrr_list) / total_q
    mean_reranked_mrr = sum(reranked_mrr_list) / total_q
    mean_baseline_map = sum(baseline_map_list) / total_q
    mean_reranked_map = sum(reranked_map_list) / total_q

    summary = {
        "tenant": tenant,
        "corpus_size": corpus_size,
        "total_questions": total_q,
        "model": "BAAI/bge-reranker-base",
        "comparison": {
            "baseline_recall@5": mean_baseline_r5,
            "reranked_recall@5": mean_reranked_r5,
            "recall@5_delta": mean_reranked_r5 - mean_baseline_r5,
            "baseline_mrr": mean_baseline_mrr,
            "reranked_mrr": mean_reranked_mrr,
            "mrr_delta": mean_reranked_mrr - mean_baseline_mrr,
            "baseline_map": mean_baseline_map,
            "reranked_map": mean_reranked_map,
            "map_delta": mean_reranked_map - mean_baseline_map
        },
        "details": results
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved reranker evaluation results to {output_path}")

    print("\n" + "=" * 70)
    print("PHASE 5C: RERANKER EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Corpus Size         : {corpus_size} chunks (Tenant: {tenant})")
    print(f"Questions Evaluated : {total_q}")
    print(f"Baseline Recall@5   : {mean_baseline_r5 * 100:.2f}%")
    print(f"Reranked Recall@5   : {mean_reranked_r5 * 100:.2f}% (Delta: {(mean_reranked_r5 - mean_baseline_r5) * 100:+.2f}%)")
    print(f"Baseline MRR        : {mean_baseline_mrr:.4f}")
    print(f"Reranked MRR        : {mean_reranked_mrr:.4f} (Delta: {mean_reranked_mrr - mean_baseline_mrr:+.4f})")
    print(f"Baseline MAP        : {mean_baseline_map:.4f}")
    print(f"Reranked MAP        : {mean_reranked_map:.4f} (Delta: {mean_reranked_map - mean_baseline_map:+.4f})")
    print("=" * 70)

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Cross-Encoder Reranker")
    parser.add_argument("--dataset", default="eval.json", help="Path to evaluation dataset")
    parser.add_argument("--tenant", default="default", help="Tenant ID")
    parser.add_argument("--top-k", type=int, default=5, help="Top K cutoff")
    parser.add_argument("--output", default="evaluation_results/reranker_results.json", help="Output path")
    args = parser.parse_args()

    run_reranker_evaluation(
        dataset_path=args.dataset,
        tenant=args.tenant,
        top_k=args.top_k,
        output_path=args.output
    )
