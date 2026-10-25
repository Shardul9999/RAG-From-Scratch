"""
Phase 5E: Latency & Semantic Cache Evaluation
Measures latency breakdown across RAG pipeline stages using time.perf_counter().
Distinguishes cold-start vs warmed-up trials, measures cache hit vs miss latency,
and benchmarks semantic cache hit rate in a non-destructive controlled test.
"""
import os
import json
import time
import argparse
import statistics
from typing import List, Dict, Any, Tuple
import numpy as np

from rag import embed, generate_answer
from db import search_chunks
from reranker import rerank
from cache import get_cached_answer, store_answer, redis_client, CACHE_THRESHOLD

def calculate_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
    """Computes mean, median, and p95 latency in milliseconds."""
    if not latencies_ms:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "samples": 0}
    sorted_l = sorted(latencies_ms)
    p95_idx = int(0.95 * len(sorted_l))
    p95_val = sorted_l[min(p95_idx, len(sorted_l) - 1)]
    return {
        "mean_ms": round(statistics.mean(latencies_ms), 2),
        "median_ms": round(statistics.median(latencies_ms), 2),
        "p95_ms": round(p95_val, 2),
        "min_ms": round(min(latencies_ms), 2),
        "max_ms": round(max(latencies_ms), 2),
        "samples": len(latencies_ms)
    }

def benchmark_embedding(sample_query: str, warm_trials: int = 5) -> Dict[str, Any]:
    print("[1/5] Benchmarking Query Embedding (Gemini Embedding 2)...")
    # Cold start
    t0 = time.perf_counter()
    _ = embed(sample_query)
    cold_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Warmed trials
    latencies = []
    for i in range(warm_trials):
        t0 = time.perf_counter()
        _ = embed(sample_query)
        latencies.append((time.perf_counter() - t0) * 1000)
        time.sleep(1)  # small pause to be gentle on API

    stats = calculate_percentiles(latencies)
    stats["cold_start_ms"] = cold_ms
    print(f"      Cold Start: {cold_ms}ms | Warmed Mean: {stats['mean_ms']}ms | Median: {stats['median_ms']}ms")
    return stats

def benchmark_pgvector(sample_query: str, tenant: str = "default", trials: int = 10) -> Dict[str, Any]:
    print("[2/5] Benchmarking pgvector Retrieval...")
    # Pre-embed once so we isolate DB retrieval time
    vector = embed(sample_query)

    # Cold start
    t0 = time.perf_counter()
    candidates = search_chunks(tenant=tenant, embedding=vector, limit=20)
    cold_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Warmed trials
    latencies = []
    for _ in range(trials):
        t0 = time.perf_counter()
        _ = search_chunks(tenant=tenant, embedding=vector, limit=20)
        latencies.append((time.perf_counter() - t0) * 1000)

    stats = calculate_percentiles(latencies)
    stats["cold_start_ms"] = cold_ms
    print(f"      Cold Start: {cold_ms}ms | Warmed Mean: {stats['mean_ms']}ms | Median: {stats['median_ms']}ms")
    return stats, candidates

def benchmark_reranker(sample_query: str, candidates: list, trials: int = 5) -> Dict[str, Any]:
    print("[3/5] Benchmarking Neural Cross-Encoder Reranker...")
    # Cold start
    t0 = time.perf_counter()
    _ = rerank(sample_query, candidates, top_k=5)
    cold_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Warmed trials
    latencies = []
    for _ in range(trials):
        t0 = time.perf_counter()
        _ = rerank(sample_query, candidates, top_k=5)
        latencies.append((time.perf_counter() - t0) * 1000)

    stats = calculate_percentiles(latencies)
    stats["cold_start_ms"] = cold_ms
    print(f"      Cold Start: {cold_ms}ms | Warmed Mean: {stats['mean_ms']}ms | Median: {stats['median_ms']}ms")
    return stats

def benchmark_llm(sample_query: str, reranked_chunks: list) -> Dict[str, Any]:
    print("[4/5] Benchmarking LLM Generation (Gemini 2.5 Flash via LiteLLM)...")
    latencies = []
    # Test 1 live generation trial to respect 5 RPM Free Tier limit
    t0 = time.perf_counter()
    try:
        _, _ = generate_answer(sample_query, reranked_chunks)
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)
        print(f"      Measured Generation Latency: {lat_ms:.2f}ms")
    except Exception as e:
        print(f"      Generation trial encountered: {e}")

    stats = calculate_percentiles(latencies)
    return stats

def benchmark_semantic_cache(sample_query: str) -> Dict[str, Any]:
    print("[5/5] Benchmarking Redis Semantic Cache (Hit vs Miss & Similarity Gate)...")
    # Non-destructive test key with unique prefix
    test_key = "semantic_cache:eval_perf_probe_isolated"
    query_vector = embed(sample_query)

    # 1. Measure cache miss lookup latency (before key is stored)
    t0 = time.perf_counter()
    res_miss = get_cached_answer(query_vector)
    miss_latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    # 2. Store isolated probe data (TTL = 60s)
    probe_response = {"answer": "Evaluation probe response", "citations": []}
    probe_data = json.dumps({
        "query": sample_query,
        "embedding": query_vector,
        "response": probe_response
    })
    redis_client.setex(test_key, 60, probe_data)

    # 3. Measure cache hit lookup latency (exact vector match)
    hit_latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        res_hit = get_cached_answer(query_vector)
        hit_latencies.append((time.perf_counter() - t0) * 1000)

    hit_stats = calculate_percentiles(hit_latencies)

    # 4. Controlled Sequence for Hit Rate Verification:
    # Query A (exact probe): should hit
    # Query A_similar (minor perturbation/paraphrase): should hit if >= 0.90
    # Query B (completely orthogonal query vector): should miss (< 0.90)
    orthogonal_query = "Recipe for chocolate chip cookies with sea salt"
    orthogonal_vec = embed(orthogonal_query)

    slightly_perturbed_vec = [v + 0.0001 for v in query_vector]
    # Re-normalize
    norm = np.linalg.norm(slightly_perturbed_vec)
    slightly_perturbed_vec = (np.array(slightly_perturbed_vec) / norm).tolist()

    seq_results = [
        {"query_type": "exact_match", "expected_hit": True, "actual_hit": get_cached_answer(query_vector) is not None},
        {"query_type": "high_similarity_paraphrase", "expected_hit": True, "actual_hit": get_cached_answer(slightly_perturbed_vec) is not None},
        {"query_type": "unrelated_query", "expected_hit": False, "actual_hit": get_cached_answer(orthogonal_vec) is not None}
    ]

    # Non-destructive clean up of ONLY our test probe key
    redis_client.delete(test_key)

    hits = sum(1 for s in seq_results if s["actual_hit"])
    expected_matches = sum(1 for s in seq_results if s["actual_hit"] == s["expected_hit"])

    print(f"      Cache Miss Lookup Latency : {miss_latency_ms}ms")
    print(f"      Cache Hit Lookup Latency  : Mean={hit_stats['mean_ms']}ms | Median={hit_stats['median_ms']}ms")
    print(f"      Controlled Test Accuracy  : {expected_matches}/{len(seq_results)} semantic routing assertions verified")

    return {
        "miss_lookup_latency_ms": miss_latency_ms,
        "hit_lookup_stats": hit_stats,
        "threshold_tested": CACHE_THRESHOLD,
        "controlled_sequence": seq_results,
        "sequence_hit_rate": hits / len(seq_results),
        "sequence_accuracy": expected_matches / len(seq_results)
    }

def run_performance_evaluation(
    sample_query: str = "What is Shardul Hingane's CGPA in B.Tech?",
    tenant: str = "default",
    output_path: str = "evaluation_results/performance_results.json"
) -> Dict[str, Any]:
    print("=" * 70)
    print("PHASE 5E: SYSTEM LATENCY & SEMANTIC CACHE EVALUATION")
    print("=" * 70)

    # 1. Embeddings
    embed_stats = benchmark_embedding(sample_query)

    # 2. pgvector
    pg_stats, candidates = benchmark_pgvector(sample_query, tenant=tenant)

    # 3. Reranker
    rerank_stats = benchmark_reranker(sample_query, candidates)

    # 4. LLM Generation
    reranked = rerank(sample_query, candidates, top_k=5)
    llm_stats = benchmark_llm(sample_query, reranked)

    # 5. Semantic Cache
    cache_stats = benchmark_semantic_cache(sample_query)

    # Estimated uncached end-to-end latency
    e2e_warmed_est = round(
        embed_stats["mean_ms"] +
        pg_stats["mean_ms"] +
        rerank_stats["mean_ms"] +
        llm_stats["mean_ms"], 2
    )

    report = {
        "tenant": tenant,
        "sample_query": sample_query,
        "benchmarks": {
            "query_embedding_gemini": embed_stats,
            "pgvector_retrieval": pg_stats,
            "cross_encoder_reranker": rerank_stats,
            "llm_generation_gemini_flash": llm_stats,
            "semantic_cache": cache_stats
        },
        "end_to_end_summary": {
            "estimated_uncached_warmed_mean_ms": e2e_warmed_est,
            "cached_lookup_warmed_mean_ms": cache_stats["hit_lookup_stats"]["mean_ms"],
            "speedup_factor_from_cache": round(e2e_warmed_est / max(cache_stats["hit_lookup_stats"]["mean_ms"], 0.1), 1)
        }
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved performance evaluation results to {output_path}")

    print("\n" + "=" * 70)
    print("PHASE 5E: PERFORMANCE EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Query Embedding Latency (Mean)  : {embed_stats['mean_ms']} ms")
    print(f"pgvector Retrieval Latency (Mean): {pg_stats['mean_ms']} ms")
    print(f"Cross-Encoder Reranker (Mean)   : {rerank_stats['mean_ms']} ms")
    print(f"LLM Generation Latency (Mean)   : {llm_stats['mean_ms']} ms")
    print(f"Total Uncached Pipeline (Est)   : {e2e_warmed_est} ms")
    print(f"Semantic Cache Hit Latency      : {cache_stats['hit_lookup_stats']['mean_ms']} ms")
    print(f"Cache Speedup Factor            : ~{report['end_to_end_summary']['speedup_factor_from_cache']}x")
    print("=" * 70)

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG Performance and Semantic Cache")
    parser.add_argument("--query", default="What is Shardul Hingane's CGPA in B.Tech?", help="Sample query")
    parser.add_argument("--tenant", default="default", help="Tenant ID")
    parser.add_argument("--output", default="evaluation_results/performance_results.json", help="Output path")
    args = parser.parse_args()

    run_performance_evaluation(
        sample_query=args.query,
        tenant=args.tenant,
        output_path=args.output
    )
