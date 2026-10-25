"""
Unit tests for RAG evaluation metrics and citation verification logic.
Tests Recall@K, MRR, MAP, Token F1, Exact Match, and Citation Validation.
No external API calls are made in these deterministic tests.
"""
import unittest
from evaluate_retrieval import (
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_average_precision
)
from evaluate_answers import (
    normalize_text,
    compute_token_metrics,
    verify_citations
)

class DummyChunk:
    def __init__(self, chunk_id, source="temp/My_Resume.pdf", title="temp/My_Resume.pdf"):
        self.id = chunk_id
        self.source = source
        self.title = title

class TestRetrievalMetrics(unittest.TestCase):
    def test_recall_at_k_perfect(self):
        expected = [1, 2]
        retrieved = [1, 2, 3, 4, 5]
        self.assertEqual(compute_recall_at_k(expected, retrieved, 2), 1.0)
        self.assertEqual(compute_recall_at_k(expected, retrieved, 5), 1.0)

    def test_recall_at_k_partial(self):
        expected = [1, 2]
        retrieved = [1, 3, 4, 5]
        self.assertEqual(compute_recall_at_k(expected, retrieved, 2), 0.5)
        self.assertEqual(compute_recall_at_k(expected, retrieved, 4), 0.5)

    def test_recall_at_k_zero(self):
        expected = [1, 2]
        retrieved = [3, 4, 5]
        self.assertEqual(compute_recall_at_k(expected, retrieved, 3), 0.0)

    def test_recall_at_k_empty_expected(self):
        self.assertEqual(compute_recall_at_k([], [1, 2, 3], 5), 0.0)

    def test_reciprocal_rank_first_hit(self):
        expected = [2, 3]
        retrieved = [2, 1, 4]
        self.assertEqual(compute_reciprocal_rank(expected, retrieved), 1.0)

    def test_reciprocal_rank_second_hit(self):
        expected = [2, 3]
        retrieved = [1, 3, 4]
        self.assertEqual(compute_reciprocal_rank(expected, retrieved), 0.5)

    def test_reciprocal_rank_third_hit(self):
        expected = [2]
        retrieved = [4, 5, 2, 1]
        self.assertAlmostEqual(compute_reciprocal_rank(expected, retrieved), 1.0 / 3.0)

    def test_reciprocal_rank_no_hit(self):
        expected = [1]
        retrieved = [2, 3, 4]
        self.assertEqual(compute_reciprocal_rank(expected, retrieved), 0.0)

    def test_average_precision_perfect_order(self):
        # 2 expected items at ranks 1 and 2
        expected = [1, 2]
        retrieved = [1, 2, 3, 4]
        # P@1 = 1/1 = 1.0, P@2 = 2/2 = 1.0 -> AP = (1.0 + 1.0)/2 = 1.0
        self.assertEqual(compute_average_precision(expected, retrieved), 1.0)

    def test_average_precision_ranks_1_and_3(self):
        expected = [1, 2]
        retrieved = [1, 4, 2, 5]
        # P@1 = 1/1 = 1.0, P@3 = 2/3 = 0.6667 -> AP = (1.0 + 2/3)/2 = 5/6 ~= 0.8333
        self.assertAlmostEqual(compute_average_precision(expected, retrieved), 5.0 / 6.0)

    def test_average_precision_empty_expected(self):
        self.assertEqual(compute_average_precision([], [1, 2]), 0.0)

class TestAnswerMetrics(unittest.TestCase):
    def test_normalize_text(self):
        raw = "  Hello, World! This is a TEST.  "
        norm = normalize_text(raw)
        self.assertEqual(norm, "hello world this is a test")

    def test_token_metrics_identical(self):
        pred = "The CGPA is 8.5/10."
        gt = "The CGPA is 8.5/10."
        metrics = compute_token_metrics(pred, gt)
        self.assertEqual(metrics["exact_match"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["jaccard"], 1.0)

    def test_token_metrics_case_and_punctuation_invariant(self):
        pred = "Shardul Hingane's CGPA is 8.5/10."
        gt = "shardul hingane's cgpa is 8.5/10"
        metrics = compute_token_metrics(pred, gt)
        self.assertEqual(metrics["exact_match"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_token_metrics_partial_overlap(self):
        pred = "FastAPI Celery and Redis"
        gt = "FastAPI Celery Redis and PostgreSQL"
        metrics = compute_token_metrics(pred, gt)
        self.assertEqual(metrics["exact_match"], 0.0)
        self.assertGreater(metrics["f1"], 0.7)
        self.assertLess(metrics["f1"], 1.0)

    def test_token_metrics_disjoint(self):
        pred = "Apples and oranges"
        gt = "PostgreSQL pgvector"
        metrics = compute_token_metrics(pred, gt)
        self.assertEqual(metrics["exact_match"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)

class TestCitationVerification(unittest.TestCase):
    def test_out_of_bounds_citation(self):
        reranked = [DummyChunk(1), DummyChunk(2)]
        res = verify_citations([5], reranked, tenant="default")
        self.assertFalse(res["all_valid"])
        self.assertTrue(any("out of bounds" in iss for cit in res["verified_citations"] for iss in cit["issues"]))

    def test_empty_citation_list(self):
        reranked = [DummyChunk(1)]
        res = verify_citations([], reranked, tenant="default")
        self.assertFalse(res["all_valid"])
        self.assertTrue(any("No citations" in iss for iss in res["issues"]))

if __name__ == "__main__":
    unittest.main()
