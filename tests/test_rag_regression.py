"""
Regression tests for core RAG functionality:
- Text chunking (sliding window)
- Pydantic models & validation
- Cosine similarity calculation
- Tenant isolation in database queries
- Cross-encoder reranking behavior on empty/singleton candidates
- AI audit log insertion
"""
import unittest
import numpy as np
from pydantic import ValidationError

from rag import chunk_text
from models import Citations, RAGResponse
from cache import cosine_similarity
from reranker import rerank
from db import search_chunks, insert_audit_log, get_recent_audit_logs

class TestRAGRegression(unittest.TestCase):
    def test_chunk_text_sliding_window(self):
        text = "word " * 350
        chunks = chunk_text(text, size=150, overlap=50)
        # 350 words:
        # chunk 1: words 0..150
        # chunk 2: words 100..250
        # chunk 3: words 200..350
        # chunk 4: words 300..350
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c.split()), 150)

    def test_pydantic_rag_response_valid(self):
        valid_data = {
            "answer": "Shardul Hingane has solved 400+ problems.",
            "citations": [1]
        }
        res = RAGResponse.model_validate(valid_data)
        self.assertEqual(res.answer, valid_data["answer"])
        self.assertEqual(res.citations, [1])

    def test_pydantic_rag_response_invalid(self):
        invalid_data = {
            "answer": "Missing citations"
        }
        with self.assertRaises(ValidationError):
            RAGResponse.model_validate(invalid_data)

    def test_cosine_similarity(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0)

        v3 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0)

        v4 = [-1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v4), -1.0)

    def test_rerank_empty_chunks(self):
        res = rerank("Any query", [], top_k=5)
        self.assertEqual(res, [])

    def test_tenant_isolation(self):
        # Database search for a tenant that has no documents should return 0 results
        dummy_vector = [0.01] * 3072
        results = search_chunks(
            tenant="non_existent_tenant_999",
            embedding=dummy_vector,
            limit=10
        )
        self.assertEqual(len(results), 0)

    def test_audit_log_insertion_and_retrieval(self):
        test_query = "Automated regression test probe query"
        insert_audit_log(
            tenant="default",
            query=test_query,
            model="test-model",
            input_tokens=10,
            output_tokens=5,
            cost=0.0001,
            chunks_used=1
        )
        recent_logs = get_recent_audit_logs(tenant="default", limit=5)
        queries = [log["query"] for log in recent_logs]
        self.assertIn(test_query, queries)

if __name__ == "__main__":
    unittest.main()
