"""
Phase 5A: Evaluation Dataset Validation
Validates dataset schema, expected chunk IDs against Postgres DB, and question/answer completeness.
"""
import os
import json
from typing import Dict, Any, List
from db import engine, text

def load_evaluation_dataset(dataset_path: str = None) -> List[Dict[str, Any]]:
    """Loads evaluation dataset from eval.json or evaluation.json."""
    if dataset_path and os.path.exists(dataset_path):
        target = dataset_path
    elif os.path.exists("eval.json"):
        target = "eval.json"
    elif os.path.exists("evaluation.json"):
        target = "evaluation.json"
    else:
        raise FileNotFoundError("Could not find eval.json or evaluation.json")

    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data, target

def get_db_chunk_ids(tenant: str = "default") -> set:
    """Fetches all existing chunk IDs for the specified tenant from the database."""
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id FROM chunks WHERE tenant = :tenant"),
            {"tenant": tenant}
        ).fetchall()
        return {r[0] for r in rows}

def validate_dataset(dataset_path: str = None, tenant: str = "default") -> Dict[str, Any]:
    """
    Validates evaluation dataset items:
    - Presence of question
    - Expected chunk IDs (and their presence in DB)
    - Presence of expected answer
    """
    data, loaded_path = load_evaluation_dataset(dataset_path)
    db_chunk_ids = get_db_chunk_ids(tenant)

    validation_results = {
        "dataset_path": loaded_path,
        "tenant": tenant,
        "database_chunk_count": len(db_chunk_ids),
        "database_chunk_ids": sorted(list(db_chunk_ids)),
        "total_test_cases": len(data),
        "valid_test_cases": 0,
        "invalid_test_cases": 0,
        "cases": []
    }

    for idx, item in enumerate(data, start=1):
        case_info = {
            "index": idx,
            "question": item.get("question"),
            "expected_chunks": item.get("Expected chunk") or item.get("expected_chunk") or item.get("expected_chunks"),
            "expected_answer": item.get("answer") or item.get("expected_answer"),
            "is_valid": True,
            "issues": []
        }

        # Check question
        if not case_info["question"] or not isinstance(case_info["question"], str) or not case_info["question"].strip():
            case_info["is_valid"] = False
            case_info["issues"].append("Missing or empty 'question' field.")

        # Check expected chunks
        exp_chunks = case_info["expected_chunks"]
        if exp_chunks is None:
            case_info["is_valid"] = False
            case_info["issues"].append("Missing 'Expected chunk' field.")
        else:
            if isinstance(exp_chunks, int):
                exp_chunks = [exp_chunks]
                case_info["expected_chunks"] = exp_chunks
            elif not isinstance(exp_chunks, list):
                case_info["is_valid"] = False
                case_info["issues"].append(f"'Expected chunk' must be a list or int, got {type(exp_chunks).__name__}.")

            # Verify chunk IDs exist in database
            if isinstance(exp_chunks, list):
                missing_in_db = [cid for cid in exp_chunks if cid not in db_chunk_ids]
                if missing_in_db:
                    case_info["is_valid"] = False
                    case_info["issues"].append(f"Expected chunk IDs {missing_in_db} do not exist in database (tenant='{tenant}').")

        # Check expected answer
        if not case_info["expected_answer"] or not isinstance(case_info["expected_answer"], str) or not case_info["expected_answer"].strip():
            # Answer missing is a warning/notice rather than complete invalidation if retrieval-only, but recorded
            case_info["issues"].append("Notice: 'answer' is missing or empty (answer quality evaluation will skip ground-truth comparison).")

        if case_info["is_valid"]:
            validation_results["valid_test_cases"] += 1
        else:
            validation_results["invalid_test_cases"] += 1

        validation_results["cases"].append(case_info)

    return validation_results

if __name__ == "__main__":
    report = validate_dataset()
    print("=" * 60)
    print("PHASE 5A: EVALUATION DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Dataset File        : {report['dataset_path']}")
    print(f"Target Tenant       : {report['tenant']}")
    print(f"Corpus Chunks in DB : {report['database_chunk_count']} {report['database_chunk_ids']}")
    print(f"Total Test Cases    : {report['total_test_cases']}")
    print(f"Valid Test Cases    : {report['valid_test_cases']}")
    print(f"Invalid Test Cases  : {report['invalid_test_cases']}")
    print("-" * 60)
    for c in report["cases"]:
        status = "PASSED" if c["is_valid"] else "FAILED"
        print(f"[{status}] Case #{c['index']}: {c['question'][:50]}...")
        print(f"         Expected Chunks: {c['expected_chunks']}")
        if c["issues"]:
            print(f"         Issues: {c['issues']}")
    print("=" * 60)
