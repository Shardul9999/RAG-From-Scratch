import torch
from sentence_transformers import CrossEncoder

import os
import torch
from sentence_transformers import CrossEncoder

def _initialize_reranker():
    device = "cpu"
    if os.environ.get("FORCE_CPU", "").lower() not in ("1", "true") and torch.cuda.is_available():
        try:
            free_mem, _ = torch.cuda.mem_get_info()
            # Require at least 500MB free GPU memory
            if free_mem >= 500 * 1024 * 1024:
                device = "cuda"
        except Exception:
            pass

    try:
        model = CrossEncoder(
            "BAAI/bge-reranker-base",
            device=device,
            max_length=512,
            activation_fn=torch.nn.Sigmoid(),
            model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {}
        )
        return model, device
    except Exception as e:
        print(f"Warning: Reranker failed to load on {device} ({e}), falling back to CPU.")
        model = CrossEncoder(
            "BAAI/bge-reranker-base",
            device="cpu",
            max_length=512,
            activation_fn=torch.nn.Sigmoid()
        )
        return model, "cpu"

reranker, DEVICE = _initialize_reranker()


def rerank(query, chunks, top_k=5):
    if not chunks:
        return []

    pairs = [
        (query, chunk.content) for chunk in chunks
    ]
    
    scores = reranker.predict(
        pairs,
        batch_size=8,
        show_progress_bar=False
    )

    ranked = list(zip(chunks, scores))

    ranked.sort(
        key = lambda x: float(x[1]),
        reverse = True
    )

    print("\n   Rerranking   ")

    for i, (chunk, score) in enumerate(ranked[:top_k], start=1):
        print(
            f"{i}. "
            f"score={float(score):.4f} | "
            f"title={chunk.title}"
        )

    
    print("-----------------\n")

    return[chunk for chunk, score in ranked[:top_k]]