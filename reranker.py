import torch
from sentence_transformers import CrossEncoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

reranker = CrossEncoder(
    "BAAI/bge-reranker-base",
    device=DEVICE,
    max_length=512,
    activation_fn=torch.nn.Sigmoid(),
    model_kwargs={
        "torch_dtype": torch.float16
    } if DEVICE == "cuda" else {}
)


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