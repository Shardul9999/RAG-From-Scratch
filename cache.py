import redis
import json
import numpy as np

redis_client = redis.Redis(
    host = "localhost",
    port = 6379,
    decode_responses = True
)

CACHE_THRESHOLD = 0.90
CACHE_TTL = 3600

def cosine_similarity(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    return float(similarity)

def get_cached_answer(query_embedding):
    keys = redis_client.keys("semantic_cache:*")

    best_similarity = 0
    best_result = None

    for key in keys:
        data = redis_client.get(key)

        if not data:
            continue

        data = json.loads(data)
        cached_embedding = data["embedding"]
        similarity = cosine_similarity(query_embedding, cached_embedding)

        if similarity > best_similarity:
            best_similarity = similarity
            best_result = data

    if best_similarity >= CACHE_THRESHOLD:
        print(
            f"SEMANTIC CACHE HIT"
            f"(similarity = {best_similarity:.3f})"
        )

        return best_result["response"]

    print(
            f"SEMANTIC CACHE MISS"
            f"(best_similarity = {best_similarity:.3f})"
        )
    
    return None

def store_answer(query, query_embedding, response):
    cache_id = abs(hash(query))
    key = f"semantic_cache:{cache_id}"

    data = {
        "query" : query,
        "embedding" : query_embedding,
        "response" : response
    }

    redis_client.setex(key, CACHE_TTL, json.dumps(data))

    print("Stored response in semantic cache")


def flush_semantic_cache():
    keys = redis_client.keys("semantic_cache:*")
    if keys:
        redis_client.delete(*keys)
    return len(keys)


def get_cache_stats():
    try:
        keys = redis_client.keys("semantic_cache:*")
        return {
            "status": "online",
            "cached_entries": len(keys)
        }
    except Exception as e:
        return {
            "status": "offline",
            "error": str(e),
            "cached_entries": 0
        }




