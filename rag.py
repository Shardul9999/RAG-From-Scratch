from litellm import completion
# from click import shell_completion
# from click import shell_completion
# from click import shell_completion
from dotenv import load_dotenv
from google import genai
from models import RAGResponse

load_dotenv()
client = genai.Client()

def embed(text):
    response = client.models.embed_content(
        model = "gemini-embedding-2",
        contents = text
    )

    return response.embeddings[0].values

# print(embed("Hello Monkeys"))

def chunk_text(text, size=150, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), size - overlap):
        chunks.append(
            " ".join(words[i:i + size])
        )
    return chunks

def generate_answer(query, chunks):
    context = "\n\n".join(
        f"[{i + 1}] {chunk.content if hasattr(chunk, 'content') else chunk}"
        for i, chunk in enumerate(chunks)
    )

    prompt = f"""
Answer the question using only the provided content.
context:
{context}

Question:
{query}

Rules:
-Answer only from the context
-Do not invent information
-citations must contain only the numbers of chunks that support the answer.
"""

    response = completion(
        model="gemini/gemini-2.5-flash",  # LiteLLM requires provider prefix
        num_retries=3,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful RAG assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format = RAGResponse
    )

    result =  RAGResponse.model_validate_json(
        response.choices[0].message.content
    )
    return result, response
    # return response.choices[0].message.content