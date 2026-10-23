import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

def main():
    import numpy as np
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    from sentence_transformers import SentenceTransformer
    from groq import Groq

    from ingestion import load_documents

    trace = {
        "schema_version": 1,
        "run_id": uuid.uuid4().hex,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "model_calls": [],
    }


    def save_trace():
        trace["duration_ms"] = round((time.perf_counter() - run_started) * 1000)
        trace_path = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "rag_demo_trace.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
        print(f"Trace saved to: {trace_path}")


    run_started = time.perf_counter()


    # 1. Load environment variables


    load_dotenv()

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )
    groq_client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Local embedding model
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )



    # 2. Load controlled corpus


    index_started = time.perf_counter()
    chunks = load_documents()

    print(f"Number of chunks: {len(chunks)}")


    # 3. Create local embeddings for document chunks

    document_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"Creating embeddings for {len(document_texts)} chunks..."
    )

    document_embeddings = embedding_model.encode(
        document_texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    trace["indexing"] = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "document_chunks": len(chunks),
        "duration_ms": round((time.perf_counter() - index_started) * 1000),
    }

    print(
        f"Embedding matrix shape: {document_embeddings.shape}"
    )



    # 4. Ask a student question


    student_question = "did i get an admission into the university?"


    # 5. Create embedding for the question


    retrieval_started = time.perf_counter()
    query_embedding = embedding_model.encode(
        student_question,
        convert_to_numpy=True
    )


    # 6. Calculate cosine similarity


    def cosine_similarity(vector_a, vector_b):

        return np.dot(vector_a, vector_b) / (
            np.linalg.norm(vector_a)
            * np.linalg.norm(vector_b)
        )


    similarities = []

    for i, embedding in enumerate(document_embeddings):

        score = cosine_similarity(
            query_embedding,
            embedding
        )

        result = chunks[i].copy()

        result["score"] = score

        similarities.append(result)



    # 7. Sort by relevance


    similarities.sort(
        key=lambda item: item["score"],
        reverse=True
    )



    # 8. Inspect retrieval scores


    print("\nTop 10 similarity scores:\n")

    for result in similarities[:10]:
        print(
            f"{result['score']:.4f} | "
            f"{result['document_title']} | "
            f"{result['chunk_id']}"
        )



    # 9. Display retrieved evidence


    top_chunks = similarities[:3]
    trace["retrieval"] = {
        "method": "cosine_similarity",
        "requested_top_k": 3,
        "returned_chunks": len(top_chunks),
        "duration_ms": round((time.perf_counter() - retrieval_started) * 1000),
        "sources": [
            {
                "document_title": result["document_title"],
                "source_id": result["source_id"],
                "chunk_id": result["chunk_id"],
                "similarity": round(float(result["score"]), 6),
            }
            for result in top_chunks
        ],
    }

    print("\nRetrieved evidence:\n")

    for result in top_chunks:
        print("=" * 70)
        print(f"Source: {result['document_title']}")
        print(f"Source ID: {result['source_id']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Similarity: {result['score']:.4f}")
        print()
        print(result["text"])
        print()


    # 10. Construct grounded context


    retrieved_context = ""
    if not top_chunks:

        print("Final answer:")
        print(
            "I could not find sufficiently relevant "
            "information in the approved knowledge sources."
        )

    else:

        retrieved_context = "\n\n".join(
            f"""
    SOURCE:
    {result['document_title']}

    SOURCE ID:
    {result['source_id']}

    SOURCE URL:
    {result['source_url']}

    CONTENT:
    {result['text']}
    """
            for result in top_chunks
        )



    # 11. Grounded generation


    system_instruction = """
    You are a university student-support assistant.

    Your answers must be grounded only in the
    retrieved university information provided to you.

    Use the retrieved information as the authoritative
    knowledge context for university-specific questions.

    Do not use general model knowledge to fill gaps.

    Do not invent university policies, procedures,
    dates, departments, fees, requirements, or
    other university-specific information.

    If the retrieved evidence does not contain enough
    information to answer the question, clearly state
    that the information is unavailable.

    Do not make decisions about admissions, grading,
    fees, disciplinary matters, or other high-impact
    university matters.

    When answering, identify the source or sources
    used for the answer.

    Respond clearly, briefly, and politely.
    """

    prompt_payload = json.dumps(
        {"system_instruction": system_instruction},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    trace["prompt_version"] = "sha256:" + hashlib.sha256(prompt_payload).hexdigest()[:16]

    try:

        model_started = time.perf_counter()
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",

            contents=f"""
    RETRIEVED UNIVERSITY EVIDENCE:

    {retrieved_context}


    STUDENT QUESTION:

    {student_question}
    """,

            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )

        trace["model_calls"].append({
            "provider": "gemini",
            "model": "gemini-3.1-pro-preview",
            "latency_ms": round((time.perf_counter() - model_started) * 1000),
            "outcome": "success",
        })
        trace["outcome"] = {
            "status": "success",
            "answer_characters": len(response.text or ""),
        }
        print("Generation model: Gemini 3.1 Pro Preview")
        print("Final answer:")
        print(response.text)


    except Exception as gemini_error:

        print("\nGemini generation failed.")
        print(f"Error type: {type(gemini_error).__name__}")
        trace["model_calls"].append({
            "provider": "gemini",
            "model": "gemini-3.1-pro-preview",
            "latency_ms": round((time.perf_counter() - model_started) * 1000),
            "outcome": "error",
            "error_type": type(gemini_error).__name__,
        })
        print("\nFalling back to Groq...\n")

        groq_started = time.perf_counter()
        try:
            groq_response = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "system",
                        "content": system_instruction
                    },
                    {
                        "role": "user",
                        "content": f"""
    RETRIEVED UNIVERSITY EVIDENCE:

    {retrieved_context}


    STUDENT QUESTION:

    {student_question}
    """
                    }
                ],
            )
        except Exception as groq_error:
            trace["model_calls"].append({
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "latency_ms": round((time.perf_counter() - groq_started) * 1000),
                "outcome": "error",
                "error_type": type(groq_error).__name__,
            })
            trace["outcome"] = {
                "status": "failed",
                "error_type": type(groq_error).__name__,
            }
            save_trace()
            raise

        answer = groq_response.choices[0].message.content or ""
        trace["model_calls"].append({
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
            "latency_ms": round((time.perf_counter() - groq_started) * 1000),
            "outcome": "success",
        })
        trace["outcome"] = {
            "status": "success",
            "answer_characters": len(answer),
        }
        print("Generation model: Groq / openai/gpt-oss-120b")
        print("Final answer:")
        print(answer)

    save_trace()

if __name__ == "__main__":
    main()
