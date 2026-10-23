"""Manually verify the local sentence-embedding model.

The first run downloads the configured model; this is not an offline unit test.
"""


def main():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sentences = [
        "How do students register at the university?",
        "Students complete registration through the university registration process.",
    ]
    embeddings = model.encode(sentences)
    print("Number of embeddings:", len(embeddings))
    print("Embedding dimensions:", embeddings.shape)


if __name__ == "__main__":
    main()