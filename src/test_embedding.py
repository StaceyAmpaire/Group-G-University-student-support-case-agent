from sentence_transformers import SentenceTransformer


# Load the local embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Test sentences
sentences = [
    "How do students register at the university?",
    "Students complete registration through the university registration process."
]

# Generate embeddings
embeddings = model.encode(sentences)

print("Number of embeddings:", len(embeddings))
print("Embedding dimensions:", embeddings.shape)