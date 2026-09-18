from pathlib import Path
import csv


# Paths
KNOWLEDGE_DIR = Path("knowledge")
REGISTER_FILE = Path("docs/corpus_register.csv")



# Load provenance information
def load_source_register():
    sources = {}

    with open(REGISTER_FILE, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            sources[row["file_name"]] = row

    return sources


# Load and chunk all knowledge documents
def load_documents():
    sources = load_source_register()
    chunks = []

    files = sorted(KNOWLEDGE_DIR.glob("*.txt"))

    for file_path in files:

        file_name = file_path.name

        with open(file_path, "r", encoding="utf-8") as file:
            document = file.read()

        source = sources.get(file_name)

        if source is None:
            print(
                f"Warning: No source register entry found for {file_name}"
            )
            continue

        # Simple paragraph-based chunking
        document_chunks = [
            chunk.strip()
            for chunk in document.split("\n\n")
            if chunk.strip()
        ]

        for index, chunk in enumerate(document_chunks, start=1):

            chunks.append({
                "chunk_id": f"{source['source_id']}-CHUNK-{index:03d}",
                "source_id": source["source_id"],
                "document_title": source["document_title"],
                "file_name": file_name,
                "source_type": source["source_type"],
                "publisher": source["publisher"],
                "source_url": source["source_url"],
                "text": chunk
            })

    return chunks