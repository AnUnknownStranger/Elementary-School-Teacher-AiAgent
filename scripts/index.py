import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

CORPUS_PATH = Path("data_processed/corpus.jsonl")
OUT_DIR = Path("vector")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_docs():
    docs = []
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(
                Document(
                    page_content=obj["text"],
                    metadata={
                        "doc_id": obj["doc_id"],
                        "subject": obj.get("subject"),
                        "grade": obj.get("grade"),
                        "source": obj.get("source"),
                    },
                )
            )
    return docs

def main():
    docs = load_docs()
    print(f"Loaded {len(docs)} chunks from {CORPUS_PATH}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vs = FAISS.from_documents(docs, embeddings)

    vs.save_local(str(OUT_DIR))
    print(f"Saved FAISS index to {OUT_DIR}")

if __name__ == "__main__":
    main()
