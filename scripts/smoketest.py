from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vs = FAISS.load_local("vector", embeddings, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    queries = [
        "How do you add fractions with different denominators?",
        "What is a noun? Give examples.",
        "Explain the water cycle in simple terms.",
        "What caused the American Revolution?"
    ]

    for q in queries:
        print("\n" + "="*80)
        print("Query:", q)
        docs = retriever.invoke(q)
        for i, d in enumerate(docs, 1):
            subj = d.metadata.get("subject")
            grade = d.metadata.get("grade")
            src = d.metadata.get("source")
            doc_id = d.metadata.get("doc_id")

            print(f"\n[{i}] subject={subj} grade={grade} source={src} doc_id={doc_id}")
            print(d.page_content[:220], "...")
if __name__ == "__main__":
    main()