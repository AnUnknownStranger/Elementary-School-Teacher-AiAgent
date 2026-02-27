from retriever_logic import retrieve

docs = retrieve(
    question="What is photosynthesis?",
    grade=4,
    subject="Science",
    k=5
)

for i, d in enumerate(docs):
    print(f"Result {i+1}")
    print(d.metadata)
    print(d.page_content[:200])#print first 200 characters
    print("="*50)