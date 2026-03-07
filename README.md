# Elementary-School-Teacher-AiAgent

## RAG Knowledge Base & Vector Index

This module builds the Retrieval-Augmented Generation (RAG) knowledge base for the Elementary School Teacher AI Agent.

### what this module does

1. Extracts text from curriculum PDFs.
2. Cleans and chunks text into passages.
3. Automatically infers metadata:
   - subject
   - grade 
   - source
   - doc_id (stable unique id)
4. Embeds chunks using:
   sentence-transformers/all-MiniLM-L6-v2
5. Builds and saves a FAISS vector store.

### current corpus states

- 20 source PDF files
- 2,618 text chunks 

### How to run this module
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/preprocess.py
python scripts/index.py
streamlit run server.py 

### How to use the system
wait for the UI to be load up on streamlit
once everything is loaded, user can start typing in inputs
