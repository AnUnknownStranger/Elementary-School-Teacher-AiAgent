# Elementary-School-Teacher-AiAgent

## Retrieval & Evaluation
* This document explains how to:
  * Test the retrieval logic
  * Run retrieval evaluation
## How to test retrieval logic
* Make sure retriever_logic.py, evaluation.py, and vector exist
* How to test retriever
   make sure test_retriever.py and retriever_logic are under the same folder
   * run:
        - **python test_retriever.py**
## How to test evaluation
* run evaluation.py
   make sure there is a eval_set.jsonl under the same folder
   * for single k-value(for example,5):
        - **python evaluation.py --eval eval_set.jsonl --k 5**
   * for multiple k-value(for example, 3,5,8):
        - **python evaluation.py --eval eval_set.jsonl --k 3 5 8**