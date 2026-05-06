"""
experiments.py

Purpose:
- Test RAG retrieval quality
- Debug embeddings
- Try prompts

NOT USED in production
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTOR_PATH = "vectorstore"

print("Loading vector DB...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    VECTOR_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def debug_query(query):
    docs = retriever.invoke(query)

    print("\n Retrieved chunks:\n")

    for i, d in enumerate(docs, 1):
        print(f"[{i}] Page {d.metadata.get('page')}")
        print(d.page_content[:300])
        print("-" * 60)


if __name__ == "__main__":
    while True:
        q = input("\nEnter query (or exit): ")
        if q.lower() == "exit":
            break
        debug_query(q)