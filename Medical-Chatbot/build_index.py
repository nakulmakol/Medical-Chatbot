import os
import re
from pypdf import PdfReader
from tqdm import tqdm

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_PATH = os.path.join(BASE_DIR, "research", "Medical_book.pdf")
SAVE_PATH = os.path.join(BASE_DIR, "vectorstore")



def clean_text(text: str) -> str:
    text = re.sub(r'\n+', '\n', text)         
    text = re.sub(r'\s+', ' ', text)          
    return text.strip()


print(" Reading PDF...")

reader = PdfReader(PDF_PATH)
docs = []

for i, page in enumerate(tqdm(reader.pages, desc="Extracting")):
    text = page.extract_text()

    if text and text.strip():
        text = clean_text(text)

        docs.append(
            Document(
                page_content=text,
                metadata={"page": i + 1}   
            )
        )

print(f"Extracted {len(docs)} pages")



print(" Splitting into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = splitter.split_documents(docs)

print(f" Created {len(chunks)} chunks")



print(" Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



print(" Building FAISS index...")

vectorstore = FAISS.from_documents(chunks, embeddings)


vectorstore.save_local(SAVE_PATH)

print(f" FAISS index saved at: {SAVE_PATH}")