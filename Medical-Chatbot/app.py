import os
import json
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage


import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not loaded")
VECTOR_PATH = "vectorstore"

MODEL_NAME = "llama-3.3-70b-versatile"
print("Loaded key:", GROQ_API_KEY)



print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Loading vector DB...")
vectorstore = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

print("Loading LLM...")
llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0.2,
    max_tokens=1000,
    groq_api_key=GROQ_API_KEY   
)

print(" Ready")


def get_rag_context(query):
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])



def call_mcp_tool(tool_name, args):
    """
    Simple local MCP simulation
    (In real setup, this would connect to MCP server)
    """

   
    from research.medical_mcp_server import (
        _search_pubmed,
        _search_fda_drug,
        _search_medlineplus,
        _search_icd
    )

    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        if tool_name == "pubmed":
            return loop.run_until_complete(_search_pubmed(args))

        elif tool_name == "drug":
            return loop.run_until_complete(_search_fda_drug(args))

        elif tool_name == "medline":
            return loop.run_until_complete(_search_medlineplus(args))

        elif tool_name == "icd":
            return loop.run_until_complete(_search_icd(args))

    finally:
        loop.close()

    return ""



def get_answer(question):

  
    rag_context = get_rag_context(question)

    
    tool_context = ""

    if "drug" in question or "medicine" in question:
        tool_context += call_mcp_tool("drug", question)

    if "disease" in question or "symptom" in question:
        tool_context += "\n" + call_mcp_tool("pubmed", question)

   
    prompt = f"""
You are MediBot, a medical assistant.

RULES:
- Be clear and short
- No technical jargon unless needed
- Do NOT mention sources
- End with: "⚕️ Consult a doctor for medical advice."

Knowledge:
{rag_context}

Extra info:
{tool_context}

Question: {question}
Answer:
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()



app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Empty question"}), 400

    try:
        answer = get_answer(question)
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )