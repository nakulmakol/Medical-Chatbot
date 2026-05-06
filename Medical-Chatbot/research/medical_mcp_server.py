import asyncio
import json
import httpx
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types


from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


app = Server("medical-knowledge-server")

VECTOR_PATH = "vectorstore"

if os.path.exists(VECTOR_PATH):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    vector_db = None


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="rag_search",
            description="Search local medical book (RAG-based retrieval)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        ),

        types.Tool(
            name="search_pubmed",
            description="Search PubMed research articles",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        ),

        types.Tool(
            name="search_fda_drug",
            description="Search FDA drug database",
            inputSchema={
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string"},
                    "search_type": {"type": "string", "default": "label"}
                },
                "required": ["drug_name"]
            }
        ),

        types.Tool(
            name="search_medlineplus",
            description="Search MedlinePlus",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        ),

        types.Tool(
            name="search_icd",
            description="Search ICD-11 diseases",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        )
    ]



async def _rag_search(query: str) -> str:
    if vector_db is None:
        return " Vector DB not found. Run build_index.py first."

    docs = vector_db.similarity_search(query, k=3)

    context = "\n\n".join([d.page_content for d in docs])

    return f""" RAG Context:

{context}

 Use this context to answer the query: "{query}"
"""



async def _search_pubmed(query: str, max_results: int = 5) -> str:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    async with httpx.AsyncClient(timeout=10) as client:
        search = await client.get(f"{base}/esearch.fcgi", params={
            "db": "pubmed", "term": query,
            "retmax": max_results, "retmode": "json"
        })

        ids = search.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return "No results found."

        summary = await client.get(f"{base}/esummary.fcgi", params={
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json"
        })

        data = summary.json()["result"]

        results = []
        for i in ids:
            art = data.get(i, {})
            title = art.get("title", "")
            results.append(f"• {title}")

        return "\n".join(results)


async def _search_fda_drug(drug_name: str, search_type: str = "label") -> str:
    async with httpx.AsyncClient(timeout=10) as client:
        url = "https://api.fda.gov/drug/label.json"

        resp = await client.get(url, params={
            "search": f'openfda.generic_name:"{drug_name}"',
            "limit": 1
        })

        if resp.status_code != 200:
            return "No data found."

        data = resp.json()["results"][0]

        return f"""
Drug: {drug_name}

Purpose: {data.get("purpose", ["N/A"])[0]}
Warnings: {data.get("warnings", ["N/A"])[0][:300]}
"""


async def _search_medlineplus(query: str) -> str:
    return f"Search MedlinePlus externally for: {query}"


async def _search_icd(query: str) -> str:
    return f"Search ICD manually for: {query}"


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    try:
        if name == "rag_search":
            result = await _rag_search(arguments["query"])

        elif name == "search_pubmed":
            result = await _search_pubmed(arguments["query"])

        elif name == "search_fda_drug":
            result = await _search_fda_drug(arguments["drug_name"])

        elif name == "search_medlineplus":
            result = await _search_medlineplus(arguments["query"])

        elif name == "search_icd":
            result = await _search_icd(arguments["query"])

        else:
            result = "Unknown tool"

    except Exception as e:
        result = str(e)

    return [types.TextContent(type="text", text=result)]



async def main():
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())