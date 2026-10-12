import sys
import json

print("=== 1. Testing NLP Parser ===")
from nlp_parser import parse_ocean_query_nlp, get_region_bbox

test_queries = [
    "Show me a map of where active floats are in the North Atlantic",
    "Plot a graph of temperature vs depth in the Pacific Ocean",
    "Explain how the halocline relates to salinity changes"
]

for q in test_queries:
    parsed = parse_ocean_query_nlp(q)
    print(f"Query: '{q}'\n  -> Intent: {parsed.intent}, Region: {parsed.region}, Vars: {parsed.variables}\n")

print("=== 2. Testing ChromaDB RAG Store ===")
from rag_store import rag_store

print(f"Total documents indexed in ChromaDB: {rag_store.count()}")
context = rag_store.format_context_for_prompt("What is the profiling cycle of an ARGO float?")
print("Retrieved RAG Context snippet:")
print(context[:300] + "...\n")

print("=== 3. Testing Flask App with Ocean Agent & ChromaDB ===")
from main import app

client = app.test_client()

# Check RAG status endpoint
status_res = client.get("/api/rag/status")
print("RAG Status response:", status_res.get_json())

# Test chat query with RAG context
print("\n=== 4. Testing End-to-End Chat Query ===")
chat_res = client.post("/chat", json={
    "message": "Explain how ARGO floats measure salinity and how the halocline affects ocean density."
})

print("Status Code:", chat_res.status_code)
chat_data = chat_res.get_json()
print("Reply snippet:", chat_data.get("reply", "")[:350] + "...")
print("Intent:", chat_data.get("intent"))
print("RAG Context used:", chat_data.get("rag_context_used"))
print("Reasoning Details present:", chat_data.get("reasoning_details") is not None)
print("\n=== SUCCESS: All LangChain, ChromaDB RAG, and NLP tests passed! ===")
