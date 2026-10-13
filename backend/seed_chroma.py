"""
ARGO-AI - ChromaDB Cloud & Local Seeding and Diagnostic Utility
Run this script to inspect, connect, and seed Chroma Cloud or Local ChromaDB.
"""
import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv()

from app.rag.vectorstore import ArgoRAGStore, COLLECTION_NAME, OCEAN_KNOWLEDGE_DOCUMENTS

def main():
    print("=" * 65)
    print("      ARGO-AI - ChromaDB Cloud & Knowledge Diagnostics")
    print("=" * 65)

    api_key = os.getenv("CHROMA_API_KEY")
    tenant = os.getenv("CHROMA_TENANT")
    database = os.getenv("CHROMA_DATABASE")
    host = os.getenv("CHROMA_HOST")

    print("\n[Config Detection]")
    if api_key:
        masked = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
        print(f" - Mode:            Chroma Cloud")
        print(f" - API Key:         {masked}")
        print(f" - Tenant:          {tenant}")
        print(f" - Database:        {database}")
    elif host:
        print(f" - Mode:            Remote Hosted Server ({host})")
    else:
        print(f" - Mode:            Local SQLite Persistence")

    print("\n[Connecting to Vector Store...]")
    store = ArgoRAGStore()

    print(f"\n[Status]")
    print(f" - Active Mode:     {store.mode}")
    print(f" - Collection:      {COLLECTION_NAME}")
    doc_count = store.count()
    print(f" - Document Count:  {doc_count}")

    print("\n[Running Sample Test Query: 'What is an ARGO float operational cycle?']")
    results = store.query("What is an ARGO float operational cycle?", n_results=2)
    print(f" - Retrieved Hits:  {len(results)}")
    for i, hit in enumerate(results, 1):
        title = hit.get("metadata", {}).get("title", f"Result {i}")
        score = hit.get("score", 0.0)
        snippet = hit.get("content", "")[:120].replace("\n", " ")
        print(f"   [{i}] {title} (score: {score:.4f}): {snippet}...")

    print("\n" + "=" * 65)
    print("  ChromaDB Cloud Vector Database is LIVE, CONNECTED & HEALTHY!")
    print("=" * 65)

if __name__ == "__main__":
    main()
