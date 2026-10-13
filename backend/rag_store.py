"""
Compatibility shim - forwards to app.rag.vectorstore
"""
from app.rag.vectorstore import (
    ArgoRAGStore,
    rag_store,
    OCEAN_KNOWLEDGE_DOCUMENTS,
)

__all__ = [
    "ArgoRAGStore",
    "rag_store",
    "OCEAN_KNOWLEDGE_DOCUMENTS",
]
