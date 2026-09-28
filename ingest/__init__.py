"""Offline ingestion pipeline: load -> chunk -> embed -> store.

Run once:  python -m ingest.run_ingestion [--offline]

Modules arrive in phase 2 (loader, chunker) and phase 3 (they call
rag/embeddings.py and rag/store.py).
"""
