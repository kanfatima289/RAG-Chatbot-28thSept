"""RAG FAQ Assistant - mutual fund facts, from 5 approved public sources.

Layout follows architecture.md section 9:

    rag/      online pipeline  - guards, embed, retrieve, generate, post-process
    ingest/   offline pipeline - load, chunk, embed, store (run once)
    eval/     measures the PRD success criteria
    app/      Streamlit UI
    tests/    unit tests for the guards and the pipeline

The entry point to read first is rag/pipeline.py.
"""
