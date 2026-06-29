"""Covered Call Agentic AI — a locally-hosted LangGraph multi-agent system.

Package layout
--------------
    app.config            central configuration (paths, secrets, business rules)
    app.state             LangGraph state schema shared across nodes
    app.llm               local Qwen2.5 GGUF loader (llama-cpp-python)
    app.engine            deterministic math engine (no LLM math, ever)
    app.data              Schwab market-data + massive.com news clients
    app.memory            ChromaDB vector memory + SQL decision logger
    app.nodes             the four agent nodes (Scout, Quant, News, Risk)
    app.notify            Discord HITL webhook
    app.graphs            LangGraph wiring (entry screener, defense monitor)
"""

__version__ = "0.1.0"
