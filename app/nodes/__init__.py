"""The four LangGraph agent nodes (Scout, Quant, News, Risk Manager).

Each node is built by a ``build_*_node(...)`` factory that takes its
dependencies (data clients, the LLM, config rules) and returns a plain
``state -> dict`` function. This dependency-injection pattern keeps nodes pure
and unit-testable: tests pass stubs/mocks, the graph wiring passes real clients.
"""
