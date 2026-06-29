"""Deterministic math engine.

The design doc's safety plan is explicit: *"The LLM must be stripped of
mathematical responsibilities. The system must rely on a deterministic Python
engine to calculate yields and spreads, enforcing flawless logic."*

Every function here is a PURE function: same inputs → same outputs, no I/O, no
randomness, no global state. That makes them trivially unit-testable and means
the LangGraph nodes can call them directly for guaranteed-correct numbers.
"""

from app.engine.math_engine import *  # noqa: F401,F403
