"""The Covered Call management agent — a multi-step (ReAct) tool-using expert.

Unlike the old one-shot router, this agent reasons over several turns: it can
call a tool, read the result, then call another or answer. It drives the local
LLM via ``structured()`` (JSON each step), so it works with llama-cpp without
native function-calling, and is fully testable with a fake backend.

The system prompt makes it a comprehensive covered-call *system manager*, not a
mere DB assistant: it knows the strategy, can manage account data, run exact
quantitative analysis (via tools — never mental math), recall past lessons, and
report performance.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.agent.tools import Tool, tool_catalog_text
from app.llm import LocalLLM

logger = logging.getLogger("agent")

# Tools that change persistent state. If the agent claims a change but none of
# these succeeded, the guard re-prompts it to actually perform the action.
_MUTATING_TOOLS = {"set_cash", "update_holding_status"}
_CHANGE_CLAIMS = ("recorded", "been recorded", "have been updated", "has been updated",
                  "marked as", "have been marked", "cash balance is now", "set your cash",
                  "set the cash", "closed the position", "rolled", "liquidated", "assigned")


def _claims_change(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in _CHANGE_CLAIMS)

_SYSTEM = """You are the Covered Call Command Center — an expert assistant that manages an \
automated covered-call income system.

Strategy you operate: buy 100 shares of a liquid, dividend-paying, optionable stock and sell \
1 out-of-the-money call (≈0.30–0.40 delta, 30–45 days to expiration) when implied volatility is \
rich, targeting >10% annualized income, intending to let shares be called away. You defend \
downside breaches with a Tree-of-Thoughts (Hard Eject / Roll Down / Hold). A human approves every \
trade; the system records cash, positions, decisions, and learned lessons in a database.

Your capabilities (via TOOLS):
- Manage account data: cash balance, holdings/positions (incl. closing them), recent decisions.
- Fetch LIVE market data: Schwab quotes/fundamentals/price history/option chains/expirations,
  news headlines, and earnings dates — then act on it.
- Run EXACT quantitative analysis: yields/AROC, Greeks, probability of profit, expected move,
  IV rank, technical indicators, composite scoring, and the three defense branches.
- Use composites: `technical_analysis(symbol)` (fetch history + indicators) and
  `find_best_covered_call(symbol)` (fetch the chain + pick the optimal strike).
- Recall past lessons from semantic memory and report portfolio performance.

When asked about a specific stock, fetch its real data (quote, chain, news, earnings) with tools
rather than relying on training knowledge — prices and chains change constantly.

Hard rules:
- NEVER do arithmetic yourself — always use the analysis tools for any number.
- NEVER invent cash balances, holdings, prices, or stats — fetch them with tools.
- To CHANGE anything (record/execute a trade, update a holding, set cash, close/roll a position) you
  MUST call the tool that performs it. NEVER say a change was made unless a tool you called RETURNED a
  successful result — claiming an un-performed action is a critical failure.
- When the user says holdings were executed / assigned / called away / sold, call
  `update_holding_status` for EACH holding (one call per symbol, status ASSIGNED unless told
  otherwise), then report the ACTUAL results the tool returned.
- Use one tool at a time; read its result before deciding the next step.
- Be concise and decision-useful. Explain reasoning, surface risks, and cite the numbers tools return.

Respond with a SINGLE JSON object each turn:
- To use a tool:  {"thought": "...", "tool": "<name>", "args": { ... }}
- When finished:  {"thought": "...", "final": "<your answer to the user>"}
"""


class CoveredCallAgent:
    def __init__(self, llm: LocalLLM, tools: Dict[str, Tool], max_steps: int = 6) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.system_prompt = _SYSTEM + "\n\nAvailable tools:\n" + tool_catalog_text(tools)

    def _filter_args(self, tool: Tool, args: Dict[str, Any]) -> Dict[str, Any]:
        allowed = set(tool.parameters.get("properties", {}))
        return {k: v for k, v in (args or {}).items() if k in allowed}

    def _build_prompt(self, message: str, history: List[Dict[str, str]], scratch: str) -> str:
        parts = []
        if history:
            convo = "\n".join(f"{h['role']}: {h['content']}" for h in history[-6:])
            parts.append(f"Conversation so far:\n{convo}")
        parts.append(f"User request: {message}")
        if scratch:
            parts.append(f"Your work so far:{scratch}")
        parts.append("Decide the next step (tool call or final answer) as JSON.")
        return "\n\n".join(parts)

    def _mutated(self, steps: List[Dict[str, Any]]) -> bool:
        """Did any state-changing tool succeed this turn?"""
        for s in steps:
            if s["tool"] in _MUTATING_TOOLS:
                obs = s.get("observation")
                if not (isinstance(obs, dict) and obs.get("error")):
                    return True
        return False

    def chat(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Run the agent loop and return {answer, steps}."""
        history = history or []
        steps: List[Dict[str, Any]] = []
        scratch = ""
        corrected = False

        for i in range(self.max_steps):
            user = self._build_prompt(message, history, scratch)
            try:
                obj = self.llm.structured(self.system_prompt, user, max_tokens=700)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Agent JSON step failed: %s", exc)
                break

            final = obj.get("final") or obj.get("answer")
            if final:
                # Anti-hallucination guard: if the answer claims a state change but
                # no mutating tool actually succeeded, nudge the agent once to do it.
                if not corrected and _claims_change(final) and not self._mutated(steps):
                    corrected = True
                    logger.info("Agent claimed a change with no mutating tool — re-prompting to act.")
                    scratch += ("\nSYSTEM CHECK: your answer claims a change was made, but you have "
                                "NOT successfully called a tool that performs it. If the user wants a "
                                "change (e.g. recording executed trades), call update_holding_status "
                                "now — once per holding — and then report the tool's actual results.")
                    continue
                return {"answer": final, "steps": steps}

            tool_name = obj.get("tool")
            args = obj.get("args", {}) or {}
            if tool_name not in self.tools:
                obs = {"error": f"unknown tool '{tool_name}'", "available": list(self.tools)}
            else:
                try:
                    obs = self.tools[tool_name].run(**self._filter_args(self.tools[tool_name], args))
                except Exception as exc:  # noqa: BLE001
                    obs = {"error": str(exc)}
            steps.append({"tool": tool_name, "args": args, "observation": obs})
            logger.info("Agent step %d: %s(%s)", i + 1, tool_name, json.dumps(args, default=str))
            scratch += (f"\nAction: {tool_name}({json.dumps(args, default=str)})"
                        f"\nObservation: {json.dumps(obs, default=str)[:1500]}")

        # Out of steps → force a plain-text final answer from what we have.
        try:
            final = self.llm.chat(
                self.system_prompt,
                self._build_prompt(message, history, scratch)
                + "\n\nProvide your final answer to the user now in plain text.",
                max_tokens=500).strip()
        except Exception as exc:  # noqa: BLE001
            final = f"(Could not complete the request: {exc})"
        return {"answer": final, "steps": steps}
