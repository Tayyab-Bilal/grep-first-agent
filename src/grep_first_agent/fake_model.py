"""A scripted chat model: deterministic stand-in for a real LLM in tests and examples."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage


def tool_turn(*calls: tuple[str, dict[str, Any]]) -> AIMessage:
    """One model turn that requests several tool calls at once."""
    return AIMessage(
        content="",
        tool_calls=[{"name": n, "args": a, "id": f"call{i}"} for i, (n, a) in enumerate(calls)],
    )


class ScriptedModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls += 1
        return self._responses.pop(0)  # IndexError when the script runs out: tests see it
