"""The discovery agent: a small ReAct loop that ends in a strict JSON pointer contract."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from .contract import KINDS, ContractError, DiscoveryResult, Pointer, parse_result
from .items import rows_of
from .security import scrub_tool_result
from .tools import PinResolver

log = logging.getLogger("grep_first_agent")

MAX_ITERATIONS = 8
MAX_RETRIES = 3
STRONG = 0.7
SYSTEM = (
    "You locate workspace records. Never answer the user, only point.\n"
    "Tools come in three layers: typed searches (search_*), fallback snapshots (describe_*) for vague "
    "requests, and name resolvers (resolve_*) that turn a name into an id.\n"
    "Parallel by default: call every search that could be relevant in the same step, they run in "
    "parallel. Missing an entity is worse than over-fetching.\n"
    f"If a search fails or comes back empty, you may retry it at most {MAX_RETRIES} times, then move on. "
    "If a result says `complete: false`, the list is partial: say so, never treat it as the whole set.\n"
    "Finish with ONLY a JSON object: "
    '{"intent": str, "scope": str|null, "pointers": {bucket: [{"id", "name", "confidence"}]}, '
    '"confidence": float, "ambiguity": null}. '
    f"The only buckets are: {', '.join(KINDS)}."
)


class ChatModel(Protocol):
    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage: ...


@dataclass
class _RunState:
    seen: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)  # every record tools returned
    bad: Counter[str] = field(default_factory=Counter)  # per tool: failed or empty results so far
    incomplete: set[str] = field(default_factory=set)  # kinds a tool said were partial


class DiscoveryAgent:
    def __init__(self, model: ChatModel, tools: list[BaseTool], resolve_pin: PinResolver | None = None) -> None:
        self.tools = {t.name: t for t in tools}
        self.resolve_pin = resolve_pin
        # Real chat models need tools bound; the scripted fake doesn't.
        self.model = model.bind_tools(tools) if hasattr(model, "bind_tools") else model

    async def run(self, query: str, pinned: list[tuple[str, str]] | None = None) -> DiscoveryResult:
        """Find records for `query`. `pinned` is a list of (kind, id) the user already attached."""
        try:
            return await self._from_pins(query, pinned) or await self._run(query)
        except Exception:  # noqa: BLE001 - the contract is "never raises into the chat turn"
            return _safe_empty(query)

    async def _from_pins(self, query: str, pinned: list[tuple[str, str]] | None) -> DiscoveryResult | None:
        """Pinned-context shortcut: the user already said which project/document they mean, so look
        those up directly. No model turn, no fan-out, no global scan. None means "fall through"."""
        if not pinned or self.resolve_pin is None:
            return None
        recs = [(k, r) for k, i in pinned if (r := await self.resolve_pin(k, i)) is not None]
        if not recs:  # pins not visible to this user: do the normal scan instead
            return None
        pointers: dict[str, list[Pointer]] = {}
        for kind, rec in recs:
            pointers.setdefault(kind, []).append(Pointer(id=rec["id"], name=rec["name"], confidence=1.0))
        scope = ", ".join(f"{k} {r['name']}" for k, r in recs)
        return DiscoveryResult(intent=query, scope=f"pinned: {scope}", pointers=pointers, confidence=1.0)

    async def _run(self, query: str) -> DiscoveryResult:
        messages: list[BaseMessage] = [SystemMessage(SYSTEM), HumanMessage(query)]
        state = _RunState()
        result: DiscoveryResult | None = None

        for _ in range(MAX_ITERATIONS):
            reply = await self.model.ainvoke(messages)
            messages.append(reply)
            if not reply.tool_calls:
                result = _try_parse(reply.content)
                break
            # Parallel by default: every requested search runs concurrently.
            messages += await asyncio.gather(*(self._call(tc, state) for tc in reply.tool_calls))

        if result is None:
            # JSON drift (or out of iterations): exactly one repair turn. Here it is one more model
            # call in this loop; production forced one extra graph turn from a post-model hook.
            messages.append(HumanMessage("Return only the JSON contract."))
            result = _try_parse((await self.model.ainvoke(messages)).content)
        if result is None:
            return _safe_empty(query)
        return _finalize(result, state)

    async def _call(self, tc: dict[str, Any], state: _RunState) -> ToolMessage:
        name = tc["name"]
        if state.bad[name] >= MAX_RETRIES:
            out = f"stop: {name} failed or came back empty {MAX_RETRIES} times. Do not call it again."
            return ToolMessage(content=out, tool_call_id=tc["id"])
        try:
            out = await self.tools[name].ainvoke(tc["args"])
            parsed = json.loads(out)
            rows = rows_of(parsed)
            for item in rows:
                if "kind" in item:  # resolver candidates are ids to use, not pointers
                    state.seen[(item["kind"], item["id"])] = item
            if isinstance(parsed, dict) and parsed.get("complete") is False:
                state.incomplete.add(parsed["kind"])
            if not rows:
                state.bad[name] += 1
        except Exception as e:  # noqa: BLE001 - a failing tool is information for the model
            out = f"error: {e}"
            state.bad[name] += 1
        # Logs get the scrubbed copy: no snippets, no addresses.
        log.info("tool=%s result=%s", name, scrub_tool_result(out))
        return ToolMessage(content=out, tool_call_id=tc["id"])


def _try_parse(raw: Any) -> DiscoveryResult | None:
    try:
        return parse_result(raw)
    except ContractError:
        return None


def _safe_empty(query: str) -> DiscoveryResult:
    return DiscoveryResult(intent=query, pointers={}, confidence=0.0, ambiguity="could not determine")


def _finalize(result: DiscoveryResult, state: _RunState) -> DiscoveryResult:
    seen = state.seen
    # Hallucinated-id filter: a pointer is only real if a tool returned it in this run.
    pointers = {
        kind: [p for p in ps if (kind, p.id) in seen] for kind, ps in result.pointers.items()
    }
    result.pointers = {k: ps for k, ps in pointers.items() if ps}
    result.incomplete = sorted(state.incomplete)  # type: ignore[assignment]  # set by tools, not the model
    if result.ambiguity is None:
        for kind, ps in result.pointers.items():
            strong = [p for p in ps if p.confidence >= STRONG]
            if len(strong) >= 2:
                # Breadcrumbs ("Project > Task") tell twin names apart. Built in code here; the
                # model's own `ambiguity`, when it supplies one, wins.
                names = [seen[(kind, p.id)].get("path") or p.name for p in strong]
                result.ambiguity = f"Which {kind} do you mean: " + " or ".join(f"'{n}'" for n in names) + "?"
                break
    return result
