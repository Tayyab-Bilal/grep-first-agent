# Usage guide

Every `python` block here is executed by `tests/test_docs.py`.

## Build tools and run the agent

`make_tools` returns 17 LangChain tools (16 without a chat history) bound to one user's token. The agent needs a chat model;
tests and demos use `ScriptedModel`. With a real model, pass any LangChain chat model that supports
tool calling (use temperature 0.2). The agent calls `bind_tools` for you.

```python
import asyncio
import json

from langchain_core.messages import AIMessage

from grep_first_agent import DiscoveryAgent, FakeWorkspaceAPI, ProjectionCache, make_tools
from grep_first_agent.chat_history import ChatHistory
from grep_first_agent.fake_model import ScriptedModel, tool_turn

api = FakeWorkspaceAPI()
chats = ChatHistory()
chats.add("c1", "acme", "alice", "Margin chat", "asked about profit margins")
tools = make_tools(api, ProjectionCache(api), "token-alice", chats, tenant="acme", user="alice")
assert len(tools) == 17

reply = {"intent": "find", "confidence": 0.9, "pointers": {
    "email": [{"id": "e1", "name": "Q3 numbers", "confidence": 0.9}],
    "chat": [{"id": "c1", "name": "Margin chat", "confidence": 0.8}]}}
model = ScriptedModel([
    tool_turn(("search_emails", {"query": "profit margins"}), ("search_chats", {"query": "profit margins"})),
    AIMessage(content=json.dumps(reply)),
])
result = asyncio.run(DiscoveryAgent(model, tools).run("profit margins"))
assert result.pointers["email"][0].id == "e1" and result.pointers["chat"][0].id == "c1"
```

Without a `ChatHistory`, `search_chats` is not offered, so the set has 16 tools.

## The three tool layers

```python
import asyncio
import json

from grep_first_agent import FakeWorkspaceAPI, ProjectionCache, make_tools

api = FakeWorkspaceAPI()
tools = {t.name: t for t in make_tools(api, ProjectionCache(api), "token-alice")}


def call(name, query=""):
    return json.loads(asyncio.run(tools[name].ainvoke({"query": query})))


# 1. typed search: pointers
assert call("search_kpis", "incident response")[0]["id"] == "k1"
# 2. fallback snapshot: a summary plus pointers; fans out 2-5 backend calls in parallel
snapshot = call("describe_project", "operations")
assert snapshot["summary"]["project"] == "Operations"
# 3. name resolver: spoken name -> id (candidates, not pointers)
assert call("resolve_user", "dana")["candidates"][0]["id"] == "u2"
```

## Hybrid document text

`search_document_text` runs a keyword index and a dense index at the same time, drops hits the user
may not see, then fuses with weighted RRF. Bring your own indexes by implementing
`async search(query) -> list[dict]`.

```python
import asyncio
import json

from grep_first_agent import FakeWorkspaceAPI, ProjectionCache, make_tools
from grep_first_agent.docsearch import FakeDenseIndex, FakeKeywordIndex, default_chunks

events = []
chunks = default_chunks()
api = FakeWorkspaceAPI()
tools = {t.name: t for t in make_tools(
    api, ProjectionCache(api), "token-alice",
    keyword_index=FakeKeywordIndex(chunks, delay=0.01, events=events),
    dense_index=FakeDenseIndex(chunks, delay=0.01, events=events),
)}
hits = json.loads(asyncio.run(tools["search_document_text"].ainvoke({"query": "outage"})))
assert hits[0]["id"] == "d1"  # no document says "outage": the dense side found it
assert events[:2] == ["keyword:start", "dense:start"]  # both started together
```

## Hierarchy filtering

The seed returns a violation from project `p7`, which is outside Alice's hierarchy. It never
reaches the model:

```python
import asyncio

from grep_first_agent import FakeWorkspaceAPI, ProjectionCache
from grep_first_agent.hierarchy import visible

api = FakeWorkspaceAPI()
cache = ProjectionCache(api)
data = asyncio.run(cache.get("token-alice"))
assert "v3" in {v["id"] for v in data["violation"]}  # the backend returned it
allowed = set(data["hierarchy"])
assert "v3" not in {v["id"] for v in visible("violation", data["violation"], allowed)}
```

## Pinned context

If the user has attached a project or document, skip the scan. Pass `(kind, id)` pairs:

```python
import asyncio

from grep_first_agent import DiscoveryAgent, FakeWorkspaceAPI, make_pin_resolver
from grep_first_agent.fake_model import ScriptedModel

api = FakeWorkspaceAPI()
model = ScriptedModel([])  # no model turn is needed
agent = DiscoveryAgent(model, [], make_pin_resolver(api, "token-alice"))
result = asyncio.run(agent.run("summarise this", pinned=[("project", "p2")]))
assert result.pointers["project"][0].name == "Mobile App Beta" and model.calls == 0
```

A pin the user cannot see is ignored and the normal scan runs.

## Impersonation

A background job that acts as a user sets the ambient identity. Tools made with
`impersonating=True` refuse to run unless it matches.

```python
import asyncio

from grep_first_agent import FakeWorkspaceAPI, ProjectionCache, make_tools
from grep_first_agent.security import Identity, ambient_identity

api = FakeWorkspaceAPI()
tools = {t.name: t for t in make_tools(api, ProjectionCache(api), "token-alice",
                                       tenant="acme", user="alice", impersonating=True)}
search = tools["search_tasks"]


async def main():
    try:
        await search.ainvoke({"query": "launch"})  # no ambient identity set
    except PermissionError:
        pass
    else:
        raise AssertionError("ran without an ambient identity")

    ambient_identity.set(Identity("acme", "alice"))  # a web layer does this per request
    await search.ainvoke({"query": "launch"})  # now it proceeds


asyncio.run(main())
```

## Logging without leaking

The agent logs each tool result to the `grep_first_agent` logger, already scrubbed. You can scrub
your own strings too:

```python
from grep_first_agent.security import scrub_tool_result

out = scrub_tool_result('[{"id": "e1", "snippet": "private text", "name": "mail a.b@acme.example"}]')
assert "private" not in out and "acme.example" not in out
```

## Matching on its own

```python
from grep_first_agent.matching import search

records = [{"id": "a", "name": "Q4 campaign brief"}, {"id": "b", "name": "Team lunch"}]
hits = search("q4 campain", records, ["name"])  # note the typo
assert [rec["id"] for rec, score in hits] == ["a"]
```

## Redis cache

`RedisBackend` takes any `redis.asyncio` client. Install the extra with `pip install -e ".[redis]"`.
The example uses fakeredis, as the tests do.

```python
import asyncio

import fakeredis

from grep_first_agent import FakeWorkspaceAPI, ProjectionCache, RedisBackend

api = FakeWorkspaceAPI()
cache = ProjectionCache(api, RedisBackend(fakeredis.FakeAsyncRedis()))


async def main():
    await asyncio.gather(*(cache.get("token-alice") for _ in range(10)))


asyncio.run(main())
assert cache.fetches == 1  # ten callers, one backend fetch
```

## Parsing model output

`parse_result` finds the first valid JSON object in prose or content blocks and validates it:

```python
from grep_first_agent import parse_result

text = 'Sure! Here you go: {"intent": "find", "pointers": {"task": [{"id": "t1", "name": "X"}]}}'
assert parse_result(text).pointers["task"][0].id == "t1"
blocks = [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": text}]
assert parse_result(blocks).intent == "find"
```
