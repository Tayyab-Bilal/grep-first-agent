"""End-to-end walkthrough with fakes: no LLM, no network, no index to build."""

import asyncio
import json

from langchain_core.messages import AIMessage, ToolMessage

from grep_first_agent import DiscoveryAgent, FakeWorkspaceAPI, ProjectionCache, make_pin_resolver, make_tools
from grep_first_agent.chat_history import ChatHistory
from grep_first_agent.fake_model import tool_turn
from grep_first_agent.items import rows_of


class GrepEchoModel:
    """Stands in for the LLM: fans out every typed search, then points at what came back."""

    def __init__(self, tools):
        self.names = [t.name for t in tools if t.name.startswith("search_")]

    async def ainvoke(self, messages):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        query = messages[1].content
        if not results:
            return tool_turn(*[(n, {"query": query}) for n in self.names])
        pointers = {}
        for m in results:
            hits = [h for h in rows_of(json.loads(m.content)) if not h["why"].startswith("fuzzy")]  # skip typo noise
            keep = [h for h in hits if h["why"] == "exact phrase" or h["why"].startswith("hybrid")] or hits[:1]
            for h in keep:
                conf = 0.9 if h["why"] == "exact phrase" else 0.6
                bucket = pointers.setdefault(h["kind"], [])
                if h["id"] not in {p["id"] for p in bucket}:  # two tools can return the same record
                    bucket.append({"id": h["id"], "name": h["name"], "confidence": conf})
        return AIMessage(content=json.dumps({"intent": query, "pointers": pointers, "confidence": 0.8}))


def show(query, result):
    print(f"\n> {query!r}")
    for kind, ptrs in result.pointers.items():
        print(f"  {kind:9} " + ", ".join(f"{p.id} ({p.name})" for p in ptrs))
    if result.incomplete:
        print(f"  partial lists: {result.incomplete}")
    if result.ambiguity:
        print(f"  ask user: {result.ambiguity}")


async def main() -> None:
    api = FakeWorkspaceAPI()
    cache = ProjectionCache(api)
    chats = ChatHistory()
    chats.add("c1", "acme", "alice", "Margin chat", "asked the assistant about profit margins")
    tools = make_tools(api, cache, "token-alice", chats, tenant="acme", user="alice")
    agent = DiscoveryAgent(GrepEchoModel(tools), tools, make_pin_resolver(api, "token-alice"))
    print(f"{len(tools)} tools: {sum(t.name.startswith('search_') for t in tools)} typed searches, "
          "3 fallback snapshots, 3 name resolvers")

    for query in ["Q4 featherfootwear task", "proffit margins email", "everything about the OPS Manual", "launch plan"]:
        show(query, await agent.run(query))

    print(f"\nbackend fetches: {cache.fetches}  (4 queries x 11 searches, one cached projection)")

    print("\nHierarchy filter: the API also returned a violation from a project Alice cannot see.")
    by_name = {t.name: t for t in tools}
    hits = json.loads(await by_name["search_violations"].ainvoke({"query": "supplier contract"}))
    print(f"  search_violations('supplier contract') -> {hits}")

    print("\nDocument text (hybrid keyword + dense in parallel, weighted RRF). Nothing says 'outage':")
    for h in json.loads(await by_name["search_document_text"].ainvoke({"query": "outage"})):
        print(f"  {h['id']} {h['name']}: {h['why']} | {h['snippet']}")

    print("\nWorkflow search with a short page size, so the list is partial:")
    small = make_tools(FakeWorkspaceAPI(page_size=1), ProjectionCache(FakeWorkspaceAPI()), "token-alice")
    flow = json.loads(await {t.name: t for t in small}["search_workflows"].ainvoke({"query": "onboarding"}))
    print(f"  complete={flow['complete']} fetched={flow['fetched']} of {flow['total']}")

    print("\nPinned context: the user attached project p2, so no scan runs.")
    api2 = FakeWorkspaceAPI()
    pinned_agent = DiscoveryAgent(GrepEchoModel([]), [], make_pin_resolver(api2, "token-alice"))
    show("status of this project", await pinned_agent.run("status of this project", pinned=[("project", "p2")]))
    print(f"  list calls made: {sum(v for k, v in api2.calls.items() if not k.startswith('get_') and k != 'hierarchy')}")


asyncio.run(main())
