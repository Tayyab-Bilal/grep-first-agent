import json

from grep_first_agent.backend import FakeWorkspaceAPI
from grep_first_agent.cache import ProjectionCache
from grep_first_agent.chat_history import ChatHistory
from grep_first_agent.tools import make_tools


def test_like_wildcards_escaped():
    h = ChatHistory()
    h.add("a", "acme", "alice", "Pricing", "we offered a discount 50% off")
    h.add("b", "acme", "alice", "Pricing 2", "we offered a discount 500 off")
    h.add("c", "acme", "alice", "Snake", "my_var and myXvar")
    assert [r["id"] for r in h.search("acme", "alice", "50%")] == ["a"]  # '%' is not a wildcard
    assert h.search("acme", "alice", "my_var")[0]["id"] == "c"
    assert h.search("acme", "alice", "zzz%") == []


def test_scoped_by_tenant_and_user():
    h = ChatHistory()
    h.add("1", "acme", "alice", "Roadmap", "budget talk")
    h.add("2", "acme", "bob", "Roadmap", "budget talk")
    h.add("3", "other", "alice", "Roadmap", "budget talk")
    assert [r["id"] for r in h.search("acme", "alice", "budget")] == ["1"]


def test_token_not_in_tool_schema():
    tools = make_tools(FakeWorkspaceAPI(), ProjectionCache(FakeWorkspaceAPI()), "token-alice", ChatHistory())
    assert len(tools) == 17  # 11 typed searches + 3 snapshots + 3 resolvers
    for t in tools:
        assert t.name.split("_")[0] in {"search", "describe", "resolve"}
    for t in tools:
        assert list(t.args) == ["query"]
        assert "token" not in json.dumps(t.args_schema.model_json_schema()).lower()


async def _search(token, tool, query):
    api = FakeWorkspaceAPI()
    tools = {t.name: t for t in make_tools(api, ProjectionCache(api), token)}
    return json.loads(await tools[tool].ainvoke({"query": query}))


async def test_user_b_never_sees_user_a_records():
    own = await _search("token-alice", "search_tasks", "launch plan")
    assert [h['id'] for h in own[:2]] == ['t2', 't3']
    assert await _search("token-bob", "search_tasks", "launch plan") == []
    # and the reverse, plus every kind, using a word unique to Bob's data
    for tool in ("search_tasks", "search_projects", "search_documents", "search_emails", "search_meetings"):
        assert await _search("token-alice", tool, "acquisition") == []
    assert (await _search("token-bob", "search_tasks", "acquisition"))[0]["id"] == "t9"


async def test_tool_returns_compact_pointer_shape():
    hit = (await _search("token-alice", "search_emails", "proffit margins"))[0]
    assert {"id", "kind", "name", "snippet", "why"} <= hit.keys() and hit["id"] == "e1"


def test_typo_chat_query_falls_back_to_scoped_fuzzy():
    h = ChatHistory()
    h.add("1", "acme", "alice", "Margins", "we discussed profit margins")
    h.add("2", "acme", "bob", "Margins", "we discussed profit margins")
    h.add("3", "other", "alice", "Margins", "we discussed profit margins")
    assert [r["id"] for r in h.search("acme", "alice", "proffit margns")] == ["1"]
    assert h.search("acme", "carol", "proffit margns") == []
