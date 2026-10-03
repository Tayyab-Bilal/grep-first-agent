import json

from langchain_core.messages import AIMessage, ToolMessage

from grep_first_agent.agent import MAX_ITERATIONS, DiscoveryAgent
from grep_first_agent.backend import FakeWorkspaceAPI
from grep_first_agent.cache import ProjectionCache
from grep_first_agent.fake_model import ScriptedModel, tool_turn
from grep_first_agent.tools import make_tools


def agent_for(responses, token="token-alice"):
    api = FakeWorkspaceAPI()
    model = ScriptedModel(responses)
    return DiscoveryAgent(model, make_tools(api, ProjectionCache(api), token)), model


def contract(**pointers):
    ps = {k: [{"id": i, "name": n, "confidence": c} for i, n, c in v] for k, v in pointers.items()}
    return json.dumps({"intent": "find", "pointers": ps, "confidence": 0.9})


async def test_parallel_tool_calls_then_contract():
    seen_messages = []

    class Spy(ScriptedModel):
        async def ainvoke(self, messages):
            seen_messages.append(list(messages))
            return await super().ainvoke(messages)

    api = FakeWorkspaceAPI()
    model = Spy([
        tool_turn(("search_tasks", {"query": "ops manual"}), ("search_documents", {"query": "ops manual"}),
                  ("search_meetings", {"query": "ops manual"})),
        AIMessage(content=contract(document=[("d1", "OPS Manual", 0.95)])),
    ])
    result = await DiscoveryAgent(model, make_tools(api, ProjectionCache(api), "token-alice")).run("ops manual")
    assert result.pointers["document"][0].id == "d1" and result.ambiguity is None
    tool_msgs = [m for m in seen_messages[1] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 3  # three typed searches answered in one turn
    assert model.calls == 2
    assert api.calls["task"] == 1  # all three shared one cached fetch


async def test_json_drift_one_repair_turn():
    agent, model = agent_for([
        tool_turn(("search_tasks", {"query": "login"})),
        AIMessage(content="I found the login bug task for you!"),
        AIMessage(content=contract(task=[("t4", "Fix login bug", 0.9)])),
    ])
    result = await agent.run("login")
    assert [p.id for p in result.pointers["task"]] == ["t4"]
    assert model.calls == 3


async def test_double_drift_returns_safe_empty():
    agent, model = agent_for([
        tool_turn(("search_tasks", {"query": "login"})),
        AIMessage(content="prose"),
        AIMessage(content="still prose"),
    ])
    result = await agent.run("login")
    assert result.pointers == {} and result.ambiguity == "could not determine"
    assert model.calls == 3  # exactly one repair turn, no more


async def test_never_raises_and_caps_iterations():
    endless = [tool_turn(("search_tasks", {"query": "x"})) for _ in range(MAX_ITERATIONS + 1)]
    agent, model = agent_for(endless)
    result = await agent.run("x")
    assert result.ambiguity == "could not determine"
    assert model.calls == MAX_ITERATIONS + 1  # 8 loop turns + 1 repair turn
    agent, _ = agent_for([])  # model blows up immediately
    assert (await agent.run("x")).pointers == {}


async def test_hallucinated_id_dropped():
    agent, _ = agent_for([
        tool_turn(("search_tasks", {"query": "login"})),
        AIMessage(content=contract(task=[("t4", "Fix login bug", 0.9), ("t999", "Invented", 0.9)],
                                   email=[("e1", "Real id, but never returned this run", 0.9)])),
    ])
    result = await agent.run("login")
    assert [p.id for p in result.pointers["task"]] == ["t4"]
    assert "email" not in result.pointers


async def test_two_strong_matches_produce_clarifying_question():
    agent, _ = agent_for([
        tool_turn(("search_tasks", {"query": "launch plan"})),
        AIMessage(content=contract(task=[("t2", "Launch plan", 0.9), ("t3", "Launch plan", 0.9)])),
    ])
    result = await agent.run("launch plan")
    q = result.ambiguity
    assert q and q.endswith("?")
    assert "Website Relaunch > Launch plan" in q and "Mobile App Beta > Launch plan" in q


async def test_one_strong_match_is_not_ambiguous():
    agent, _ = agent_for([
        tool_turn(("search_tasks", {"query": "launch plan"})),
        AIMessage(content=contract(task=[("t2", "Launch plan", 0.9), ("t3", "Launch plan", 0.4)])),
    ])
    assert (await agent.run("launch plan")).ambiguity is None
