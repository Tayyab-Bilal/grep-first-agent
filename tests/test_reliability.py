import json
import logging

import pytest
from helpers import call, tools_for
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from grep_first_agent import DiscoveryAgent, DiscoveryResult, make_pin_resolver
from grep_first_agent.agent import MAX_RETRIES, SYSTEM
from grep_first_agent.backend import FakeWorkspaceAPI
from grep_first_agent.contract import KINDS, ContractError, parse_result
from grep_first_agent.fake_model import ScriptedModel, tool_turn
from grep_first_agent.security import Identity, ambient_identity, scrub_tool_result


def contract(**pointers):
    ps = {k: [{"id": i, "name": n, "confidence": c} for i, n, c in v] for k, v in pointers.items()}
    return AIMessage(content=json.dumps({"intent": "find", "pointers": ps, "confidence": 0.9}))


# --- pointer contract ---------------------------------------------------------------------------

def test_exactly_nine_typed_buckets():
    assert set(KINDS) == {"task", "project", "document", "email", "meeting", "chat", "workflow", "kpi", "violation"}
    assert len(KINDS) == 9
    for kind in KINDS:
        DiscoveryResult(intent="x", pointers={kind: [{"id": "1", "name": "n"}]})


def test_unknown_bucket_is_rejected():
    with pytest.raises(ValidationError):
        DiscoveryResult(intent="x", pointers={"people": [{"id": "1", "name": "n"}]})
    with pytest.raises(ContractError):
        parse_result('{"intent": "x", "pointers": {"tasks": []}}')  # plural is not a bucket


async def test_unknown_bucket_from_model_goes_to_repair_turn():
    _api, tools = tools_for()
    model = ScriptedModel([
        tool_turn(("search_tasks", {"query": "login"})),
        AIMessage(content='{"intent": "x", "pointers": {"people": []}}'),
        contract(task=[("t4", "Fix login bug", 0.9)]),
    ])
    result = await DiscoveryAgent(model, list(tools.values())).run("login")
    assert model.calls == 3 and result.pointers["task"][0].id == "t4"


# --- prompt -------------------------------------------------------------------------------------

def test_system_prompt_says_parallel_by_default_and_over_fetch():
    assert "Parallel by default" in SYSTEM
    assert "Missing an entity is worse than over-fetching" in SYSTEM
    assert "at most 3 times" in SYSTEM
    assert all(k in SYSTEM for k in KINDS)


# --- retries capped at 3 ------------------------------------------------------------------------

async def test_retries_capped_at_three_per_tool():
    runs = []

    async def empty(query: str = "") -> str:
        runs.append(query)
        return "[]"

    flaky = StructuredTool.from_function(coroutine=empty, name="search_tasks", description="always empty")
    seen = []

    class Spy(ScriptedModel):
        async def ainvoke(self, messages):
            seen.append(messages)
            return await super().ainvoke(messages)

    model = Spy([tool_turn(("search_tasks", {"query": f"try{i}"})) for i in range(5)] + [contract()])
    await DiscoveryAgent(model, [flaky]).run("x")
    assert runs == ["try0", "try1", "try2"]  # the 4th and 5th attempts never reached the tool
    stops = [m.content for m in seen[-1] if isinstance(m, ToolMessage) and m.content.startswith("stop")]
    assert len(stops) == 2 and str(MAX_RETRIES) in stops[0]


async def test_failures_count_toward_the_cap_and_other_tools_are_unaffected():
    _api, tools = tools_for(api=FakeWorkspaceAPI(fail={"email"}))
    model = ScriptedModel([tool_turn(("search_emails", {"query": "q3"})) for _ in range(4)]
                          + [tool_turn(("search_tasks", {"query": "login"}))] + [contract(task=[("t4", "Fix login bug", 0.9)])])
    result = await DiscoveryAgent(model, list(tools.values())).run("q3")
    assert result.pointers["task"][0].id == "t4"  # search_tasks still ran after search_emails was cut off


# --- completeness flags reach the contract ------------------------------------------------------

async def test_partial_workflow_list_marks_result_incomplete():
    _api, tools = tools_for(api=FakeWorkspaceAPI(page_size=1))
    model = ScriptedModel([tool_turn(("search_workflows", {"query": "onboarding"})),
                           contract(workflow=[("w3", "New hire onboarding", 0.9)])])
    result = await DiscoveryAgent(model, list(tools.values())).run("onboarding")
    assert result.incomplete == ["workflow"] and result.pointers["workflow"][0].id == "w3"


async def test_complete_workflow_list_is_not_marked_incomplete():
    _api, tools = tools_for()
    model = ScriptedModel([tool_turn(("search_workflows", {"query": "onboarding"})),
                           contract(workflow=[("w3", "New hire onboarding", 0.9)])])
    assert (await DiscoveryAgent(model, list(tools.values())).run("onboarding")).incomplete == []


# --- pinned-context shortcut --------------------------------------------------------------------

async def test_pinned_context_skips_global_scan():
    api, tools = tools_for()
    model = ScriptedModel([])  # any model turn would raise IndexError
    agent = DiscoveryAgent(model, list(tools.values()), make_pin_resolver(api, "token-alice"))
    result = await agent.run("what is the status here?", pinned=[("project", "p2"), ("document", "d1")])
    assert model.calls == 0
    assert [p.id for p in result.pointers["project"]] == ["p2"] and result.pointers["document"][0].id == "d1"
    assert result.scope.startswith("pinned:")
    assert not any(k in api.calls for k in ("task", "email", "meeting", "kpi"))  # no scan, no cache fill


async def test_pin_outside_hierarchy_falls_back_to_normal_scan():
    api, tools = tools_for()
    model = ScriptedModel([contract()])
    agent = DiscoveryAgent(model, list(tools.values()), make_pin_resolver(api, "token-alice"))
    result = await agent.run("x", pinned=[("document", "d3")])  # d3 is in project p7
    assert model.calls == 1 and result.pointers == {}


# --- telemetry scrubbing ------------------------------------------------------------------------

def test_scrub_removes_snippets_and_addresses():
    raw = json.dumps([{"id": "e1", "name": "mail dana.reyes@acme.example", "snippet": "profit margins"}])
    out = scrub_tool_result(raw)
    assert "profit" not in out and "acme.example" not in out and "[email]" in out
    assert "acme.example" not in scrub_tool_result("error: bounce from sam.ito@acme.example")


async def test_agent_logs_never_contain_snippets_or_addresses(caplog):
    _api, tools = tools_for()
    model = ScriptedModel([tool_turn(("search_emails", {"query": "profit margins"})), contract()])
    with caplog.at_level(logging.INFO, logger="grep_first_agent"):
        await DiscoveryAgent(model, list(tools.values())).run("profit margins")
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "tool=search_emails" in logged and '"e1"' in logged  # something was logged...
    assert "shipping costs" not in logged and "acme.example" not in logged  # ...but scrubbed


# --- impersonation guard ------------------------------------------------------------------------

async def test_impersonated_call_blocked_when_ambient_identity_differs():
    _, tools = tools_for(tenant="acme", user="alice", impersonating=True)
    token = ambient_identity.set(Identity("acme", "bob"))
    try:
        with pytest.raises(PermissionError):
            await call(tools, "search_tasks", "launch")
    finally:
        ambient_identity.reset(token)


async def test_impersonated_call_blocked_with_no_ambient_identity():
    _, tools = tools_for(tenant="acme", user="alice", impersonating=True)
    with pytest.raises(PermissionError):
        await call(tools, "search_tasks", "launch")


async def test_impersonated_call_proceeds_when_ambient_identity_matches():
    _, tools = tools_for(tenant="acme", user="alice", impersonating=True)
    token = ambient_identity.set(Identity("acme", "alice"))
    try:
        assert (await call(tools, "search_tasks", "launch plan"))[0]["id"] == "t2"
    finally:
        ambient_identity.reset(token)


async def test_non_impersonated_call_ignores_ambient_identity():
    _, tools = tools_for(tenant="acme", user="alice")
    assert (await call(tools, "search_tasks", "launch plan"))[0]["id"] == "t2"
