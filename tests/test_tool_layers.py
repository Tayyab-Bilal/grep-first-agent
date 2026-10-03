import asyncio
import time

from helpers import call, tools_for

from grep_first_agent.backend import FakeWorkspaceAPI
from grep_first_agent.chat_history import ChatHistory
from grep_first_agent.docsearch import FakeDenseIndex, FakeKeywordIndex, default_chunks

TYPED = {"search_tasks", "search_projects", "search_documents", "search_document_text", "search_emails",
         "search_meetings", "search_chats", "search_group_chats", "search_workflows", "search_kpis",
         "search_violations"}
SNAPSHOTS = {"describe_user_context", "describe_project", "describe_space"}
RESOLVERS = {"resolve_user", "resolve_project", "resolve_space"}


def test_three_layers_of_tools():
    _, tools = tools_for(chats=ChatHistory())
    assert set(tools) == TYPED | SNAPSHOTS | RESOLVERS


async def test_each_typed_search_finds_its_kind():
    _, tools = tools_for()
    assert (await call(tools, "search_kpis", "incident response"))[0]["id"] == "k1"
    assert (await call(tools, "search_violations", "safety inspection"))[0]["id"] == "v1"
    chat = (await call(tools, "search_group_chats", "launch war room"))[0]
    assert chat["id"] == "g1" and chat["kind"] == "chat"  # group chats land in the chat bucket


async def test_snapshot_fans_out_in_parallel():
    api = FakeWorkspaceAPI(delay=0.1)
    _, tools = tools_for(api=api)
    await call(tools, "describe_user_context")  # warm the cache so only the fan-out is timed
    api.calls.clear()
    start = time.perf_counter()
    snap = await call(tools, "describe_project", "operations")
    elapsed = time.perf_counter() - start
    fanned = [k for k in api.calls if k.startswith("project_")]
    assert len(fanned) == 5 and api.calls["project_members"] == 1
    assert elapsed < 0.25  # five 0.1 s calls in parallel, not 0.5 s in a row
    assert snap["summary"]["project"] == "Operations" and snap["summary"]["tasks"] == 1


async def test_describe_space_and_user_context_fan_out_2_to_5_calls():
    api = FakeWorkspaceAPI()
    _, tools = tools_for(api=api)
    ctx = await call(tools, "describe_user_context")
    assert ctx["summary"]["open_tasks"] == 4 and ctx["summary"]["name"] == "Alice Park"
    api.calls.clear()
    space = await call(tools, "describe_space", "operations")
    assert space["summary"]["projects"] == ["Operations", "Marketing"]
    assert {"space_projects", "space_members", "kpi"} <= set(api.calls)


async def test_resolvers_turn_names_into_ids():
    _, tools = tools_for()
    assert (await call(tools, "resolve_user", "dana"))["candidates"][0]["id"] == "u2"
    assert (await call(tools, "resolve_project", "mobile beta"))["candidates"][0]["id"] == "p2"
    assert (await call(tools, "resolve_space", "operations"))["candidates"][0]["id"] == "s2"
    _, bob = tools_for("token-bob")
    assert (await call(bob, "resolve_project", "mobile beta"))["candidates"] == []  # not in Bob's hierarchy


async def test_workflow_matches_source_prompt_and_document():
    _, tools = tools_for()
    out = await call(tools, "search_workflows", "summarise open tasks per project")
    assert [i["id"] for i in out["items"]] == ["w1"] and "prompt or document" in out["items"][0]["why"]
    out = await call(tools, "search_workflows", "OPS Manual")
    assert out["items"][0]["id"] == "w2" and out["complete"] is True


async def test_partial_workflow_list_is_flagged_incomplete():
    # Pages of 1, at most 3 pages: 3 of Alice's 6 workflows are fetched, so the list is partial.
    _, tools = tools_for(api=FakeWorkspaceAPI(page_size=1))
    out = await call(tools, "search_workflows", "onboarding")
    assert out["complete"] is False and out["fetched"] == 3 and out["total"] == 6
    _, tools = tools_for(api=FakeWorkspaceAPI(page_size=2))
    assert (await call(tools, "search_workflows", "onboarding"))["complete"] is True


async def test_keyword_and_dense_searches_run_in_parallel():
    events: list[str] = []
    chunks = default_chunks()
    _, tools = tools_for(
        keyword_index=FakeKeywordIndex(chunks, delay=0.05, events=events),
        dense_index=FakeDenseIndex(chunks, delay=0.05, events=events),
    )
    await call(tools, "search_document_text", "escalation")
    assert events[:2] == ["keyword:start", "dense:start"]  # both started before either finished
    assert set(events[2:]) == {"keyword:end", "dense:end"}


async def test_document_text_finds_meaning_the_keywords_miss():
    _, tools = tools_for()
    hits = await call(tools, "search_document_text", "outage")  # no document says "outage"
    assert [h["id"] for h in hits] == ["d1"] and hits[0]["why"].startswith("hybrid")
    hits = await call(tools, "search_document_text", "escalation")
    assert hits[0]["id"] == "d1" and len(hits[0]["snippet"].split(" ... ")) <= 2  # best two snippets
    assert await call(tools, "search_document_text", "zzzz") == []


async def test_search_runs_concurrently_across_tools():
    api, tools = tools_for(api=FakeWorkspaceAPI(delay=0.05))
    await asyncio.gather(*(call(tools, n, "ops") for n in ("search_tasks", "search_kpis", "search_emails")))
    assert sum(api.calls.values()) == 9  # one shared fetch (8 kinds + hierarchy), not three
