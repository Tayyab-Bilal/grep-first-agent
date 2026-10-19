"""Regression tests for the cross-hierarchy leak: a record outside the user's hierarchy is dropped."""

from helpers import call, tools_for

from grep_first_agent.hierarchy import is_visible, visible


async def test_violation_search_never_returns_item_from_outside_hierarchy():
    # The backend over-returns v3 (project p7). Alice's hierarchy is p1-p4.
    _, tools = tools_for()
    assert await call(tools, "search_violations", "supplier contract") == []
    ids = {h["id"] for h in await call(tools, "search_violations", "violation sign-off inspection colour review")}
    assert "v3" not in ids and "v1" in ids


async def test_record_with_no_project_is_dropped_on_miss():
    _, tools = tools_for()
    assert await call(tools, "search_violations", "access review") == []  # v4 has no project_id


async def test_hierarchy_applies_to_every_scoped_search_and_snapshot():
    _, tools = tools_for()
    assert await call(tools, "search_documents", "merger") == []
    wf = await call(tools, "search_workflows", "merger data room")
    assert wf["items"] == []
    assert await call(tools, "resolve_project", "p7") == {"candidates": []}


async def test_hybrid_document_text_never_leaks_other_users_or_out_of_hierarchy_docs():
    # The shared index holds d3 (Alice's backend, project p7) and d9 (Bob's). Alice may see neither.
    _, alice = tools_for()
    assert await call(alice, "search_document_text", "offer price negotiation") == []
    _, bob = tools_for("token-bob")
    assert [h["id"] for h in await call(bob, "search_document_text", "offer price negotiation")] == ["d9"]


def test_visibility_rule_unit():
    allowed = {"p1"}
    assert is_visible("task", {"project_id": "p1"}, allowed)
    assert not is_visible("task", {"project_id": "p2"}, allowed)
    assert not is_visible("task", {}, allowed)
    assert is_visible("project", {"id": "p1"}, allowed) and not is_visible("project", {"id": "p2"}, allowed)
    assert is_visible("email", {}, allowed)  # emails belong to the user, not a project
    assert visible("kpi", [{"project_id": "p1", "id": "a"}, {"project_id": "x", "id": "b"}], allowed) == [
        {"project_id": "p1", "id": "a"}
    ]
