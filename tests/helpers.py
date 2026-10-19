"""Shared test helpers."""

import json

from grep_first_agent.backend import FakeWorkspaceAPI
from grep_first_agent.cache import ProjectionCache
from grep_first_agent.tools import make_tools


def tools_for(token="token-alice", api=None, **kw):
    api = api or FakeWorkspaceAPI()
    return api, {t.name: t for t in make_tools(api, ProjectionCache(api), token, **kw)}


async def call(tools, name, query=""):
    return json.loads(await tools[name].ainvoke({"query": query}))
