# API reference

All names import from `grep_first_agent` unless a module is shown. Every `python` block is run by
the test suite.

## Agent

### `DiscoveryAgent(model, tools, resolve_pin=None)`
ReAct loop (max 8 iterations). `model` is any object with `async ainvoke(messages) -> AIMessage`
(a LangChain chat model, or `ScriptedModel`). `tools` is the list from `make_tools`.
`resolve_pin` is the callable from `make_pin_resolver`.

- `async run(query: str, pinned: list[tuple[str, str]] | None = None) -> DiscoveryResult`
  Never raises. Returns an empty result with `ambiguity="could not determine"` on failure.
  With `pinned`, looks those `(kind, id)` records up directly and skips the scan.
- Module constants in `grep_first_agent.agent`: `MAX_ITERATIONS = 8`, `MAX_RETRIES = 3`,
  `STRONG = 0.7`, `SYSTEM` (the shipped system prompt).

## Tools (`grep_first_agent.tools`)

### `make_tools(api, cache, token, chats=None, tenant="", user="", *, keyword_index=None, dense_index=None, impersonating=False) -> list[StructuredTool]`
Builds the tool set for one user. Every tool takes one argument, `query: str`.

| Layer | Tools |
|---|---|
| Typed searches | `search_tasks`, `search_projects`, `search_documents` (by name), `search_document_text` (hybrid), `search_emails`, `search_meetings`, `search_group_chats`, `search_workflows`, `search_kpis`, `search_violations`, `search_chats` (needs `chats`) |
| Fallback snapshots | `describe_user_context`, `describe_project`, `describe_space` |
| Name resolvers | `resolve_user`, `resolve_project`, `resolve_space` |

Output shapes: a list of pointers `{id, kind, name, snippet, why, path?}`; or an object with `items`
(snapshots, workflows) or `candidates` (resolvers). Workflow output also has `kind`, `complete`,
`fetched`, `total`.

### `make_pin_resolver(api, token) -> Callable[[kind, id], Awaitable[dict | None]]`
Single-record lookup that checks the hierarchy. Returns `None` if not visible.

## Contract (`grep_first_agent.contract`)

- `Kind`: `Literal["task", "project", "document", "email", "meeting", "chat", "workflow", "kpi", "violation"]`. `KINDS` is the tuple of the same nine.
- `Pointer(id: str, name: str, confidence: float = 1.0)`
- `DiscoveryResult(intent, scope=None, pointers: dict[Kind, list[Pointer]] = {}, confidence=0.0, ambiguity=None, incomplete: list[Kind] = [])`
  Unknown bucket keys raise `ValidationError`. `incomplete` lists kinds whose results may be partial.
- `parse_result(raw: str | list) -> DiscoveryResult` raises `ContractError` if no valid object is found.

```python
from grep_first_agent import ContractError, parse_result

try:
    parse_result("no json here")
except ContractError:
    pass
else:
    raise AssertionError
```

## Cache

- `ProjectionCache(api, backend=None, ttl=300.0, jitter=30.0, rng=None)`
  - `async get(token) -> dict` kind -> records, plus `"hierarchy"` (allowed project ids). One
    fetch fills all kinds. Partial, empty or failed fetches are returned but not stored.
  - `fetches: int` number of backend fetches so far.
- `InMemoryBackend(clock=time.monotonic)`; `RedisBackend(client)`. Both implement
  `async get(key) -> str | None` and `async set(key, value, ttl)`.

## Matching (`grep_first_agent.matching`)

- `search(query, records, fields, fuzzy_threshold=0.6, limit=10) -> list[(record, MatchScore)]`
  Best first. Prose fields are cut to 200 characters.
- `score(query, fields) -> MatchScore`; `MatchScore(phrase_hit, token_hits, fuzzy)` with `.key`.
- `tokenize(text) -> list[str]` (Unicode-aware); `STOPWORDS`.

## Fusion (`grep_first_agent.rrf`)

- `weighted_rrf(rankings, weights, k=60) -> list[(doc_id, score)]`
- `hybrid_content_search(keyword_hits, dense_hits, keyword_weight=3, dense_weight=1, dense_min=0.45, relevance_floor=0.20, k=60) -> list[{doc_id, score, snippets}]`
  Dense hits need a `score` (cosine). Output is one row per document, best two snippets.

```python
from grep_first_agent.rrf import hybrid_content_search

out = hybrid_content_search(
    [{"doc_id": "d1", "snippet": "a"}],
    [{"doc_id": "d1", "snippet": "b", "score": 0.9}, {"doc_id": "d2", "snippet": "c", "score": 0.2}],
)
assert [r["doc_id"] for r in out] == ["d1"]  # d2 is under the 0.45 dense floor
```

## Document indexes (`grep_first_agent.docsearch`)

- `KeywordIndex`, `DenseIndex`: protocols with `async search(query) -> list[dict]`.
- `FakeKeywordIndex(chunks, delay=0.0, events=None)`, `FakeDenseIndex(chunks, delay=0.0, events=None)`.
- `Chunk(doc_id, text)`; `default_chunks()` builds one per seeded sentence.

## Hierarchy (`grep_first_agent.hierarchy`)

- `visible(kind, records, allowed: set[str]) -> list`; `is_visible(kind, record, allowed) -> bool`.
- `SCOPED`: kinds that live inside a project. A scoped record with no project, or one outside
  `allowed`, is dropped.

## Security (`grep_first_agent.security`)

- `Identity(tenant, user)`; `ambient_identity: ContextVar[Identity | None]`.
- `check_impersonation(claimed: Identity)` raises `PermissionError` unless the ambient identity equals `claimed`.
- `scrub_tool_result(raw: str) -> str` removes `snippet` keys and masks email addresses.

## Chat history

- `ChatHistory(conn=None)` (`grep_first_agent.chat_history`)
  - `add(id, tenant, user, title, body)`
  - `search(tenant, user, query, limit=10) -> list[dict]` scoped by tenant and user.

## Backend

- `WorkspaceAPI`: protocol the agent searches through (`list_tasks`, `list_projects`, `list_documents`, `list_emails`, `list_meetings`, `list_group_chats`, `list_kpis`, `list_violations`, `list_workflows`, `allowed_project_ids`, `list_users`, `list_spaces`, `get_profile`, `project_members`, `project_items`, `space_projects`, `space_members`, `get_record`).
- `FakeWorkspaceAPI(delay=0.0, fail=None, page_size=2)`: seeded in memory. `delay` slows every call,
  `fail` is a set of call labels that raise, `calls` is a `Counter` of calls made.
- `ScriptedModel(responses)` and `tool_turn(*calls)` in `grep_first_agent.fake_model`.
