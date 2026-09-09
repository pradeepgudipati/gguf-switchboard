# context-harbor — "GGUF-Switchboard"

## Setup (one step, zero hops)
- MCP server `context-harbor` must be configured (Streamable HTTP `POST /mcp` with a bearer MCP API key). No CLI, JWT, or extra config needed.
- This skill is pre-scoped: always pass `project_name="GGUF-Switchboard"` (or `projectName` for ingest tools). Do not ask the user for it and do not call `list_projects` to discover it.

## Purpose
Search and retrieve documents from the "GGUF-Switchboard" project using context-harbor.

## Decision tree (fewest calls)
1. New task / plan / estimate? → `planning_context` first. It returns overview + requirements + prior decisions with sources. Do not invent requirements.
2. Exact requirement or spec question? → `requirement_lookup` (defaults to spec/plan docs).
3. Broad / exploratory question? → `search_project_docs`.
4. Hit looks truncated or references nearby context ("as shown above", "this approach")? → `read_chunk_neighbors` on that hit only.
5. Need structure before deep search? → `get_project_brief`, then `get_project_structure` → `get_area_brief` / `search_area` / `get_knowledge_item`.
6. Reusable instruction? → `list_prompts` / `get_prompt` before writing your own.
7. Prior decision / constraint / rejected approach? → `search_memory` before deciding; `add_memory` after (see Memory).
8. Code question (symbols, callers, blast radius)? → `code_context` / `code_impact` after `get_code_brief`. If the index is missing/stale, run `setup_code_graph` once, then retry.
9. Nothing indexed yet or doc missing? → `list_files` / `status` to diagnose; `ingest_file` / `ingest_data` to add (see Roles).

## Tools Available (pre-scoped to "GGUF-Switchboard")
### Plan & search
- `planning_context(project_name, task)` - Structured planning context: overview + requirements + decisions with sources. Call before planning, coding, estimating, or refactoring.
- `requirement_lookup(project_name, requirement, limit?, docTypes?, tags?)` - Exact requirement lookup (defaults to spec/plan docs).
- `search_project_docs(project_name, query, limit?, docTypes?, tags?)` - Project-scoped hybrid keyword + semantic search. Returns filePath, chunkIndex, text, fileTitle, score (0 = best).
- `get_project_brief(project_name)` - Project overview (structured hierarchy when available, else top overview/requirements chunks).
- `read_chunk_neighbors(chunkIndex, filePath | source, before?, after?)` - Expand one search hit with surrounding chunks. Pass chunkIndex plus exactly one of filePath (ingest_file docs) or source (ingest_data docs). Defaults before=2, after=2.
- `list_projects()` - Fallback only: call if a query fails on the project id ("GGUF-Switchboard" not found / no access). Otherwise skip — the project id is already scoped in this file.
- `query_documents(query, limit?, scope?, docTypes?, tags?)` - Use directly: the project id is scoped in this file, so no extra project lookup is needed. Prefer `search_project_docs` when you need explicit project_name + score-ranked hits.
- `list_files()` / `status()` - Ingestion status and index stats. Use to diagnose "nothing found".

### Knowledge hierarchy (API/Postgres deployment)
- `get_project_structure(project_name)` - Domain → module → submodule tree (area_id, name, type, status, depth). Orient here before drilling in.
- `get_area_brief(project_name, area_id | area_name, detail_level?)` - Consolidated brief for one area (summary | standard | full).
- `search_area(project_name, query, area_id | area_name?, include_children?)` - Search within an area, optionally including descendants.
- `get_knowledge_item(project_name, item_id | display_key)` - One consolidated item with provenance.

### Prompt library (reusable instructions)
- Rule: `list_prompts` (filter by `tag?`, check `usageCount`) → `get_prompt(title)` before writing your own; only save a new prompt when the workflow is reusable. Use `add_memory` for decisions, `save_prompt` for reusable instructions.
- `list_prompts(project_name, tag?)` - List saved prompts { id, title, body, tags, usageCount }.
- `get_prompt(project_name, title)` - Fetch one prompt by exact title.
- `save_prompt(project_name, title, body, tags?)` / `rename_prompt` / `update_prompt` / `delete_prompt` - Manage prompts (see Roles).

### Memory (durable project knowledge)
- `search_memory(project_name, query, limit?)` - Recall decisions, preferences, rejected approaches, unresolved issues.
- `add_memory(project_name, messages | text)` - Record durable memory distilled from the conversation (see Roles).
- `forget_memory(project_name, memory_id)` - Delete one memory by id (audited, irreversible; see Roles).

### Code intelligence (when enabled)
- `code_context(project_name, query, area_id? | area_name?, limit?)` - Find symbols, files, signatures, callers by meaning.
- `code_impact(project_name, symbol_name? | file_path?, depth?)` - Blast radius: affected callers, dependent modules, related tests, linked requirements.
- `get_code_brief(project_name)` - Index overview: repos, analysis status, counts, staleness. Call before trusting code_context/code_impact, and after every upload to verify ready.
- `setup_code_graph(project_name)` - Returns a self-install shell script + upload URLs to index the repo. Never writes to your filesystem — write scriptContent to scriptPath, run it (auth reuses your harness MCP key for context-harbor: CH_MCP_TOKEN env wins, else auto-detected from MCP config — never ask the user for a key) with CH_BRANCH set to the deploy branch, then verify with get_code_brief. First run uploads a full bundle; reruns auto-send a delta (base_commit — only added/changed files re-embed). Shortcut for later syncs: `ch code sync` (same delta path).

### Setup helpers
- `setup_agent_harness(project_name, harness?, targets?, task?, repoPath?)` - Fastest install: returns the harness-specific install set with target paths in one call (claude-code: skill file + AGENTS.md + CLAUDE.md + Stop hook; other harnesses: skill file + AGENTS.md). Write the returned files yourself (skill: overwrite; AGENTS.md/CLAUDE.md: merge between markers; settings: merge-json). This tool never writes.
- `setup_harness(harness, project_name)` - Claude Code memory-capture enforcement (settings fragment + check script). The caller writes the files; this tool never writes.

### Write / ingest (see Roles)
- `ingest_file(filePath, projectName?, docType?, tags?, visual?, visualQuality?)` - Ingest a local file (PDF, DOCX, TXT, MD). Absolute path only; re-ingest replaces. PDF figures: visual=true (fast default, quality for in-image text).
- `ingest_data(content, metadata{source, format}, projectName?, docType?, tags?)` - Ingest raw text/HTML/Markdown. Source: URL for web pages, else {type}://{date}[/{detail}].
- `delete_file(filePath | source)` - Remove ingested content. Either filePath or source must be provided.

## Usage Pattern (copy-paste, already scoped)

```
planning_context(project_name="GGUF-Switchboard", task="<current task>")
requirement_lookup(project_name="GGUF-Switchboard", requirement="<question>")
search_project_docs(project_name="GGUF-Switchboard", query="<your search query>", limit=5)
```

## Examples

### Plan a feature (1 call, no hops)
```
planning_context(project_name="GGUF-Switchboard", task="implement user login feature")
```

### Finding requirements
```
requirement_lookup(project_name="GGUF-Switchboard", requirement="authentication requirements")
```

### General search
```
search_project_docs(project_name="GGUF-Switchboard", query="how does the API work", limit=10)
```

### Reading more context around a result
```
read_chunk_neighbors(chunkIndex=<chunkIndex from a search_project_docs hit>, filePath="<filePath from the same hit>", before=2, after=2)
```

### Orient in a large project, then drill in
```
get_project_brief(project_name="GGUF-Switchboard")
get_project_structure(project_name="GGUF-Switchboard")
get_area_brief(project_name="GGUF-Switchboard", area_name="<module>", detail_level="standard")
search_area(project_name="GGUF-Switchboard", query="<question>", area_name="<module>", include_children=true)
get_knowledge_item(project_name="GGUF-Switchboard", display_key="<display_key from search_area>")
```

### Reuse a saved prompt
```
list_prompts(project_name="GGUF-Switchboard")
get_prompt(project_name="GGUF-Switchboard", title="<exact title>")
```

### Recalling a prior decision
```
search_memory(project_name="GGUF-Switchboard", query="what did we decide about authentication")
```

### Recording a durable decision
```
add_memory(project_name="GGUF-Switchboard", messages=[{ "role": "assistant", "content": "Decided to use JWT auth with 24h expiry; rejected server-side sessions because ..." }])
```

### Code search and blast radius
```
get_code_brief(project_name="GGUF-Switchboard")
code_context(project_name="GGUF-Switchboard", query="where is auth token validation")
code_impact(project_name="GGUF-Switchboard", symbol_name="validateToken")
```

### Index the repo (full first, delta after, then verify)
```
setup_code_graph(project_name="GGUF-Switchboard")
# write returned scriptContent to returned scriptPath, then run:
# sh ./ch-code-graph-setup.sh <repo-dir> (auth reuses your harness MCP key for context-harbor — CH_MCP_TOKEN env overrides; CH_BRANCH sets the deploy branch)
# First run = full bundle upload; reruns = auto delta (base_commit). Later syncs: ch code sync.
get_code_brief(project_name="GGUF-Switchboard")
```

## Memory (durable project knowledge)
- Prefer the canonical `ch_memory_*` / `ch_session_*` tools (`ch_memory_search` to recall, `ch_memory_remember` to record, `ch_session_handoff` for handoffs); the legacy `search_memory` / `add_memory` / `forget_memory` aliases keep working during migration. These are the only durable-memory tools for "GGUF-Switchboard". Do not use any other memory MCP or plugin for this project's knowledge.
- Memory is separate from document search: use it for decisions, constraints, and rationale that should survive across sessions — not for document content, which belongs in ingest/RAG.
- **Capture is mandatory, not optional.** A task that reached a decision, resolved an ambiguity, or accepted a trade-off is not complete until memory has been reviewed. Purely mechanical work (formatting, typo fixes, routine dependency bumps) is exempt — default to reviewing when in doubt.
- Sequence: `ch_memory_search` (or `search_memory`) first to avoid duplicates, then `ch_memory_remember` (or `add_memory`) if nothing durable already covers it — otherwise treat it as explicitly NOT REQUIRED, don't just skip it silently.
- If remembering fails with a rate limit (HTTP 429), retry once; if it still fails, report the failure rather than dropping it silently.
- Only the top-level session should write memory (`ch_memory_remember` / `add_memory`, `ch_session_handoff` / `forget_memory`) — a delegated subagent should not write memory on its own.
- State whether memory was reviewed (and what was stored, or that it wasn't required) whenever a task completes — don't omit it.

## Roles & fallbacks (avoids failed-call hops)
- Manager-only (fails for viewers): `ingest_file`, `ingest_data`, `delete_file`, `add_memory`, `forget_memory`, `save_prompt`, `delete_prompt`, `rename_prompt`, `update_prompt`. If one fails with unauthorized, report it and continue read-only — don't retry.
- `code_context` / `code_impact` / `get_code_brief` fall back to an instructional message when code intelligence is disabled — run `setup_code_graph` first or continue with document search.
- `setup_harness` currently supports harness="claude-code" only.
- Scores: lower is better (0 = best). <0.3 use directly; 0.3–0.5 include if same concept; 0.5–0.7 only if directly relevant; >0.7 skip unless nothing better.

## Notes
- Always specify the project name: "GGUF-Switchboard"
- Use specific queries for better results
- Adjust the limit parameter based on how much context you need
- Prefer `docTypes` filters: ["spec","plan"] for requirements, ["transcript","minutes"] for history/decisions, ["api-doc"] for endpoints, ["design"] for architecture, ["code-summary"] for code, ["test"] for QA, ["status"] for progress
