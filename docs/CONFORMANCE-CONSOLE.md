# Conformance Console

> [← Back to README](../README.md)

The Conformance Console is a built-in diagnostic surface at **http://localhost:9090/swagger-ui/conformance.html**. It answers two questions that standard chat UIs cannot:

1. **"Did this model actually call the tool, or did it just talk about calling it?"** — Local models frequently fail to produce OpenAI-compatible structured `tool_calls` and instead dump JSON as plain text, leak tool-call JSON into `reasoning_content`, or produce nothing at all.

2. **"What does this model's embedded Jinja chat template actually resolve to?"** — Some models ship broken or missing chat templates. The only way to catch it is to inspect the rendered prompt directly.

Every run is automatically saved to `conformance.db` (SQLite) so you can compare results across model versions, llama.cpp builds, or template tweaks.

## Custom endpoint target

Every tab (except History) has a **Target** widget that lets you switch between:

- **GGUF Switchboard (local):** Uses the switchboard-managed model via the scheduler.
- **Custom OpenAI-compatible endpoint:** Send requests to any external OpenAI-compatible API (OpenAI, another proxy, a different local server). Configure with a base URL, model name, and optional API key.

API keys are held **in browser memory only** — never written to localStorage or the history database. Base URL and model name are persisted in localStorage per tab.

## Tabs

### Inspect

**What it's for:** Find out whether a model can actually produce a structured tool call, or whether it cheats by dumping JSON as plain text or leaking it into its reasoning trace. Use this when a model claims to support tools but your agent code never receives a `tool_calls` field — the console shows you exactly where the tool call ended up (or didn't).

**How it works:** Send any chat completion request (the body shape matches `/v1/chat/completions` — you can paste straight from Swagger). The console runs the request, then classifies every response choice into one of four outcomes:

| Badge | Meaning | What happened |
|-------|---------|---------------|
| **Structured tool call** (green) | Correct | The tool call landed in `message.tool_calls` where OpenAI-compatible clients expect it |
| **Dumped as plain text** (red) | Broken | The model wrote the tool call as a JSON blob inside `message.content` — your agent code won't parse it as a tool call |
| **Leaked into reasoning** (red) | Broken | The tool call JSON ended up in `message.reasoning_content` instead of `tool_calls` — common with thinking/reasoning models |
| **No tool call detected** (grey) | Missing | The model didn't produce a tool call anywhere in the response |

The request is forced to non-streaming so the full response can be inspected. A run passes only when every choice produces a structured tool call.

**Endpoint:** `POST /v1/conformance/inspect`

---

### Resolved Template

**What it's for:** See the actual prompt string that gets sent to the model after its Jinja chat template runs. Use this when a model produces garbled or nonsensical output — the problem is often a broken or missing embedded chat template, not the model itself. Models like GLM and DeepSeek variants are common offenders.

**How it works:** Provide a model name, a list of messages, and optionally some tools. The console asks the backend to render the Jinja template and shows you the resulting prompt string. If the backend doesn't support template rendering, it falls back to showing the raw template source so you can inspect it manually.

| Result | Meaning |
|--------|---------|
| **Resolved** (pass) | The backend rendered the template successfully — you see the exact prompt the model will receive |
| **Template-only** (fail) | The backend couldn't render the template, but returned the raw Jinja source for manual inspection |
| **Error** (fail) | Both template rendering and source retrieval failed |

**Endpoint:** `POST /v1/conformance/resolve-template`

---

### Battery

**What it's for:** Quickly check whether a model handles the four most common tool-calling patterns correctly. Instead of manually crafting requests and reading raw JSON, run the battery and get a clear pass/fail table. Use this when evaluating a new model, comparing llama.cpp builds, or verifying that a chat template change didn't break tool calling.

**How it works:** Four fixed scenarios run sequentially against one model. Each tests a different aspect of tool-calling behavior:

#### Case 1: Single tool call

**Prompt:** "Call the echo tool with message set to 'hello'." (tool_choice: required)

**What it checks:** Can the model produce a basic structured tool call at all? This is the same check the load-time tool probe runs.

**Pass:** Response has a structured `tool_calls` entry with the correct echo call.

---

#### Case 2: Parallel tool calls

**Prompt:** "Call get_weather for both Paris and Tokyo." (one get_weather tool, tool_choice: required)

**What it checks:** Can the model make more than one tool call in a single response? Some models collapse parallel calls into a single call or duplicate the same call twice.

**Pass:** Response has 2+ structured `tool_calls` with distinct arguments.

---

#### Case 3: Tool call with reasoning

**Prompt:** "Think step by step about which tool to use, then call the echo tool with message set to 'hello'." (max_tokens: 512)

**What it checks:** Can the model reason about tool use and still emit the structured call? Some models think about calling the tool in their reasoning chain but never actually emit the structured call, or can do one but not the other.

**Pass:** Response has both a structured `tool_calls` entry and non-empty `reasoning_content`.

---

#### Case 4: Multi-turn tool result

**Prompt:** A multi-turn conversation where the user asks about weather, the assistant makes a tool call, the tool returns JSON data, and the model must summarize the result.

**What it checks:** Can the model summarize a tool result in natural language, or does it just echo back the raw JSON? This is the most common failure mode when using tools with local models.

**Pass:** Final content is non-empty natural language (not a raw JSON dump).

---

**Overall:** All 4 cases must pass for the battery to report success.

**Endpoint:** `POST /v1/conformance/battery/{model_id}`

---

### Compare

**What it's for:** Put two models side by side on the exact same task and see which one handles tool calling better. Use this when deciding between two quantizations, two models of similar size, or before and after a llama.cpp update.

**How it works:** Pick two models (Model A and Model B), then choose either a specific battery case or write a custom request. The console runs the same request against both models and shows the results in a side-by-side grid.

Two modes:

- **Battery case:** Runs the selected battery case against each model and shows whether each passed or failed.
- **Custom request:** Runs an inspect classification on each model's response and shows where each model placed the tool call.

**Important:** Since the switchboard loads one model at a time, this means two full model swaps — expect it to take as long as loading each model.

**Endpoint:** `POST /v1/conformance/compare`

---

### History

**What it's for:** Review past conformance runs over time. Use this to track whether a model's tool-calling behavior has changed across llama.cpp versions, to audit which models have been tested, or to find a specific run you want to revisit.

**How it works:** Every Inspect, Resolved Template, Battery, and Compare run is automatically saved to `conformance.db` (a local SQLite database, sibling of the token-usage `usage.db`). The History tab shows all recorded runs with:

- Timestamp of the run
- Kind (inspect / battery / compare / resolve_template)
- Model(s) involved
- Summary text
- Pass/fail badge

Filter by kind or model, click a row to expand the full stored response, or clear the entire history.

**Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/v1/conformance/history` | List recent runs (with optional `kind` and `model` filters) |
| `GET` | `/v1/conformance/history/{id}` | Get one run with full detail |
| `DELETE` | `/v1/conformance/history/{id}` | Delete one run |
| `DELETE` | `/v1/conformance/history` | Clear all history |
