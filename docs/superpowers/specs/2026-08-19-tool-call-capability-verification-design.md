# Tool-Call Capability Verification

## Goal

Stop trusting the static, HF-tag-derived `capabilities: ["tools"]` flag
([src/config/hf_sync.rs:149](../../../src/config/hf_sync.rs)) as a promise that a
served model can actually round-trip OpenAI-compatible tool calls. Verify it at
load time with a real probe against the running backend, gate tool-bearing
requests on the verdict, and fix the one known llama.cpp serialization defect
that would otherwise cause a verified model to fail anyway.

## Background

`gguf-switchboard` is a proxy in front of `llama-server`. Today it is a pure
passthrough for `tools` / `tool_calls` — [src/sanitize.rs](../../../src/sanitize.rs)
only strips Swagger UI placeholder values, and
[2026-07-30-responses-tool-calling-design.md](2026-07-30-responses-tool-calling-design.md)
explicitly decided the `/v1/responses` translation layer would not parse,
normalize, or repair model-generated arguments. That decision holds for
*model*-generated malformed output. It does not cover a llama.cpp
*serialization* bug: `tool_calls[].function.arguments` is sometimes emitted as
a JSON object instead of the OpenAI-required JSON string, which breaks
strict clients (including OpenCode) regardless of whether the model's tool
call was itself well-formed. This spec treats that case as upstream
transport corruption, not model behavior, and fixes it — see "Argument
Normalization" below for why this narrowly supersedes the prior passthrough
decision.

The `capabilities` tag is otherwise inferred once, offline, from HF
`pipeline_tag`/`tags` string matching
([capabilities_from_hf](../../../src/config/hf_sync.rs)) and never checked
against real backend behavior. A model can carry `tools` in its capability
list and still fail every tool call once served, with no signal to the
caller beyond a malformed-JSON error deep in a client like OpenCode.

## Scope

The implementation will:

- run a one-shot tool-call probe against a model's backend after it becomes
  healthy and before it is reported ready, but only for models whose
  registry `capabilities` already includes `tools`;
- classify the result as `Verified`, `Failed(reason)`, or `Skipped` (no
  `tools` capability declared) and hold it in in-memory runtime state, keyed
  by model id, recomputed on every load;
- surface the verdict in `/v1/models` output (e.g. `tools_verified: bool`)
  alongside the existing static `capabilities` list;
- reject inbound chat-completion requests that include `tools`/`tool_choice`
  for a model whose current verdict is `Failed`, with a clear 4xx error
  naming the model and reason, instead of forwarding a request that will
  produce malformed output;
- normalize `tool_calls[].function.arguments` from a JSON object to its
  string-encoded form, both in the probe's own validation and on the normal
  request path, whenever the backend emits the object form; and
- add unit tests for probe response validation, the request-gate logic, and
  the argument normalization.

The implementation will not:

- persist verdicts to `models.toml` or across process restarts;
- build a per-model-family parser registry (Ollama-style `qwen3`/`gemma4`/
  `harmony` parsers) — out of scope for this slice; the probe either trusts
  llama.cpp's own template/parser output or it doesn't;
- repair any other form of malformed model output (missing tool name,
  invalid JSON *content* inside the arguments string, etc.) — those remain
  `Failed` verdicts, not repair targets;
- change the static `capabilities` tag written by `sync-hf-metadata`; it
  remains the input that decides whether a probe runs at all; or
- add capability gating for `/v1/responses` (the existing translation layer
  is a separate concern; this spec's gate applies at the Chat Completions
  entry point both endpoints funnel through).

## Design

### Probe

A new `tool_probe` module sends one non-streaming chat-completion request to
the backend's own HTTP port (the same port the scheduler already health-checks)
once `wait_until_healthy` returns `Ok`
([src/scheduler/mod.rs:961](../../../src/scheduler/mod.rs)), at each of its
three call sites (:1071, :1207, :1331). The probe request:

```json
{
  "model": "<alias>",
  "messages": [
    {"role": "user", "content": "Call the echo tool with message set to \"hello\"."}
  ],
  "tools": [{
    "type": "function",
    "function": {
      "name": "echo",
      "description": "Echo a message back.",
      "parameters": {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"]
      }
    }
  }],
  "tool_choice": "required"
}
```

Validation requires: a `tool_calls` array of length 1, `function.name ==
"echo"`, and `function.arguments` deserializes (after normalization, see
below) to `{"message": "hello"}` or a semantically equivalent value. Anything
else — empty `tool_calls`, the call text landing in `content` instead
(the Gemma-family failure mode from the motivating report), a timeout, or a
non-2xx response — is `Failed(reason)`, where `reason` is a short static
string (`"no_tool_calls"`, `"malformed_json"`, `"backend_error"`,
`"timeout"`) for logging and the eventual 4xx message. The probe runs with a
5-second timeout and does not retry; a `Failed` verdict from a transient
backend hiccup is acceptable since the model still serves chat traffic — only
tool-bearing requests are affected, and a subsequent reload re-probes.

Models without `tools` in their static `capabilities` are `Skipped` — no
probe request is sent, avoiding load-time cost for the majority of chat/coder
models.

### Runtime state

`AppState` gains a `tool_capability: RwLock<HashMap<String, ToolCapability>>`
(or the existing state's established pattern for per-model runtime data, to
be confirmed against [src/state/mod.rs](../../../src/state/mod.rs) during
implementation):

```rust
enum ToolCapability {
    Verified,
    Failed(&'static str),
    Skipped,
}
```

This is intentionally not part of `RegistryEntry`/`models.toml` — it is a
property of the *running* backend (quant, template flags, and llama-server
version can all change what a given alias actually does), not a static
config fact. It is dropped on process restart and recomputed on the next
load, same lifecycle as other scheduler runtime state.

### Request gate

The existing chat-completions handler checks, before forwarding: if the
request has a non-empty `tools` array and `AppState.tool_capability.get(model)
== Some(Failed(reason))`, return HTTP 400 with a body naming the model, the
failure reason, and (if any) other aliases currently `Verified`. Requests
without `tools`, or against a `Verified`/`Skipped` model, are forwarded
unchanged — chat-only traffic is never affected by this feature.

### Argument normalization

`tool_calls[].function.arguments` must be an OpenAI-compatible JSON string.
When the backend instead emits it as a JSON object (the known llama.cpp
defect referenced in the motivating report), both the probe's own validator
and the normal chat-completions response path re-serialize it to a string
before it reaches the caller. This is a narrow, targeted fix for one
confirmed upstream serialization bug — not general model-output repair — and
is applied uniformly regardless of probe verdict, since it corrects a
transport-level defect rather than a model competence issue. This
supersedes the "no normalize/repair" boundary in
[2026-07-30-responses-tool-calling-design.md](2026-07-30-responses-tool-calling-design.md)
for this one case only; that spec's refusal to repair *model*-generated
malformed arguments (bad JSON content, wrong types, etc.) still stands.

## Error Handling

- Probe transport failure (connection refused, timeout) → `Failed("timeout")`
  or `Failed("backend_error")`; logged at `warn!` with model id; does not
  block the model from becoming ready for chat-only traffic.
- Probe returns a well-formed non-tool response (model ignored `tool_choice:
  required`) → `Failed("no_tool_calls")`.
- Probe returns tool_calls with unparseable arguments even after object→string
  normalization → `Failed("malformed_json")`.
- Gated request rejection is a client-facing 400, not a 5xx — this is a
  request the caller should not have sent to this model, not a server fault.

## Testing

1. Unit tests for the probe response validator: valid string args, valid
   object args (normalized), missing `tool_calls`, wrong function name, tool
   call text leaked into `content` instead of `tool_calls`.
2. Unit tests for the request-gate: `Verified`/`Failed`/`Skipped` crossed with
   request-with-tools/without-tools, confirming only
   `Failed` + `tools present` produces a 400.
3. Unit test for argument normalization applied on the live request path
   (not just inside the probe).
4. Scheduler-level test (using the existing mock-backend pattern near the
   `load_model_id` tests in [src/scheduler/mod.rs](../../../src/scheduler/mod.rs))
   confirming the probe runs exactly once per load and the verdict is cached
   for subsequent `ensure_loaded` calls without re-probing.
5. `./precommit.sh` gates formatting, compilation, linting, and the full
   test suite before completion, per repository convention.

## Compatibility Impact

- `/v1/models` output gains a new field (`tools_verified` or similar) under
  each model's capability info — additive, non-breaking for existing
  consumers.
- A model previously advertised as `tools`-capable but that fails the probe
  now rejects tool-bearing requests with a 400 instead of silently returning
  malformed tool-call JSON. This is an intentional behavior change: callers
  relying on the old silent-failure behavior (if any) will need to check
  `tools_verified` or handle the new 400.
- No change to models without a `tools` capability tag, and no change to
  chat-only traffic for any model.
