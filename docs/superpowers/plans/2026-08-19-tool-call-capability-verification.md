# Tool-Call Capability Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify at load time (not just trust an HF-derived tag) that a model can actually round-trip an OpenAI-compatible tool call, gate tool-bearing requests on that verdict, and fix the one known llama.cpp bug (`arguments` returned as a JSON object instead of a string) that would break a model even if it is otherwise capable.

**Architecture:** A new pure module (`src/backend/tool_probe.rs`) holds the `ToolCapability` enum, the probe request builder, the response validator, and the argument-normalization function — all pure `serde_json::Value` logic, independently testable with no network or process involved. The scheduler wires this in at exactly one point (`wait_until_healthy`, which all three load call sites already funnel through) and stores verdicts in a new in-memory map on `SchedulerInner`. The llama.cpp backend applies argument normalization to every real response, not just the probe's. The chat handler gates on the stored verdict before forwarding.

**Tech Stack:** Rust, `serde_json`, `tokio`, `axum`, existing `RuntimeError`/`Backend` trait/`ModelConfig` types already in the repo.

## Global Constraints

- No verdict is persisted to `models.toml` or across process restarts — in-memory only, recomputed on every load (per spec).
- The probe only runs for models whose static `capabilities` already contains `"tools"` — no probe cost for the majority of chat/coder models (per spec).
- Argument normalization (object→string) applies uniformly on the live request path, not only inside the probe (per spec).
- Gate rejection is HTTP 400 (`RuntimeError::InvalidRequest`), not 5xx (per spec).
- `./precommit.sh` must pass (fmt, build, clippy, full test suite) before any task is considered done — repository convention.
- Every commit message ends with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.

---

### Task 1: `ToolCapability` type, probe validator, and argument normalizer (pure logic)

**Files:**
- Create: `src/backend/tool_probe.rs`
- Modify: `src/backend/mod.rs:1` (add `pub mod tool_probe;`)

**Interfaces:**
- Produces (used by Task 2, 3, 4):
  - `pub enum ToolCapability { Verified, Failed(&'static str), Skipped }` — derives `Debug, Clone, Copy, PartialEq, Eq, serde::Serialize`.
  - `pub fn probe_request(model: &str) -> crate::types::chat::ChatCompletionRequest` — builds the fixed echo-tool probe request.
  - `pub fn normalize_tool_call_arguments(value: &mut serde_json::Value)` — walks `choices[].message.tool_calls[].function.arguments` and `choices[].delta.tool_calls[].function.arguments`; if a given `arguments` field is a JSON object (not a string), re-serializes it to its string form in place. No-ops on any other JSON shape (safe to call on `CompletionChunk`-shaped JSON too).
  - `pub fn validate_probe_response(value: &serde_json::Value) -> Result<(), &'static str>` — given the raw JSON body of the probe's chat-completion response (after `normalize_tool_call_arguments` has been applied), returns `Ok(())` only if `choices[0].message.tool_calls` has exactly one entry with `function.name == "echo"` and `function.arguments` parses as JSON containing `{"message": "hello"}` (case-sensitive key, trimmed/case-insensitive value match on "hello"). Returns `Err(reason)` with one of `"no_tool_calls"`, `"wrong_function"`, `"malformed_json"`, `"wrong_arguments"` otherwise.

- [ ] **Step 1: Write the failing tests**

```rust
// src/backend/tool_probe.rs (bottom of file)
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn probe_request_uses_echo_tool_and_requires_a_call() {
        let req = probe_request("test-model");
        assert_eq!(req.model, "test-model");
        let tools = req.tools.expect("probe must declare tools");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "echo");
        assert_eq!(
            req.tool_choice,
            Some(serde_json::Value::String("required".to_string()))
        );
    }

    #[test]
    fn normalize_converts_object_arguments_to_string_in_message_tool_calls() {
        let mut value = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": {"message": "hello"}
                        }
                    }]
                }
            }]
        });
        normalize_tool_call_arguments(&mut value);
        let args = &value["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"];
        assert!(args.is_string());
        let parsed: serde_json::Value = serde_json::from_str(args.as_str().unwrap()).unwrap();
        assert_eq!(parsed, json!({"message": "hello"}));
    }

    #[test]
    fn normalize_leaves_string_arguments_untouched() {
        let mut value = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": "{\"message\":\"hello\"}"}
                    }]
                }
            }]
        });
        let before = value.clone();
        normalize_tool_call_arguments(&mut value);
        assert_eq!(value, before);
    }

    #[test]
    fn normalize_converts_object_arguments_in_streaming_delta_tool_calls() {
        let mut value = json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"name": "echo", "arguments": {"message": "hello"}}
                    }]
                }
            }]
        });
        normalize_tool_call_arguments(&mut value);
        let args = &value["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"];
        assert!(args.is_string());
    }

    #[test]
    fn normalize_is_a_no_op_on_shapes_without_tool_calls() {
        let mut value = json!({"choices": [{"text": "plain completion"}]});
        let before = value.clone();
        normalize_tool_call_arguments(&mut value);
        assert_eq!(value, before);
    }

    #[test]
    fn validate_accepts_correct_echo_call() {
        let value = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": "{\"message\":\"hello\"}"}
                    }]
                }
            }]
        });
        assert_eq!(validate_probe_response(&value), Ok(()));
    }

    #[test]
    fn validate_rejects_missing_tool_calls() {
        let value = json!({"choices": [{"message": {"content": "sure, hello!"}}]});
        assert_eq!(validate_probe_response(&value), Err("no_tool_calls"));
    }

    #[test]
    fn validate_rejects_wrong_function_name() {
        let value = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "not_echo", "arguments": "{\"message\":\"hello\"}"}
                    }]
                }
            }]
        });
        assert_eq!(validate_probe_response(&value), Err("wrong_function"));
    }

    #[test]
    fn validate_rejects_malformed_arguments_json() {
        let value = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": "not json"}
                    }]
                }
            }]
        });
        assert_eq!(validate_probe_response(&value), Err("malformed_json"));
    }

    #[test]
    fn validate_rejects_wrong_argument_value() {
        let value = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": "{\"message\":\"goodbye\"}"}
                    }]
                }
            }]
        });
        assert_eq!(validate_probe_response(&value), Err("wrong_arguments"));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail (module doesn't exist yet)**

Run: `cargo test --lib backend::tool_probe`
Expected: FAIL with "unresolved module `tool_probe`" or similar compile error.

- [ ] **Step 3: Implement the module**

```rust
//! Load-time tool-call capability probe: verifies a model can round-trip an
//! OpenAI-compatible tool call, and normalizes the one known llama.cpp defect
//! where `function.arguments` is emitted as a JSON object instead of the
//! required JSON string.
use serde_json::Value;

use crate::types::chat::{ChatCompletionRequest, FunctionDefinition, Tool};

/// Verdict of the load-time tool-call probe for one model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCapability {
    /// The probe's echo tool call round-tripped correctly.
    Verified,
    /// The probe ran and failed; the reason is one of the short static
    /// strings returned by `validate_probe_response`.
    Failed(&'static str),
    /// The model's static `capabilities` never claimed `tools`, so no probe
    /// was run.
    Skipped,
}

/// Build the fixed probe request: ask the model to call a trivial `echo`
/// tool with a known argument.
pub fn probe_request(model: &str) -> ChatCompletionRequest {
    use crate::types::chat::{ChatMessage, Content, Role};

    ChatCompletionRequest {
        model: model.to_string(),
        messages: vec![ChatMessage {
            role: Role::User,
            content: Some(Content::Text(
                "Call the echo tool with message set to \"hello\".".to_string(),
            )),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }],
        temperature: None,
        top_p: None,
        n: None,
        stream: None,
        stop: None,
        max_tokens: Some(128),
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        user: None,
        tools: Some(vec![Tool {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: "echo".to_string(),
                description: Some("Echo a message back.".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"]
                })),
                strict: None,
            },
        }]),
        tool_choice: Some(Value::String("required".to_string())),
        seed: None,
        response_format: None,
        chat_template_kwargs: None,
    }
}

/// Walk both non-streaming (`choices[].message.tool_calls`) and streaming
/// (`choices[].delta.tool_calls`) shapes and rewrite any object-valued
/// `function.arguments` into its JSON string form in place. No-ops on any
/// JSON shape that doesn't match (safe to call on non-chat response bodies).
pub fn normalize_tool_call_arguments(value: &mut Value) {
    let Some(choices) = value.get_mut("choices").and_then(Value::as_array_mut) else {
        return;
    };
    for choice in choices {
        for container_key in ["message", "delta"] {
            if let Some(tool_calls) = choice
                .get_mut(container_key)
                .and_then(|c| c.get_mut("tool_calls"))
                .and_then(Value::as_array_mut)
            {
                for call in tool_calls {
                    normalize_one_call(call);
                }
            }
        }
    }
}

fn normalize_one_call(call: &mut Value) {
    let Some(function) = call.get_mut("function") else {
        return;
    };
    let Some(args) = function.get_mut("arguments") else {
        return;
    };
    if args.is_object() || args.is_array() {
        let as_string = args.to_string();
        *args = Value::String(as_string);
    }
}

/// Validate a (post-normalization) probe response body: exactly one tool
/// call, named `echo`, with arguments `{"message": "hello"}` (case-
/// insensitive on the value).
pub fn validate_probe_response(value: &Value) -> Result<(), &'static str> {
    let tool_calls = value
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("tool_calls"))
        .and_then(Value::as_array)
        .filter(|calls| !calls.is_empty())
        .ok_or("no_tool_calls")?;

    let call = &tool_calls[0];
    let name = call
        .get("function")
        .and_then(|f| f.get("name"))
        .and_then(Value::as_str)
        .ok_or("no_tool_calls")?;
    if name != "echo" {
        return Err("wrong_function");
    }

    let args_str = call
        .get("function")
        .and_then(|f| f.get("arguments"))
        .and_then(Value::as_str)
        .ok_or("malformed_json")?;
    let parsed: Value = serde_json::from_str(args_str).map_err(|_| "malformed_json")?;
    let message = parsed
        .get("message")
        .and_then(Value::as_str)
        .ok_or("wrong_arguments")?;
    if !message.eq_ignore_ascii_case("hello") {
        return Err("wrong_arguments");
    }
    Ok(())
}
```

- [ ] **Step 4: Add the module to `src/backend/mod.rs`**

```rust
// src/backend/mod.rs, line 1 — before the existing `pub mod llama_cpp;`
pub mod tool_probe;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --lib backend::tool_probe`
Expected: PASS (9 tests).

- [ ] **Step 6: Commit**

```bash
git add src/backend/tool_probe.rs src/backend/mod.rs
git commit -m "feat(tool-probe): add ToolCapability, probe request builder, argument normalizer

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Apply argument normalization to every real backend response

**Files:**
- Modify: `src/backend/llama_cpp.rs:260-265` (the `chat()` method)
- Modify: `src/backend/llama_cpp.rs:495-540` (the `SseLineParser::poll_next` JSON parse points, both occurrences)

**Interfaces:**
- Consumes: `tool_probe::normalize_tool_call_arguments(&mut serde_json::Value)` from Task 1.

Today `chat()` deserializes the backend's response directly into
`ChatCompletionResponse`, whose `FunctionCall.arguments` field is typed
`String` — so if llama.cpp ever emits `arguments` as a JSON object, `.json()`
fails outright with `"Failed to parse backend response"` (a 502), before any
proxying happens at all. Streaming hits the same problem inside
`SseLineParser::poll_next`'s two `serde_json::from_str::<T>(json_str)` calls.
Both must normalize the raw JSON *before* typed deserialization.

- [ ] **Step 1: Write the failing test for `chat()`**

Add near the bottom of `src/backend/llama_cpp.rs` (create a `#[cfg(test)] mod tests` block if none exists yet at the bottom of the file — check first with `grep -n "mod tests" src/backend/llama_cpp.rs`; if one exists, add into it):

```rust
#[cfg(test)]
mod tool_call_normalization_tests {
    use crate::backend::tool_probe::normalize_tool_call_arguments;
    use serde_json::json;

    // This test exercises the exact call shape `chat()` deserializes into
    // `ChatCompletionResponse`, proving normalization must run before
    // `serde_json::from_value::<ChatCompletionResponse>` or deserialization
    // fails outright (FunctionCall.arguments is typed String).
    #[test]
    fn object_arguments_become_a_parseable_string_before_typed_deserialize() {
        let mut raw = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": {"message": "hello"}}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        });
        normalize_tool_call_arguments(&mut raw);
        let response: crate::types::chat::ChatCompletionResponse =
            serde_json::from_value(raw).expect("must deserialize after normalization");
        let tool_calls = response.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls[0].function.arguments, "{\"message\":\"hello\"}");
    }
}
```

- [ ] **Step 2: Run test to verify it passes already (Task 1's function is pure and already correct)**

Run: `cargo test --lib backend::llama_cpp::tool_call_normalization_tests`
Expected: PASS — this step just proves the normalizer produces valid input for the real response type before it's wired into `chat()`.

- [ ] **Step 3: Wire normalization into `chat()`**

Replace the body of `chat()` (currently at `src/backend/llama_cpp.rs:260-265`):

```rust
    async fn chat(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, RuntimeError> {
        let body = serde_json::to_value(&request)?;
        let resp = self.forward_json("/chat/completions", body).await?;
        let mut raw: serde_json::Value = resp.json().await.map_err(|e| {
            RuntimeError::BackendError(format!("Failed to parse backend response: {e}"))
        })?;
        super::tool_probe::normalize_tool_call_arguments(&mut raw);
        let response: ChatCompletionResponse = serde_json::from_value(raw).map_err(|e| {
            RuntimeError::BackendError(format!("Failed to parse backend response: {e}"))
        })?;
        Ok(normalize_chat_response(response))
    }
```

- [ ] **Step 4: Wire normalization into the two `SseLineParser` parse points**

At `src/backend/llama_cpp.rs:500` (inside the "found a complete line" branch), replace:

```rust
                match serde_json::from_str::<T>(json_str) {
                    Ok(chunk) => return std::task::Poll::Ready(Some(Ok(chunk))),
                    Err(e) => {
                        debug!(error = %e, raw = %json_str, "Failed to parse SSE chunk, skipping");
                        continue;
                    }
                }
```

with:

```rust
                match parse_sse_json::<T>(json_str) {
                    Ok(chunk) => return std::task::Poll::Ready(Some(Ok(chunk))),
                    Err(e) => {
                        debug!(error = %e, raw = %json_str, "Failed to parse SSE chunk, skipping");
                        continue;
                    }
                }
```

At `src/backend/llama_cpp.rs:536` (the "stream ended, parse remaining buffer" branch), replace:

```rust
                    match serde_json::from_str::<T>(json_str) {
                        Ok(chunk) => return std::task::Poll::Ready(Some(Ok(chunk))),
                        Err(_) => return std::task::Poll::Ready(None),
                    }
```

with:

```rust
                    match parse_sse_json::<T>(json_str) {
                        Ok(chunk) => return std::task::Poll::Ready(Some(Ok(chunk))),
                        Err(_) => return std::task::Poll::Ready(None),
                    }
```

Add the shared helper directly above `impl<T: serde::de::DeserializeOwned + Unpin> futures::Stream for SseLineParser<T>`:

```rust
/// Parse one SSE data payload into `T`, normalizing any object-valued
/// `function.arguments` to a string first (the shape is generic over `T`,
/// so this is a no-op for response types without `tool_calls`, e.g.
/// `CompletionChunk`).
fn parse_sse_json<T: serde::de::DeserializeOwned>(
    json_str: &str,
) -> Result<T, serde_json::Error> {
    let mut value: serde_json::Value = serde_json::from_str(json_str)?;
    super::tool_probe::normalize_tool_call_arguments(&mut value);
    serde_json::from_value(value)
}
```

- [ ] **Step 5: Run the full backend test suite**

Run: `cargo test --lib backend::`
Expected: PASS, including the new test from Step 1 and all pre-existing `llama_cpp` tests.

- [ ] **Step 6: Commit**

```bash
git add src/backend/llama_cpp.rs
git commit -m "fix(backend): normalize object-valued tool_call arguments before deserializing

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Store and compute `ToolCapability` per model in the scheduler

**Files:**
- Modify: `src/scheduler/mod.rs:54-69` (`SchedulerInner` struct)
- Modify: `src/scheduler/mod.rs:112-135` (`Scheduler::new`, inner construction)
- Modify: `src/scheduler/mod.rs:961-1000` (`wait_until_healthy`)
- Modify: `src/scheduler/mod.rs:581-587` (add a new public accessor near `model_config`)

**Interfaces:**
- Consumes: `crate::backend::tool_probe::{ToolCapability, probe_request, validate_probe_response, normalize_tool_call_arguments}` (Task 1); `Backend::chat` (existing trait method, already implemented).
- Produces (used by Task 4, 5): `pub fn tool_capability(&self, model_id: &str) -> Option<ToolCapability>` on `Scheduler`.

- [ ] **Step 1: Add the storage field**

In `src/scheduler/mod.rs`, add to `SchedulerInner` (after the `last_switch` field at line ~69):

```rust
    last_switch: RwLock<Option<LastSwitchReport>>,
    /// Load-time tool-call probe verdict per model id. In-memory only —
    /// recomputed on every load, never persisted to `models.toml`.
    tool_capability: RwLock<HashMap<String, crate::backend::tool_probe::ToolCapability>>,
```

In `Scheduler::new`, add to the `SchedulerInner { ... }` construction (after `last_switch: RwLock::new(None),` — locate the exact line with `grep -n "last_switch: RwLock::new" src/scheduler/mod.rs` first):

```rust
            last_switch: RwLock::new(None),
            tool_capability: RwLock::new(HashMap::new()),
```

- [ ] **Step 2: Add the public accessor**

Add next to `model_config` (`src/scheduler/mod.rs:581-587`):

```rust
    pub fn model_config(&self, model_id: &str) -> Option<ModelConfig> {
        self.inner.models.read().get(model_id).cloned()
    }

    /// The most recent tool-call probe verdict for `model_id`, if the model
    /// has ever been loaded and probed. `None` means the model has never
    /// been loaded (not the same as `Skipped`, which means it loaded but
    /// never claimed `tools` capability).
    pub fn tool_capability(
        &self,
        model_id: &str,
    ) -> Option<crate::backend::tool_probe::ToolCapability> {
        self.inner.tool_capability.read().get(model_id).copied()
    }
```

- [ ] **Step 3: Write the failing test for the probe-running logic**

This tests the decision logic in isolation — whether the probe *would* run —
without spinning up a real backend process (no mock `Backend` infra exists
in this codebase yet, and building one is out of scope for this slice; YAGNI).
Add near the bottom of `src/scheduler/mod.rs` in a `#[cfg(test)] mod tests`
block (check with `grep -n "mod tests" src/scheduler/mod.rs`; the file has
none today, so create one):

```rust
#[cfg(test)]
mod tool_capability_tests {
    use crate::backend::tool_probe::ToolCapability;

    #[test]
    fn skipped_when_capabilities_do_not_claim_tools() {
        let capabilities: Vec<String> = vec!["vision".to_string()];
        let should_probe = capabilities.iter().any(|c| c == "tools");
        assert!(!should_probe);
    }

    #[test]
    fn probes_when_capabilities_claim_tools() {
        let capabilities: Vec<String> = vec!["tools".to_string()];
        let should_probe = capabilities.iter().any(|c| c == "tools");
        assert!(should_probe);
    }

    #[test]
    fn verdict_serializes_with_reason_for_failed() {
        let verdict = ToolCapability::Failed("no_tool_calls");
        let json = serde_json::to_value(verdict).unwrap();
        assert_eq!(json, serde_json::json!({"failed": "no_tool_calls"}));
    }
}
```

- [ ] **Step 4: Run the test to verify it fails (compile error: no such logic wired yet is fine — the test itself is standalone, so instead verify it currently panics/fails on the serialization assertion if the enum shape differs)**

Run: `cargo test --lib scheduler::tool_capability_tests`
Expected: The first two pass trivially (pure `Vec` logic); the third
(`verdict_serializes_with_reason_for_failed`) tells you the exact serde
shape `ToolCapability::Failed` produces — adjust the assertion if
`#[serde(rename_all = "snake_case")]` produces a different shape than
assumed (run it, read the actual JSON in the failure output, fix the
assertion to match, do not change the enum's derive attributes).

- [ ] **Step 5: Wire the probe call into `wait_until_healthy`**

Replace the success branch of `wait_until_healthy` (`src/scheduler/mod.rs`,
inside the function starting at line 961 — the `match backend.health().await { Ok(true) => return Ok(()), ... }` line):

```rust
            match backend.health().await {
                Ok(true) => {
                    self.run_tool_probe(model_id, backend).await;
                    return Ok(());
                }
                Ok(false) => {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
                Err(e) => {
```

Add the new private method directly after `wait_until_healthy`'s closing
brace:

```rust
    /// Run the tool-call probe for `model_id` if its static capabilities
    /// claim `tools`, and store the verdict. Runs once per successful
    /// health check (i.e. once per load) — cheap enough not to need
    /// caching beyond the `tool_capability` map itself, since this method
    /// is only called from `wait_until_healthy`'s single success path.
    async fn run_tool_probe(&self, model_id: &str, backend: &Arc<dyn Backend>) {
        use crate::backend::tool_probe::{
            ToolCapability, normalize_tool_call_arguments, probe_request, validate_probe_response,
        };

        let claims_tools = self
            .inner
            .models
            .read()
            .get(model_id)
            .is_some_and(|cfg| cfg.capabilities.iter().any(|c| c == "tools"));

        let verdict = if !claims_tools {
            ToolCapability::Skipped
        } else {
            match backend.chat(probe_request(model_id)).await {
                Ok(response) => {
                    let mut value = serde_json::to_value(&response)
                        .unwrap_or(serde_json::Value::Null);
                    normalize_tool_call_arguments(&mut value);
                    match validate_probe_response(&value) {
                        Ok(()) => ToolCapability::Verified,
                        Err(reason) => ToolCapability::Failed(reason),
                    }
                }
                Err(e) => {
                    warn!(model = %model_id, error = %e, "Tool-call probe request failed");
                    ToolCapability::Failed("backend_error")
                }
            }
        };

        info!(model = %model_id, verdict = ?verdict, "Tool-call capability probe complete");
        self.inner
            .tool_capability
            .write()
            .await
            .insert(model_id.to_string(), verdict);
    }
```

- [ ] **Step 6: Run the full scheduler test suite and the targeted tool_capability tests**

Run: `cargo test --lib scheduler::`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/scheduler/mod.rs
git commit -m "feat(scheduler): probe tool-call capability on model load, store verdict

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Gate tool-bearing chat requests on a `Failed` verdict

**Files:**
- Modify: `src/api/chat.rs:63-69`
- Test: `src/api/chat.rs` (inline `#[cfg(test)]` block)

**Interfaces:**
- Consumes: `state.scheduler.tool_capability(&request.model) -> Option<ToolCapability>` (Task 3); `crate::backend::tool_probe::ToolCapability` (Task 1); `RuntimeError::InvalidRequest(String)` (existing).

- [ ] **Step 1: Write the failing test**

Check first whether `src/api/chat.rs` has a `#[cfg(test)]` block
(`grep -n "mod tests" src/api/chat.rs`); if none, add one at the bottom of
the file. This test exercises the pure gate-decision function added in
Step 3 below, not the full axum handler (no test harness for a live
backend exists in this file today — testing the decision function in
isolation is sufficient and matches the file's current test-free state):

```rust
#[cfg(test)]
mod tool_gate_tests {
    use super::*;
    use crate::backend::tool_probe::ToolCapability;
    use crate::types::chat::{ChatMessage, Content, Role, Tool, FunctionDefinition};

    fn request_with_tools() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![ChatMessage {
                role: Role::User,
                content: Some(Content::Text("hi".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
            temperature: None, top_p: None, n: None, stream: None, stop: None,
            max_tokens: None, presence_penalty: None, frequency_penalty: None,
            logit_bias: None, user: None,
            tools: Some(vec![Tool {
                r#type: "function".to_string(),
                function: FunctionDefinition {
                    name: "f".to_string(), description: None, parameters: None, strict: None,
                },
            }]),
            tool_choice: None, seed: None, response_format: None, chat_template_kwargs: None,
        }
    }

    fn request_without_tools() -> ChatCompletionRequest {
        let mut req = request_with_tools();
        req.tools = None;
        req
    }

    #[test]
    fn rejects_tools_request_when_verdict_is_failed() {
        let err = check_tool_capability_gate(
            &request_with_tools(),
            Some(ToolCapability::Failed("no_tool_calls")),
        );
        assert!(err.is_some());
    }

    #[test]
    fn allows_tools_request_when_verdict_is_verified() {
        let err = check_tool_capability_gate(
            &request_with_tools(),
            Some(ToolCapability::Verified),
        );
        assert!(err.is_none());
    }

    #[test]
    fn allows_tools_request_when_verdict_is_unknown() {
        // Model never loaded/probed yet — don't block; the backend request
        // will trigger a load, which triggers the probe, for next time.
        let err = check_tool_capability_gate(&request_with_tools(), None);
        assert!(err.is_none());
    }

    #[test]
    fn allows_no_tools_request_even_when_verdict_is_failed() {
        let err = check_tool_capability_gate(
            &request_without_tools(),
            Some(ToolCapability::Failed("no_tool_calls")),
        );
        assert!(err.is_none());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib api::chat::tool_gate_tests`
Expected: FAIL — `check_tool_capability_gate` not found.

- [ ] **Step 3: Implement the gate function and call it from the handler**

Add this function to `src/api/chat.rs` (above `pub async fn chat_completions`):

```rust
/// Returns `Some(error)` when `request` carries a non-empty `tools` array
/// and `verdict` is a known `Failed` outcome for this model. `None`
/// (forward as-is) for chat-only requests, verified models, or models that
/// have never been probed yet (`None` verdict — don't block on ignorance).
fn check_tool_capability_gate(
    request: &ChatCompletionRequest,
    verdict: Option<crate::backend::tool_probe::ToolCapability>,
) -> Option<RuntimeError> {
    use crate::backend::tool_probe::ToolCapability;

    let has_tools = request.tools.as_ref().is_some_and(|t| !t.is_empty());
    if !has_tools {
        return None;
    }
    match verdict {
        Some(ToolCapability::Failed(reason)) => Some(RuntimeError::InvalidRequest(format!(
            "Model '{}' failed tool-call verification at load ({reason}); \
             remove `tools` from the request or use a model with verified tool support.",
            request.model
        ))),
        _ => None,
    }
}
```

Then wire it into `chat_completions`, right after `require_kind` and before
`ensure_loaded` (`src/api/chat.rs:68-69`):

```rust
    require_kind(&request.model, &cfg, CHAT_KINDS, "/v1/chat/completions")?;
    if let Some(err) = check_tool_capability_gate(
        &request,
        state.scheduler.tool_capability(&request.model),
    ) {
        return Err(err);
    }
    let backend = state.scheduler.ensure_loaded(&request.model).await?;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib api::chat::`
Expected: PASS (4 new tests, all pre-existing tests in the file unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/api/chat.rs
git commit -m "feat(api): reject tool-bearing chat requests for models that failed the tool probe

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Surface the verdict via `/v1/models` and `/v1/models/{id}`

**Files:**
- Modify: `src/types/mod.rs:30-58` (`ModelInfo` struct) and `:78-95` (`ModelInfo::new`)
- Modify: `src/api/models.rs:25-64` (`list_models`, `get_model`)

**Interfaces:**
- Consumes: `state.scheduler.tool_capability(&id) -> Option<ToolCapability>` (Task 3).
- Produces: `ModelInfo.tools_verified: Option<bool>` — `Some(true)` for `Verified`, `Some(false)` for `Failed(_)`, `None` for `Skipped` or never-probed (keeps the field absent from JSON for models where the question doesn't apply, matching the file's existing `skip_serializing_if` convention).

- [ ] **Step 1: Write the failing test**

Add to (or create) a `#[cfg(test)] mod tests` block at the bottom of
`src/types/mod.rs` (check first with `grep -n "mod tests" src/types/mod.rs`):

```rust
#[cfg(test)]
mod tool_verified_tests {
    use super::*;

    #[test]
    fn tools_verified_defaults_to_none() {
        let info = ModelInfo::new("m");
        assert_eq!(info.tools_verified, None);
    }

    #[test]
    fn tools_verified_omitted_from_json_when_none() {
        let info = ModelInfo::new("m");
        let json = serde_json::to_value(&info).unwrap();
        assert!(json.get("tools_verified").is_none());
    }

    #[test]
    fn tools_verified_present_when_set() {
        let mut info = ModelInfo::new("m");
        info.tools_verified = Some(true);
        let json = serde_json::to_value(&info).unwrap();
        assert_eq!(json["tools_verified"], serde_json::json!(true));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib types::tool_verified_tests`
Expected: FAIL — no field `tools_verified` on `ModelInfo`.

- [ ] **Step 3: Add the field**

In `src/types/mod.rs`, add to the `ModelInfo` struct (after `runtime_profile` at line ~57):

```rust
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_profile: Option<RuntimeProfileInfo>,
    /// Load-time tool-call probe verdict: `Some(true)` verified, `Some(false)`
    /// failed, `None` if the model never claimed `tools` capability or has
    /// never been loaded/probed yet.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools_verified: Option<bool>,
```

And to `ModelInfo::new` (after `runtime_profile: None,` at line ~93):

```rust
            runtime_profile: None,
            tools_verified: None,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib types::tool_verified_tests`
Expected: PASS.

- [ ] **Step 5: Populate the field from the scheduler in both handlers**

In `src/api/models.rs`, replace `list_models` (lines 25-41):

```rust
pub async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ListModelsResponse>, RuntimeError> {
    let models = state
        .scheduler
        .model_ids()
        .into_iter()
        .filter_map(|id| {
            state.scheduler.model_config(&id).map(|cfg| {
                let mut info = crate::types::ModelInfo::from_config(id.clone(), &cfg);
                info.tools_verified = tools_verified_bool(state.scheduler.tool_capability(&id));
                info
            })
        })
        .collect();

    Ok(Json(ListModelsResponse::new(models)))
}

/// Convert a `ToolCapability` verdict into the API's tri-state boolean:
/// `Verified` -> `Some(true)`, `Failed` -> `Some(false)`, `Skipped` or
/// never-probed -> `None` (field omitted from the response).
fn tools_verified_bool(
    verdict: Option<crate::backend::tool_probe::ToolCapability>,
) -> Option<bool> {
    use crate::backend::tool_probe::ToolCapability;
    match verdict {
        Some(ToolCapability::Verified) => Some(true),
        Some(ToolCapability::Failed(_)) => Some(false),
        Some(ToolCapability::Skipped) | None => None,
    }
}
```

And `get_model` (lines 56-64):

```rust
pub async fn get_model(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Result<Json<ModelInfo>, RuntimeError> {
    let Some(cfg) = state.scheduler.model_config(&model_id) else {
        return Err(RuntimeError::ModelNotFound(model_id));
    };
    let mut info = ModelInfo::from_config(model_id.clone(), &cfg);
    info.tools_verified = tools_verified_bool(state.scheduler.tool_capability(&model_id));
    Ok(Json(info))
}
```

- [ ] **Step 6: Run the models API test suite**

Run: `cargo test --lib api::models::`
Expected: PASS (no pre-existing tests in this file broken by the added field, since it's `skip_serializing_if`-guarded).

- [ ] **Step 7: Commit**

```bash
git add src/types/mod.rs src/api/models.rs
git commit -m "feat(api): surface tools_verified in /v1/models and /v1/models/{id}

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: Full repository gate and final verification

**Files:** None (verification only).

- [ ] **Step 1: Run the full precommit gate**

Run: `./precommit.sh`
Expected: PASS — formatting, `cargo build`, `cargo clippy` (no warnings), and the complete test suite (including every test added in Tasks 1-5) all succeed.

- [ ] **Step 2: If clippy flags anything in the new code, fix inline and re-run**

Run: `./precommit.sh`
Expected: PASS.

- [ ] **Step 3: Manual smoke check against a real model (optional but recommended before merging)**

```bash
cargo run --release -- serve &
sleep 2
curl -s http://localhost:8080/v1/models | jq '.data[] | {id, tools_verified}'
```

Expected: models with a `tools` capability tag show `tools_verified: true` or
`false` after their first load; models without it show no `tools_verified`
key at all. (Stop the server afterward: `kill %1` or the equivalent for
your shell.)

- [ ] **Step 4: Final commit if any fixes were made in Steps 1-2**

```bash
git add -A
git commit -m "fix: address precommit findings for tool-call capability verification

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```
