# Strict Responses API Tool Calling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Translate Responses API function tools and llama.cpp Chat Completions tool calls into strict Responses API request, output-item, and streaming-event shapes.

**Architecture:** Keep llama.cpp behind the existing Chat Completions backend interface. Add typed Responses function-tool and output-item models, isolate request/output conversion in testable functions, and add a request-local streaming state machine that emits typed Responses SSE events.

**Tech Stack:** Rust 2024, Axum 0.8, Serde, Futures, Utoipa, Tokio, existing llama.cpp backend abstraction.

## Global Constraints

- Support function tools only.
- Do not execute tools or add persistent state.
- Preserve model-generated `arguments` as opaque JSON strings.
- Support choice index zero only.
- Use strict Responses API output items and streaming event names.
- Reject malformed or unsupported tools with HTTP 400.
- Follow red-green-refactor for every production behavior.

## File Map

- Modify `src/types/responses.rs`: typed request tools, output-item enum, response content, stream event serialization, and unit tests for wire shapes.
- Modify `src/api/responses.rs`: request conversion, non-streaming conversion, streaming state machine, handler integration, and focused unit tests.
- Modify `tests/integration.rs`: update public type construction and serialization coverage.
- Modify `src/api/mod.rs`: register new response schemas with Utoipa.
- Modify `README.md` and `docs/COMPATIBILITY.md`: accurately describe the supported Responses function-tool subset.

---

### Task 1: Typed Function Tools and Strict Output Items

**Files:**
- Modify: `src/types/responses.rs`
- Modify: `tests/integration.rs`
- Modify: `src/api/mod.rs`

**Interfaces:**
- Produces: `ResponseTool::Function(ResponseFunctionTool)`.
- Produces: `ResponseOutput::Message(ResponseMessageOutput)`.
- Produces: `ResponseOutput::FunctionCall(ResponseFunctionCallOutput)`.
- Produces: `ResponseContent::OutputText { text, annotations }`.

- [ ] **Step 1: Write failing serialization tests**

Add tests that deserialize a flattened Responses function tool and serialize a
function-call output item:

```rust
#[test]
fn deserializes_responses_function_tool() {
    let tool: ResponseTool = serde_json::from_value(serde_json::json!({
        "type": "function",
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object"},
        "strict": true
    }))
    .unwrap();
    assert!(matches!(tool, ResponseTool::Function(_)));
}

#[test]
fn serializes_function_call_output_item() {
    let item = ResponseOutput::FunctionCall(ResponseFunctionCallOutput {
        id: "fc_1".into(),
        call_id: "call_1".into(),
        name: "get_weather".into(),
        arguments: "{\"city\":\"Pune\"}".into(),
        status: "completed".into(),
    });
    let value = serde_json::to_value(item).unwrap();
    assert_eq!(value["type"], "function_call");
    assert_eq!(value["call_id"], "call_1");
}
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cargo test types::responses -- --nocapture
```

Expected: compilation fails because the typed Responses tool and output variants
do not exist.

- [ ] **Step 3: Implement the minimal typed wire models**

Use internally tagged Serde enums:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseTool {
    Function(ResponseFunctionTool),
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponseFunctionTool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseOutput {
    Message(ResponseMessageOutput),
    FunctionCall(ResponseFunctionCallOutput),
}
```

Add `status` to message items and `annotations: Vec<serde_json::Value>` to
`output_text`. Change `ResponseRequest.tools` to `Option<Vec<ResponseTool>>`.

- [ ] **Step 4: Update Utoipa registrations and integration constructors**

Register every new schema in `src/api/mod.rs`. Update existing tests to construct
the typed request without changing their text-only assertions.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
cargo test types::responses
cargo test test_response
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/types/responses.rs src/api/mod.rs tests/integration.rs
git commit -m "feat: model strict Responses tool items"
```

### Task 2: Request and Non-streaming Response Translation

**Files:**
- Modify: `src/api/responses.rs`

**Interfaces:**
- Produces: `to_chat_request(request: &ResponseRequest) -> Result<ChatCompletionRequest, RuntimeError>`.
- Produces: `to_response_output(message: &ChatMessage) -> Result<Vec<ResponseOutput>, RuntimeError>`.
- Consumes: typed models from Task 1.

- [ ] **Step 1: Write failing request translation test**

Construct a Responses request with `get_weather`, call `to_chat_request`, and
assert the backend tool shape nests the function:

```rust
assert_eq!(chat.tools.as_ref().unwrap()[0].r#type, "function");
assert_eq!(chat.tools.as_ref().unwrap()[0].function.name, "get_weather");
assert_eq!(chat.tool_choice, Some(json!("auto")));
```

- [ ] **Step 2: Run request test and verify RED**

Run:

```bash
cargo test api::responses::tests::translates_function_tools_to_chat_request -- --exact
```

Expected: compilation fails because `to_chat_request` does not exist.

- [ ] **Step 3: Implement minimal request translation**

Move current message construction into `to_chat_request`. Translate each
`ResponseFunctionTool` into `chat::Tool { type: "function", function:
FunctionDefinition { ... } }`. Forward `tool_choice` unchanged because the
supported Responses values and llama.cpp Chat Completions values share the
accepted string/object wire forms.

- [ ] **Step 4: Run request test and verify GREEN**

Run the exact test from Step 2. Expected: PASS.

- [ ] **Step 5: Write failing output translation tests**

Add separate tests for:

- tool-only output;
- text plus one tool call;
- two tool calls;
- backend tool call with an empty id or name returning `RuntimeError::BackendError`.

Assert `id` starts with `fc_`, `call_id` preserves the backend id, arguments are
byte-for-byte unchanged, and text precedes calls.

- [ ] **Step 6: Run output tests and verify RED**

Run:

```bash
cargo test api::responses::tests::translates_ -- --nocapture
```

Expected: tests fail because the existing handler emits only a message item.

- [ ] **Step 7: Implement minimal non-streaming translation**

Implement `to_response_output` and use it in the handler. Return no message item
when content is absent or empty. Generate one `fc_` output id per backend call.
Reject tool calls with an empty id, empty name, or non-`function` type.

- [ ] **Step 8: Run focused tests and verify GREEN**

Run:

```bash
cargo test api::responses::tests
```

Expected: all Responses translation tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/api/responses.rs
git commit -m "feat: translate Responses function calls"
```

### Task 3: Strict Responses Streaming Events

**Files:**
- Modify: `src/types/responses.rs`
- Modify: `src/api/responses.rs`

**Interfaces:**
- Produces: serializable `ResponseStreamEvent`.
- Produces: `ResponseStreamState::new(response_id, model, created_at)`.
- Produces: `ResponseStreamState::start_events()`.
- Produces: `ResponseStreamState::apply_chunk(chunk)`.
- Produces: `ResponseStreamState::finish_events()`.

- [ ] **Step 1: Write failing text streaming state test**

Feed role, text delta, and stop chunks into the wished-for state machine. Assert
the serialized event types contain, in order:

```text
response.created
response.in_progress
response.output_item.added
response.content_part.added
response.output_text.delta
response.output_text.done
response.content_part.done
response.output_item.done
response.completed
```

Also assert sequence numbers are `0..n` with no duplicates.

- [ ] **Step 2: Run text stream test and verify RED**

Run:

```bash
cargo test api::responses::tests::streams_strict_text_events -- --exact
```

Expected: compilation fails because `ResponseStreamState` does not exist.

- [ ] **Step 3: Implement text event state machine**

Add request-local state containing response metadata, next sequence number,
next output index, optional text item state, and accumulated usage. Serialize
each SSE record as:

```text
event: <event.type>
data: <event-json>

```

Do not append `[DONE]`.

- [ ] **Step 4: Run text stream test and verify GREEN**

Run the exact test from Step 2. Expected: PASS.

- [ ] **Step 5: Write failing fragmented tool-call streaming test**

Feed chunks where tool index zero arrives as:

```json
{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"loc"}}
{"id":"","type":"","function":{"name":"","arguments":"ation\":\"Pune\"}"}}
```

Assert one stable output item, concatenated arguments, delta events for both
fragments, a done event containing the complete arguments, and a completed
function-call output item.

- [ ] **Step 6: Run tool stream test and verify RED**

Run:

```bash
cargo test api::responses::tests::streams_fragmented_function_arguments -- --exact
```

Expected: fails because tool-call stream assembly is missing.

- [ ] **Step 7: Implement tool-call stream assembly**

Track calls by their position in each chunk's `tool_calls` vector. Preserve the
first non-empty id and name, append every argument fragment, and assign a stable
Responses output index and `fc_` item id. Emit `response.output_item.added` once,
argument delta events per non-empty fragment, and done/item-done events at
finish.

- [ ] **Step 8: Add and pass multiple-call streaming test**

Feed two calls and assert distinct `call_id`, `fc_` ids, and output indices.

Run:

```bash
cargo test api::responses::tests::streams_
```

Expected: all streaming tests pass.

- [ ] **Step 9: Wire the state machine into the handler**

Replace the current partial-response-object mapping with `stream::unfold` or an
equivalent stateful adapter that emits the start events, converted backend
events, and completion events while retaining scheduler and active-request
guards.

- [ ] **Step 10: Run all Responses tests**

Run:

```bash
cargo test responses -- --nocapture
```

Expected: all selected tests pass with no panic or warning.

- [ ] **Step 11: Commit**

```bash
git add src/types/responses.rs src/api/responses.rs
git commit -m "feat: emit strict Responses streaming events"
```

### Task 4: Compatibility Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/COMPATIBILITY.md`

**Interfaces:**
- Consumes: completed endpoint behavior from Tasks 1 through 3.
- Produces: accurate public compatibility claims.

- [ ] **Step 1: Update documentation**

State that Chat Completions forwards llama.cpp tool calling and Responses
translates function tools/function calls. Mark Responses as partial because
stateful chaining, built-in tools, multimodal items, and other full API features
remain unsupported.

- [ ] **Step 2: Run documentation consistency searches**

Run:

```bash
rg -n "Tool calling|POST /v1/responses|Responses API|\\[DONE\\]" README.md docs src/api/responses.rs
```

Expected: no claim says all Responses API tools are supported and no Responses
stream implementation appends `[DONE]`.

- [ ] **Step 3: Run formatting**

Run:

```bash
cargo fmt --all
```

- [ ] **Step 4: Run the full repository gate**

Run:

```bash
./precommit.sh
```

Expected: format, clippy with warnings denied, build, all tests, and doc tests
pass.

- [ ] **Step 5: Inspect final diff**

Run:

```bash
git status --short
git diff --check
git diff --stat HEAD~3
```

Expected: only scoped source, test, and documentation changes plus the plan;
`repomix-output.xml` remains untracked and uncommitted.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/COMPATIBILITY.md
git commit -m "docs: document Responses function tools"
```
