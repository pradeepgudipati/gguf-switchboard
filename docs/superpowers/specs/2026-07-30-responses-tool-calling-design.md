# Strict Responses API Tool Calling

## Goal

Make `POST /v1/responses` translate function tools and function calls between the
OpenAI Responses API contract and the llama.cpp Chat Completions backend without
executing tools or retaining conversation state.

## Scope

The implementation will:

- accept Responses API function-tool definitions and `tool_choice`;
- validate and translate them into the existing Chat Completions request types;
- translate backend assistant `tool_calls` into top-level Responses API
  `function_call` output items;
- preserve assistant text as a separate `message` output item;
- support text-only, tool-only, mixed text and tools, and multiple tool calls;
- translate streaming text and function-call deltas into Responses API events;
- return a client error for malformed or unsupported tool definitions; and
- add focused translation and handler-level regression tests.

The implementation will not:

- execute tools;
- add sessions, `previous_response_id`, response persistence, or compaction;
- add retained reasoning;
- implement hosted, built-in, MCP, computer-use, or custom tools;
- add `function_call_output` continuation beyond the existing basic message
  input representation; or
- broaden support for multimodal Responses input.

## API Contract

### Request

This slice supports function tools with the Responses API shape:

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get weather for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": { "type": "string" }
    },
    "required": ["location"]
  },
  "strict": true
}
```

Supported `tool_choice` values will map to the backend representation without
changing their meaning. Unsupported tool types or structurally invalid function
definitions will return HTTP 400 instead of being silently discarded.

### Non-streaming output

Assistant text remains a message item:

```json
{
  "type": "message",
  "id": "msg_...",
  "status": "completed",
  "role": "assistant",
  "content": [
    {
      "type": "output_text",
      "text": "I will check.",
      "annotations": []
    }
  ]
}
```

Each backend function call becomes a separate top-level item:

```json
{
  "type": "function_call",
  "id": "fc_...",
  "call_id": "call_...",
  "name": "get_weather",
  "arguments": "{\"location\":\"Hyderabad\"}",
  "status": "completed"
}
```

The response may contain only messages, only function calls, or both. Multiple
backend tool calls produce multiple `function_call` items. When text and tool
calls coexist, the message item precedes the function-call items because Chat
Completions represents them within one assistant choice and provides no
interleaving metadata.

The backend tool-call identifier becomes `call_id`. The Responses output-item
`id` uses a distinct generated `fc_` identifier.

### Streaming output

The handler will stop emitting partial response objects and emit typed Responses
API SSE events. It will maintain per-request stream assembly state for the first
backend choice and translate:

- text deltas into `response.output_item.added`,
  `response.content_part.added`, `response.output_text.delta`,
  `response.output_text.done`, `response.content_part.done`, and
  `response.output_item.done`;
- function-call deltas into `response.output_item.added`,
  `response.function_call_arguments.delta`,
  `response.function_call_arguments.done`, and
  `response.output_item.done`; and
- stream lifecycle into `response.created`, `response.in_progress`, and
  `response.completed`.

Each event includes a monotonically increasing `sequence_number`. Output indices
remain stable for the duration of the response. The stream terminates after
`response.completed`; it will not append the Chat Completions `[DONE]` sentinel.

Only choice index zero is supported because Responses API returns one response
output sequence and the current endpoint does not expose `n`.

## Internal Design

`src/types/responses.rs` will replace the message-only output representation
with a tagged output-item enum containing:

- `Message`;
- `FunctionCall`.

Request tools will use explicit Responses tool structs or a narrowly tagged enum
instead of untyped `serde_json::Value`. Translation functions will live beside
the Responses types or in `src/api/responses.rs` when handler-specific. They
will convert:

1. Responses function tools to `types::chat::Tool`;
2. Responses `tool_choice` to the backend value;
3. a Chat Completions assistant message to Responses output items; and
4. Chat Completions streaming deltas to Responses streaming events.

No state will be added to `AppState`. Streaming assembly state exists only for
the lifetime of one request.

## Error Handling

Deserialization rejects missing function names, missing parameter schemas, and
unsupported tool types before model loading. Translation rejects backend tool
calls without a usable function name or identifier as backend errors rather than
returning malformed Responses output.

Backend argument strings remain opaque JSON strings. The proxy will not parse,
normalize, or repair model-generated arguments.

## Testing

Tests will follow red-green-refactor:

1. A request translation test proves function tools and `tool_choice` reach the
   Chat Completions request.
2. A non-streaming translation test covers one function call.
3. Separate tests cover mixed text and tools and multiple calls.
4. A malformed-tool test proves the request fails rather than dropping tools.
5. Streaming tests cover text events, fragmented function arguments, stable
   output indices, and completion events.
6. Existing text-only Responses behavior remains covered.

After focused tests pass, the repository gate `./precommit.sh` will verify
formatting, compilation, linting, and the full test suite.

## Compatibility Impact

This intentionally changes `/v1/responses` streaming from the repository's
custom partial-response-object SSE format to the OpenAI Responses event format.
It also changes the public Rust `ResponseOutput` representation. The HTTP
contract becomes more compatible, but Rust consumers importing the old concrete
struct must migrate to the tagged enum.

The README and compatibility matrix will be updated only where necessary to stop
claiming that Responses tools are forwarded before this implementation and to
describe the supported function-tool subset accurately.
