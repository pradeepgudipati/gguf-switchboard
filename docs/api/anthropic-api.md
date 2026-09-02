# Anthropic Messages API

> [← Back to README](../../README.md)

GGUF Switchboard translates the Anthropic Messages API onto the loaded `llama-server` OpenAI backend. Streaming and tool calling are supported.

## Non-streaming

```bash
curl http://localhost:9090/v1/messages \
    -H "Content-Type: application/json" \
    -H "x-api-key: not-needed" \
    -H "anthropic-version: 2023-06-01" \
    -d '{
        "model": "gemma-4-e4b",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Explain the difference between threads and processes."}
        ]
    }'
```

## Streaming

```bash
curl http://localhost:9090/v1/messages \
    -H "Content-Type: application/json" \
    -H "x-api-key: not-needed" \
    -H "anthropic-version: 2023-06-01" \
    -d '{
        "model": "gemma-4-e4b",
        "max_tokens": 1024,
        "stream": true,
        "messages": [
            {"role": "user", "content": "Write a haiku about Rust programming."}
        ]
    }'
```

## Tool calling

```bash
curl http://localhost:9090/v1/messages \
    -H "Content-Type: application/json" \
    -H "x-api-key: not-needed" \
    -H "anthropic-version: 2023-06-01" \
    -d '{
        "model": "gemma-4-e4b",
        "max_tokens": 1024,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        ],
        "messages": [
            {"role": "user", "content": "What is the weather in San Francisco?"}
        ]
    }'
```

The request is translated to OpenAI format, forwarded to `llama-server`, and the response is translated back to Anthropic format. Tool definitions, tool calls, and content blocks are mapped bidirectionally.

## Notes

- The `x-api-key` header is required by the Anthropic SDK but not validated by GGUF Switchboard. Any non-empty string works.
- The `anthropic-version` header should be set to `2023-06-01`.
- The model name should match an alias configured in `models.toml`.
