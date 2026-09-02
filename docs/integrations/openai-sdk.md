# OpenAI SDK Integration

> [← Back to README](../../README.md)

Use the OpenAI SDK with GGUF Switchboard for Python, Node.js, and curl.

## What this enables

The OpenAI SDK connects to GGUF Switchboard's OpenAI-compatible API, giving you programmatic access to any locally loaded GGUF or SafeTensors model.

## Prerequisites

- GGUF Switchboard running on `http://localhost:9090`
- At least one model registered and loaded
- OpenAI SDK installed (Python or Node.js)

## Verify available models

```bash
curl -s http://localhost:9090/v1/models | jq -r '.data[].id'
```

## Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9090/v1",
    api_key="not-needed",  # any string works
)

# Chat completion
response = client.chat.completions.create(
    model="qwen3.5-9b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain Rust ownership."}
    ],
    temperature=0.7,
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="qwen3.5-9b",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
print()
```

## Node.js / TypeScript

```typescript
import OpenAI from "openai";

const client = new OpenAI({
    baseURL: "http://localhost:9090/v1",
    apiKey: "not-needed",
});

// Chat completion
const response = await client.chat.completions.create({
    model: "qwen3.5-9b",
    messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "Explain Rust ownership." },
    ],
});
console.log(response.choices[0].message.content);

// Streaming
const stream = await client.chat.completions.create({
    model: "qwen3.5-9b",
    messages: [{ role: "user", content: "Tell me a story." }],
    stream: true,
});
for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}
console.log();
```

## curl

```bash
# Chat completion
curl http://localhost:9090/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.5-9b",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }'

# Streaming
curl http://localhost:9090/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.5-9b",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "stream": true
    }'

# Embeddings
curl http://localhost:9090/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.5-9b",
        "input": "The quick brown fox jumps over the lazy dog."
    }'
```

## Tool calling

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9090/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="qwen3.5-9b",
    messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }],
    tool_choice="auto"
)

print(response.choices[0].message.tool_calls)
```

## Notes

- The `api_key` parameter is required by the SDK but not validated by GGUF Switchboard. Any string works.
- The model name should match an alias configured in `models.toml`.
- The client experience is identical regardless of whether the model runs through llama.cpp (GGUF) or vLLM (SafeTensors).

## Troubleshooting

**Model not found:** Ensure the model name matches an alias in `models.toml`.

**Connection refused:** Verify GGUF Switchboard is running: `curl http://localhost:9090/health`

**Streaming issues:** Ensure your SDK version supports SSE streaming.
