# OpenAI API compatibility

> [← Back to README](../README.md)

gguf-switchboard forwards requests to `llama-server`. Compatibility depends on both this proxy and your installed llama.cpp build. Treat this matrix as **best-effort** — verify with your client and backend version.

## Endpoints

| Endpoint | Status | Notes |
|----------|--------|-------|
| `GET /v1/models` | Supported | Lists configured aliases |
| `GET /v1/models/registry.json` | Supported | Portable registry export |
| `POST /v1/chat/completions` | Supported | Streaming and non-streaming |
| `POST /v1/completions` | Supported | Streaming and non-streaming |
| `POST /v1/embeddings` | Supported | Depends on model/backend |
| `POST /v1/responses` | Partial | Mapped to chat completions internally |
| `POST /v1/audio/transcriptions` | Partial | Proxied when backend exposes endpoint |
| `POST /v1/audio/speech` | Partial | Proxied when backend exposes endpoint |
| `GET /health`, `GET /status` | Supported | Includes `llama_server_version` when detected |
| `GET /metrics` | Supported | Prometheus text format |
| `GET /v1/usage` | Supported | SQLite-backed usage history (extension) |

## Features

| Feature | Status | Notes |
|---------|--------|-------|
| SSE streaming | Supported | `[DONE]` terminator emitted |
| Tool calling | Untested | Passed through if backend supports |
| `response_format` / JSON mode | Untested | Depends on llama-server |
| Structured outputs | Not supported | — |
| Reasoning fields | Partial | `reasoning_content` promoted in chat types |
| Streaming usage counts | Not supported | Usage recorded as zero for streams |
| Multimodal content arrays | Untested | — |
| Logprobs | Untested | — |
| Request cancellation | Not supported | — |
| Batch API | Not supported | — |

## Tested clients

| Client | Version tested | Notes |
|--------|----------------|-------|
| curl / OpenAI Python SDK | ad hoc | Primary development path |
| Cursor / Cline / Continue | community | Report issues with request shapes |

Report gaps in [GitHub Issues](https://github.com/pradeepgudipati/gguf-switchboard/issues).
