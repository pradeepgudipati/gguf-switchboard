# OpenAI API compatibility

> [← Back to README](../README.md)

gguf-switchboard forwards requests to the selected llama.cpp or vLLM process. Compatibility has three layers: the proxy contract, the selected backend version, and the model/tokenizer/chat template. Treat this matrix as **best-effort** and verify the exact model/backend combination used in production.

## Endpoints

| Endpoint | Status | Notes |
|----------|--------|-------|
| `GET /v1/models` | Supported | Lists configured aliases with `kind`, `description`, `max_context_length`, `min_vram_gb`, `capabilities`, `hf_repo`, `tools_verified` |
| `GET /v1/models/{model_id}` | Supported | Individual model info with `tools_verified` field |
| `GET /v1/models/{model_id}/runtime` | Supported | Runtime profile: effective `context_size`, `ngl`, `split_mode`, `kv_cache_type`, `profile_source` |
| `GET /v1/models/registry.json` | Supported | Portable registry export |
| `POST /v1/models/refresh` | Supported | Hot-reload model registry from disk |
| `POST /v1/chat/completions` | Supported | Streaming and non-streaming; tool calling forwarded when model supports it |
| `POST /v1/completions` | Supported | Streaming and non-streaming |
| `POST /v1/embeddings` | Supported | Depends on model/backend; batch_size/ubatch_size auto-configure `-b`/`-ub` |
| `POST /v1/rerank` | Supported | Requires a `reranker` model and a backend exposing the compatible rerank route |
| `POST /v1/responses` | Supported | Function tools, `tool_choice`, streaming events; mapped to/from Chat Completions |
| `POST /v1/messages` | Supported | Anthropic Messages API — streaming, tool calling, content blocks translated to/from OpenAI format |
| `POST /v1/audio/transcriptions` | Partial | Proxied when backend exposes endpoint |
| `POST /v1/audio/speech` | Partial | Proxied when backend exposes endpoint |
| `GET /health`, `GET /status` | Supported | Includes `llama_server_version` and `last_switch` timing |
| `GET /metrics` | Supported | Prometheus text format |
| `GET /v1/usage` | Supported | SQLite-backed usage history (extension) |
| `GET /v1/usage/recent` | Supported | Recent usage summary |

## Features

| Feature | Status | Notes |
|---------|--------|-------|
| SSE streaming | Supported | Chat Completions emits `[DONE]`; Responses terminates after `response.completed`; Anthropic emits `message_stop` |
| Chat Completions tool calling | Supported | Passed through when the model/backend supports it; load-time tool-call probe verifies capability |
| Responses function tools | Supported | Function definitions and `tool_choice` translated to Chat Completions; function calls returned as top-level output items |
| Anthropic Messages API tool calling | Supported | Tool definitions, tool calls, and content blocks translated bidirectionally to/from OpenAI format |
| Responses built-in/hosted tools | Not supported | Function tools only |
| `response_format` / JSON mode | Untested | Depends on the selected backend and model |
| Structured outputs | Not supported | — |
| Reasoning fields | Partial | `reasoning_content` promoted in chat types; when `max_tokens` is too low and reasoning consumes the budget, `reasoning_content` is promoted into `content` |
| Streaming usage counts | Not supported | Usage recorded as zero for streams |
| Multimodal content arrays | Untested | — |
| Logprobs | Untested | — |
| Request cancellation | Not supported | — |
| Batch API | Not supported | — |
| Auto GPU layers (`auto_ngl`) | Supported | Opt-in: picks `-ngl` from free VRAM + GGUF size |
| ModelFitPlanner | Supported for GGUF | Opt-in `[fit]` section: llama.cpp bounded fallback ladder with profile caching; vLLM uses a pre-launch VRAM weight-fit gate |
| Memory-pressure eviction | Supported | Monitors system RAM; unloads at critical threshold |
| OOM context fallback | Supported for GGUF | Auto-reduces llama.cpp `-c` on OOM-class failures down to `context_fallback_min` |
| Idle priority model | Supported | Preferred model auto-loads after configurable idle timeout |
| Tool-call capability probe | Supported | `tools`-tagged models probed at load time; verdict on `tools_verified` |
| HF metadata enrichment | Supported | Automatic sync on launch/refresh; `sync-hf-metadata` CLI |
| Model management CLI | Supported | GGUF: `models search/files/pull`; Safetensors: `models search vllm` and `models pull vllm` |
| Swagger UI model picker | Supported | Live model dropdown from OpenAPI spec; hides endpoints by model kind |
| Known-good profile caching | Supported | Successful load profiles cached to `model-profiles.json` |

## Backend and format boundaries

| Source | Managed backend | Automatic setup | Notes |
|--------|-----------------|:---------------:|-------|
| GGUF | llama.cpp | Yes | Supports CPU/GPU layer splitting; fit planner and OOM context fallback apply |
| Hugging Face Safetensors | vLLM | Yes | Requires weights to fit the selected vLLM hardware topology; supports registry-declared quantization and tensor parallelism |
| Pickle-backed PyTorch `.bin` | None | No | Deliberately excluded from the automated download path |

The installer never enables vLLM `trust_remote_code`. Models requiring repository-provided Python code need a separate explicit security decision.

## Tested clients

| Client | Version tested | Notes |
|--------|----------------|-------|
| curl / OpenAI Python SDK | ad hoc | Primary development path |
| Cursor / Cline / Continue | community | Report issues with request shapes |
| Anthropic Python SDK | ad hoc | `POST /v1/messages` translation layer |

Report gaps in [GitHub Issues](https://github.com/pradeepgudipati/gguf-switchboard/issues).
