# Benchmarks

> [← Back to README](../README.md)

## Throughput test

```bash
# Install hey: go install github.com/rakyll/hey@latest

# Non-streaming throughput
hey -n 100 -c 4 \
    -m POST \
    -H "Content-Type: application/json" \
    -d '{"model":"local-gemma-code","messages":[{"role":"user","content":"Say hello"}],"max_tokens":50}' \
    http://localhost:9090/v1/chat/completions

# Streaming throughput
hey -n 50 -c 2 \
    -m POST \
    -H "Content-Type: application/json" \
    -d '{"model":"local-gemma-code","messages":[{"role":"user","content":"Count to 10"}],"stream":true,"max_tokens":100}' \
    http://localhost:9090/v1/chat/completions
```

### Model switching latency

```bash
# Time a cold model switch
time curl -s http://localhost:9090/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"local-qwen-coder","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}' \
    > /dev/null
```

### Proxy overhead vs llama-swap

Both tools are thin proxies in front of the same `llama-server` — token generation speed is identical. Measure proxy overhead with [`scripts/bench-vs-llama-swap.sh`](../scripts/bench-vs-llama-swap.sh). Feature differences: [Comparison — vs llama-swap](COMPARISON.md#vs-llama-swap).

```bash
LLAMA_SERVER_BIN=/usr/local/bin/llama-server \
MODEL_A_PATH=/models/model-a.gguf \
MODEL_B_PATH=/models/model-b.gguf \
./scripts/bench-vs-llama-swap.sh
```

Reports request latency, swap latency, RSS, and optional throughput via [`hey`](https://github.com/rakyll/hey). Results land in `.bench/results-<timestamp>/report.md`.
