# Hardware-Aware Model Search Design

## Goal

Extend `ggs models search <query>` with a `SUPPORTED` column that estimates whether the current machine can run at least one complete GGUF model from each Hugging Face repository in the result set.

The estimate describes installed hardware capacity, not memory currently free. It is a discovery aid rather than a runtime guarantee.

## User-visible behavior

The existing search columns remain and `SUPPORTED` is added to each result row. A row shows `Yes` when the smallest runnable model option found in that repository fits the machine's estimated model-memory budget. It shows `No` otherwise, including when the repository does not expose enough size information to make the estimate.

The command prints a concise note explaining that support is estimated from model file size, total system RAM, total NVIDIA VRAM, and a runtime safety margin. Context length, KV cache configuration, GPU offload settings, and other runtime allocations can still affect whether a model starts.

CPU-only systems remain supported. Their budget consists only of total system RAM. When `nvidia-smi` is unavailable or reports no usable devices, NVIDIA VRAM contributes zero.

## Hardware capacity

Reuse the existing platform-specific memory probe to obtain total system RAM. Extend the NVIDIA probe with a parser for:

```text
nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits
```

Sum valid positive values from all reported GPUs because llama.cpp can distribute model layers across multiple devices. Convert MiB to bytes with binary units. The model-memory budget is total system RAM plus total NVIDIA VRAM.

This calculation intentionally uses total capacity rather than free memory so search results do not change with transient system load.

## Repository sizing

The Hugging Face model-search response does not reliably expose the size of each sibling GGUF file. After the search request, fetch the repository tree for each returned repository through the existing tree API helper. Run these requests concurrently with a small fixed bound so the default ten-result search does not become serially slow or place an unbounded load on Hugging Face.

From each tree:

1. Exclude projector and auxiliary GGUF files using the existing model-file classification.
2. Treat a normal single `.gguf` file as one runnable option.
3. Group split GGUF shards by their shared model prefix and sum all shard sizes. A single shard is never treated as a complete runnable option.
4. Select the smallest complete option in the repository.

If the repository tree request fails or no complete sized model option exists, preserve the search result and mark it `No`. One repository failure must not fail the entire search command.

## Support estimate

Reserve 20 percent runtime headroom for model metadata, allocator overhead, and a modest KV cache. Expressed without floating-point arithmetic, a model is supported when:

```text
model_bytes * 120 <= capacity_bytes * 100
```

Use checked or saturating arithmetic to avoid overflow on malformed metadata.

The estimate deliberately does not use the repository's advertised maximum context length. gguf-switchboard can run a lower context than the model maximum, so sizing the maximum KV cache would incorrectly reject otherwise runnable models.

## Structure

Keep hardware probing, support calculation, and repository-option selection as small pure helpers where possible. `cmd_search` coordinates the search, bounded repository enrichment, and rendering. Network and process boundaries remain outside the pure decision functions so tests can exercise behavior deterministically.

No new command-line flag or configuration setting is introduced.

## Error handling

- Failure to read system RAM yields a zero RAM contribution rather than aborting search.
- Missing or failing `nvidia-smi` yields zero VRAM.
- Malformed GPU lines are ignored; valid GPU lines still contribute.
- A failed repository tree lookup produces `SUPPORTED=No` for that row and does not discard other results.
- A failed initial Hugging Face search retains the existing command-level error behavior.

## Testing

Tests will be written before production changes and will cover:

- parsing one and multiple NVIDIA total-VRAM rows;
- ignoring malformed GPU output;
- CPU-only capacity behavior;
- the 20 percent support threshold on both sides of the boundary;
- selecting the smallest normal GGUF;
- excluding `mmproj` and auxiliary files;
- grouping split GGUF shards and rejecting incomplete shard sets;
- unknown repository sizing producing `No`;
- rendering the new `SUPPORTED` header and Yes/No values.

After focused tests pass, run the full repository validation gate and manually run `ggs models search "gemma"` when network access and the built alias are available.
