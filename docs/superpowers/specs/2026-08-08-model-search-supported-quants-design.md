# Model Search Supported Quants Design

## Goal

Extend `ggs models search` with a `QUANT` column listing the standalone GGUF quantizations that fit the detected hardware, and print one directly usable `ggs models pull` command after the results.

## Runnable model options

Replace the single smallest-model calculation with a pure option extractor that returns each complete runnable option as its quant label and total bytes.

- A normal model GGUF is one option.
- A split GGUF is one option only when every declared shard exists; its bytes are the checked sum of all shard sizes.
- Derive the split option's quant from the common filename prefix.
- Recognize quant labels preceded by either `-` or `.`, including filenames such as `nomic-embed-text-v1.5.Q4_K_M.gguf`. The same parser serves search, files, and pull commands.
- Accept only standard quant-shaped tokens (`Q<digit>_...`, `IQ<digit>_...`) and explicit precision labels (`BF16`, `FP16`, `FP32`, `F16`, `F32`, `MXFP4`, `NVFP4`). Descriptive fragments such as `Q8Out` and `Q2KDown` are not pullable quant labels.
- Continue excluding projectors, adapters, tokenizers, vocabularies, and other non-model GGUF files.
- Exclude clearly auxiliary options within otherwise normal repositories: filenames with an `MTP` token, filenames beginning with `dspark-`, and filenames containing both `DSpark` and `support` tokens.
- Ignore options whose filename has no recognized quant label for the `QUANT` column and pull recommendation. An unnamed option must not make `SUPPORTED=Yes` because it cannot produce the required `--quant` command reliably.
- When several files represent the same quant, retain the smallest complete byte size for that quant.

## Per-row result

For a standalone-eligible repository, filter its named options through the existing capacity rule of total system RAM plus total NVIDIA VRAM with 20 percent runtime headroom. Sort fitting quants by complete model bytes from smallest to largest, breaking equal sizes by quant name.

The `QUANT` column contains the fitting labels joined with commas. It contains `-` when the repository is auxiliary, its tree lookup fails, it has no recognized complete quant, or no quant fits.

`SUPPORTED=Yes` exactly when the fitting quant list is non-empty. The existing standalone eligibility and hardware probes do not change.

## Pull recommendation

After the table and explanatory support note, choose the first supported repository in search-result order that has a named fitting quant.

Choose its recommended quant as follows:

1. Use `Q4_K_M` when it is in the fitting list.
2. Otherwise use the fitting option with the largest complete byte size.

Print:

```text
Try: ggs models pull <repo-id> --quant <quant>
```

Omit the line when no result has a named fitting quant. Repository ids and quant labels originate from the Hugging Face API and recognized filename tokens; render them as separate fixed command arguments without shell interpolation.

## Structure

Introduce a small model-option type and keep extraction, capacity filtering, recommendation, and rendering independently testable. The existing bounded tree-fetch stage returns a per-row assessment containing `supported`, ordered fitting quants, and an optional recommended quant. Preserve row order after concurrent fetches.

## User-visible table

Before the table, print the exact detected capacity inputs used by the support calculation:

```text
Hardware: System RAM 62.0 GiB | NVIDIA VRAM 24.0 GiB | Total 86.0 GiB
```

These values describe total hardware capacity, not currently free memory. Use binary GiB formatting with one decimal place. CPU-only systems show `NVIDIA VRAM 0.0 GiB`. Build the support capacity from these same MiB inputs so the displayed header and row decisions cannot diverge.

Use a delimiter-based table with ` | ` between columns. Compute each core column's width from its heading and the values in the complete result set. Left-align repository, support, context, and architecture values; right-align file counts and sizes.

Render columns in this order:

```text
REPO | FILES | SIZE | SUPPORTED | CONTEXT | ARCH | QUANT
```

Keep `QUANT` last so a long list cannot shift or misalign the core columns. Long repository ids expand the repository column rather than overflowing into `FILES`. Long quant lists remain complete rather than truncated, even if a row exceeds terminal width.

Update `docs/USAGE.md` to explain the column and recommendation rule.

## Tests and verification

Write failing tests first for:

- extracting named normal options;
- extracting a quant separated from the model name by a dot;
- rejecting descriptive Q-prefixed filename fragments and auxiliary DSpark/MTP files;
- grouping complete split options and rejecting incomplete shards;
- deduplicating a quant to its smallest complete option;
- filtering options with the 20 percent headroom rule;
- excluding auxiliary repositories;
- ordering fitting quants by size and then name;
- preferring `Q4_K_M` for the sample command;
- falling back to the largest fitting option;
- omitting a command when no named quant fits;
- rendering `QUANT`, `-`, and the chosen pull command.
- dynamically aligning rows containing both short and long repository ids without column overlap.
- rendering system RAM, NVIDIA VRAM, and their total from the same capacity inputs used for support decisions.

Run the full repository gate, then live searches for `gemma` and `DSpark Drafter`. Verify Gemma rows list fitting quants and print a valid recommendation, while auxiliary drafter rows remain `No`, show `-`, and produce no recommendation when all results are auxiliary.
