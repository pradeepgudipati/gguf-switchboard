# Model Search Standalone Eligibility Design

## Goal

Correct `ggs models search` so `SUPPORTED=Yes` means a repository contains at least one complete GGUF that both fits the detected hardware and can generate independently. Auxiliary speculative-decoding models such as DSpark drafters must show `No` even when their files fit memory.

## Eligibility decision

Evaluate standalone eligibility from the Hugging Face search hit before applying the existing memory estimate. A repository is ineligible when any of these positive auxiliary signals exists:

- the normalized Hugging Face tags contain `draft-model` or `auxiliary-model`;
- the normalized GGUF architecture is exactly `dflash` or `deepseek4-dspark`;
- the normalized GGUF architecture contains `draft` or `speculator`;
- as a fallback for poorly tagged repositories, the repository id or a candidate GGUF filename contains the token `drafter` or `speculator`.

Do not reject a repository merely because its name contains `dspark`, `speculative-decoding`, `mtp`, `draft`, or `support`. Full target checkpoints can reference these concepts, and broad substring matching would create false negatives. Strong filename/repository fallback terms use token boundaries so unrelated words do not match.

When no positive auxiliary signal exists, retain the current default that the repository is standalone-eligible. The subsequent complete-file and hardware-capacity checks remain unchanged.

## Data flow

Add a pure `is_standalone_model(hit, entries)` decision helper in `models_cmd.rs`. Each bounded repository-tree result passes the original Hugging Face hit and fetched GGUF entries to this helper. The final result is:

```text
supported = standalone_eligible && smallest_complete_gguf_fits_capacity
```

A failed tree lookup continues to produce `No`. Search result ordering, concurrency, RAM/VRAM probing, and the 20 percent headroom rule do not change.

## User-visible behavior

Keep the `SUPPORTED` Yes/No column. Update the note and usage documentation to state that `Yes` requires a standalone GGUF as well as sufficient estimated capacity.

## Tests

Write failing tests first for:

- `draft-model` and `auxiliary-model` tags producing `No` eligibility;
- `dflash`, `deepseek4-dspark`, and draft/speculator architectures producing `No` eligibility;
- `drafter` and `speculator` repository or GGUF filename tokens producing `No` eligibility when tags are absent;
- a full target model whose repository name contains `DSpark` remaining eligible without an auxiliary signal;
- the final support decision requiring both standalone eligibility and memory fit;
- the explanatory output mentioning standalone eligibility.

Run the full repository gate, then verify a live DSpark/drafter search and a normal model search before completion.
