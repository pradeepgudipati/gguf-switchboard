# Quant Selector Design

`models pull --quant` accepts four selector forms:

- An exact quant such as `Q4_K_M` selects that quant, case-insensitively.
- A quant family such as `Q4` selects the first available preference in `Q4_K_M`, `Q4_K_S`, `Q4_0`, `Q4_1`.
- The `K_M` alias selects `Q4_K_M` when available. If it is absent, the command reports the available matching `*_K_M` variants instead of silently changing quality.
- `auto` evaluates complete, standalone model files against total system RAM plus NVIDIA VRAM using the existing 20 percent runtime headroom rule, then selects the largest fitting quant by file size.

Quant labels come from GGUF filenames when repository metadata does not provide them. Exact or deterministic selectors never fall through to substring ambiguity. Errors list available quantizations and explain when no candidate fits the detected hardware.

Tests exercise the selector as a pure function using real Hugging Face tree-entry values, including exact, family, alias, automatic fit, and failure cases. The pull command delegates only selection to that function and retains its existing download, validation, and registration flow.
