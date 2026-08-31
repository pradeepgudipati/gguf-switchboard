# Default Dual-Backend Installation Design

## Goal

Make `./deploy.sh` the single Linux installation entry point for GGUF/llama.cpp
and Hugging Face Safetensors/vLLM models, then reorganize the README so the
product value and working installation path are immediately clear.

## User-Facing Contract

A default deployment installs or updates all three runtime components:

1. CUDA-enabled `llama-server` under `/usr/local`;
2. an isolated vLLM environment managed by `uv`; and
3. the gguf-switchboard binary and systemd service.

The primary installation remains three commands:

```bash
git clone --branch main https://github.com/pradeepgudipati/gguf-switchboard.git
cd gguf-switchboard
./deploy.sh
```

The default is intentionally comprehensive and therefore slower than the
current deployment. Advanced update and recovery workflows may use
`--skip-llama-cpp` or `--skip-vllm`; neither skip flag changes the default.
Existing `--skip-pull`, `--refresh-models`, and `--migrate-models` behavior
remains intact.

## Backend Setup

### llama.cpp

`deploy.sh` invokes the existing `scripts/update-llama-cpp.sh` rather than
duplicating its CUDA build and installation logic. It disables that helper's
service restart because the parent deploy owns service lifecycle. A failed
llama.cpp build fails the default deployment with the helper's actionable
error. `--skip-llama-cpp` retains an already-installed `llama-server` and
requires its version check to pass when a configured GGUF source needs it.

### vLLM

A focused, sourceable `scripts/setup-vllm.sh` owns vLLM installation and
verification. It installs `uv` to a system-visible path when missing, creates
an isolated project under `/opt/gguf-switchboard/vllm-runtime`, installs the
repository-approved vLLM dependency there, and verifies:

```bash
/usr/local/bin/uv run --project /opt/gguf-switchboard/vllm-runtime vllm --version
```

The setup is idempotent. It does not install vLLM into the deploy user's global
Python environment. The version constraint lives in a tracked runtime project
file so upgrades are reviewed in the repository rather than silently chosen at
service start. `--skip-vllm` preserves an existing environment without
modifying it.

The generated or existing registry receives these defaults without replacing
unrelated user configuration:

```toml
[defaults]
vllm_command = "/usr/local/bin/uv"
vllm_project = "/opt/gguf-switchboard/vllm-runtime"
```

Deployment validates both backends as the `ggs` service account. Runtime model
selection remains registry-driven: vLLM is preferred when a Safetensors source
fits VRAM, with GGUF/llama.cpp fallback when the same alias has both sources.

## Model-Free First Install

Installing both engines does not require downloading model weights. An empty
first install succeeds, installs the unit without starting it, and prints one
GGUF pull example and one vLLM/Safetensors pull example. The service starts on a
subsequent deployment after either usable source type is registered.

Model detection must not assume that every valid installation has a `.gguf`
file or working llama.cpp source. A registry entry with a valid vLLM source is
sufficient to start the service.

## README Information Architecture

The README opening uses this order:

1. **Top features:** one API, GGUF plus Safetensors, single-slot swapping,
   OpenAI and Anthropic compatibility, hardware-aware model selection,
   eviction/fallback, model management, and observability.
2. **Installation:** the three-command default deployment block, followed by
   one GGUF pull and one vLLM pull example.
3. **Details:** how it works, backend behavior, configuration, API examples,
   comparison, compatibility, platform alternatives, updating, and
   troubleshooting.

The opening copy must no longer say the product is GGUF-only or llama.cpp-only.
It must still state the current operational boundaries: experimental,
single-GPU, one resident model, trusted LAN, and Linux/NVIDIA as the primary
automated deployment target.

## Documentation Updates

`docs/ARCHITECTURE.md` documents the backend trait, llama.cpp and vLLM process
lifecycles, and registry-driven selection. `docs/CONFIGURATION.md` documents
all vLLM registry fields and dual-source fallback. `docs/USAGE.md` documents
search, pull, startup, update, and verification for both formats.
`docs/COMPATIBILITY.md` separates proxy endpoint support from backend/model
support. `docs/COMPARISON.md` removes stale llama.cpp-only claims and compares
the actual dual-backend product.

## Failure Handling and Security

- Every dependency check fails before a model download begins.
- vLLM setup reports unsupported platform, Python, CUDA, or wheel resolution
  errors directly and leaves existing environments intact when possible.
- The installer never enables `trust_remote_code` automatically.
- Safetensors remains the accepted automated Hugging Face weight format.
  Pickle-backed model files require an explicit future design and are not part
  of this change.
- Existing user-owned `config.toml` and `models.toml` values are preserved
  except for missing deployment-managed absolute paths.

## Verification

Deployment regression tests cover default dual setup, both skip flags,
idempotent vLLM setup, service-account access, preservation of custom registry
values, vLLM-only model candidates, and model-free next-step output. Tests use
fake backend executables and temporary directories; they do not download or
compile llama.cpp/vLLM.

Documentation checks assert the README heading order, the three-command install
block, both model-pull paths, and removal of stale llama.cpp-only positioning.
The final repository gate is:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
./scripts/test-deploy-models.sh
```

Real CUDA llama.cpp and vLLM installation remains a Linux deployment
verification step and must be reported separately from local mocked tests.
