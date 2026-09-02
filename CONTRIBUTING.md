# Contributing to GGUF Switchboard

Thank you for your interest in contributing to GGUF Switchboard! This document provides guidelines and information for contributors.

## Table of Contents

- [Development Environment](#development-environment)
- [Repository Architecture](#repository-architecture)
- [Running Locally](#running-locally)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Submitting PRs](#submitting-prs)
- [Adding Model Support](#adding-model-support)
- [Adding Runtime Support](#adding-runtime-support)
- [Adding Integrations](#adding-integrations)
- [Documentation Standards](#documentation-standards)
- [Issue Workflow](#issue-workflow)

## Development Environment

### Prerequisites

- **Rust** 1.88+ (installed automatically by `deploy.sh` if missing)
- **Git**
- **NVIDIA GPU + CUDA toolkit** (for testing GPU features)
- **llama.cpp** (for testing GGUF models)
- **Python 3.10-3.14 + uv** (for testing vLLM/SafeTensors models)

### Setup

```bash
# Clone the repository
git clone https://github.com/pradeepgudipati/gguf-switchboard.git
cd gguf-switchboard

# Install git hooks (format, clippy, build, tests run before each commit)
./scripts/install-hooks.sh

# Build in debug mode
cargo build

# Run the server locally
cargo run --release -- config.toml
```

## Repository Architecture

```
gguf-switchboard/
├── src/
│   ├── main.rs              # Entry point, CLI handling
│   ├── lib.rs               # Library root
│   ├── api/                 # HTTP route handlers
│   ├── backend/             # llama.cpp and vLLM backend implementations
│   ├── config/              # Configuration, model registry, HF integration
│   ├── scheduler/           # Model switching, priority, memory watcher
│   ├── types/               # Request/response type definitions
│   └── ...                  # Other modules (fit, gpu, metrics, etc.)
├── docs/                    # Documentation
├── tests/                   # Integration tests
├── scripts/                 # Helper scripts
├── releases/                # Per-tag release notes
└── swagger-ui-overrides/    # Swagger UI customizations
```

### Key Subsystems

| Subsystem | Location | Description |
|-----------|----------|-------------|
| **API Layer** | `src/api/` | Axum routes for OpenAI/Anthropic APIs |
| **Backends** | `src/backend/` | llama.cpp and vLLM process management |
| **Scheduler** | `src/scheduler/` | Single-slot model switching, drain, rollback |
| **Config** | `src/config/` | Configuration loading, model registry, HF integration |
| **Fit Planner** | `src/fit.rs` | Hardware-aware model fit planning |
| **GPU** | `src/gpu.rs` | nvidia-smi VRAM probing |

## Running Locally

```bash
# Create configuration files
cp config.example.toml config.toml
cp models.example.toml models.toml

# Edit config.toml to set your VRAM and paths
# Edit models.toml to add your models

# Run with cargo
cargo run --release -- config.toml

# Run with debug logging
RUST_LOG=debug cargo run --release -- config.toml
```

## Running Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test <test_name>

# Run integration tests only
cargo test --test integration

# Run scheduler tests
cargo test --test scheduler_switch

# Run responses tests
cargo test --test responses
```

## Code Style

- **Rust 2024 edition** with `rustfmt` defaults
- **Clippy warnings are errors** (`clippy::all = "warn"` in Cargo.toml)
- **No comments** unless explicitly asked
- **Prefer `thiserror`** for error types
- **Prefer `tracing`** for structured logging

### Formatting

```bash
# Check formatting
cargo fmt --all -- --check

# Apply formatting
cargo fmt --all
```

### Linting

```bash
# Run clippy
cargo clippy --all-targets --all-features -- -D warnings
```

## Submitting PRs

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the code style guidelines
3. **Run the full verification gate**:
   ```bash
   cargo fmt --all -- --check
   cargo clippy --all-targets --all-features -- -D warnings
   cargo test
   ```
4. **Write a clear commit message** describing your changes
5. **Push to your fork** and create a pull request
6. **Fill out the PR template** with details about your changes

### PR Guidelines

- Keep changes focused and atomic
- Include tests for new functionality
- Update documentation if needed
- Follow existing code patterns
- Do not introduce new dependencies without discussion

## Adding Model Support

To add support for a new model architecture:

1. Check if the architecture is supported by llama.cpp or vLLM
2. Add the architecture to the kind detection in `src/config/models_local.rs`
3. Add any required backend arguments in `src/config/models_registry.rs`
4. Test with a model of the new architecture
5. Update documentation in `docs/models/`

## Adding Runtime Support

To add a new runtime backend:

1. Implement the `Backend` trait in `src/backend/`
2. Add backend selection logic in `src/config/models_registry.rs`
3. Add configuration options to `models.example.toml`
4. Update documentation in `docs/runtimes/`
5. Add integration tests

## Adding Integrations

To add a new client integration:

1. Create a new file in `docs/integrations/`
2. Follow the integration guide template:
   - What this enables
   - Prerequisites
   - Configure the client
   - Test the connection
   - Troubleshooting
3. Add the integration to the README table
4. Test the configuration against a running instance

## Documentation Standards

- Use GitHub-flavored Markdown
- Include a `[← Back to README]` link at the top of each doc
- Use code blocks with language hints
- Include copy-paste-ready examples
- Keep documentation version-controlled
- Update internal links when moving files

## Issue Workflow

1. **Search existing issues** before creating a new one
2. **Use issue templates** when available
3. **Provide complete information** (version, OS, GPU, model, logs)
4. **Respond to questions** from maintainers
5. **Close issues** when resolved

## Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Documentation**: Check the [docs/](docs/) directory

## License

By contributing to GGUF Switchboard, you agree that your contributions will be licensed under the MIT License.
