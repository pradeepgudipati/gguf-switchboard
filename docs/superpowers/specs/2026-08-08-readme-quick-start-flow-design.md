# README Quick Start Flow Design

## Goal

Reorder the README Quick Start so a new user installs each dependency before the guide asks them to run its commands.

## Scope

Change only the organization and connective wording of `README.md`. Preserve the existing llama.cpp, deployment, model-management, update, alias, and verification instructions unless a small wording change is required for the new order.

## Structure

The Quick Start will progress in this order:

1. Explain prerequisites.
2. Install llama.cpp and verify `llama-server`.
3. Install gguf-switchboard, with Linux/systemd as the primary path and links or adjacent alternatives for prebuilt binaries and source builds.
4. Search, inspect, download, validate, and register GGUF models using the installed `gguf-switchboard` command.
5. Start or refresh the service as appropriate and verify its health, status, and model registry.
6. Cover updates and the optional `ggs` shell alias after the first-run path.

The model section must not invoke `gguf-switchboard models ...` before an installation path has made that command available.

## Validation

Review the rendered Markdown hierarchy and anchor links, confirm every setup command appears after its prerequisite installation, and inspect the final diff for accidental content loss or unrelated edits.
