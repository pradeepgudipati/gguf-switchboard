# Deploy Runtime Configuration Design

## Goal

Make a first `./deploy.sh` invocation create the runtime configuration and default model directory while ensuring later deployments do not replace user-owned configuration.

## File Ownership

The repository will track `config.example.toml` and `models.example.toml` as installation defaults. Runtime `config.toml`, `models.toml`, and `models.json` will be ignored by Git and owned by the user.

Existing tracked `config.toml` and `models.toml` will be renamed to their `.example.toml` counterparts. A checkout that already has runtime files must retain their contents through migration and subsequent deployments.

## First Installation

When `MODELS_DIR` is unset, `deploy.sh` will use and create `~/models`. When `MODELS_DIR` is set, it will create the explicitly configured directory when the value identifies one local directory; comma-separated discovery paths will retain their existing validation behavior.

If runtime `config.toml` is missing, deployment will copy `config.example.toml`. If runtime `models.toml` is missing, deployment will run model discovery against the resolved model directory. If discovery cannot produce a registry because no GGUF files exist, deployment will copy `models.example.toml` so the runtime file still exists and print the existing next-step guidance.

## Subsequent Deployment

Deployment will preserve existing runtime `config.toml` and `models.toml`. Only `--refresh-models` may regenerate and merge the model registry. It will not replace `config.toml`.

The systemd service will continue reading the resolved runtime `config.toml`. `GGUF_SWITCHBOARD_CONFIG_DIR` remains supported for users who keep runtime configuration outside the checkout.

## Documentation and Validation

Update README installation and configuration guidance to distinguish tracked examples from runtime files. Extend the deployment regression script to verify first-run directory and file creation, preservation on a second run, and explicit model refresh behavior. Run the repository precommit gate before completion.
