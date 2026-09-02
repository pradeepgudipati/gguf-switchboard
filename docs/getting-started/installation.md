# Installation

> [← Back to README](../../README.md)

Platform-specific installation guides for GGUF Switchboard.

## Linux (Primary)

Linux with NVIDIA/CUDA is the primary deployment target.

See [Linux Installation](linux.md) for detailed instructions.

## macOS

Build from source with Apple Metal support.

See [macOS Installation](macos.md) for detailed instructions.

## Windows

Run via WSL2 with NVIDIA CUDA support.

See [Windows Installation](windows.md) for detailed instructions.

## Prebuilt binaries

Prebuilt binaries are available on the [GitHub Releases](https://github.com/pradeepgudipati/gguf-switchboard/releases) page:

- `gguf-switchboard-linux-amd64`
- `gguf-switchboard-linux-arm64`
- `gguf-switchboard-darwin-amd64`
- `gguf-switchboard-darwin-arm64`

Download the appropriate binary, make it executable, and place it in your PATH:

```bash
chmod +x gguf-switchboard-linux-amd64
sudo mv gguf-switchboard-linux-amd64 /usr/local/bin/gguf-switchboard
```

## Verify installation

```bash
gguf-switchboard --help
```
