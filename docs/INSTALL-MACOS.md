# Install on macOS

> [← Back to README](../README.md)

macOS supports GGUF Switchboard as a source build with llama.cpp and Metal. The Linux `deploy.sh` systemd installer and the vLLM backend are not supported by this guide.

## Prerequisites

Install the Xcode command-line tools, Rust, CMake, and `jq`:

```bash
xcode-select --install
brew install cmake jq rustup
rustup-init
```

Open a new terminal after `rustup-init` if `cargo` is not yet on `PATH`.

## Build llama.cpp with Metal

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j"$(sysctl -n hw.ncpu)"
sudo cp build/bin/llama-server /usr/local/bin/
llama-server --version
cd ..
```

## Build GGUF Switchboard

```bash
git clone --branch main https://github.com/pradeepgudipati/gguf-switchboard.git
cd gguf-switchboard
cargo build --release

cp config.example.toml config.toml
cp models.example.toml models.toml
```

Download a GGUF model into a user-owned directory, then generate the registry:

```bash
mkdir -p "$HOME/models"
./target/release/gguf-switchboard models search "Qwen3.5 9B"
./target/release/gguf-switchboard models pull \
  lmstudio-community/Qwen3.5-9B-GGUF \
  --quant Q4_K_M \
  --dir "$HOME/models"
./target/release/gguf-switchboard discover-models "$HOME/models" -o models.toml
```

Start GGUF Switchboard in the foreground:

```bash
./target/release/gguf-switchboard config.toml
```

macOS has no included background-service definition. Keep the process in a terminal or create a user-owned `launchd` service.

## Verify

In another terminal:

```bash
curl -s http://localhost:9090/health
curl -s http://localhost:9090/v1/models | jq '.data[].id'
```

Open `http://localhost:9090/swagger-ui/` to use the interactive API documentation.

## Optional shell alias

```bash
echo "alias ggs='$PWD/target/release/gguf-switchboard'" >> ~/.zshrc
source ~/.zshrc
```

This alias points to the current checkout. Move the binary to a stable location before changing the alias if you plan to delete or relocate the repository.
