# Install on Windows

> [← Back to README](../README.md)

GGUF Switchboard does not currently provide a native Windows deployment. Use WSL2 with an Ubuntu distribution so the Linux build, CUDA tooling, and systemd workflow remain consistent with the primary deployment target.

## WSL2 prerequisites

From an elevated PowerShell terminal:

```powershell
wsl --install -d Ubuntu
```

Restart Windows if requested, finish the Ubuntu account setup, and confirm that WSL uses version 2:

```powershell
wsl --list --verbose
```

For NVIDIA acceleration, install a current Windows NVIDIA driver with WSL CUDA support. In the Ubuntu terminal, verify that the GPU is visible:

```bash
nvidia-smi
```

## Install inside Ubuntu

Run the normal Linux installation from the WSL Ubuntu terminal, not PowerShell:

```bash
sudo apt update
sudo apt install -y git curl

git clone --branch main https://github.com/pradeepgudipati/gguf-switchboard.git
cd gguf-switchboard
./deploy.sh
```

If systemd is disabled in the WSL distribution, enable it in `/etc/wsl.conf`:

```ini
[boot]
systemd=true
```

Then restart WSL from PowerShell:

```powershell
wsl --shutdown
```

Reopen Ubuntu and rerun `./deploy.sh`.

## Verify

From the Ubuntu terminal:

```bash
curl -s http://localhost:9090/health
curl -s http://localhost:9090/v1/models | jq '.data[].id'
```

Windows normally forwards WSL localhost ports, so `http://localhost:9090/swagger-ui/` should open in a Windows browser.

## Optional alias

Create the alias inside WSL because GGUF Switchboard runs there:

```bash
echo "alias ggs='gguf-switchboard'" >> ~/.bashrc
source ~/.bashrc
```

The previous native PowerShell alias is intentionally not used: it would point at a Windows executable that this project does not ship.
