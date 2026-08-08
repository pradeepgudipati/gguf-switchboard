# README Quick Start Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorder the README Quick Start so users install gguf-switchboard before running its model-management commands.

**Architecture:** Keep the existing instructions and commands in `README.md`, but move the first-install paths ahead of model acquisition. Use cross-links for alternative installation methods and keep updating, verification, and alias guidance after the first-run sequence.

**Tech Stack:** Markdown, shell command examples, Git

## Global Constraints

- Change only the organization and connective wording of `README.md`.
- Preserve the existing llama.cpp, deployment, model-management, update, alias, and verification instructions unless a small wording change is required for the new order.
- Do not invoke `gguf-switchboard models ...` before an installation path has made the command available.
- Preserve unrelated working-tree content, including `.zcode/`.

---

### Task 1: Reorder and validate the Quick Start

**Files:**
- Modify: `README.md`
- Test: `README.md` structural and diff inspection

**Interfaces:**
- Consumes: Existing llama.cpp installation, deployment, prebuilt binary, source build, model-management, verification, updating, and alias instructions in `README.md`.
- Produces: A linear first-run guide in which every command appears after its required executable is installed.

- [x] **Step 1: Move gguf-switchboard installation ahead of model management**

Place the Linux/systemd installation immediately after the llama.cpp prerequisite. Keep prebuilt and source-build alternatives adjacent to the primary installation path, then place GGUF model search and download instructions after those paths.

- [x] **Step 2: Adjust connective wording**

Remove instructions that tell users to acquire GGUF files before installing gguf-switchboard. Make the first-install comments and model refresh guidance consistent with the new sequence without changing command semantics.

- [x] **Step 3: Verify structure and command ordering**

Run:

```bash
rg -n '^#{2,4} |gguf-switchboard models (search|files|pull)|git clone --branch main|./deploy.sh' README.md
git diff --check
git diff -- README.md
```

Expected: an installation heading and its commands precede the first `gguf-switchboard models` command; Markdown headings remain coherent; `git diff --check` reports no errors; the diff contains no unrelated content changes.

- [x] **Step 4: Commit**

```bash
git add README.md docs/superpowers/plans/2026-08-08-readme-quick-start-flow.md
git commit -m "docs: organize Quick Start installation flow"
```
