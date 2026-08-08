# Stable Model Port Allocation Design

## Goal

Prevent model-switch failures caused by multiple registered models receiving the same llama-server port after discovery or registry refresh.

## Port allocation

The default model backend port range starts at `18081`. Models without an explicit `port` are ordered deterministically using the registry's normalized ordering and receive the next available consecutive port from that base.

Explicit per-model ports remain unchanged. The allocator reserves every explicit port before assigning automatic ports, so an automatic assignment skips any reserved value. Allocation fails with a configuration error if the `u16` port range is exhausted rather than silently saturating or producing duplicates.

This contract applies both when expanding a registry for runtime use and when discovery or refresh regenerates registry artifacts. Repeating expansion with the same model set and configuration must produce the same model-to-port mapping.

## Compatibility

Existing registries that set `defaults.base_port` retain their configured starting point. Existing `[[models]].port` values retain precedence. Registries relying on the old implicit default of `8081` move to `18081` after updating.

The tracked example configuration and user documentation will describe the new default and collision-safe allocation behavior.

## Verification

Regression tests will cover:

- the new implicit default base port;
- deterministic consecutive allocation for multiple models;
- preservation of explicit ports;
- skipping an explicit port that falls inside the automatic range;
- port-range exhaustion returning a configuration error;
- stable assignments across repeated registry expansion.

The repository precommit gate will run after implementation.
