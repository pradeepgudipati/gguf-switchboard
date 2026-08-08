# Stable Model Port Allocation Design

## Goal

Prevent model-switch failures caused by multiple registered models receiving the same llama-server port after discovery or registry refresh.

## Port allocation

The default model backend port range starts at `18081`. Registered models are ordered deterministically using the registry's normalized ordering and receive consecutive ports from that base.

Model backend ports are internal runtime details and are not part of the public API or UI. Discovery and refresh therefore normalize existing per-model `port` values into the same consecutive range instead of preserving manual pins. Allocation fails with a configuration error if the `u16` port range is exhausted rather than silently saturating or producing duplicates.

This contract applies both when expanding a registry for runtime use and when discovery or refresh regenerates registry artifacts. Repeating expansion with the same model set and configuration must produce the same model-to-port mapping.

## Compatibility

Existing registries that set `defaults.base_port` retain their configured starting point. Registries relying on the old implicit default of `8081` move to `18081` after updating. Discovery and refresh replace existing `[[models]].port` values with the normalized consecutive assignment; ordinary startup expansion uses the normalized registry values.

The tracked example configuration and user documentation will describe the new default and collision-safe allocation behavior.

## Verification

Regression tests will cover:

- the new implicit default base port;
- deterministic consecutive allocation for multiple models;
- replacement of existing explicit ports during discovery and refresh;
- port-range exhaustion returning a configuration error;
- stable assignments across repeated registry expansion.

The repository precommit gate will run after implementation.
