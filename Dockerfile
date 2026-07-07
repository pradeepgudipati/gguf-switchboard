# ─── Stage 1: Build ───────────────────────────────────────────────────────────
FROM rust:1.85-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src/ src/
COPY swagger-ui-overrides/ swagger-ui-overrides/
COPY .cargo/ .cargo/

RUN cargo build --release && strip /app/target/release/openai-runtime

# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /var/lib/openai-runtime && \
    chown appuser:appuser /var/lib/openai-runtime
USER appuser
WORKDIR /home/appuser

COPY --from=builder /app/target/release/openai-runtime /usr/local/bin/openai-runtime

# Default config location (mount your own at runtime)
COPY --chown=appuser:appuser config.toml /home/appuser/config.toml

EXPOSE 9090

ENTRYPOINT ["openai-runtime"]
CMD ["config.toml"]
