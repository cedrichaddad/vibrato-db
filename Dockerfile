FROM rust:1.84-bookworm AS builder

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends pkg-config libsqlite3-dev ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml ./
COPY src ./src
COPY crates ./crates
COPY benches ./benches
COPY tests ./tests
COPY README.md LICENSE ./

RUN cargo build --release --bin vibrato-db

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates libsqlite3-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /var/lib/vibrato

COPY --from=builder /app/target/release/vibrato-db /usr/local/bin/vibrato-db

ENV RUST_LOG=info
EXPOSE 8080

ENTRYPOINT ["vibrato-db"]
CMD ["serve-v2", "--data-dir", "/var/lib/vibrato", "--collection", "default", "--dim", "128", "--host", "0.0.0.0", "--port", "8080"]
