# Flora: Privacy-Preserving Cycle Detection for Federated Graphs

This repository contains the implementation of **Flora**, a privacy-preserving
cycle detection service for federated graphs. Three parties (`S1`, `S2`, `S3`)
jointly detect cycles in a graph that is secret-shared across `S1` and `S2`,
while `S3` acts as a dealer serving pre-generated correlations (Beaver triples
and GF zero-test correlations). Both CPU and GPU (CUDA) backends are provided.

## Repository layout

| Directory | Description |
|---|---|
| [`common/`](./common/) | Shared constants, config parsing, and networking helpers. |
| [`secret-share/`](./secret-share/) | Secret-shared tuple types and (de)serialization. |
| [`computing-server/`](./computing-server/) | End-to-end cycle-detection protocol (CPU `server` + GPU `server_gpu`), preprocessing, and configs. |
| [`micro-eval/`](./micro-eval/) | Microbenchmarks for the **filter** and **neighbor-passing** sub-protocols. |
| [`dataset-scripts/`](./dataset-scripts/) | Scripts for generating / transforming graph datasets. |
| [`demo-graph/`](./demo-graph/) | A tiny pre-shared demo graph for quick tests. |
| [`local-cycle-detect/`](./local-cycle-detect/) | Plaintext local cycle counting / degree analysis tools. |
| [`emp-tool/`](./emp-tool/), [`emp-ot/`](./emp-ot/), [`emp-sh2pc/`](./emp-sh2pc/) | Vendored [EMP toolkit](https://github.com/emp-toolkit) dependencies (the `emp-tool` threading patch is already applied). |

## Quick start

### 1. Install dependencies and build EMP

See [`install.md`](./install.md) for system packages and how to build/install the
vendored `emp-tool` / `emp-ot` / `emp-sh2pc`.

### 2. Build Flora

```bash
# End-to-end server (CPU + GPU; pass -DBUILD_GPU=OFF to skip CUDA)
cd computing-server
cmake -B build
cmake --build build -j

# Microbenchmarks
cd ../micro-eval
cmake -B build
cmake --build build -j
```

### 3. Generate preprocessing correlations

`S3` streams pre-generated triples to `S1`/`S2` during the protocol. Generate
them once with the built-in tool (run from `computing-server/`):

```bash
./build/bin/generate_flora_triples 0 20000000 20000000 \
    preprocess/sort_s0.bin     preprocess/sort_s1.bin \
    preprocess/neighbor_s0.bin preprocess/neighbor_s1.bin \
    preprocess/filter_s0.bin   preprocess/filter_s1.bin
```

The first number is the sort-triple count (only needed by the GPU backend; `0`
is fine for CPU/micro-eval). Triple files automatically rewind when exhausted,
so the counts only need to be large enough to avoid excessive rewinding.

### 4. Run — three terminals

All protocols run as three independent processes, one per party. Open three
terminals and start `S3` first (it is the dealer), then `S1` and `S2`:

```bash
# from computing-server/
# terminal 1 (S1)
./build/bin/server configs/config_s1.txt > logs/s1.log 2>&1
# terminal 2 (S2)
./build/bin/server configs/config_s2.txt > logs/s2.log 2>&1
# terminal 3 (S3)
./build/bin/server configs/config_s3.txt > logs/s3.log 2>&1
```

Use `server_gpu` instead of `server` for the GPU backend. The microbenchmarks
follow the same pattern; see [`micro-eval/README.md`](./micro-eval/README.md).

> Each party binds/connects according to its own config file. For a local
> smoke test the shipped configs already use `127.0.0.1`; for a distributed
> deployment update the `ip_1/ip_2/ip_3` fields and `share_file*` paths.

## Further reading

- [`install.md`](./install.md) — environment and EMP build instructions.
- [`computing-server/README.md`](./computing-server/README.md) — end-to-end
  build options, config reference, and output format.
- [`micro-eval/README.md`](./micro-eval/README.md) — filter / neighbor
  microbenchmarks.
- [`dataset-scripts/README.md`](./dataset-scripts/README.md) — dataset
  preparation.