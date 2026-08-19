# End-to-end cycle detection

This directory contains the full Flora protocol. `S1` and `S2` hold 2-out-of-2
shares of the federated graph and perform the online 2PC computation; `S3` is a
dealer that streams pre-generated Beaver triples (neighbor passing / sort) and
GF zero-test correlations (filter) to `S1`/`S2`.

## Build

```bash
# from computing-server/
cmake -B build
cmake --build build -j
```

Binaries are produced under `build/bin/`:

| Binary | Description |
|---|---|
| `server` | CPU backend. |
| `server_gpu` | GPU (CUDA) backend. Built automatically when CUDA ≥ 11.1 is detected; pass `-DBUILD_GPU=OFF` to skip. |
| `generate_flora_triples` | Offline preprocessing tool (see below). |

## Preprocessing

`S3` loads correlation files from `preprocess/` at startup (paths are relative
to the working directory, so run from `computing-server/`).

Generate them once:

```bash
./build/bin/generate_flora_triples \
    <sort_count> <neighbor_count> <filter_count> \
    preprocess/sort_s0.bin     preprocess/sort_s1.bin \
    preprocess/neighbor_s0.bin preprocess/neighbor_s1.bin \
    preprocess/filter_s0.bin   preprocess/filter_s1.bin
```

- `sort_count` is only used by `server_gpu`; set it to `0` for the CPU backend.
- `neighbor_count` / `filter_count` should be large enough for your graph.
  Files automatically rewind when exhausted (correlations are reused), so a few
  tens of millions is usually sufficient for benchmarking.
- Each record is 12 bytes; e.g. 20 M records ≈ 240 MB per file.

Example (CPU + micro-eval):

```bash
./build/bin/generate_flora_triples 0 20000000 20000000 \
    preprocess/sort_s0.bin     preprocess/sort_s1.bin \
    preprocess/neighbor_s0.bin preprocess/neighbor_s1.bin \
    preprocess/filter_s0.bin   preprocess/filter_s1.bin
```

## Config files

Templates for the three parties are in [`configs/`](./configs/). Each config is
a sequence of `key` / `value` line pairs. Key fields:

| Field | Meaning |
|---|---|
| `debugging` | `1` to reconstruct and print plaintext shares at each stage (S1 only, verbose); `0` for normal runs. |
| `server_id` | `1`, `2`, or `3`. |
| `max_degree` | Maximum node degree `d`. |
| `detect_length_k` | Initial cycle length to detect. |
| `total_detect_length` | Detect cycles up to this length (inclusive). |
| `*_parallel_num` | Thread counts for local compute / neighbor pass / sort / filter / extract. |
| `sort_elem_threshold` | Partitions no larger than this are sorted directly without further pivoting. |
| `num_to_find_median` | Number of random candidates sampled when choosing a pivot. |
| `seed12_permute`, `seed23_permute`, `seed31_permute`, `seedA`, `seedB` | Shuffle permutation seeds. The shipped configs use the correct per-party values; keep them in sync across parties. |
| `ip_1`, `ip_2`, `ip_3` | Party addresses. Each party sets its **own** entry to `0.0.0.0:<port>` (bind) and the others to the connectable `host:port`. |
| `*_batch_num` | Batch size for filter / neighbor / extract / partition; `0` means process everything in one batch. |
| `share_file1`, `share_file2` | Paths to this party's two input share files. S1 holds (out1, out2), S2 holds (out2, out3), S3 holds (out3, out1). |

> For distributed deployment, update the IPs and `share_file*` paths in all
> three configs. The shipped configs use `127.0.0.1` and the bundled
> `dataset/out*_small_100` shares for a local smoke test.

## Run — three terminals

Start **S3 first** (it must be listening as dealer), then S1 and S2:

```bash
# from computing-server/
mkdir -p logs

# terminal 1 — S1
./build/bin/server configs/config_s1.txt > logs/s1.log 2>&1

# terminal 2 — S2
./build/bin/server configs/config_s2.txt > logs/s2.log 2>&1

# terminal 3 — S3
./build/bin/server configs/config_s3.txt > logs/s3.log 2>&1
```

For the GPU backend, replace `server` with `server_gpu` (and make sure
`sort_s0.bin` / `sort_s1.bin` exist).

Follow progress with `tail -f logs/s1.log`.

## Output

Each round prints a `[TIMING PROFILE]` block per module, e.g.:

```
┌────────────────────────────────────────────────────────
│  [TIMING PROFILE] Module: ObShuffle-1
├────────────────────────────────────────────────────────
│  1. computation time:     ... s  (xx%)
│  2. communication time:   ... s  (xx%)
│  3. Serial time:          ... s  (xx%)
├────────────────────────────────────────────────────────
│  End-to-end time:         ... s
└────────────────────────────────────────────────────────
```

The detection iterates `k = detect_length_k … total_detect_length`. Each round
reports the number of detected cycles and finishes with:

```
[TIME] ──>  Round k takes: ... seconds.
```

When no active paths remain, or `total_detect_length` is reached:

```
[Flora] Done.
```

## Datasets

Sample share files are included under [`dataset/`](./dataset/). To prepare
other graphs, see [`../dataset-scripts/README.md`](../dataset-scripts/README.md).
Each share file may have a companion `<file>_meta` containing the number of
pristine edge tuples (the `V` boundary); if absent, the whole file is treated
as edges (only safe for `k = 2`).
