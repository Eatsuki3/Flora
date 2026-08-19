# Microbenchmarks

This directory contains microbenchmarks for two Flora sub-protocols:

- **`filter`** — oblivious zero-test / cycle filtering (`obliviousFilter`).
  `S1`/`S2` hold shares of synthetic filled paths; `S3` serves GF zero-test correlations.
- **`neighbor`** — oblivious prefix neighbor passing (`obliviousPrefixNeighborPass`).
  `S1`/`S2` hold shares of synthetic edge + path tuples; `S3` serves Boolean Beaver triples.

In both cases `S1` generates the random input shares and sends one copy to `S2`; `S3` holds no input and only acts as dealer.

## Build

```bash
# from micro-eval/
cmake -B build
cmake --build build -j
```

Binaries are produced under `build/bin/`:

- `build/bin/filter`
- `build/bin/neighbor`

## Preprocessing

The microbenchmarks load the same correlation files as the end-to-end server,
from `../computing-server/preprocess/` (relative to the `micro-eval/` working
directory). If you have not generated them yet, do so once:

```bash
bash scripts/gen_triples.sh
```

This builds `generate_flora_triples` (if needed) and writes
`neighbor_s{0,1}.bin` and `filter_s{0,1}.bin` into
`../computing-server/preprocess/`. Set `NEIGHBOR_COUNT` / `FILTER_COUNT` to
override the default 20 M records each, e.g.:

```bash
NEIGHBOR_COUNT=50000000 FILTER_COUNT=50000000 bash scripts/gen_triples.sh
```

Files rewind when exhausted, so the counts only need to be large enough to
avoid excessive rewinding for your benchmark sizes.

## Config files

| Program | Config directory | Microbench-specific fields |
|---|---|---|
| `filter` | [`filter_configs/`](./filter_configs/) | `path_num` (number of paths), `valid_ratio`, `cycle_ratio`, `run_times`. `max_degree` must be `1`. |
| `neighbor` | [`neighbor_configs/`](./neighbor_configs/) | `nodes_num` (number of nodes), `path_num` (paths per node), `run_times`. Total tuples = `nodes_num * (1 + path_num)`. |

The common fields (`*_parallel_num`, seeds, `ip_*`, `*_batch_num`) have the
same meaning as in the end-to-end configs; see
[`../computing-server/README.md`](../computing-server/README.md). The shipped
configs use `127.0.0.1` and are ready for a local three-terminal run.

> All three parties must use identical data-size parameters (`path_num`,
> `nodes_num`, `max_degree`, `detect_length_k`, `total_detect_length`). Only
> `server_id`, the seeds, and the bind/connect IPs differ across parties.

## Run — three terminals

Start **S3 first**, then S1 and S2. You can use the convenience wrappers,
which run the binary with the right config and redirect output to
`logs_filter/` or `logs_neighbor/`:

```bash
# from micro-eval/
# terminal 1 — S1
bash scripts/run_filter.sh 1
# terminal 2 — S2
bash scripts/run_filter.sh 2
# terminal 3 — S3
bash scripts/run_filter.sh 3
```

Replace `run_filter.sh` with `run_neighbor.sh` for the neighbor benchmark. An
optional second argument overrides the log path:

```bash
bash scripts/run_filter.sh 1 logs_filter/filter_p24_k5_s1.log
```

Equivalently, run the binaries directly:

```bash
# terminal 1
./build/bin/filter   filter_configs/config_s1.txt   > logs_filter/filter_s1.log 2>&1
# terminal 2
./build/bin/filter   filter_configs/config_s2.txt   > logs_filter/filter_s2.log 2>&1
# terminal 3
./build/bin/filter   filter_configs/config_s3.txt   > logs_filter/filter_s3.log 2>&1
```

For neighbor, swap `filter` / `filter_configs` / `logs_filter` for `neighbor`
/ `neighbor_configs` / `logs_neighbor`.

## Sweeping parameters

To benchmark different sizes, edit `path_num` / `nodes_num` / `detect_length_k`
/ `valid_ratio` in the configs (keep all three parties in sync) and re-run.
For example, `path_num = 2^24` is written as `16777216`.

## Output

Each repetition prints a timing line:

```
[TIME] round 0 filter takes: 2.58133 seconds, with tuples of 4194304, valid ratio: 0.05, cycle ratio: 0.005
```

```
[TIME] round 0 neighbor passing takes: ... seconds, with nodes_num: 131072, path_num: 1, in total 262144 tuples
```

`S1`/`S2` also print `[COMM]` lines with on-the-wire byte counts and
round counts.

## Optional traffic capture

`scripts/network_analysis.py` parses a `tcpdump` capture and separates
MPC traffic (ports 12345–12857) from setup traffic (port 5555):

```bash
sudo tcpdump -i any -nn host <peer_ip> -w capture.pcap &
# ... run the benchmark ...
sudo pkill -SIGINT tcpdump
python3 scripts/network_analysis.py capture.pcap
```
