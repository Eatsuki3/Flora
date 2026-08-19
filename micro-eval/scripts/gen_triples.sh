#!/bin/bash
#
# Generate preprocessing correlations for the microbenchmarks.
#
# Writes neighbor_s{0,1}.bin and filter_s{0,1}.bin into
# ../computing-server/preprocess/ (the path hard-coded in eval_server.hpp).
# Sort triples are not needed by the CPU microbenchmarks, so sort_count=0.
#
# Override counts with environment variables, e.g.:
#   NEIGHBOR_COUNT=50000000 FILTER_COUNT=50000000 bash scripts/gen_triples.sh
#
set -euo pipefail

SORT_COUNT="${SORT_COUNT:-0}"
NEIGHBOR_COUNT="${NEIGHBOR_COUNT:-20000000}"
FILTER_COUNT="${FILTER_COUNT:-20000000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICRO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER_DIR="$(cd "$MICRO_DIR/../computing-server" && pwd)"
PREPROCESS_DIR="$SERVER_DIR/preprocess"
GENERATOR="$SERVER_DIR/build/bin/generate_flora_triples"

echo "[*] Repository layout:"
echo "    computing-server: $SERVER_DIR"
echo "    preprocess dir:   $PREPROCESS_DIR"

# Build the generator if it is missing.
if [[ ! -x "$GENERATOR" ]]; then
    echo "[*] Building generate_flora_triples ..."
    cmake -S "$SERVER_DIR" -B "$SERVER_DIR/build"
    cmake --build "$SERVER_DIR/build" -j --target generate_flora_triples
fi

mkdir -p "$PREPROCESS_DIR"

echo "[*] Generating triples:"
echo "    sort_count     = $SORT_COUNT"
echo "    neighbor_count = $NEIGHBOR_COUNT"
echo "    filter_count   = $FILTER_COUNT"

"$GENERATOR" \
    "$SORT_COUNT" "$NEIGHBOR_COUNT" "$FILTER_COUNT" \
    "$PREPROCESS_DIR/sort_s0.bin"     "$PREPROCESS_DIR/sort_s1.bin" \
    "$PREPROCESS_DIR/neighbor_s0.bin" "$PREPROCESS_DIR/neighbor_s1.bin" \
    "$PREPROCESS_DIR/filter_s0.bin"   "$PREPROCESS_DIR/filter_s1.bin"

echo "[*] Done. Files in $PREPROCESS_DIR:"
ls -lh "$PREPROCESS_DIR"/*.bin
