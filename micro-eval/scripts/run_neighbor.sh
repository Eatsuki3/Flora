#!/bin/bash
#
# Launch the neighbor-passing microbenchmark for one party.
#
# Usage:
#   bash scripts/run_neighbor.sh <1|2|3> [log_file]
#
# Open three terminals and run with party 1, 2, 3 respectively.
# Start S3 first (it is the dealer).
#
set -euo pipefail

PARTY="${1:?Usage: bash scripts/run_neighbor.sh <1|2|3> [log_file]}"
if [[ ! "$PARTY" =~ ^[123]$ ]]; then
    echo "Party must be 1, 2 or 3." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICRO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MICRO_DIR"

BIN="build/bin/neighbor"
CONFIG="neighbor_configs/config_s${PARTY}.txt"
LOG="${2:-logs_neighbor/neighbor_s${PARTY}.log}"

if [[ ! -x "$BIN" ]]; then
    echo "[!] Binary not found: $BIN" >&2
    echo "    Build it first: cmake -B build && cmake --build build -j" >&2
    exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "[!] Config not found: $CONFIG" >&2
    exit 1
fi

mkdir -p "$(dirname "$LOG")"
ulimit -n 1048576 || true

echo "[*] S${PARTY}: $BIN $CONFIG > $LOG 2>&1"
exec "$BIN" "$CONFIG" > "$LOG" 2>&1
