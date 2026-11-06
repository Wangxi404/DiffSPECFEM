#!/usr/bin/env bash
set -euo pipefail
prefix=$1
shift
exec /usr/bin/time -v -o "${prefix}.rank${OMPI_COMM_WORLD_RANK}.time.txt" "$@"
