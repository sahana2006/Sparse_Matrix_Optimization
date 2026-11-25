#!/usr/bin/env bash
# Usage: capture_gpu_utilization.sh [seconds] [output_file]
set -euo pipefail
seconds="${1:-60}"
out="${2:-gpu_utilization.log}"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found" >&2
  exit 1
fi
echo "# sampled at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$out"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 1 -lms 1000 | head -n "$((seconds + 1))" >>"$out"
echo "Wrote $out"
