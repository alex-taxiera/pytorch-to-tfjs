#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WORK_HOST="${WORK_DIR:-./work}"
ONNX_HOST="$WORK_HOST/out/model.onnx"

docker compose run --rm export

if [[ ! -f "$ONNX_HOST" ]]; then
  echo "error: expected ONNX at $ONNX_HOST after export" >&2
  exit 1
fi

if [[ -n "${ONNX2TF_ARGS:-}" ]]; then
  docker compose run --rm -e "ONNX2TF_ARGS=$ONNX2TF_ARGS" convert
else
  docker compose run --rm convert
fi

echo "Pipeline finished. Outputs under $WORK_HOST/out/"
