#!/usr/bin/env bash
set -euo pipefail

ONNX_INPUT="${ONNX_INPUT:-/work/out/model.onnx}"
SAVED_MODEL_DIR="${SAVED_MODEL_DIR:-/work/out/saved_model}"
TFJS_OUTPUT_DIR="${TFJS_OUTPUT_DIR:-/work/out/tfjs_model}"
ONNX2TF_ARGS="${ONNX2TF_ARGS:--osd -v info}"

if [[ ! -f "$ONNX_INPUT" ]]; then
  echo "error: ONNX not found: $ONNX_INPUT" >&2
  exit 1
fi

rm -rf "$SAVED_MODEL_DIR" "$TFJS_OUTPUT_DIR"
mkdir -p "$(dirname "$SAVED_MODEL_DIR")" "$(dirname "$TFJS_OUTPUT_DIR")"

echo "running onnx2tf..."
# ONNX2TF_ARGS is intentionally unquoted to allow multiple CLI flags from env
onnx2tf -i "$ONNX_INPUT" -o "$SAVED_MODEL_DIR" $ONNX2TF_ARGS

if [[ ! -d "$SAVED_MODEL_DIR" ]] || [[ -z "$(ls -A "$SAVED_MODEL_DIR" 2>/dev/null)" ]]; then
  echo "error: onnx2tf did not produce output under $SAVED_MODEL_DIR" >&2
  exit 1
fi
if [[ ! -f "$SAVED_MODEL_DIR/saved_model.pb" ]]; then
  echo "error: missing $SAVED_MODEL_DIR/saved_model.pb (ensure onnx2tf is run with -osd / output_signaturedefs)" >&2
  exit 1
fi

echo "running tensorflowjs_converter..."
if ! command -v tensorflowjs_converter >/dev/null 2>&1; then
  echo "error: tensorflowjs_converter not on PATH" >&2
  exit 1
fi
tensorflowjs_converter \
  --input_format tf_saved_model \
  --output_format tfjs_graph_model \
  "$SAVED_MODEL_DIR" \
  "$TFJS_OUTPUT_DIR"

echo "done: SavedModel -> $SAVED_MODEL_DIR, TF.js -> $TFJS_OUTPUT_DIR"
