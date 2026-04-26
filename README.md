# PyTorch → ONNX → TensorFlow SavedModel → TensorFlow.js (Docker)

End-to-end, Dockerized conversion pipeline:

1. **PyTorch** (optional) — you provide `export_onnx.py` that loads your weights (`.pth`, `.safetensors`, etc.) and writes **`out/model.onnx`**, *or* you **start from an existing `.onnx` file** (see [Starting from existing ONNX](#starting-from-existing-onnx-skip-pytorch)).
2. **ONNX Runtime** — optional sanity check after export (disable with `SKIP_ORT_VALIDATION=1`; skipped when you do not run the export container).
3. **onnx2tf** — ONNX → TensorFlow SavedModel (+ TFLite artifacts in the same folder).
4. **tensorflowjs_converter** — SavedModel → TF.js graph model.

References: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ONNX Runtime](https://github.com/microsoft/onnxruntime), [onnx2tf](https://github.com/PINTO0309/onnx2tf), [TensorFlow.js](https://github.com/tensorflow/tfjs).

## Why you must supply `export_onnx.py`

A `.pth` file is almost always weights or a checkpoint. Loading it requires the **same `nn.Module` architecture** (and preprocessing contract) that created it. Mount a working directory that contains your weights **and** a small script that builds the model, loads the checkpoint, and calls `torch.onnx.export`.

### Using `.safetensors` weights

Safetensors is still **weights only**—the pipeline does not change: your `export_onnx.py` must define (or import) the model, load tensors from disk, then export ONNX. The export image includes [`safetensors`](https://github.com/huggingface/safetensors); typical usage:

```python
from safetensors.torch import load_file

state = load_file("/work/your_weights.safetensors")
model.load_state_dict(state)  # or strip/rename keys if the file uses a prefix (common with Hugging Face checkpoints)
```

Put the file under `WORK_DIR` and point `load_file` at the path your script uses. The minimal example prefers `model.safetensors` over `model.pth` when both could exist. If you run export scripts on the host instead of Docker, install `safetensors` in that environment (or list it in `requirements-export.txt` for the container).

## Layout (mounted at `/work` in containers)

| Path | Purpose |
|------|---------|
| `export_onnx.py` | Your entrypoint (name overridable via `EXPORT_SCRIPT`). |
| `*.pth` / `*.safetensors` / checkpoint | Your weights (any name your script expects). |
| `requirements-export.txt` | Optional; installed with `pip` before export if present. |
| `out/model.onnx` | **Written by your script**; read by the convert stage. |
| `out/saved_model/` | Produced by `onnx2tf`. |
| `out/tfjs_model/` | Produced by `tensorflowjs_converter`. |

## Quick start

1. Create a directory (e.g. `./work`) with `export_onnx.py` and your weights.
2. In `export_onnx.py`, write ONNX to **`out/model.onnx`** (create `out/` if needed).
3. Run:

```bash
export WORK_DIR=./work   # default if unset: ./work
./scripts/convert.sh
```

### Minimal smoke test

```bash
WORK_DIR=./example/minimal ./scripts/convert.sh
```

## Starting from existing ONNX (skip PyTorch)

If you already have ONNX files (e.g. `4x-UltraSharp-fp32-opset17.onnx`), you only need the **`convert`** service. Do **not** run `./scripts/convert.sh` for that case — it always runs the **export** container first, which expects `export_onnx.py` on disk.

Paths below use the container view: the host directory `WORK_DIR` (default `./work`) is mounted at **`/work`**.

### Option 1: Default input path (`out/model.onnx`)

Copy or symlink your ONNX file so it appears as `out/model.onnx` inside the mounted folder:

```bash
export WORK_DIR=./work
mkdir -p "$WORK_DIR/out"
cp /path/to/4x-UltraSharp-fp32-opset17.onnx "$WORK_DIR/out/model.onnx"

docker compose run --rm convert
```

`convert` uses the default `ONNX_INPUT=/work/out/model.onnx`, which matches that layout.

### Option 2: Any path under `WORK_DIR` (`ONNX_INPUT`)

Keep filenames as-is and point `ONNX_INPUT` at the path **as seen inside the container** (`/work/...`):

```bash
export WORK_DIR=./work
mkdir -p "$WORK_DIR/models"
cp /path/to/4x-UltraSharp-fp16-opset17.onnx "$WORK_DIR/models/"

docker compose run --rm \
  -e ONNX_INPUT=/work/models/4x-UltraSharp-fp16-opset17.onnx \
  convert
```

You can set `SAVED_MODEL_DIR` and `TFJS_OUTPUT_DIR` the same way if you want outputs somewhere other than `out/saved_model` and `out/tfjs_model`.

### Opset and dtype hints

- **Opset:** Prefer **opset 17** if a lower opset fails or produces odd graphs; [onnx2tf](https://github.com/PINTO0309/onnx2tf) is aligned with current ONNX/TensorFlow stacks.
- **FP16 vs FP32:** Try **FP32** ONNX first for fewer conversion surprises; use FP16 ONNX if you need smaller graphs and the run succeeds end-to-end.

### Sanity check (optional)

Validate the file with [ONNX Runtime](https://github.com/microsoft/onnxruntime) on your machine before conversion, or temporarily run the export image against a tiny script that only loads and runs the ONNX — the export container is not required for the ONNX → TF.js path.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WORK_DIR` | `./work` | Host directory mounted as `/work`. |
| `EXPORT_SCRIPT` | `export_onnx.py` | Script under `WORK_DIR` to run for ONNX export. |
| `SKIP_ORT_VALIDATION` | unset | Set to `1` to skip ONNX Runtime inference check. |
| `ONNX_INPUT` | `/work/out/model.onnx` | Input ONNX inside container. |
| `SAVED_MODEL_DIR` | `/work/out/saved_model` | onnx2tf output directory. |
| `TFJS_OUTPUT_DIR` | `/work/out/tfjs_model` | TF.js graph model output. |
| `ONNX2TF_ARGS` | `-osd -v info` (in `run_convert.sh`) | Extra onnx2tf flags; **include `-osd`** so `saved_model.pb` exists for TF.js. |

Pass `ONNX2TF_ARGS` from the host (compose forwards it):

```bash
export ONNX2TF_ARGS="-osd -v info -ois input:1,3,224,224 -dgc"
WORK_DIR=./work ./scripts/convert.sh
```

## Manual Docker Compose steps

Full pipeline (PyTorch → ONNX → TF.js):

```bash
docker compose build
docker compose run --rm export
docker compose run --rm convert
```

ONNX only (SavedModel → TF.js):

```bash
docker compose build
# Ensure WORK_DIR/out/model.onnx exists, or pass -e ONNX_INPUT=/work/...
docker compose run --rm convert
```

## Real-ESRGAN notes

For [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), reuse the same network classes and loading logic as their inference scripts (e.g. RRDBNet and checkpoint keys from `inference_realesrgan.py`). Export a **fixed** low-resolution input size for smoother conversion. Tiled inference is often implemented **outside** the graph in the reference code; exporting a single-tile ONNX is the usual approach for web deployment.

## Troubleshooting

- **Conversion / layout errors:** onnx2tf transposes NCHW image inputs to NHWC; for audio or other layouts try `-kat` / `--keep_shape_absolutely_input_names` / `--overwrite_input_shape` per [onnx2tf CLI](https://github.com/PINTO0309/onnx2tf#cli-parameter).
- **Group convolution issues:** try `-dgc` / `--disable_group_convolution`.
- **Wrong or dynamic shapes:** set static shapes with `-ois name:1,3,H,W` or fix the dummy input in `torch.onnx.export`.
- **Debug accuracy:** onnx2tf `-cotof` compares ONNX vs TensorFlow per-op (memory-heavy on large models).
- **Missing `saved_model.pb`:** ensure `-osd` / `output_signaturedefs` is enabled for onnx2tf (default in `scripts/run_convert.sh`).

## Convert image notes

The `convert` service extends [onnx2tf’s Docker image](https://github.com/PINTO0309/onnx2tf). It installs `tensorflowjs` 4.x for `tensorflowjs_converter`, plus `jax` / `jaxlib` (required for package import). A **stub** `tensorflow_decision_forests` package is included so the converter can start without pulling TF-DF’s strict TensorFlow pins (which can fail on some platforms). That stub is only appropriate for **SavedModel → TF.js** flows like this pipeline—not for models that actually use TensorFlow Decision Forests.

## Risks

- Not every ONNX operator combination converts cleanly; you may need export flags, onnx-simplifier, or onnx2tf `-prf` replacement JSON.
- TF.js graph models work best with **static** (or tightly bounded) input shapes.

## License

Scripts and Docker glue in this repository are provided as-is for orchestration; respect the licenses of PyTorch, onnx2tf, TensorFlow.js, and your own model checkpoints.
