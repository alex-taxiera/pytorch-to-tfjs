# realesr-animevideov3 → ONNX

Matches the official **SRVGGNetCompact** ×4 model from [Real-ESRGAN `inference_realesrgan.py`](https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py) (`realesr-animevideov3`).

## Weights

Place `realesr-animevideov3.pth` under `models/` in this folder (or set `WEIGHTS_PATH` to a path relative to this directory). The published checkpoint uses the `params` key (not `params_ema`).

## Export

From the repository root:

```bash
WORK_DIR=./example/realesr_animevideov3 docker compose run --rm export
```

Optional: `EXPORT_INPUT_H`, `EXPORT_INPUT_W` (default 64×64), `EXPORT_OPSET` (default 17).

Output: `out/model.onnx`. The graph expects **RGB**, **float32**, range **\[0, 1\]**, NCHW — same convention as upstream `RealESRGANer`.

## Full pipeline to TF.js

After ONNX exists, run the `convert` service (see main README) with this directory as `WORK_DIR`, or set `ONNX_INPUT` to the ONNX path inside the container.
