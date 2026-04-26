# RealESRGAN_x4plus_anime_6B → ONNX

Matches the official **x4 RRDBNet with 6 blocks** from [Real-ESRGAN anime model docs](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/anime_model.md).

## Weights

Place `RealESRGAN_x4plus_anime_6B.pth` under `models/` in this folder (or set `WEIGHTS_PATH` to a path relative to this directory).

## Export

From the repository root:

```bash
WORK_DIR=./example/realesrgan_x4plus_anime_6b docker compose run --rm export
```

Optional: `EXPORT_INPUT_H`, `EXPORT_INPUT_W` (default 64×64), `EXPORT_OPSET` (default 17).

Output: `out/model.onnx`. The graph expects **RGB**, **float32**, range **\[0, 1\]**, NCHW — the same tensor you would feed after `img / 255.0` and `BGR → RGB` as in upstream `RealESRGANer`.

## Full pipeline to TF.js

After ONNX exists, run convert (see main README) with `WORK_DIR` pointing at this folder, or set `ONNX_INPUT` to the ONNX path inside the container.
