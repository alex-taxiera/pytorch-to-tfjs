# Minimal example

Used by the repo smoke test: a tiny CNN exports to `out/model.onnx`, then the convert image produces `out/saved_model/` and `out/tfjs_model/`.

From the repository root:

```bash
WORK_DIR=./example/minimal ./scripts/convert.sh
```

The first run creates `model.pth` with random weights if no weights are present. If you add `model.safetensors` (same `state_dict` layout as `TinyNet`), that file is loaded instead of `model.pth`.
