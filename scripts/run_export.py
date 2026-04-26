#!/usr/bin/env python3
"""Run user export_onnx.py from /work and optionally validate with ONNX Runtime."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> int:
    work = Path(os.environ.get("WORK_DIR", "/work")).resolve()
    out = work / "out"
    out.mkdir(parents=True, exist_ok=True)

    script_name = os.environ.get("EXPORT_SCRIPT", "export_onnx.py")
    script_path = work / script_name
    if not script_path.is_file():
        print(f"error: missing {script_path} (mount your project at {work})", file=sys.stderr)
        return 1

    req = work / "requirements-export.txt"
    if req.is_file():
        import subprocess

        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req)],
            cwd=str(work),
        )
        if r.returncode != 0:
            return r.returncode

    sys.path.insert(0, str(work))
    runpy.run_path(str(script_path), run_name="__main__")

    onnx_path = out / "model.onnx"
    if not onnx_path.is_file():
        print(f"error: export did not write {onnx_path}", file=sys.stderr)
        return 1

    if os.environ.get("SKIP_ORT_VALIDATION", "").lower() in ("1", "true", "yes"):
        print("skip ort validation (SKIP_ORT_VALIDATION set)")
        return 0

    return 0 if _validate_ort(onnx_path) else 1


def _validate_ort(onnx_path: Path) -> bool:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        print(f"error: onnxruntime not available: {e}", file=sys.stderr)
        return False

    try:
        sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        print(f"error: failed to load ONNX in ORT: {e}", file=sys.stderr)
        return False

    feeds = {}
    for inp in sess.get_inputs():
        shape = []
        for d in inp.shape:
            if d is None or (isinstance(d, str) and d):
                shape.append(1)
            elif isinstance(d, int) and d <= 0:
                shape.append(1)
            else:
                shape.append(int(d))

        if inp.type == "tensor(float)":
            arr = np.random.randn(*shape).astype(np.float32)
        elif inp.type == "tensor(double)":
            arr = np.random.randn(*shape).astype(np.float64)
        elif inp.type in ("tensor(int64)", "tensor(int32)"):
            arr = np.zeros(shape, dtype=np.int64 if "int64" in inp.type else np.int32)
        else:
            arr = np.random.randn(*shape).astype(np.float32)

        feeds[inp.name] = arr

    try:
        sess.run(None, feeds)
    except Exception as e:
        print(f"error: ORT inference failed: {e}", file=sys.stderr)
        return False

    print("onnx runtime validation ok")
    return True


if __name__ == "__main__":
    raise SystemExit(main())
