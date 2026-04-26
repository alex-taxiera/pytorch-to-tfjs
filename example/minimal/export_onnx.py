"""Minimal smoke-test export: tiny CNN -> work/out/model.onnx."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv(x)))
        return self.fc(x.flatten(1))


def main() -> None:
    work = Path(__file__).resolve().parent
    out = work / "out"
    out.mkdir(parents=True, exist_ok=True)
    weights_st = work / "model.safetensors"
    weights_pt = work / "model.pth"

    model = TinyNet().eval()
    if weights_st.is_file():
        model.load_state_dict(load_file(str(weights_st)))
    elif weights_pt.is_file():
        state = torch.load(weights_pt, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    else:
        torch.save(model.state_dict(), weights_pt)

    dummy = torch.randn(1, 3, 64, 64)
    onnx_path = out / "model.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
