"""
Export realesr-animevideov3 (.pth) -> out/model.onnx (FP32, static NCHW RGB [0,1]).

Architecture matches upstream:
  SRVGGNetCompact(3, 3, 64, num_conv=16, upscale=4, act_type='prelu')
  https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py

Network structure from:
  https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/srvgg_arch.py

Env:
  WEIGHTS_PATH     default: models/realesr-animevideov3.pth
  EXPORT_INPUT_H   default: 64
  EXPORT_INPUT_W   default: 64
  EXPORT_OPSET     default: 17
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class SRVGGNetCompact(nn.Module):
    """Compact VGG-style SR net (same layout as upstream for state_dict compatibility)."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
        act_type: str = "prelu",
    ) -> None:
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        if act_type == "relu":
            activation: nn.Module = nn.ReLU(inplace=True)
        elif act_type == "prelu":
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == "leakyrelu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            raise ValueError(act_type)
        self.body.append(activation)

        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            if act_type == "relu":
                activation = nn.ReLU(inplace=True)
            elif act_type == "prelu":
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == "leakyrelu":
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=float(self.upscale), mode="nearest")
        return out + base


class _ClampWrapper(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x).clamp(0.0, 1.0)


def main() -> None:
    work = Path(__file__).resolve().parent
    weights = work / os.environ.get("WEIGHTS_PATH", "models/realesr-animevideov3.pth")
    if not weights.is_file():
        raise SystemExit(f"missing weights: {weights}")

    h = int(os.environ.get("EXPORT_INPUT_H", "64"))
    w = int(os.environ.get("EXPORT_INPUT_W", "64"))
    opset = int(os.environ.get("EXPORT_OPSET", "17"))

    net = SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=16,
        upscale=4,
        act_type="prelu",
    )
    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise SystemExit("unexpected checkpoint format")
    if "params_ema" in ckpt:
        state = ckpt["params_ema"]
    elif "params" in ckpt:
        state = ckpt["params"]
    else:
        state = ckpt
    net.load_state_dict(state, strict=True)
    model = _ClampWrapper(net.eval().float())

    dummy = torch.rand(1, 3, h, w, dtype=torch.float32)
    out_dir = work / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_onnx = out_dir / "model.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(out_onnx),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamo=False,
    )
    print(
        f"wrote {out_onnx} (RGB 0-1, NCHW 1x3x{h}x{w} -> 1x3x{h * 4}x{w * 4}, opset {opset})"
    )


if __name__ == "__main__":
    main()
