"""
Export RealESRGAN_x4plus_anime_6B (.pth) -> out/model.onnx (FP32, static NCHW RGB [0,1]).

Architecture matches upstream:
  RRDBNet(3, 3, 64, num_block=6, num_grow_ch=32, scale=4)
  https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py

RRDB blocks adapted from BasicSR (Apache-2.0):
  https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py

Env:
  WEIGHTS_PATH     default: models/RealESRGAN_x4plus_anime_6B.pth
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
from torch.nn import init

# --- arch_util (subset) -----------------------------------------------------


def _pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


@torch.no_grad()
def _default_init_weights(module_list: nn.Module | list, scale: float = 1) -> None:
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()


def _make_layer(block: type[nn.Module], n: int, **kwarg) -> nn.Sequential:
    return nn.Sequential(*[block(**kwarg) for _ in range(n)])


# --- rrdbnet (subset) -------------------------------------------------------


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        _default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Same contract as basicsr.archs.rrdbnet_arch.RRDBNet for ESRGAN / Real-ESRGAN."""

    def __init__(
        self,
        num_in_ch: int,
        num_out_ch: int,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = _make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = _pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = _pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class _ClampWrapper(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x).clamp(0.0, 1.0)


def main() -> None:
    work = Path(__file__).resolve().parent
    weights = work / os.environ.get(
        "WEIGHTS_PATH", "models/RealESRGAN_x4plus_anime_6B.pth"
    )
    if not weights.is_file():
        raise SystemExit(f"missing weights: {weights}")

    h = int(os.environ.get("EXPORT_INPUT_H", "64"))
    w = int(os.environ.get("EXPORT_INPUT_W", "64"))
    opset = int(os.environ.get("EXPORT_OPSET", "17"))

    net = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=6,
        num_grow_ch=32,
        scale=4,
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
