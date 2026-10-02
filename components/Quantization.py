"""逐张量 INT8 量化：q=clamp(round(x/scale)+zero_point)，x̂=(q-zp)·scale。

仅用于展示量化数学：不实现校准、逐通道量化、QAT 或 INT8 矩阵乘 kernel。
量化张量真正存为 torch.int8；参数与反量化在 FP32 中计算。
round 使用 PyTorch 的四舍六入五成双；量化是不可微的离散操作。
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Int8Quantized:
    values: torch.Tensor       # 输入形状，torch.int8
    scale: torch.Tensor        # 标量，torch.float32，严格为正
    zero_point: torch.Tensor   # 标量，torch.int64，整数零点


def _fp32_input(x):
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise TypeError("量化输入必须是浮点 Tensor")
    x = x.detach().float()
    if x.numel() == 0 or not torch.isfinite(x).all():
        raise ValueError("量化输入必须非空，且能表示为有限 FP32 值")
    return x


def symmetric_quantize(x):
    """对称 INT8，q∈[-127,127]，zp=0，scale=max(abs(x))/127。

    不使用 -128，使正负范围完全对称。全零张量取 scale=1。
    非零极小范围用 FP32 tiny 下界避免 scale 下溢成 0。
    """
    x = _fp32_input(x)
    magnitude = x.abs().max()
    tiny = torch.finfo(torch.float32).tiny
    scale = torch.where(magnitude == 0, torch.ones_like(magnitude), (magnitude / 127).clamp_min(tiny))
    zp = torch.zeros((), dtype=torch.int64, device=x.device)
    values = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return Int8Quantized(values, scale, zp)


def asymmetric_quantize(x):
    """非对称 INT8，q∈[-128,127]，采用整数 zero_point。

    min/max 范围先扩展到包含 0，再计算 scale=(max-min)/255，
    zp=clamp(round(-128-min/scale), -128,127)。这样正/负常数也能量化，
    并且 x=0 总能精确反量化为 0。全零取 scale=1、zp=-128。
    """
    x = _fp32_input(x)
    minimum = x.min().clamp_max(0)
    maximum = x.max().clamp_min(0)
    # 先除再减，避免 max-min 对接近 FP32 极限的有限输入溢出。
    raw_scale = maximum / 255 - minimum / 255
    scale = torch.where((maximum == 0) & (minimum == 0), torch.ones_like(raw_scale),
                        raw_scale.clamp_min(torch.finfo(torch.float32).tiny))
    zp = torch.round(-128 - minimum / scale).clamp(-128, 127).to(torch.int64)
    values = (torch.round(x / scale) + zp).clamp(-128, 127).to(torch.int8)
    return Int8Quantized(values, scale, zp)


def dequantize(quantized):
    """重建与输入同形状的 FP32 Tensor，不会还原量化丢失的信息。

    接近 FP32 最大值时，量化格点舍入后可能略微超出可表示范围，
    因此把重建值饱和到 FP32 有限端点，避免有限输入反量化产生 inf。
    """
    result = (quantized.values.float() - quantized.zero_point.float()) * quantized.scale
    limit = torch.finfo(torch.float32).max
    return result.clamp(-limit, limit)


if __name__ == "__main__":
    values = torch.tensor([-2., -.3, 0., 1.2, 3.])
    for quantize in (symmetric_quantize, asymmetric_quantize):
        result = quantize(values)
        print(quantize.__name__, result.values, "scale:", result.scale.item(),
              "zero_point:", result.zero_point.item(), "reconstructed:", dequantize(result))
