"""逐张量 INT8 量化：q=round(x/scale)+zero_point，x̂=(q-zero_point)*scale。"""

import torch


def symmetric_quantize(x):
    """对称范围 [-127,127]；返回 (整数张量, scale, zero_point)。"""
    scale = x.abs().max().item() / 127
    if scale == 0:
        scale = 1.0
    values = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return values, scale, 0


def asymmetric_quantize(x):
    """非对称范围 [-128,127]，先将浮点范围扩展到包含零。"""
    minimum = min(x.min().item(), 0.0)
    maximum = max(x.max().item(), 0.0)
    scale = (maximum - minimum) / 255
    if scale == 0:
        scale = 1.0
    zp = max(-128, min(127, round(-128 - minimum / scale)))
    values = (torch.round(x / scale) + zp).clamp(-128, 127).to(torch.int8)
    return values, scale, zp


def dequantize(values, scale, zero_point=0):
    """重建与输入同形状的 FP32 Tensor，不会还原量化丢失的信息。"""
    return (values.float() - zero_point) * scale


if __name__ == "__main__":
    values = torch.tensor([-2., -.3, 0., 1.2, 3.])
    for quantize in (symmetric_quantize, asymmetric_quantize):
        quantized, scale, zero_point = quantize(values)
        print(quantize.__name__, quantized, "scale:", scale, "zero_point:", zero_point)
        print("reconstructed:", dequantize(quantized, scale, zero_point))
