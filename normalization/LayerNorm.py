"""
层归一化（Layer Normalization）

Transformer 中常用的归一化技术，在每个样本的特征维度上进行归一化。
与 BatchNorm 不同，LayerNorm 不依赖 batch 统计量，适用于可变长度序列。

参考论文: Layer Normalization
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    层归一化模块

    公式: LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta

    Args:
        model_dim: 归一化的特征维度
        eps: 数值稳定性常数，防止除零，默认 1e-5
    """

    def __init__(self, model_dim, eps=1e-5):
        super().__init__()
        if model_dim <= 0 or eps <= 0:
            raise ValueError("model_dim 与 eps 必须为正")
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(model_dim))   # 可学习的缩放参数
        self.beta = nn.Parameter(torch.zeros(model_dim))   # 可学习的偏移参数

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, model_dim]

        Returns:
            归一化后的张量 [batch_size, seq_len, model_dim]
        """
        if not x.is_floating_point() or x.ndim == 0 or x.shape[-1] != self.gamma.numel():
            raise ValueError("x 必须是最后一维为 model_dim 的浮点张量")
        # 半精度先在 FP32 中统计；FP64 保留精度，便于数值梯度验证。
        stats_x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x

        # 计算均值
        # mean: [batch_size, seq_len, 1]
        mean = stats_x.mean(-1, keepdim=True)

        # 计算方差
        # 【面试大坑】torch.var 默认是 unbiased=True（除以 N-1）
        # 但 LayerNorm 的定义通常是除以 N（unbiased=False）
        # var: [batch_size, seq_len, 1]
        var = stats_x.var(-1, keepdim=True, unbiased=False)

        # 归一化：零均值、单位方差
        # x_normalized: [batch_size, seq_len, model_dim]
        x_normalized = (stats_x - mean) * torch.rsqrt(var + self.eps)

        # 应用可学习的仿射变换
        return (x_normalized * self.gamma + self.beta).to(x.dtype)
