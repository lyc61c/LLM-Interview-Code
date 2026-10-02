"""使用批内负样本的 InfoNCE：第 i 个 query 与第 i 个 key 为正样本。

参考 CPC（2018）https://arxiv.org/abs/1807.03748
仅适用于每行一个正样本的成对输入；相同语义的其他行会成为假负样本。
"""

import math
import torch
from torch import nn

from .EntropyLoss import cross_entropy_loss


def info_nce_loss(queries, keys, temperature=0.1, symmetric=False):
    """输入 queries/keys: [B, D]；返回 scalar。

    s_ij = cosine(q_i,k_j)/temperature，L_qk = mean_i[-log softmax(s_i)_i]。
    symmetric=True 时返回 (L_qk+L_kq)/2。B=1 没有负样本，损失为 0。
    范数、温度缩放或损失超出计算 dtype 范围时抛 ValueError。
    """
    if queries.ndim != 2 or queries.shape != keys.shape or queries.shape[0] == 0 or queries.shape[1] == 0:
        raise ValueError("queries、keys 必须是相同的非空 [B, D] 张量")
    if queries.device != keys.device or not queries.is_floating_point() or not keys.is_floating_point():
        raise ValueError("queries、keys 必须是同设备的浮点张量")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature 必须为正且有限")
    if not torch.isfinite(queries).all() or not torch.isfinite(keys).all():
        raise ValueError("特征必须有限")
    if queries.dtype in (torch.float16, torch.bfloat16):
        queries = queries.float()
    if keys.dtype in (torch.float16, torch.bfloat16):
        keys = keys.float()
    if queries.dtype != keys.dtype:
        raise ValueError("queries、keys 必须具有相同 dtype")
    if temperature < 1 / torch.finfo(queries.dtype).max:
        raise ValueError("temperature 的倒数超出计算 dtype 可表示范围")
    # autocast 可能把 matmul 再降为半精度，导致 temperature 缩放时溢出。
    # 归一化与相似度/CE 均保留上面确定的 float32 或 float64 计算精度。
    with torch.autocast(device_type=queries.device.type, enabled=False):
        query_norms = queries.norm(dim=-1, keepdim=True)
        key_norms = keys.norm(dim=-1, keepdim=True)
        if not torch.isfinite(query_norms).all() or not torch.isfinite(key_norms).all():
            raise ValueError("特征范数超出计算 dtype 可表示范围")
        # clamp_min 允许零向量输入：其归一化结果仍为零，不会出现除零。
        queries = queries / query_norms.clamp_min(1e-12)
        keys = keys / key_norms.clamp_min(1e-12)
        similarities = queries @ keys.T / temperature
        if not torch.isfinite(similarities).all():
            raise ValueError("temperature 缩放后的相似度超出计算 dtype 可表示范围")
        targets = torch.arange(queries.shape[0], device=queries.device)
        loss = cross_entropy_loss(similarities, targets)
        if symmetric:
            loss = (loss + cross_entropy_loss(similarities.T, targets)) / 2
        if not torch.isfinite(loss):
            raise ValueError("InfoNCE 损失超出计算 dtype 可表示范围")
    return loss


class InfoNCELoss(nn.Module):
    """函数版的 nn.Module 封装，可放入训练代码。"""

    def __init__(self, temperature=0.1, symmetric=False):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature 必须为正且有限")
        self.temperature, self.symmetric = temperature, symmetric

    def forward(self, queries, keys):
        return info_nce_loss(queries, keys, self.temperature, self.symmetric)
