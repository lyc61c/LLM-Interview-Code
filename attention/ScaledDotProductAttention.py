"""
缩放点积注意力（Scaled Dot-Product Attention）

Transformer 注意力机制的核心计算模块。
公式: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _prepare_mask(mask, shape, device):
    """2D [Q,K]、3D [B,Q,K] 或 4D mask；数值 mask 仅接受 0/1。"""
    if not isinstance(mask, torch.Tensor) or mask.ndim not in (2, 3, 4):
        raise ValueError("mask must be a tensor with 2, 3 or 4 dimensions")
    if mask.dtype != torch.bool and not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must contain only 0/1; additive masks are not supported")
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    try:
        return torch.broadcast_to(mask.to(device=device, dtype=torch.bool), shape)
    except RuntimeError as exc:
        raise ValueError(f"mask shape {tuple(mask.shape)} cannot broadcast to {shape}") from exc


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """手写 SDPA，支持 V 的维度不同于 Q/K；全屏蔽行定义为零。

    fp16/bf16 下用 fp32 计算点积、softmax 和加权和，最后恢复输入 dtype。
    在 softmax 前将全屏蔽行替换为零，随后将屏蔽权重清零，避免
    softmax([-inf, ..., -inf]) 产生 NaN，也避免使用 -1e9 导致半精度溢出。
    """
    if any(not isinstance(t, torch.Tensor) or t.ndim != 4 for t in (q, k, v)):
        raise ValueError("q, k, v must have shape [batch, heads, seq_len, dim]")
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        raise ValueError("q, k, v must have the same batch and head counts")
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError("Q/K dimensions and K/V sequence lengths must match")
    if min(q.shape[-2:]) <= 0 or min(k.shape[-2:]) <= 0 or v.shape[-1] <= 0:
        raise ValueError("sequence lengths and head dimensions must be positive")
    if not q.is_floating_point() or q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, v must have the same floating dtype")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, v must be on the same device")

    work_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    # 外层可能启用了 autocast；显式 .float() 不足以防止 matmul 又被降精度。
    with torch.autocast(device_type=q.device.type, enabled=False):
        scores = torch.matmul(q.to(work_dtype), k.to(work_dtype).transpose(-2, -1)) / math.sqrt(q.shape[-1])
        allowed = None if mask is None else _prepare_mask(mask, scores.shape, scores.device)
        if allowed is not None:
            scores = scores.masked_fill(~allowed, float("-inf"))
            scores = scores.masked_fill(~allowed.any(dim=-1, keepdim=True), 0.0)
        weights = F.softmax(scores, dim=-1)
        if allowed is not None:
            weights = weights.masked_fill(~allowed, 0.0)
        if dropout is not None:
            weights = dropout(weights)
        output = torch.matmul(weights, v.to(work_dtype))
    return output.to(q.dtype), weights.to(q.dtype)


def zero_fully_masked_queries(output, mask, num_heads, key_len):
    """输出投影有 bias 时也保持全屏蔽 query 为零，不让 bias 泄漏到 padding。"""
    if mask is None:
        return output
    batch, query_len = output.shape[:2]
    allowed = _prepare_mask(mask, (batch, num_heads, query_len, key_len), output.device)
    query_valid = allowed.any(dim=-1).any(dim=1)
    return output.masked_fill(~query_valid[:, :, None], 0.0)


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力模块

    计算查询和键的点积，除以缩放因子后应用 softmax 得到注意力权重，
    最后用注意力权重对值进行加权求和。

    Args:
        dropout_p: Dropout 概率，默认 0.0
    """

    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, q, k, v, mask=None):
        """
        前向传播

        Args:
            q: 查询张量 [batch_size, num_heads, seq_len_q, head_dim]
            k: 键张量 [batch_size, num_heads, seq_len_k, head_dim]
            v: 值张量 [batch_size, num_heads, seq_len_k, value_dim]
            mask: True/1 允许注意，False/0 屏蔽；支持 [Q,K]、[B,Q,K] 或
                  可广播到 [B,H,Q,K] 的 4D 张量（不接受 additive mask）

        Returns:
            output: 注意力输出 [batch_size, num_heads, seq_len_q, value_dim]
            attn_weights: 注意力权重 [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        return scaled_dot_product_attention(q, k, v, mask, self.dropout)


# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟输入: batch_size=2, num_heads=4, seq_len=8, head_dim=64
    batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 64
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # 构造一个 Causal Mask (下三角矩阵)，模拟 GPT 生成过程
    # shape: (seq_len, seq_len), 上三角为 0, 下三角为 1
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)

    attention_layer = ScaledDotProductAttention()
    output, weights = attention_layer(q, k, v, mask=causal_mask)

    print(f"Output shape: {output.shape}")  # Should be (2, 4, 8, 64)
    print(f"Weights shape: {weights.shape}")  # Should be (2, 4, 8, 8)

    # 验证 Mask 是否生效：查看第一个样本第一个头的第一行
    # 理论上只有第1个位置有值，后面全是0
    print("\nCheck Causal Masking (Row 0 should only attend to Col 0):")
    print(weights[0, 0, 0, :])
