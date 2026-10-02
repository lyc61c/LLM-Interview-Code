"""
多头注意力（Multi-Head Attention）

Transformer 的核心组件，通过并行运行多个注意力头来捕捉不同子空间的特征。
每个头独立计算注意力，最后将结果拼接并通过线性层融合。
"""

import torch
import torch.nn as nn
try:
    from .ScaledDotProductAttention import scaled_dot_product_attention, zero_fully_masked_queries
except ImportError:  # 兼容 python attention/MultiHeadAttention.py
    from ScaledDotProductAttention import scaled_dot_product_attention, zero_fully_masked_queries


class MultiHeadAttention(nn.Module):
    """
    多头注意力模块

    支持自注意力（Self-Attention）和交叉注意力（Cross-Attention）：
    - 自注意力：Q = K = V = x_query
    - 交叉注意力：Q = x_query, K = V = x_context

    Args:
        model_dim: 模型隐藏维度
        num_heads: 注意力头数
        dropout_p: Dropout 概率，默认 0.0
    """

    def __init__(self, model_dim, num_heads, dropout_p=0.0):
        super().__init__()

        if not isinstance(model_dim, int) or not isinstance(num_heads, int) or model_dim <= 0 or num_heads <= 0:
            raise ValueError("model_dim and num_heads must be positive integers")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads  # 每个头的维度

        # Q, K, V 投影层
        self.w_q = nn.Linear(model_dim, model_dim)
        self.w_k = nn.Linear(model_dim, model_dim)
        self.w_v = nn.Linear(model_dim, model_dim)

        # 输出投影层
        self.w_o = nn.Linear(model_dim, model_dim)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x_query, x_context=None, mask=None):
        """
        前向传播

        Args:
            x_query: 查询输入 [batch_size, seq_len_q, model_dim]
            x_context: 上下文输入（用于生成 K 和 V）[batch_size, seq_len_k, model_dim]
                       如果为 None，则使用 x_query（自注意力）
            mask: True/1 允许注意，False/0 屏蔽；可广播到 [B,H,Q,K]。
                  全屏蔽 query 的输出为零。默认不自动添加 causal mask。

        Returns:
            output: 注意力输出 [batch_size, seq_len_q, model_dim]
        """
        if not isinstance(x_query, torch.Tensor) or x_query.ndim != 3 or x_query.shape[-1] != self.model_dim or x_query.shape[1] == 0:
            raise ValueError("x_query must have shape [batch, query_len, model_dim]")
        if x_context is not None:
            if not isinstance(x_context, torch.Tensor) or x_context.ndim != 3 or x_context.shape[-1] != self.model_dim or x_context.shape[1] == 0:
                raise ValueError("x_context must have shape [batch, key_len, model_dim]")
            if x_query.shape[0] != x_context.shape[0]:
                raise ValueError("x_query and x_context must have the same batch size")
            if x_query.dtype != x_context.dtype or x_query.device != x_context.device:
                raise ValueError("x_query and x_context must have the same dtype and device")
        batch_size = x_query.size(0)

        # ========== 线性投影 ==========
        # 自注意力: q = k = v = x_query
        # 交叉注意力: q = x_query, k = v = x_context
        q = self.w_q(x_query)

        if x_context is not None:
            k = self.w_k(x_context)
            v = self.w_v(x_context)
        else:
            k = self.w_k(x_query)
            v = self.w_v(x_query)

        # ========== 分头处理 ==========
        # [batch_size, seq_len, model_dim] -> [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, q.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, k.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, v.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        # ========== 缩放点积注意力 ==========
        # softmax(QK^T / sqrt(d)) V；共享实现处理半精度及全屏蔽行。
        context, _ = scaled_dot_product_attention(q, k, v, mask, self.dropout)

        # ========== 合并多头 ==========
        # [batch_size, num_heads, seq_len_q, head_dim] -> [batch_size, seq_len_q, num_heads, head_dim] -> [batch_size, seq_len_q, model_dim]
        context = context.transpose(1, 2)
        context = context.contiguous()
        output = context.view(batch_size, x_query.shape[1], self.model_dim)

        # 输出投影
        output = self.w_o(output)

        return zero_fully_masked_queries(output, mask, self.num_heads, k.shape[-2])


if __name__ == "__main__":
    # batch_size=2, seq_len=10, model_dim=64, num_heads=8
    x = torch.randn(2, 10, 64)
    mha = MultiHeadAttention(model_dim=64, num_heads=8)
    out = mha(x, x)  # Self-Attention: x_query=x, x_context=x
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")  # 应该还是 (2, 10, 64)
