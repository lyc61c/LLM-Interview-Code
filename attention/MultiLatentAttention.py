"""
多头潜在注意力（Multi-Head Latent Attention, MLA）

DeepSeek 提出的高效注意力机制，通过低秩压缩 KV 缓存来减少显存占用。
核心思想：将 KV 压缩到潜在空间，推理时只需缓存潜在向量。

教学说明：本文件实现的是 MLA 的简化版本，重点演示 Q/KV 低秩投影和
RoPE 注意力的主干流程，未实现原论文中的共享解耦 RoPE Key、投影吸收
以及增量 KV Cache 接口。

参考论文: DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from position.RotaryEmbedding import RotaryEmbedding
from attention.ScaledDotProductAttention import scaled_dot_product_attention, zero_fully_masked_queries


class MultiLatentAttention(nn.Module):
    """
    简化版多头潜在注意力模块

    通过低秩投影将 Q 和 KV 压缩到潜在空间，演示压缩表示的张量变换。
    同时结合 RoPE 位置编码保持位置感知能力。

    注意：这里为了突出核心张量变换，为每个头生成 RoPE Key。DeepSeek-V2
    原始 MLA 使用共享的解耦 RoPE Key，并包含适合增量推理的缓存设计。

    Args:
        model_dim: 模型隐藏维度
        num_heads: 注意力头数
        head_dim: 每个注意力头的维度
        latent_dim: 潜在空间维度（压缩后的维度）
        rope_dim: RoPE 旋转位置编码维度
        dropout_p: Dropout 概率，默认 0.0
    """

    def __init__(self, model_dim, num_heads, head_dim, latent_dim, rope_dim, dropout_p=0.0):
        super().__init__()
        if any(not isinstance(n, int) or n <= 0 for n in (model_dim, num_heads, head_dim, latent_dim, rope_dim)):
            raise ValueError("all dimensions must be positive integers")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if rope_dim % 2 != 0:
            raise ValueError("rope_dim must be even")

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.latent_dim = latent_dim
        self.rope_dim = rope_dim

        # 1. KV 压缩投影：model_dim -> latent_dim（下采样）-> num_heads * (head_dim + rope_dim + head_dim)（上采样）
        self.kv_down_proj = nn.Linear(model_dim, latent_dim, bias=False)
        self.kv_up_proj = nn.Linear(latent_dim, num_heads * (head_dim + rope_dim + head_dim), bias=False)

        # 2. Q 压缩投影：model_dim -> latent_dim（下采样）-> num_heads * (head_dim + rope_dim)（上采样）
        self.q_down_proj = nn.Linear(model_dim, latent_dim, bias=False)
        self.q_up_proj = nn.Linear(latent_dim, num_heads * (head_dim + rope_dim), bias=False)

        # 3. 输出投影
        self.o_proj = nn.Linear(num_heads * head_dim, model_dim, bias=False)

        self.dropout = nn.Dropout(dropout_p)

        # 4. RoPE 旋转位置编码
        self.rope = RotaryEmbedding(head_dim=rope_dim)

    def forward(self, x, mask=None):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, model_dim]
            mask: True/1 允许注意，False/0 屏蔽；可广播到 [B,H,T,T]。
                  全屏蔽 query 的输出为零。默认不自动添加 causal mask。

        Returns:
            输出张量 [batch_size, seq_len, model_dim]
        """
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[-1] != self.model_dim or x.shape[1] == 0:
            raise ValueError("x must have shape [batch, seq_len, model_dim]")
        batch_size, seq_len, _ = x.size()

        # ========== KV 投影 ==========
        # [batch_size, seq_len, model_dim] -> [batch_size, seq_len, latent_dim]
        kv_latent = self.kv_down_proj(x)
        # [batch_size, seq_len, latent_dim] -> [batch_size, seq_len, num_heads * (head_dim + rope_dim + head_dim)]
        kv_full = self.kv_up_proj(kv_latent)
        # [batch_size, seq_len, num_heads * (head_dim + rope_dim + head_dim)] -> [batch_size, seq_len, num_heads, head_dim + rope_dim + head_dim]
        kv_full = kv_full.view(batch_size, seq_len, self.num_heads, -1)

        # 分离内容部分和 RoPE 部分
        # k_content: [batch_size, seq_len, num_heads, head_dim]
        # k_rope: [batch_size, seq_len, num_heads, rope_dim]
        # v_content: [batch_size, seq_len, num_heads, head_dim]
        k_content, k_rope, v_content = torch.split(kv_full, [self.head_dim, self.rope_dim, self.head_dim], dim=-1)

        # ========== Q 投影 ==========
        # [batch_size, seq_len, model_dim] -> [batch_size, seq_len, latent_dim]
        q_latent = self.q_down_proj(x)
        # [batch_size, seq_len, latent_dim] -> [batch_size, seq_len, num_heads * (head_dim + rope_dim)]
        q_full = self.q_up_proj(q_latent)
        # [batch_size, seq_len, num_heads * (head_dim + rope_dim)] -> [batch_size, seq_len, num_heads, head_dim + rope_dim]
        q_full = q_full.view(batch_size, seq_len, self.num_heads, self.head_dim + self.rope_dim)

        # 分离内容部分和 RoPE 部分
        q_content, q_rope = torch.split(q_full, [self.head_dim, self.rope_dim], dim=-1)

        # ========== 应用 RoPE ==========
        # 对 q_rope 和 k_rope 应用旋转位置编码
        q_rope, k_rope = self.rope(q_rope, k_rope)

        # ========== 合并内容和 RoPE 部分 ==========
        # [batch_size, seq_len, num_heads, head_dim + rope_dim]
        q = torch.cat([q_content, q_rope], dim=-1)
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim + rope_dim]

        # [batch_size, seq_len, num_heads, head_dim + rope_dim]
        k = torch.cat([k_content, k_rope], dim=-1)
        k = k.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim + rope_dim]

        # [batch_size, num_heads, seq_len, head_dim]
        v = v_content.transpose(1, 2)

        # ========== 缩放点积注意力 ==========
        # Q/K 的维度为 head_dim + rope_dim，V 的维度仍为 head_dim。
        # 缩放因子必须按 Q/K 的总维度计算，不能只用内容维度。
        context, _ = scaled_dot_product_attention(q, k, v, mask, self.dropout)

        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads * head_dim]
        output = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)

        # [batch_size, seq_len, num_heads * head_dim] -> [batch_size, seq_len, model_dim]
        output = self.o_proj(output)

        return zero_fully_masked_queries(output, mask, self.num_heads, seq_len)
