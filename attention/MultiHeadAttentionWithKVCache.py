"""
带 KV Cache 的多头注意力

沿用原仓库 MHA 的计算步骤，只增加历史 K/V 拼接与解码时的位置偏移。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from position.RotaryEmbedding import RotaryEmbedding


class MultiHeadAttentionWithKVCache(nn.Module):
    """首轮 prefill 计算整个输入，后续 decode 只传入新 token。

    Args:
        model_dim: 模型隐藏维度
        num_heads: 注意力头数
        dropout_p: Dropout 概率，默认 0.0
        use_rope: 是否对 Q/K 应用 RoPE，默认 True
        max_seq_len: 原版 RoPE 频率表的固定容量
        theta: RoPE 频率基数，默认 10000.0
    """

    def __init__(self, model_dim, num_heads, dropout_p=0.0, use_rope=True,
                 max_seq_len=2048, theta=10000.0):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.w_q = nn.Linear(model_dim, model_dim)
        self.w_k = nn.Linear(model_dim, model_dim)
        self.w_v = nn.Linear(model_dim, model_dim)
        self.w_o = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout_p)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, theta) if use_rope else None

    def forward(self, x, past_key_value=None, mask=None):
        """x: 本轮输入 [batch_size, query_len, model_dim]。

        past_key_value: 历史 (K, V)，shape 为 [batch_size, num_heads, past_len, head_dim]。
        mask: 0 表示屏蔽，可广播到 [batch_size, num_heads, query_len, key_len]。
        返回 output 和更新后的 (K, V)。使用 RoPE 时总长度不超过 max_seq_len。
        """
        batch_size, query_len, _ = x.shape
        past_len = 0 if past_key_value is None else past_key_value[0].size(2)

        # ========== 线性投影与分头 ==========
        # 仅对本轮 token 投影：[B,Q,D] -> [B,H,Q,d]。
        q = self.w_q(x).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)

        # ========== RoPE 位置编码 ==========
        # 原版 RoPE.forward 从位置 0 开始；这里按历史长度切同一张频率表。
        if self.rope is not None:
            cos = self.rope.cos[past_len:past_len + query_len].view(1, 1, query_len, self.head_dim)
            sin = self.rope.sin[past_len:past_len + query_len].view(1, 1, query_len, self.head_dim)
            q1, q2 = torch.chunk(q, 2, dim=-1)
            k1, k2 = torch.chunk(k, 2, dim=-1)
            q = q * cos + torch.cat((-q2, q1), dim=-1) * sin
            k = k * cos + torch.cat((-k2, k1), dim=-1) * sin

        # ========== 追加历史缓存 ==========
        # 缓存中的 K 已旋转，直接拼接，不重复旋转历史 K。
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        # ========== 缩放点积注意力 ==========
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 新 Query 的绝对位置是 past_len + q，不能关注更靠后的 Key。
        query_positions = torch.arange(query_len, device=x.device) + past_len
        key_positions = torch.arange(k.size(2), device=x.device)
        causal_mask = key_positions[None, :] <= query_positions[:, None]
        scores = scores.masked_fill(causal_mask == 0, -1e9)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)

        # ========== 合并多头与输出投影 ==========
        context = context.transpose(1, 2).contiguous()
        output = context.view(batch_size, query_len, self.model_dim)
        output = self.w_o(output)

        return output, (k, v)
