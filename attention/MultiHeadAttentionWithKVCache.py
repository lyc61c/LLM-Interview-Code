"""带 RoPE 和 KV Cache 的自回归 MHA：prefill / 单 token / 分块 decode。

核心面试点：只投影新 token；缓存已旋转的 K 与原始 V；RoPE 的 offset 和
causal mask 的 query 起始位置都等于历史缓存长度。缓存 shape 为 [B,H,T,D]。
教学版使用 torch.cat 更新缓存，方便理解；生产推理通常用预分配或分页缓存。
"""

import os
import sys

import torch

# 与原仓库模块一致，既可包导入也可直接运行此文件。
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention.AttentionMask import create_causal_mask
from attention.MultiHeadAttention import MultiHeadAttention
from attention.ScaledDotProductAttention import (
    _prepare_mask, scaled_dot_product_attention, zero_fully_masked_queries,
)
from position.RotaryEmbedding import RotaryEmbedding


class MultiHeadAttentionWithKVCache(MultiHeadAttention):
    """保留 MHA 的 w_q/w_k/w_v/w_o 参数命名，便于复制权重和对照验证。

    Args:
        model_dim, num_heads, dropout_p: 与 MultiHeadAttention 一致。
        use_rope: 是否对 Q/K 使用 RoPE；关闭可直接对照原 MHA。
        max_seq_len, theta: RoPE 初始表容量与频率基数；超过容量会自动扩展。
    """

    def __init__(self, model_dim, num_heads, dropout_p=0.0, use_rope=True,
                 max_seq_len=2048, theta=10000.0):
        super().__init__(model_dim, num_heads, dropout_p)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, theta) if use_rope else None

    def forward(self, x, past_key_value=None, mask=None):
        """返回 (output, present_key_value)，不会修改传入的 past_key_value。

        x: [B, Q, model_dim]，仅包含本次新增 token。
        past_key_value: None 或 (past_k, past_v)，二者均为 [B,H,past_len,D]。
        mask: True/1 允许注意，shape 可广播到 [B,H,Q,past_len+Q]；可用
              create_attention_mask(完整 token ids, past_len=...) 屏蔽 padding。
              用户 mask 总与自动生成的偏移 causal mask 取交集。
        output: [B,Q,model_dim]；present: 包含历史与新 token 的 (K,V)。

        前提：输入形状和模型维度匹配，历史缓存来自同一个模型；
        缓存与新投影 K/V 的 batch、头数、头维度、dtype 和 device 一致。

        等价性对照应使用 eval() 或 dropout_p=0；训练时不自动 detach 缓存，
        因而梯度可沿历史 K/V 回传。推理调用者通常使用 torch.no_grad()。
        """
        batch, query_len, _ = x.shape
        # 步骤1：历史序列有多长，新 token 的绝对位置就从哪里开始。
        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.shape[-2]

        # 步骤2：仅投影本轮新增 token，[B,Q,M] -> [B,Q,H,D]。
        q = self.w_q(x).view(batch, query_len, self.num_heads, self.head_dim)
        k = self.w_k(x).view(batch, query_len, self.num_heads, self.head_dim)
        v = self.w_v(x).view(batch, query_len, self.num_heads, self.head_dim)
        # 步骤3：按绝对位置旋转新 Q/K，再转成 [B,H,Q,D] 缓存布局。
        if self.rope is not None:
            q, k = self.rope(q, k, offset=past_len)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # 步骤4：在历史 K/V 之后追加本轮 K/V。
        if past_key_value is not None:
            # 缓存已旋转的历史 K；沿序列维拼接新 K/V，不重复投影历史 token。
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        present_key_value = (k, v)

        # 步骤5：按偏移构造 causal mask，并与用户的 padding 等 mask 合并。
        key_len = past_len + query_len
        causal = create_causal_mask(query_len, key_len, past_len, x.device)
        if mask is not None:
            causal = causal & _prepare_mask(mask, (batch, self.num_heads, query_len, key_len), x.device)
        context, _ = scaled_dot_product_attention(q, k, v, causal, self.dropout)
        # 步骤6：合头、输出投影；全屏蔽 query 的最终输出仍设为零。
        context = context.transpose(1, 2).contiguous().view(batch, query_len, self.model_dim)
        output = self.w_o(context)
        output = zero_fully_masked_queries(output, causal, self.num_heads, key_len)
        return output, present_key_value


if __name__ == "__main__":
    torch.manual_seed(0)
    model = MultiHeadAttentionWithKVCache(32, 4).eval()
    tokens = torch.randn(2, 6, 32)
    with torch.no_grad():
        full, _ = model(tokens)
        prefix, cache = model(tokens[:, :4])
        suffix, cache = model(tokens[:, 4:], past_key_value=cache)
    print("分块 decode 与完整 causal forward 一致:",
          torch.allclose(full, torch.cat((prefix, suffix), dim=1), atol=1e-6))
    print("KV Cache shape:", cache[0].shape)
