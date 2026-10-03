"""
旋转位置编码（Rotary Position Embedding, RoPE）

一种将位置信息编码到注意力机制中的方法，通过旋转向量来表示相对位置。
使注意力点积包含相对位置信息。

参考论文: RoFormer: Enhanced Transformer with Rotary Position Embedding
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码模块

    通过将查询和键向量按照位置进行旋转，使注意力分数包含相对位置信息。
    公式: f(x, m) = x * cos(m*θ) + rotate_half(x) * sin(m*θ)

    Args:
        head_dim: 正偶数，旋转编码的维度（通常是 attention head 的维度）
        max_seq_len: 频率表的初始容量，默认 2048，超出后自动扩展
        theta: 正的旋转频率基数，默认 10000.0
    """

    def __init__(self, head_dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE head_dim must be even"

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 预计算 cos 和 sin 值，避免重复计算
        cos, sin = self.precompute_freqs(head_dim, max_seq_len, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # .double() 仅转换已有 fp32 值，不能恢复精度；记录表的原始计算精度。
        self._table_dtype = torch.float32

    def precompute_freqs(self, head_dim, max_seq_len, theta, device=None, dtype=torch.float32):
        """
        预计算旋转位置编码的频率

        Args:
            head_dim: 编码维度
            max_seq_len: 最大序列长度
            theta: 旋转基数

        Returns:
            cos: 余弦值 [max_seq_len, head_dim]
            sin: 正弦值 [max_seq_len, head_dim]
        """
        # 计算逆频率: 1 / (theta^(2i/d))
        # inv_freqs: [head_dim/2]
        inv_freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))

        # 位置索引
        # t: [max_seq_len]
        t = torch.arange(max_seq_len, device=inv_freqs.device, dtype=dtype)

        # 计算角度矩阵: outer(t, inv_freqs)
        # angles: [max_seq_len, head_dim/2]
        angles = torch.outer(t, inv_freqs)

        # 将角度复制一份以匹配完整维度
        # angles: [max_seq_len, head_dim]
        angles = torch.cat((angles, angles), dim=-1)

        return angles.cos(), angles.sin()

    def forward(self, xq, xk, offset=0):
        """
        对查询和键应用旋转位置编码

        Args:
            xq: 查询张量 [batch_size, seq_len, num_heads, head_dim]
            xk: 键张量 [batch_size, seq_len, num_kv_heads, head_dim]
            offset: 非负整数，当前 chunk 的绝对起始位置，解码时等于历史长度。
                    历史 K 已旋转，只对新 Q/K 用 offset 旋转，不能再次旋转历史 K。

        Returns:
            xq_rotated: 旋转后的查询 [batch_size, seq_len, num_heads, head_dim]
            xk_rotated: 旋转后的键 [batch_size, seq_len, num_kv_heads, head_dim]
        """
        # 步骤1：确定本轮 token 的绝对位置区间 [offset, offset + seq_len)。
        seq_len = xq.size(1)
        end = offset + seq_len
        # 角度至少以 fp32 计算；double 输入保留 double 精度。超出初始容量时扩容，
        # 并在 module.half() 后重新生成 fp32 表，避免低精度位置频率误差。
        table_dtype = torch.float64 if xq.dtype == torch.float64 else torch.float32
        if (end > self.cos.shape[0] or self.cos.device != xq.device
                or self.cos.dtype != table_dtype or self._table_dtype != table_dtype):
            capacity = max(end, self.cos.shape[0] * 2) if end > self.cos.shape[0] else self.cos.shape[0]
            self.cos, self.sin = self.precompute_freqs(self.head_dim, capacity, self.theta, xq.device, table_dtype)
            self.max_seq_len = capacity
            self._table_dtype = table_dtype

        # 步骤2：取当前位置的 cos / sin，再补上 batch 和 head 的广播维度。
        # cos, sin: [1, seq_len, 1, head_dim]
        cos = self.cos[offset:end].to(dtype=xq.dtype).view(1, seq_len, 1, self.head_dim)
        sin = self.sin[offset:end].to(dtype=xq.dtype).view(1, seq_len, 1, self.head_dim)

        def rotate_half(x):
            """
            将张量分成两半并旋转

            Args:
                x: 输入张量 [..., head_dim]

            Returns:
                旋转后的张量 [..., head_dim]
            """
            # x1, x2: [..., head_dim/2]
            x1, x2 = torch.chunk(x, 2, dim=-1)
            # 拼接为 [-x2, x1]: [..., head_dim]
            return torch.cat((-x2, x1), dim=-1)

        # 步骤3：把前后半区组成特征对，应用二维旋转。
        # 公式: x * cos + rotate_half(x) * sin
        #
        # 详细推导:
        # 设 x = [x1, x2]，则 rotate_half(x) = [-x2, x1]
        # x * cos = [x1*cos, x2*cos]
        # rotate_half(x) * sin = [-x2*sin, x1*sin]
        # 结果 = [x1*cos - x2*sin, x2*cos + x1*sin]
        #
        # 等价于旋转矩阵: [cos, -sin; sin, cos] @ [x1; x2] = [x1'; x2']

        # RoPE 是纯旋转，保持输入 dtype；外层 autocast 不应改写旋转运算。
        # 特别是 CPU bf16 autocast + 显式 fp16 输入时，cat 可能不支持该混合路径。
        with torch.autocast(device_type=xq.device.type, enabled=False):
            xq_rotated = (xq * cos) + (rotate_half(xq) * sin)
            xk_rotated = (xk * cos) + (rotate_half(xk) * sin)

        return xq_rotated, xk_rotated
