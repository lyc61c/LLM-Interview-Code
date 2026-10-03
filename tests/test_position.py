"""原仓库 RoPE：普通 FP32 输入的二维旋转与形状验证。"""

import torch

from position.RotaryEmbedding import RotaryEmbedding


def test_rope_preserves_shape_and_vector_norm():
    torch.manual_seed(0)
    q = torch.randn(2, 5, 3, 8)
    k = torch.randn_like(q)
    rope = RotaryEmbedding(8, max_seq_len=16)
    rotated_q, rotated_k = rope(q, k)
    assert rotated_q.shape == q.shape
    assert rotated_k.shape == k.shape
    torch.testing.assert_close(rotated_q.square().sum(-1), q.square().sum(-1))
    torch.testing.assert_close(rotated_k.square().sum(-1), k.square().sum(-1))
    torch.testing.assert_close(rotated_q[:, 0], q[:, 0])


def test_rope_matches_two_dimensional_rotation():
    q = torch.tensor([[[[1., 0.]], [[1., 0.]], [[1., 0.]]]])
    rope = RotaryEmbedding(2, max_seq_len=8)
    rotated_q, _ = rope(q, q)
    positions = torch.arange(3).float()
    expected = torch.stack([positions.cos(), positions.sin()], dim=-1).view(1, 3, 1, 2)
    torch.testing.assert_close(rotated_q, expected)
