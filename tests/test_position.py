"""RoPE 保范数、相对位置性质、offset、扩容及 dtype 验证。"""

import pytest
import torch

from position.RotaryEmbedding import RotaryEmbedding


def test_rope_preserves_norm_and_chunk_offsets_match_full_sequence():
    torch.manual_seed(12)
    q = torch.randn(2, 7, 3, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 7, 1, 8, dtype=torch.float64, requires_grad=True)
    rope = RotaryEmbedding(8, max_seq_len=2)
    full_q, full_k = rope(q, k)
    chunk_q, chunk_k = rope(q[:, 4:], k[:, 4:], offset=4)
    torch.testing.assert_close(chunk_q, full_q[:, 4:], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(chunk_k, full_k[:, 4:], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(full_q.square().sum(-1), q.square().sum(-1), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(full_k.square().sum(-1), k.square().sum(-1), atol=1e-12, rtol=1e-12)
    full_q.sum().backward()
    assert torch.isfinite(q.grad).all()
    assert rope.cos.shape[0] >= 7


def test_rope_dot_product_is_invariant_to_common_position_shift():
    torch.manual_seed(13)
    q = torch.randn(2, 1, 3, 8, dtype=torch.float64)
    k = torch.randn_like(q)
    rope = RotaryEmbedding(8, max_seq_len=4)
    q_at_2, _ = rope(q, q, offset=2)
    _, k_at_5 = rope(k, k, offset=5)
    q_at_12, _ = rope(q, q, offset=12)
    _, k_at_15 = rope(k, k, offset=15)
    torch.testing.assert_close((q_at_2 * k_at_5).sum(-1), (q_at_12 * k_at_15).sum(-1), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_rope_preserves_input_dtype_after_module_cast(dtype):
    rope = RotaryEmbedding(8, max_seq_len=2).to(dtype=dtype)
    q = torch.randn(2, 3, 4, 8, dtype=dtype)
    qr, kr = rope(q, q, offset=3)
    assert qr.dtype == kr.dtype == dtype
    assert torch.isfinite(qr).all()
    assert rope.cos.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)


def test_rope_invalid_dimension_offset_and_input_layout():
    with pytest.raises(ValueError):
        RotaryEmbedding(3)
    with pytest.raises(ValueError):
        RotaryEmbedding(4, max_seq_len=0)
    with pytest.raises(ValueError):
        RotaryEmbedding(4, theta=0)
    rope = RotaryEmbedding(4)
    q = torch.randn(2, 3, 4, 4)
    with pytest.raises(ValueError, match="offset"):
        rope(q, q, offset=-1)
    with pytest.raises(ValueError, match="head_dim"):
        rope(q, q[..., :2])


def test_rope_explicit_float16_input_inside_bfloat16_autocast():
    torch.manual_seed(15)
    q = torch.randn(2, 3, 4, 8, dtype=torch.float16)
    k = torch.randn(2, 3, 2, 8, dtype=torch.float16)
    reference = RotaryEmbedding(8, max_seq_len=2)(q, k, offset=7)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = RotaryEmbedding(8, max_seq_len=2)(q, k, offset=7)
    for output, expected in zip(actual, reference):
        assert output.dtype == torch.float16
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
