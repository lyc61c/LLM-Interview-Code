"""用 PyTorch 独立基准验证注意力、mask、缓存及反向传播。"""

import copy

import pytest
import torch
import torch.nn.functional as F

from attention.AttentionMask import (
    create_attention_mask, create_causal_mask, create_padding_mask,
)
from attention.GroupQueryAttention import GroupQueryAttention
from attention.MultiHeadAttention import MultiHeadAttention
from attention.MultiHeadAttentionWithKVCache import MultiHeadAttentionWithKVCache
from attention.MultiLatentAttention import MultiLatentAttention
from attention.ScaledDotProductAttention import ScaledDotProductAttention

torch.set_num_threads(1)


@pytest.mark.parametrize("dtype,atol,rtol", [
    (torch.float64, 1e-12, 1e-12), (torch.float32, 1e-6, 1e-5),
    (torch.float16, 2e-3, 2e-3), (torch.bfloat16, 2e-2, 2e-2),
])
def test_sdpa_matches_pytorch_with_rectangular_and_fully_masked_rows(dtype, atol, rtol):
    torch.manual_seed(7)
    q = torch.randn(2, 3, 4, 6, dtype=dtype)
    k = torch.randn(2, 3, 5, 6, dtype=dtype)
    v = torch.randn(2, 3, 5, 8, dtype=dtype)  # V 维度无需等于 Q/K。
    mask = torch.rand(2, 1, 4, 5) > 0.3
    mask[0, :, 2] = False
    actual, weights = ScaledDotProductAttention()(q, k, v, mask)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    assert actual.dtype == weights.dtype == dtype
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[0, :, 2]) == 0
    assert torch.count_nonzero(weights[0, :, 2]) == 0
    assert torch.count_nonzero(weights.masked_select(~mask.expand_as(weights))) == 0


def test_sdpa_gradients_match_pytorch_including_masked_rows():
    torch.manual_seed(3)
    tensors = [torch.randn(2, 2, t, d, dtype=torch.float64, requires_grad=True)
               for t, d in [(3, 4), (5, 4), (5, 6)]]
    mask = create_causal_mask(3, 5, past_len=2).expand(2, 1, 3, 5).clone()
    mask[0, :, 1] = False
    actual, _ = ScaledDotProductAttention()(*tensors, mask)
    expected = F.scaled_dot_product_attention(*tensors, attn_mask=mask)
    actual_grads = torch.autograd.grad(actual.square().sum(), tensors)
    expected_grads = torch.autograd.grad(expected.square().sum(), tensors)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-11, rtol=1e-11)


def test_mask_builders_padding_query_masking_and_cache_offset():
    ids = torch.tensor([[0, 11, 12, 0, 13], [21, 22, 23, 24, 25]])
    padding = create_padding_mask(ids)
    assert padding.shape == (2, 1, 1, 5) and padding.dtype == torch.bool
    causal = create_causal_mask(2, past_len=3)
    assert causal[0, 0].tolist() == [[True, True, True, True, False], [True] * 5]
    mask = create_attention_mask(ids, past_len=3)
    assert mask.shape == (2, 1, 2, 5)
    assert not mask[0, 0, 0].any()  # 当前第一个 query 为 padding。
    assert mask[0, 0, 1].tolist() == [False, True, True, False, True]
    key_only = create_attention_mask(ids, past_len=3, mask_query_padding=False)
    assert key_only[0, 0, 0].tolist() == [False, True, True, False, False]


def test_mha_self_and_cross_attention_match_pytorch():
    torch.manual_seed(1)
    model = MultiHeadAttention(24, 3).double().eval()
    query = torch.randn(2, 4, 24, dtype=torch.float64)
    context = torch.randn(2, 6, 24, dtype=torch.float64)
    mask = create_causal_mask(4, 6, past_len=2)
    q, k, v = [proj(inp).reshape(2, -1, 3, 8).transpose(1, 2)
               for proj, inp in [(model.w_q, query), (model.w_k, context), (model.w_v, context)]]
    reference = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    reference = model.w_o(reference.transpose(1, 2).reshape(2, 4, 24))
    torch.testing.assert_close(model(query, context, mask), reference, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(model(query), model(query, query))


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_gqa_matches_independent_repeated_kv_reference(kv_heads):
    torch.manual_seed(2)
    model = GroupQueryAttention(16, 4, kv_heads).double().eval()
    x = torch.randn(2, 5, 16, dtype=torch.float64)
    q = model.w_q(x).reshape(2, 5, 4, 4).transpose(1, 2)
    k = model.w_k(x).reshape(2, 5, kv_heads, 4).transpose(1, 2).repeat_interleave(4 // kv_heads, dim=1)
    v = model.w_v(x).reshape(2, 5, kv_heads, 4).transpose(1, 2).repeat_interleave(4 // kv_heads, dim=1)
    mask = create_causal_mask(5)
    reference = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    reference = model.w_o(reference.transpose(1, 2).reshape(2, 5, 16))
    torch.testing.assert_close(model(x, mask), reference, atol=1e-12, rtol=1e-12)


def test_mla_matches_pytorch_with_independent_rope_formula():
    torch.manual_seed(9)
    model = MultiLatentAttention(16, 4, 4, 3, 2).double().eval()
    x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
    qc, qr = model.q_up_proj(model.q_down_proj(x)).reshape(2, 5, 4, 6).split([4, 2], -1)
    kc, kr, v = model.kv_up_proj(model.kv_down_proj(x)).reshape(2, 5, 4, 10).split([4, 2, 4], -1)
    # rope_dim=2 时唯一角频率为 1，独立计算旋转，验证内容/RoPE 切分和缩放。
    positions = torch.arange(5, dtype=torch.float64).reshape(1, 5, 1, 1)
    def rotate(t):
        return t * positions.cos() + torch.cat((-t[..., 1:], t[..., :1]), -1) * positions.sin()
    q = torch.cat((qc, rotate(qr)), -1).transpose(1, 2)
    k = torch.cat((kc, rotate(kr)), -1).transpose(1, 2)
    mask = create_causal_mask(5)
    reference = F.scaled_dot_product_attention(q, k, v.transpose(1, 2), attn_mask=mask)
    reference = model.o_proj(reference.transpose(1, 2).reshape(2, 5, 16))
    actual = model(x, mask)
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("factory", [
    lambda: MultiHeadAttention(16, 4), lambda: GroupQueryAttention(16, 4, 2),
    lambda: MultiLatentAttention(16, 4, 4, 3, 2),
])
def test_fully_masked_queries_zero_after_projection_and_have_finite_gradients(factory):
    torch.manual_seed(4)
    model = factory()
    x = torch.randn(2, 3, 16, requires_grad=True)
    mask = torch.ones(2, 1, 3, 3, dtype=torch.bool)
    mask[0, :, 1] = False
    actual = model(x, mask=mask)
    assert torch.count_nonzero(actual[0, 1]) == 0
    actual.square().sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("use_rope", [False, True])
@pytest.mark.parametrize("chunks", [[1] * 7, [3, 1, 3], [2, 5]])
def test_cached_attention_prefill_and_chunk_decode_equal_full_forward(use_rope, chunks):
    torch.manual_seed(5)
    model = MultiHeadAttentionWithKVCache(24, 3, use_rope=use_rope, max_seq_len=2).double().eval()
    x = torch.randn(2, 7, 24, dtype=torch.float64)
    full, full_cache = model(x)
    cache, pieces, start = None, [], 0
    for size in chunks:
        old_cache = None if cache is None else tuple(t.clone() for t in cache)
        piece, present = model(x[:, start:start + size], past_key_value=cache)
        if cache is not None:
            for old, current in zip(old_cache, cache):
                torch.testing.assert_close(old, current)  # 不修改传入历史缓存。
        cache = present
        pieces.append(piece)
        start += size
        assert cache[0].shape == (2, 3, start, 8)
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=1e-12, rtol=1e-12)
    for incremental, all_at_once in zip(cache, full_cache):
        torch.testing.assert_close(incremental, all_at_once, atol=1e-12, rtol=1e-12)


def test_cache_without_rope_matches_original_mha_weights():
    torch.manual_seed(6)
    original = MultiHeadAttention(16, 4).eval()
    cached = MultiHeadAttentionWithKVCache(16, 4, use_rope=False).eval()
    cached.load_state_dict(original.state_dict())
    x = torch.randn(2, 6, 16)
    actual, _ = cached(x)
    torch.testing.assert_close(actual, original(x, mask=create_causal_mask(6)))


def test_cached_padding_queries_zero_and_padding_keys_do_not_affect_valid_outputs():
    torch.manual_seed(8)
    model = MultiHeadAttentionWithKVCache(16, 4).double().eval()
    ids = torch.tensor([[0, 11, 12, 0, 13], [21, 22, 23, 24, 0]])
    x = torch.randn(2, 5, 16, dtype=torch.float64)
    altered = x.clone()
    altered[ids == 0] = 1000 * torch.randn_like(altered[ids == 0])
    mask = create_attention_mask(ids)
    full, _ = model(x, mask=mask)
    changed, _ = model(altered, mask=mask)
    torch.testing.assert_close(full, changed, atol=1e-12, rtol=1e-12)
    assert torch.count_nonzero(full[ids == 0]) == 0
    prefix, cache = model(x[:, :3], mask=create_attention_mask(ids[:, :3]))
    suffix, _ = model(x[:, 3:], past_key_value=cache, mask=create_attention_mask(ids, past_len=3))
    torch.testing.assert_close(torch.cat((prefix, suffix), dim=1), full, atol=1e-12, rtol=1e-12)


def test_cached_attention_gradients_match_full_forward():
    torch.manual_seed(10)
    full_model = MultiHeadAttentionWithKVCache(16, 4, max_seq_len=2).double().eval()
    chunk_model = copy.deepcopy(full_model)
    full_x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
    chunk_x = full_x.detach().clone().requires_grad_()
    full, _ = full_model(full_x)
    prefix, cache = chunk_model(chunk_x[:, :2])
    suffix, _ = chunk_model(chunk_x[:, 2:], past_key_value=cache)
    full.square().sum().backward()
    torch.cat((prefix, suffix), dim=1).square().sum().backward()
    torch.testing.assert_close(chunk_x.grad, full_x.grad, atol=1e-11, rtol=1e-11)
    for actual, expected in zip(chunk_model.parameters(), full_model.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-11, rtol=1e-11)


def test_sdpa_eval_disables_dropout_and_numeric_mask_equivalence():
    torch.manual_seed(11)
    q = torch.randn(2, 3, 4, 6)
    mask = create_causal_mask(4)
    model = ScaledDotProductAttention(dropout_p=0.5).eval()
    actual, _ = model(q, q, q, mask)
    repeated, _ = model(q, q, q, mask.float())
    torch.testing.assert_close(actual, repeated)
    torch.testing.assert_close(actual, F.scaled_dot_product_attention(q, q, q, attn_mask=mask))


def test_attention_accumulation_stays_float32_inside_autocast():
    # fp16 matmul 的 1e4 * 1e4 会溢出；显式禁止内部 autocast 才能保持 fp32 累加。
    q = torch.full((1, 1, 2, 4), 1e4, dtype=torch.float16)
    v = torch.randn_like(q)
    with torch.autocast("cpu", dtype=torch.float16):
        output, _ = ScaledDotProductAttention()(q, q, v)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, v.float().mean(dim=-2, keepdim=True).expand_as(v).half())


def test_cached_attention_with_autocast_projection_dtypes():
    torch.manual_seed(14)
    model = MultiHeadAttentionWithKVCache(16, 4).eval()
    x = torch.randn(2, 5, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        full, _ = model(x)
        prefix, cache = model(x[:, :3])
        suffix, _ = model(x[:, 3:], past_key_value=cache)
    assert cache[0].dtype == torch.bfloat16
    torch.testing.assert_close(torch.cat((prefix, suffix), 1), full, atol=2e-2, rtol=2e-2)


def test_core_head_divisibility_and_rope_parity():
    with pytest.raises(AssertionError):
        GroupQueryAttention(16, 4, 3)
    with pytest.raises(AssertionError):
        MultiLatentAttention(16, 4, 4, 3, 3)
