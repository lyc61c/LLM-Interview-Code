"""普通 FP32 输入下的注意力、mask 与 KV Cache 教学验证。"""

import pytest
import torch
import torch.nn.functional as F

from attention.AttentionMask import create_attention_mask, create_causal_mask, create_padding_mask
from attention.GroupQueryAttention import GroupQueryAttention
from attention.MultiHeadAttention import MultiHeadAttention
from attention.MultiHeadAttentionWithKVCache import MultiHeadAttentionWithKVCache
from attention.MultiLatentAttention import MultiLatentAttention
from attention.ScaledDotProductAttention import ScaledDotProductAttention


def test_scaled_dot_product_attention_matches_pytorch():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 4, 6)
    k = torch.randn(2, 3, 5, 6)
    v = torch.randn(2, 3, 5, 6)
    mask = create_causal_mask(4, 5, past_len=1)
    output, weights = ScaledDotProductAttention()(q, k, v, mask)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 3, 4))


def test_multi_head_attention_self_and_cross_attention():
    torch.manual_seed(1)
    model = MultiHeadAttention(16, 4)
    query = torch.randn(2, 3, 16)
    context = torch.randn(2, 5, 16)
    q = model.w_q(query).view(2, 3, 4, 4).transpose(1, 2)
    k = model.w_k(context).view(2, 5, 4, 4).transpose(1, 2)
    v = model.w_v(context).view(2, 5, 4, 4).transpose(1, 2)
    expected = F.scaled_dot_product_attention(q, k, v)
    expected = model.w_o(expected.transpose(1, 2).reshape(2, 3, 16))
    torch.testing.assert_close(model(query, context), expected)
    assert model(query, query).shape == query.shape


def test_group_query_attention_matches_shared_kv_reference():
    torch.manual_seed(2)
    model = GroupQueryAttention(16, 4, 2)
    x = torch.randn(2, 4, 16)
    q = model.w_q(x).view(2, 4, 4, 4).transpose(1, 2)
    k = model.w_k(x).view(2, 4, 2, 4).transpose(1, 2).repeat_interleave(2, dim=1)
    v = model.w_v(x).view(2, 4, 2, 4).transpose(1, 2).repeat_interleave(2, dim=1)
    expected = F.scaled_dot_product_attention(q, k, v)
    expected = model.w_o(expected.transpose(1, 2).reshape(2, 4, 16))
    torch.testing.assert_close(model(x), expected)


def test_multi_latent_attention_output_shape():
    x = torch.randn(2, 4, 16)
    model = MultiLatentAttention(16, 4, 4, 3, 2)
    assert model(x).shape == x.shape


def test_padding_and_causal_masks():
    ids = torch.tensor([[11, 12, 0, 13, 14]])
    padding = create_padding_mask(ids)
    assert padding[0, 0, 0].tolist() == [True, True, False, True, True]
    causal = create_causal_mask(2, past_len=3)
    assert causal[0, 0].tolist() == [[True, True, True, True, False], [True] * 5]
    combined = create_attention_mask(ids, past_len=3)
    assert combined[0, 0].tolist() == [
        [True, True, False, True, False], [True, True, False, True, True],
    ]


@pytest.mark.parametrize("use_rope", [False, True])
def test_cached_attention_matches_full_and_chunked_forward(use_rope):
    torch.manual_seed(3)
    model = MultiHeadAttentionWithKVCache(16, 4, use_rope=use_rope, max_seq_len=16).eval()
    x = torch.randn(2, 6, 16)
    full, full_cache = model(x)

    cache, outputs = None, []
    for index in range(x.size(1)):
        output, cache = model(x[:, index:index + 1], past_key_value=cache)
        outputs.append(output)
    torch.testing.assert_close(torch.cat(outputs, dim=1), full)
    torch.testing.assert_close(cache[0], full_cache[0])
    torch.testing.assert_close(cache[1], full_cache[1])

    first, cache = model(x[:, :2])
    rest, cache = model(x[:, 2:], past_key_value=cache)
    torch.testing.assert_close(torch.cat([first, rest], dim=1), full)
    assert cache[0].shape == (2, 4, 6, 4)


def test_cache_without_rope_matches_original_mha():
    torch.manual_seed(4)
    original = MultiHeadAttention(16, 4).eval()
    cached = MultiHeadAttentionWithKVCache(16, 4, use_rope=False).eval()
    cached.load_state_dict(original.state_dict())
    x = torch.randn(2, 4, 16)
    output, _ = cached(x)
    torch.testing.assert_close(output, original(x, x, mask=create_causal_mask(4)))


def test_padding_keys_do_not_affect_valid_token_outputs():
    torch.manual_seed(5)
    model = MultiHeadAttentionWithKVCache(16, 4).eval()
    ids = torch.tensor([[11, 12, 0, 13]])
    x = torch.randn(1, 4, 16)
    changed = x.clone()
    changed[ids == 0] = torch.randn_like(changed[ids == 0])
    mask = create_attention_mask(ids)
    output, _ = model(x, mask=mask)
    altered, _ = model(changed, mask=mask)
    torch.testing.assert_close(output[ids != 0], altered[ids != 0])
