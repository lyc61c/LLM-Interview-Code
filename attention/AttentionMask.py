"""
注意力掩码：Padding Mask 与 Causal Mask

与原仓库一致，1 / True 表示允许注意，0 / False 表示屏蔽。
"""

import torch


def create_padding_mask(input_ids, pad_token_id=0):
    """由 [batch_size, seq_len] 构造 [batch_size, 1, 1, seq_len]。

    Padding Mask 屏蔽作为 Key 的补齐位置。
    """
    return (input_ids != pad_token_id)[:, None, None, :]


def create_causal_mask(query_len, key_len=None, past_len=0, device=None):
    """构造 [1, 1, query_len, key_len] 的因果掩码。

    已缓存 past_len 个 token 时，第 q 个新 Query 的绝对位置为 past_len+q。
    允许注意的条件：key_position <= past_len + query_position。
    """
    if key_len is None:
        key_len = past_len + query_len

    # Query 从历史缓存之后开始，Key 从整个序列的开头开始。
    query_positions = torch.arange(query_len, device=device) + past_len
    key_positions = torch.arange(key_len, device=device)
    mask = key_positions[None, :] <= query_positions[:, None]

    return mask[None, None, :, :]


def create_attention_mask(input_ids, pad_token_id=0, past_len=0):
    """合并 Padding 与 Causal Mask，返回 [batch_size, 1, query_len, key_len]。

    input_ids 包含历史与本轮 token；query_len = key_len - past_len。
    用逻辑与合并：既不是 padding key，也不是未来 key，才允许注意。
    """
    key_len = input_ids.size(1)
    query_len = key_len - past_len

    padding_mask = create_padding_mask(input_ids, pad_token_id)
    causal_mask = create_causal_mask(query_len, key_len, past_len, input_ids.device)

    return padding_mask & causal_mask
