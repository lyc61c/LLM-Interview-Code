"""手写 padding / causal mask：本仓库统一 True 或 1 表示允许注意。

注意：该约定与 nn.MultiheadAttention 的布尔 mask 相反，但与
torch.nn.functional.scaled_dot_product_attention 的布尔 mask 一致。
所有函数返回可广播到 [batch, heads, query_len, key_len] 的布尔张量。
"""

import torch


def create_padding_mask(input_ids, pad_token_id=0):
    """由 [B, K] token id 构造 [B, 1, 1, K] 的 key padding mask。

    只屏蔽被关注的 padding key；需要屏蔽 padding query 时见
    create_attention_mask。input_ids 在缓存解码时应包含历史与当前 token。
    """
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, key_len]")
    return (input_ids != pad_token_id)[:, None, None, :]


def create_causal_mask(query_len, key_len=None, past_len=0, device=None):
    """允许 key_position <= past_len + query_position，返回 [1, 1, Q, K]。

    prefill：past_len=0，Q=K。解码：K=past_len+Q；例如 past_len=3、Q=2，
    两行分别允许 key 0..3 与 0..4。不能直接对矩形 [Q, K] 做 tril，
    否则新 query 将无法看到完整缓存。
    """
    if not isinstance(query_len, int) or query_len <= 0:
        raise ValueError("query_len must be a positive integer")
    if not isinstance(past_len, int) or past_len < 0:
        raise ValueError("past_len must be a non-negative integer")
    if key_len is None:
        key_len = past_len + query_len
    if not isinstance(key_len, int) or key_len <= 0:
        raise ValueError("key_len must be a positive integer")
    query_positions = torch.arange(query_len, device=device) + past_len
    key_positions = torch.arange(key_len, device=device)
    return (key_positions[None, :] <= query_positions[:, None])[None, None, :, :]


def create_attention_mask(input_ids, pad_token_id=0, query_len=None,
                          past_len=0, mask_query_padding=True):
    """合并 padding 与 causal mask，返回 [B, 1, Q, K]。

    input_ids=[B, K] 包含完整历史；当前 query 对应位置 past_len..K-1。
    默认同时屏蔽 padding query，因而它们会产生全屏蔽行；仓库的手写
    attention 明确定义这种行的权重、context 和最终输出均为零。
    mask_query_padding=False 时仅屏蔽 key，适用于只需标准 key mask 的场景。
    """
    padding_mask = create_padding_mask(input_ids, pad_token_id)
    if not isinstance(past_len, int) or past_len < 0:
        raise ValueError("past_len must be a non-negative integer")
    key_len = input_ids.shape[1]
    if query_len is None:
        query_len = key_len - past_len
    causal_mask = create_causal_mask(query_len, key_len, past_len, input_ids.device)
    if past_len + query_len != key_len:
        raise ValueError("input_ids must cover exactly past_len + query_len keys")
    mask = padding_mask & causal_mask
    if mask_query_padding:
        query_valid = input_ids[:, past_len:past_len + query_len] != pad_token_id
        mask = mask & query_valid[:, None, :, None]
    return mask
