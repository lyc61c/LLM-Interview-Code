"""GSPO 损失：长度归一化的序列比率、序列级截断、回答平均。"""

import torch


def gspo_loss(old_log_probs, new_log_probs, advantages, mask=None,
              epsilon_low=3e-4, epsilon_high=4e-4):
    """log_probs: [batch_size, seq_len]，advantages: [batch_size]。

    mask: 同形状的 0/1 张量，每条回答至少包含一个有效 token。
    """
    if mask is None:
        mask = torch.ones_like(new_log_probs)

    # 步骤1: 对每条回答的 log ratio 按有效长度平均，再取指数。
    log_ratio = new_log_probs - old_log_probs
    sequence_log_ratio = (log_ratio * mask).sum(dim=-1) / mask.sum(dim=-1)
    sequence_ratio = torch.exp(sequence_log_ratio)

    # 步骤2: 对序列比率截断，而不是对各 token 比率截断。
    clipped_ratio = torch.clamp(sequence_ratio, 1 - epsilon_low, 1 + epsilon_high)
    surrogate1 = sequence_ratio * advantages
    surrogate2 = clipped_ratio * advantages

    # 步骤3: 每条回答等权平均。
    return -torch.min(surrogate1, surrogate2).mean()
