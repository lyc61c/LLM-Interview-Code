"""DAPO 损失：token 级重要性比率、非对称截断、有效 token 平均。"""

import torch


def dapo_loss(old_log_probs, new_log_probs, advantages, mask=None,
              epsilon_low=0.2, epsilon_high=0.28):
    """log_probs: [batch_size, seq_len]，advantages: [batch_size]。

    mask: 同形状的 0/1 张量，标记有效 token，至少包含一个有效位置。
    """
    # 步骤1: 计算每个 token 的重要性采样比率。
    ratio = torch.exp(new_log_probs - old_log_probs)
    advantages = advantages.unsqueeze(-1)

    # 步骤2: 非对称截断，并取两个代理目标的较小值。
    clipped_ratio = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high)
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    token_loss = -torch.min(surrogate1, surrogate2)

    # 步骤3: 对全部有效 token 求平均。
    if mask is None:
        mask = torch.ones_like(token_loss)
    return (token_loss * mask).sum() / mask.sum()
