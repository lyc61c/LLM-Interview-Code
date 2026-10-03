"""DAPO 策略损失核心：非对称 clipping + 全有效 token 归一化。

仅展示损失计算；动态采样、奖励计算和超长回答处理属于训练流程。
"""

import torch


def dapo_loss(old_log_probs, new_log_probs, advantages, mask=None,
              epsilon_low=0.2, epsilon_high=0.28):
    """old/new_log_probs: [B,T]；advantages: [B]、[B,1] 或 [B,T]。

    mask 为同形状的 0/1 或 bool 张量，1 表示有效 response token。
    L=-sum(mask*min(r*A,clip(r)*A))/sum(mask)，r=exp(log pi_new-log pi_old)。
    rollout 的 old policy 与 advantage 不参与反传，全屏蔽返回可导的 0。
    假设有效输入和 exp 比率在输入精度内可表示，0<=epsilon_low<1、epsilon_high>=0。
    """
    # 步骤1: 准备 mask 与 token 优势；每条回答的优势沿 token 维广播。
    valid = torch.ones_like(new_log_probs, dtype=torch.bool) if mask is None else mask.bool()
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1)
    advantages = torch.where(valid, advantages.detach(), 0.0)

    # 步骤2: 计算 token 重要性比率；padding 在指数计算前置零。
    old = torch.where(valid, old_log_probs.detach(), 0.0)
    new = torch.where(valid, new_log_probs, 0.0)
    ratio = torch.exp(new - old)

    # 步骤3: 非对称截断，取未截断和截断目标中较小的值。
    clipped_ratio = ratio.clamp(1 - epsilon_low, 1 + epsilon_high)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)

    # 步骤4: 所有有效 token 等权；长回答贡献更多项。
    return -surrogate.sum() / valid.sum().clamp_min(1)
