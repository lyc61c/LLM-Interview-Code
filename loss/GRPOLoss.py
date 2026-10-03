"""GRPO：使用同一 prompt 的组内相对奖励估计优势，无需 Critic。"""

import torch

from .KLDivergence import sampled_kl_divergence


def compute_grpo_advantages(rewards, eps=1e-8):
    """rewards: 浮点 [B,G]，G>=1。A=(R-mean(R))/(std(R)+eps)，eps>0。"""
    if rewards.dtype in (torch.float16, torch.bfloat16):
        rewards = rewards.float()

    # 步骤1: 按组计算均值和总体标准差，G=1 不会出现无偏 std 的 NaN。
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)

    # 步骤2: 标准化奖励；单样本组和同奖励组都得到零优势。
    return (rewards - mean) / (std + eps)


def grpo_loss(old_log_probs, new_log_probs, advantages, clip_epsilon=0.2,
              beta=0.01, ref_kl=None):
    """输入同形状 log 概率、advantages 与可选 ref_kl，按元素平均。

    L=-mean(min(r*A,clip(r)*A))+beta*mean(KL)。old policy 和 advantage
    视为固定 rollout 数据。这里没有 response mask，变长回答需另行处理归约。
    """
    # 步骤1: 计算重要性比率，梯度只流向新策略。
    ratio = torch.exp(new_log_probs - old_log_probs.detach())
    advantages = advantages.detach()

    # 步骤2: 比较未截断与截断的目标。
    clipped_ratio = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages)

    # 步骤3: 可选参考策略 KL 惩罚。
    if ref_kl is not None:
        policy_loss = policy_loss + beta * ref_kl
    return policy_loss.mean()


def compute_kl_penalty(log_probs, ref_log_probs, mask=None, reduction="mean"):
    """样本来自 P 时，用 k3 估计 KL(P||Q)：E_P[exp(log Q-log P)-1-(log Q-log P)]。

    E_P[Q/P-1]=0，所以加入零均值控制变量不改变期望；不是由 Taylor 展开保证无偏。
    mask 与 reduction 的约定同 sampled_kl_divergence。
    """
    return sampled_kl_divergence(log_probs, ref_log_probs, "k3", mask, reduction)
