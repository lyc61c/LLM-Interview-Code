"""GSPO：先计算长度归一化的序列比率，再做序列级 clipping。"""

import torch


def gspo_loss(old_log_probs, new_log_probs, advantages, mask=None,
              epsilon_low=3e-4, epsilon_high=4e-4):
    """old/new_log_probs: [B,T]；advantages: [B]，每条回答一个优势。

    mask 为同形状的 0/1 或 bool 张量。s_i=exp(mean_t(log pi_new-log pi_old))，
    先按有效长度平均 log ratio，不能用平均 token ratio 或未经归一化的连乘。
    每条非空回答等权；全屏蔽返回可导的 0，old policy/advantage 内部 detach。
    假设有效输入和 exp 比率在输入精度内可表示，0<=epsilon_low<1、epsilon_high>=0。
    """
    # 步骤1: 计算回答长度，空回答不参与最终平均。
    valid = torch.ones_like(new_log_probs, dtype=torch.bool) if mask is None else mask.bool()
    lengths = valid.sum(dim=-1)
    valid_sequences = lengths > 0
    advantages = torch.where(valid_sequences, advantages.detach(), 0.0)

    # 步骤2: 先对 log ratio 做长度归一化，再取 exp。
    old = torch.where(valid, old_log_probs.detach(), 0.0)
    new = torch.where(valid, new_log_probs, 0.0)
    sequence_log_ratio = (new - old).sum(dim=-1) / lengths.clamp_min(1)
    sequence_ratio = sequence_log_ratio.exp()

    # 步骤3: 对整条回答的比率做非对称 clipping。
    clipped_ratio = sequence_ratio.clamp(1 - epsilon_low, 1 + epsilon_high)
    surrogate = torch.minimum(sequence_ratio * advantages, clipped_ratio * advantages)

    # 步骤4: 非空回答等权，区别于 DAPO 的有效 token 等权。
    return -surrogate.sum() / valid_sequences.sum().clamp_min(1)
