"""GSPO 序列级重要性比率与 clipping（原始 GSPO，非 GSPO-token）。

参考：https://arxiv.org/html/2507.18071v2 Eq.5/7；默认非对称 clipping
来自原论文 §5.1，不能直接照用 token-level PPO 的 0.2 范围。
"""

import torch

from ._utils import clipped_policy_surrogate, make_mask, safe_log_ratio, validate_clips, validate_pair


def gspo_loss(old_log_probs, new_log_probs, advantages, mask=None,
              epsilon_low=3e-4, epsilon_high=4e-4):
    """old/new_log_probs: [B,T]，advantages: [B]（每条序列一个优势）。

    s_i=exp(sum_t mask_it*(log pi_new-log pi_old) / length_i)，
    L=-mean_i min(s_i*A_i, clip(s_i,1-eps_low,1+eps_high)*A_i)。

    先平均 log ratio 再 exp，不能用平均 token ratio 或未归一化连乘。
    每条非空回答等权；全 padding 序列不参与平均。rollout 的 old policy
    与 advantages 内部 detach；全部为空返回可反传的 0。
    在 log 域执行等价裁剪，未被裁剪的目标真正溢出时抛 ValueError。
    """
    validate_pair(old_log_probs, new_log_probs)
    validate_clips(epsilon_low, epsilon_high)
    if new_log_probs.ndim != 2 or advantages.shape != (new_log_probs.shape[0],):
        raise ValueError("log_probs 必须为 [B,T]，advantages 必须为 [B]")
    if advantages.device != new_log_probs.device or not advantages.is_floating_point():
        raise ValueError("advantages 必须是同设备的浮点张量")
    valid = make_mask(new_log_probs, mask)
    lengths = valid.sum(dim=-1)
    valid_sequences = lengths > 0
    log_ratio = safe_log_ratio(old_log_probs, new_log_probs, valid)
    sequence_log_ratio = log_ratio.sum(dim=-1) / lengths.clamp_min(1)
    advantage = torch.where(valid_sequences, advantages.detach(), 0.0)
    if not torch.isfinite(advantage).all():
        raise ValueError("非空序列的 advantages 必须有限")
    surrogate = clipped_policy_surrogate(sequence_log_ratio, advantage, epsilon_low, epsilon_high)
    return -surrogate.sum() / valid_sequences.sum().clamp_min(1)
