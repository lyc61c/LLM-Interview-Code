"""DAPO 的策略损失核心：Clip-Higher + Token-level Policy Gradient Loss。

参考原论文 Eq.12：https://arxiv.org/html/2503.14476v1
这里不实现动态采样、奖励计算、overlong shaping 或分布式训练系统。
"""

import torch

from ._utils import clipped_policy_surrogate, make_mask, safe_log_ratio, validate_clips, validate_pair


def dapo_loss(old_log_probs, new_log_probs, advantages, mask=None,
              epsilon_low=0.2, epsilon_high=0.28):
    """old/new_log_probs: [B, T]；advantage: [B]、[B,1] 或 [B,T]。

    r_it=exp(log pi_new - log pi_old)，
    L=-sum_it mask_it * min(r_it*A_it, clip(r_it,1-eps_low,1+eps_high)*A_it)
      / sum_it mask_it。

    所有有效 token 等权，长回答占更大权重；mask=1 表示 response token。
    old policy 和 advantage 视为固定 rollout 数据，内部 detach。
    全部屏蔽返回可反传的 0；有效位置的 log_probs 必须有限。
    在 log 域执行等价裁剪，未被裁剪的目标真正溢出时抛 ValueError。
    """
    validate_pair(old_log_probs, new_log_probs)
    validate_clips(epsilon_low, epsilon_high)
    if new_log_probs.ndim != 2:
        raise ValueError("log_probs 必须为 [B, T]")
    if advantages.device != new_log_probs.device or not advantages.is_floating_point():
        raise ValueError("advantages 必须是同设备的浮点张量")
    if advantages.shape == (new_log_probs.shape[0],):
        advantages = advantages.unsqueeze(-1)
    if advantages.shape not in ((new_log_probs.shape[0], 1), new_log_probs.shape):
        raise ValueError("advantages 必须为 [B]、[B,1] 或 [B,T]")
    valid = make_mask(new_log_probs, mask)
    log_ratio = safe_log_ratio(old_log_probs, new_log_probs, valid)
    advantage = torch.where(valid, advantages.detach(), 0.0)
    if not torch.isfinite(advantage).all():
        raise ValueError("有效位置的 advantages 必须有限")
    surrogate = clipped_policy_surrogate(log_ratio, advantage, epsilon_low, epsilon_high)
    return -surrogate.sum() / valid.sum().clamp_min(1)
