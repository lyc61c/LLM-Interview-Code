"""采样策略损失共用的形状、mask 与归约规则。"""

import math
import torch


def validate_pair(old_log_probs, new_log_probs):
    if old_log_probs.shape != new_log_probs.shape or old_log_probs.device != new_log_probs.device:
        raise ValueError("两个 log_probs 必须形状相同且位于同一设备")
    if not old_log_probs.is_floating_point() or not new_log_probs.is_floating_point():
        raise ValueError("log_probs 必须是浮点张量")


def make_mask(values, mask):
    if mask is None:
        return torch.ones_like(values, dtype=torch.bool)
    if mask.shape != values.shape or mask.device != values.device:
        raise ValueError("mask 必须与 log_probs 形状、设备相同")
    if mask.is_complex():
        raise ValueError("mask 必须是实数 0/1 或 bool 张量，不能是复数")
    if mask.dtype != torch.bool and not ((mask == 0) | (mask == 1)).all():
        raise ValueError("mask 只能包含 0/1（1 表示有效位置）")
    return mask.bool()


def safe_log_ratio(old_log_probs, new_log_probs, valid):
    # 先清理 padding，再 exp；仅在 exp 后乘 0 无法消除 Inf/NaN。
    old = torch.where(valid, old_log_probs.detach(), 0.0)
    new = torch.where(valid, new_log_probs, 0.0)
    if not torch.isfinite(old).all() or not torch.isfinite(new).all():
        raise ValueError("有效 token 的 log_probs 必须有限")
    if new.dtype in (torch.float16, torch.bfloat16):
        new, old = new.float(), old.float()
    return new - old


def validate_clips(epsilon_low, epsilon_high):
    if not math.isfinite(epsilon_low) or not 0 <= epsilon_low < 1:
        raise ValueError("epsilon_low 必须满足 0 <= epsilon_low < 1")
    if not math.isfinite(epsilon_high) or epsilon_high < 0:
        raise ValueError("epsilon_high 必须非负且有限")


def clipped_policy_surrogate(log_ratio, advantages, epsilon_low, epsilon_high):
    """与 min(r*A,clip(r)*A) 等价，先按优势符号在 log 域截断再 exp。

    A>0 只受上界约束，A<0 只受下界约束。先 exp 大比率再 minimum 会让
    已裁剪的零梯度乘上 Inf，反传变 NaN；零优势也必须先清理比率。
    对 A<0 的巨大比率，上界并不限制目标，真实溢出时明确拒绝计算。
    """
    upper = math.log1p(epsilon_high)
    lower = math.log1p(-epsilon_low)
    effective_log_ratio = torch.where(
        advantages > 0, log_ratio.clamp_max(upper),
        torch.where(advantages < 0, log_ratio.clamp_min(lower), 0.0),
    )
    ratio = effective_log_ratio.exp()
    surrogate = ratio * advantages
    if not torch.isfinite(ratio).all() or not torch.isfinite(surrogate).all():
        raise ValueError("未裁剪的策略目标超出 dtype 可表示范围；请缩小策略更新或使用更高精度")
    return surrogate


def reduce_masked(values, valid, reduction):
    masked = torch.where(valid, values, 0.0)
    if reduction == "none":
        return masked
    if reduction == "sum":
        return masked.sum()
    if reduction == "mean":
        return masked.sum() / valid.sum().clamp_min(1)
    raise ValueError("reduction 必须是 none、sum 或 mean")


def causal_lm_loss(logits, labels, ignore_index=-100):
    """右移标签的 next-token CE；按有效标签数归一化，全忽略为可导的 0。"""
    from torch.nn import functional as F

    if logits.ndim != 3 or labels.shape != logits.shape[:2] or logits.shape[-1] == 0:
        raise ValueError("logits 必须为 [B,T,V]，labels 必须为 [B,T]，V>0")
    if not logits.is_floating_point() or labels.dtype != torch.long:
        raise ValueError("logits 必须为浮点张量，labels 必须为 torch.long")
    if logits.device != labels.device:
        raise ValueError("logits、labels 必须位于同一设备")
    shifted_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    shifted_labels = labels[:, 1:].reshape(-1)
    # 忽略行必须在 softmax 之前清理，不能仅依赖 CE 的 ignore_index。
    shifted_logits = torch.where((shifted_labels != ignore_index).unsqueeze(-1),
                                  shifted_logits, 0.0)
    if logits.dtype in (torch.float16, torch.bfloat16):
        shifted_logits = shifted_logits.float()
    loss_sum = F.cross_entropy(shifted_logits, shifted_labels,
                               ignore_index=ignore_index, reduction="sum")
    return loss_sum / (shifted_labels != ignore_index).sum().clamp_min(1)
