"""手写 Softmax、LogSoftmax、硬/软标签 CE 和全分布 KL。

类别维始终是最后一维，支持 [B, C] 或 [B, T, C]；半精度先升到
float32 做指数与归约。旧的两个位置参数 API 保持兼容。
"""

import torch


def _check_logits(logits, check_values=True):
    if not logits.is_floating_point() or logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("logits 必须是浮点张量且最后一维非空")
    logits = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    if check_values:
        if torch.isnan(logits).any() or torch.isposinf(logits).any():
            raise ValueError("有效 logits 不能包含 NaN 或 +Inf")
        if not torch.isfinite(logits).any(dim=-1).all():
            raise ValueError("每个有效样本必须至少有一个有限 logit，不能整行都是 -Inf")
    return logits


def softmax(logits):
    """exp(x-max(x)) / sum(exp(x-max(x)))，类别维为最后一维。"""
    logits = _check_logits(logits)
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    exp_shifted = shifted.exp()
    return exp_shifted / exp_shifted.sum(dim=-1, keepdim=True)


def log_softmax(logits):
    """x-max(x)-log(sum(exp(x-max(x))))，避免巨大公共偏移造成抵消。"""
    logits = _check_logits(logits)
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    return shifted - shifted.exp().sum(dim=-1, keepdim=True).log()


def cross_entropy_loss(logits, targets, reduction="mean", ignore_index=-100):
    """CE=-log p_y（硬标签）或 -sum_c target_c*log p_c（软标签）。

    logits: [..., C]；硬标签 targets: [...] 整数；软标签: [..., C]
    非负浮点且每行和为 1。reduction 支持 none/sum/mean。ignore_index
    仅对硬标签生效；全忽略或空批次返回可反传的 0，而非 NaN。
    """
    # 先校验 shape/dtype；忽略行在下面清理后再校验值和计算 log_softmax。
    logits = _check_logits(logits, check_values=False)
    if targets.device != logits.device:
        raise ValueError("targets 与 logits 必须位于同一设备")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError("reduction 必须是 none、sum 或 mean")
    if targets.shape == logits.shape:
        if not targets.is_floating_point():
            raise ValueError("软标签必须是浮点概率分布")
        if not torch.isfinite(targets).all() or (targets < 0).any():
            raise ValueError("软标签必须非负且有限")
        if not torch.allclose(targets.sum(dim=-1), torch.ones_like(targets[..., 0]),
                              atol=1e-3, rtol=1e-3):
            raise ValueError("软标签每行概率和必须为 1")
        log_probs = log_softmax(logits)
        # 0*log(0) 按 0 处理；适用于把某类 logits 设为 -inf 的情况。
        safe_log_probs = torch.where(targets > 0, log_probs, 0.0)
        losses = -(targets * safe_log_probs).sum(dim=-1)
        valid = torch.ones_like(losses, dtype=torch.bool)
    else:
        if targets.shape != logits.shape[:-1] or targets.dtype not in (torch.int32, torch.int64):
            raise ValueError("硬标签必须是形状 logits.shape[:-1] 的整数张量")
        valid = targets != ignore_index
        if ((targets[valid] < 0) | (targets[valid] >= logits.shape[-1])).any():
            raise ValueError("有效标签必须在 [0, num_classes) 范围内")
        safe_targets = torch.where(valid, targets, 0).long()
        safe_logits = torch.where(valid.unsqueeze(-1), logits, 0.0)
        log_probs = log_softmax(safe_logits)
        losses = -log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        losses = torch.where(valid, losses, 0.0)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.sum() / valid.sum().clamp_min(1)


def KL_divergence(p_logits, q_logits, reduction="mean"):
    """精确 KL(P||Q)=sum_c P_c(log P_c-log Q_c)。

    输入相同形状 [..., C]。mean 对分布（而非类别）平均，none 返回 [...]。
    这与 torch.nn.functional.kl_div(log Q, P, reduction='batchmean')
    在二维输入下相同。不同于 KLDivergence.py 的采样估计。
    """
    if p_logits.shape != q_logits.shape or p_logits.device != q_logits.device:
        raise ValueError("P、Q 必须形状相同且位于同一设备")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError("reduction 必须是 none、sum 或 mean")
    p_log_probs, q_log_probs = log_softmax(p_logits), log_softmax(q_logits)
    p_probs = p_log_probs.exp()
    # P=0 的类贡献为 0，即使 log P=-inf。
    difference = torch.where(p_probs > 0, p_log_probs - q_log_probs, 0.0)
    losses = (p_probs * difference).sum(dim=-1)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.sum() / max(losses.numel(), 1)
