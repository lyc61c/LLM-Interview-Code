"""手写 Softmax、LogSoftmax、交叉熵和全分布 KL。

输入为浮点 logits，类别维在最后一维，相关张量由调用者放在同一设备。
有效行应能定义概率分布；硬/软标签的形状与取值约定见各函数说明。
"""

import torch


def softmax(logits):
    """输入/输出 [..., C]；减去最大值后计算指数，避免指数溢出。"""
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()

    # 步骤1: 减去每行最大值，不改变 softmax 的结果。
    shifted = logits - logits.max(dim=-1, keepdim=True).values

    # 步骤2: 对指数结果归一化，类别概率之和为 1。
    exp_shifted = shifted.exp()
    return exp_shifted / exp_shifted.sum(dim=-1, keepdim=True)


def log_softmax(logits):
    """log p = (x-max(x)) - log(sum(exp(x-max(x))))，类别维为最后一维。"""
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()

    # 直接求 log 概率，避免先 softmax 再 log 时概率下溢。
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    return shifted - shifted.exp().sum(dim=-1, keepdim=True).log()


def cross_entropy_loss(logits, targets, reduction="mean", ignore_index=-100):
    """硬标签 CE=-log p_y；软标签 CE=-sum_c target_c * log p_c。

    logits: [..., C]。硬标签 targets: [...] 的整数类别索引；软标签:
    [..., C] 的非负浮点概率，每行和为 1。reduction 使用 none/sum/mean。
    ignore_index 仅用于硬标签；mean 只平均有效位置，全忽略时返回可导的 0。
    """
    # 步骤1: 根据标签形状区分软标签与硬标签。
    if targets.shape == logits.shape:
        log_probs = log_softmax(logits)
        # 目标概率为 0 的类别不贡献 CE，按 0*log(0)=0 处理。
        log_probs = torch.where(targets > 0, log_probs, 0.0)
        losses = -(targets * log_probs).sum(dim=-1)
        valid = torch.ones_like(losses, dtype=torch.bool)
    else:
        valid = targets != ignore_index
        # 忽略行在 softmax 前置零；忽略索引也替换成可供 gather 使用的 0。
        safe_logits = torch.where(valid.unsqueeze(-1), logits, 0.0)
        safe_targets = torch.where(valid, targets, 0).long()
        log_probs = log_softmax(safe_logits)
        losses = -log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        losses = torch.where(valid, losses, 0.0)

    # 步骤2: 保留每个位置、求和，或按有效位置数求平均。
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.sum() / valid.sum().clamp_min(1)


def KL_divergence(p_logits, q_logits, reduction="mean"):
    """精确 KL(P||Q)=sum_c P_c(log P_c-log Q_c)。

    输入相同形状 [..., C] 的 logits。none 返回 [...]，sum 求和，
    mean 对分布平均。在二维输入下对应 PyTorch kl_div(log Q, P, batchmean)。
    """
    # 步骤1: 得到两个分布的 log 概率，以及分布 P 的概率。
    p_log_probs = log_softmax(p_logits)
    q_log_probs = log_softmax(q_logits)
    p_probs = p_log_probs.exp()

    # 步骤2: 沿类别维累加；P=0 的类别贡献为 0。
    log_difference = torch.where(p_probs > 0, p_log_probs - q_log_probs, 0.0)
    losses = (p_probs * log_difference).sum(dim=-1)

    # 步骤3: 对分布归约，而不是对所有类别元素取平均。
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.sum() / max(losses.numel(), 1)
