"""KL(P||Q) 的采样估计 k1/k2/k3，样本必须来自 P。

k1/k3 在相应支撑条件下无偏；k2 为局部二阶近似，一般有偏。
"""

import torch


def sampled_kl_divergence(log_probs, ref_log_probs, estimator="k3", mask=None,
                          reduction="mean"):
    """输入同形状浮点 [...]，分别为样本的 log P(x)、log Q(x)。

    令 l=log Q-log P：k1=-l，k2=l²/2，k3=exp(l)-1-l。
    estimator 使用 k1/k2/k3；mask 为同形状 0/1 或 bool，1 表示有效样本。
    reduction 使用 none/sum/mean，mean 只平均有效样本，全屏蔽为 0。
    采样估计值的 autograd 不自动等于精确 KL 的梯度。
    """
    # 步骤1: padding 位置先置零，再计算 log ratio。
    valid = torch.ones_like(log_probs, dtype=torch.bool) if mask is None else mask.bool()
    p = torch.where(valid, log_probs, 0.0)
    q = torch.where(valid, ref_log_probs, 0.0)
    log_ratio = q - p

    # 步骤2: 选择估计公式；k3 用 expm1 减少近零时的相消误差。
    if estimator == "k1":
        values = -log_ratio
    elif estimator == "k2":
        values = log_ratio.square() / 2
    else:  # k3：加上期望为零的 Q/P-1 控制变量。
        values = torch.expm1(log_ratio) - log_ratio
    values = torch.where(valid, values, 0.0)

    # 步骤3: 对有效样本归约。
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    return values.sum() / valid.sum().clamp_min(1)


def kl_k1(log_probs, ref_log_probs, mask=None, reduction="mean"):
    return sampled_kl_divergence(log_probs, ref_log_probs, "k1", mask, reduction)


def kl_k2(log_probs, ref_log_probs, mask=None, reduction="mean"):
    return sampled_kl_divergence(log_probs, ref_log_probs, "k2", mask, reduction)


def kl_k3(log_probs, ref_log_probs, mask=None, reduction="mean"):
    return sampled_kl_divergence(log_probs, ref_log_probs, "k3", mask, reduction)
