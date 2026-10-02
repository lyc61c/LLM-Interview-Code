"""KL(P||Q) 的采样估计 k1/k2/k3；输入是样本的 log P、log Q。

来源 John Schulman：https://joschu.net/blog/kl-approx.html
必须在 x~P 下平均才能解释为 KL(P||Q)：交换采样分布会改变方向。
k1/k3 在 P、Q 相互绝对连续时无偏；k2 是有偏的局部二阶近似。
单个 k1 值可为负，k2/k3 非负；这里实现估计值，不替代完整 RL 目标。
"""

import torch

from ._utils import make_mask, reduce_masked, validate_pair


def sampled_kl_divergence(log_probs, ref_log_probs, estimator="k3", mask=None,
                          reduction="mean"):
    """输入相同形状 [...]，mask 为相同形状的 0/1 或 bool 张量。

    令 l=log Q-log P：k1=-l，k2=l^2/2，k3=exp(l)-1-l。
    k3 使用 expm1(l)-l，减轻 l~0 时直接 exp(l)-1 的消减误差。
    reduction: none（屏蔽位置返回 0）、sum、mean（按有效样本数）。
    这里不 detach log_probs：可用于检查公式的 autograd；蒙特卡洛
    估计值的梯度不自动等于精确 KL 的梯度，RL 时需结合采样目标推导。
    """
    validate_pair(log_probs, ref_log_probs)
    valid = make_mask(log_probs, mask)
    p = torch.where(valid, log_probs, 0.0)
    q = torch.where(valid, ref_log_probs, 0.0)
    if not torch.isfinite(p).all() or not torch.isfinite(q).all():
        raise ValueError("有效位置的 log_probs 必须有限")
    if p.dtype in (torch.float16, torch.bfloat16):
        p, q = p.float(), q.float()
    log_ratio = q - p
    if estimator == "k1":
        values = -log_ratio
    elif estimator == "k2":
        values = log_ratio.square() / 2
    elif estimator == "k3":
        values = torch.expm1(log_ratio) - log_ratio
    else:
        raise ValueError("estimator 必须是 k1、k2 或 k3")
    return reduce_masked(values, valid, reduction)


def kl_k1(log_probs, ref_log_probs, mask=None, reduction="mean"):
    return sampled_kl_divergence(log_probs, ref_log_probs, "k1", mask, reduction)


def kl_k2(log_probs, ref_log_probs, mask=None, reduction="mean"):
    return sampled_kl_divergence(log_probs, ref_log_probs, "k2", mask, reduction)


def kl_k3(log_probs, ref_log_probs, mask=None, reduction="mean"):
    return sampled_kl_divergence(log_probs, ref_log_probs, "k3", mask, reduction)
