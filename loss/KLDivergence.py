"""KL(P||Q) 的三种采样估计；样本来自 P，输入为 log P 与 log Q。"""

import torch


def sampled_kl_divergence(log_probs, ref_log_probs, estimator="k3"):
    """k1=-log(Q/P)，k2=log(Q/P)^2/2，k3=Q/P-log(Q/P)-1。"""
    # 步骤1: 计算 log 比率。
    log_ratio = ref_log_probs - log_probs

    # 步骤2: 选择估计公式。
    if estimator == "k1":
        kl = -log_ratio
    elif estimator == "k2":
        kl = 0.5 * log_ratio.square()
    else:  # k3
        kl = torch.exp(log_ratio) - log_ratio - 1

    # 步骤3: 对采样值取平均。
    return kl.mean()
