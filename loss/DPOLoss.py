"""DPO：直接使用 chosen/rejected 偏好优化策略相对参考策略的概率。"""

from torch.nn import functional as F


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps,
             beta=0.1, label_smoothing=0.0):
    """四组输入为同形状非空 [B] 序列 log 概率，调用前已聚合有效 token。

    beta>0，label_smoothing 位于 [0,1]。
    L=-mean(log sigmoid(beta*(chosen_ratio-rejected_ratio)))。
    """
    # 步骤1: 计算 chosen/rejected 相对参考策略的 log 比率。
    chosen_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_ratio = policy_rejected_logps - ref_rejected_logps
    preference_logits = beta * (chosen_ratio - rejected_ratio)

    # 步骤2: 用 logsigmoid 计算损失，避免先 sigmoid 再 log 的下溢。
    losses = -F.logsigmoid(preference_logits)

    # 步骤3: 标签平滑混合正向和反向偏好损失。
    if label_smoothing > 0:
        inverse_losses = -F.logsigmoid(-preference_logits)
        losses = (1 - label_smoothing) * losses + label_smoothing * inverse_losses
    return losses.mean()
