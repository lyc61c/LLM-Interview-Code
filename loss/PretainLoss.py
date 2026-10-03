"""因果语言模型预训练损失：当前位置预测下一个 token。"""

import torch
from torch import nn
from torch.nn import functional as F


class PretrainLoss(nn.Module):
    """输入浮点 logits [B,T,V]、整数 labels [B,T]，忽略标签默认为 -100。"""

    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        # 步骤1: next-token shift，最后一个 logit 与第一个 label 不参与损失。
        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        valid = shifted_labels != self.ignore_index

        # 步骤2: 清理忽略行并展平，CE 的类别维度为 vocab_size。
        shifted_logits = torch.where(valid.unsqueeze(-1), shifted_logits, 0.0)
        flat_logits = shifted_logits.reshape(-1, logits.shape[-1])
        flat_labels = shifted_labels.reshape(-1)

        # 步骤3: 先求损失之和，再除以有效 token 数；全忽略时为可导的 0。
        loss_sum = F.cross_entropy(flat_logits, flat_labels,
                                   ignore_index=self.ignore_index, reduction="sum")
        return loss_sum / valid.sum().clamp_min(1)
