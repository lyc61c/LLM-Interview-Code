"""监督微调：先屏蔽 prompt 标签，再做 next-token shift。"""

import torch
from torch import nn
from torch.nn import functional as F


class SFTLoss(nn.Module):
    """logits [B,T,V]、整数 labels [B,T]，prompt_lengths 为 [B] 整数或列表。

    prompt 长度由调用者保证在 0..T 范围内；已有 -100 padding 标签继续忽略。
    """

    def forward(self, logits, labels, prompt_lengths):
        # 步骤1: 复制 labels，只保留 response 的监督信号，不修改原输入。
        masked_labels = labels.clone()
        for batch_idx, prompt_length in enumerate(prompt_lengths):
            masked_labels[batch_idx, :prompt_length] = -100

        # 步骤2: next-token shift，将当前位置的 logits 与下一个 label 对齐。
        shifted_logits = logits[:, :-1, :]
        shifted_labels = masked_labels[:, 1:]
        valid = shifted_labels != -100

        # 步骤3: 忽略行先置零，再展平计算 CE。
        shifted_logits = torch.where(valid.unsqueeze(-1), shifted_logits, 0.0)
        flat_logits = shifted_logits.reshape(-1, logits.shape[-1])
        flat_labels = shifted_labels.reshape(-1)
        loss_sum = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="sum")

        # 步骤4: response 的有效 token 等权；全忽略时返回可导的 0。
        return loss_sum / valid.sum().clamp_min(1)
