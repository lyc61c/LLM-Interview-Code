"""
监督微调损失（Supervised Fine-Tuning Loss）

用于大语言模型监督微调的损失函数。
与预训练损失类似，但支持屏蔽 prompt 部分，只计算 response 的损失。
"""

import torch
import torch.nn as nn
from ._utils import causal_lm_loss


class SFTLoss(nn.Module):
    """
    监督微调损失模块

    在 SFT 阶段，我们通常只希望计算 response 部分的损失，
    而不计算 prompt 部分的损失。该模块支持通过 prompt_lengths 来屏蔽 prompt。

    Args:
        无
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, prompt_lengths):
        """
        前向传播

        Args:
            logits: 模型输出的未归一化对数概率 [batch_size, seq_len, vocab_size]
            labels: 真实词元索引 [batch_size, seq_len]
            prompt_lengths: 每个样本的 prompt 长度 [batch_size]

        Returns:
            loss: 标量损失值
        """
        if logits.ndim != 3 or labels.shape != logits.shape[:2]:
            raise ValueError("logits 必须为 [B,T,V]，labels 必须为 [B,T]")
        prompt_lengths = torch.as_tensor(prompt_lengths, device=labels.device)
        if prompt_lengths.shape != (labels.shape[0],) or prompt_lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("prompt_lengths 必须是 [B] 的整数张量或列表")
        if ((prompt_lengths < 0) | (prompt_lengths > labels.shape[1])).any():
            raise ValueError("prompt length 必须在 [0, seq_len]")
        # 同时保留 labels 原有的 -100（例如 padding），不修改调用者数据。
        positions = torch.arange(labels.shape[1], device=labels.device)
        masked_labels = labels.masked_fill(positions[None, :] < prompt_lengths[:, None], -100)
        return causal_lm_loss(logits, masked_labels)
