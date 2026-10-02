"""
预训练损失（Pretrain Loss）

用于大语言模型预训练的标准因果语言模型损失。
采用下一个词预测（Next Token Prediction）任务。
"""

import torch
import torch.nn as nn
from ._utils import causal_lm_loss


class PretrainLoss(nn.Module):
    """
    预训练损失模块

    计算因果语言模型的交叉熵损失。
    通过预测下一个词来训练语言模型。

    Args:
        ignore_index: 忽略的标签索引，不计入损失计算，默认 -100
    """

    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        """
        前向传播

        Args:
            logits: 模型输出的未归一化对数概率 [batch_size, seq_len, vocab_size]
            labels: 真实词元索引 [batch_size, seq_len]

        Returns:
            loss: 标量损失值
        """
        # sum / 有效标签数：全忽略或 seq_len<=1 时返回可反传的 0。
        return causal_lm_loss(logits, labels, self.ignore_index)
