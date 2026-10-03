"""InfoNCE 损失：同一行是正样本，批内其他配对是负样本。"""

import torch
import torch.nn.functional as F


def info_nce_loss(queries, keys, temperature=0.1):
    """queries、keys: [batch_size, feature_dim]。"""
    # 步骤1: 对特征做 L2 归一化。
    queries = F.normalize(queries, dim=-1)
    keys = F.normalize(keys, dim=-1)

    # 步骤2: 计算批内相似度矩阵，并用温度缩放。
    logits = queries @ keys.T / temperature

    # 步骤3: 正样本位于对角线，目标类别是每行自身的索引。
    targets = torch.arange(queries.size(0))
    return F.cross_entropy(logits, targets)
