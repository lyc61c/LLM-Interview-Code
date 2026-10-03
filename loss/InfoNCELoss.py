"""批内负样本 InfoNCE：第 i 个 query 与第 i 个 key 构成正样本。

其余配对作为负样本；相同语义的其他行可能成为假负样本。
"""

import torch
from torch import nn

from .EntropyLoss import cross_entropy_loss


def info_nce_loss(queries, keys, temperature=0.1, symmetric=False):
    """输入相同的非空浮点 [B,D] 特征，temperature 为正，返回标量。

    s_ij=cosine(q_i,k_j)/temperature，L=-mean_i log softmax(s_i)_i。
    symmetric=True 时平均 q→k 与 k→q 两个方向；B=1 时没有负样本，损失为 0。
    假设特征范数和温度缩放处于计算精度可表示的范围。
    """
    if queries.dtype in (torch.float16, torch.bfloat16):
        queries = queries.float()
        keys = keys.float()

    # 在当前计算精度下归一化、做点积；避免 autocast 再降低点积精度。
    with torch.autocast(device_type=queries.device.type, enabled=False):
        # 步骤1: L2 归一化，clamp_min 使零向量不会除零。
        queries = queries / queries.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # 步骤2: 计算所有配对的相似度，正样本位于对角线。
        similarities = queries @ keys.T / temperature
        targets = torch.arange(queries.shape[0], device=queries.device)

        # 步骤3: 每行以第 i 类为目标做 CE，可选对两个方向取平均。
        loss = cross_entropy_loss(similarities, targets)
        if symmetric:
            loss = (loss + cross_entropy_loss(similarities.T, targets)) / 2
    return loss


class InfoNCELoss(nn.Module):
    """info_nce_loss 的模块封装。"""

    def __init__(self, temperature=0.1, symmetric=False):
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric

    def forward(self, queries, keys):
        return info_nce_loss(queries, keys, self.temperature, self.symmetric)
