"""手写全连接层：y = x @ W.T + b，保留任意前导维度。"""

import torch
from torch import nn


class Linear(nn.Module):
    """输入 [..., in_features]，输出 [..., out_features]。

    weight 的布局与 nn.Linear 一致：[out_features, in_features]。
    nn.Parameter 使参数自动出现在 parameters()/state_dict() 中；bias=False
    时不使用偏置。前向直接使用矩阵乘法。
    前提：输入/输出维度为正整数，输入最后一维等于 in_features。
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        output = x @ self.weight.T
        return output if self.bias is None else output + self.bias


if __name__ == "__main__":
    layer = Linear(4, 3, bias=False)
    print(layer(torch.randn(2, 5, 4)).shape)  # [2, 5, 3]
