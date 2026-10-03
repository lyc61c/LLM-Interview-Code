"""手写全连接层：y = x @ W.T + b，保留任意前导维度。"""

import math

import torch
from torch import nn


class Linear(nn.Module):
    """输入 [..., in_features]，输出 [..., out_features]。

    weight 的布局与 nn.Linear 一致：[out_features, in_features]。
    nn.Parameter 使参数自动出现在 parameters()/state_dict() 中；bias=False
    时注册 None。前向没有调用 nn.Linear 或 functional.linear。
    前提：输入/输出维度为正整数，输入最后一维等于 in_features。
    """

    def __init__(self, in_features, out_features, bias=True, *, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # 与 nn.Linear 默认初始化相同：U(-1/sqrt(fan_in), 1/sqrt(fan_in))。
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        output = x @ self.weight.transpose(-1, -2)
        return output if self.bias is None else output + self.bias


if __name__ == "__main__":
    layer = Linear(4, 3, bias=False)
    print(layer(torch.randn(2, 5, 4)).shape)  # [2, 5, 3]
