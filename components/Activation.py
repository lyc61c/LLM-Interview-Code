"""手写 Sigmoid 与 SiLU 激活函数。"""

import torch


def sigmoid(x):
    """σ(x) = 1 / (1 + exp(-x))，输入输出形状相同。"""
    return 1 / (1 + torch.exp(-x))


def silu(x):
    """SiLU(x) = x * σ(x)，又称 Swish。"""
    return x * sigmoid(x)


if __name__ == "__main__":
    values = torch.tensor([-2., -1., 0., 1., 2.])
    print("sigmoid:", sigmoid(values))
    print("silu:", silu(values))
