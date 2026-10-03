"""从零实现 Sigmoid 和 SiLU；避免大幅负数导致 exp(-x) 溢出。"""

import torch


def sigmoid(x):
    """σ(x)=1/(1+exp(-x))，输入为浮点 Tensor，输出形状和 dtype 相同。

    正半轴使用 exp(-x)，负半轴使用 exp(x)/(1+exp(x))。
    torch.where 会计算两条分支，因此先把每条分支的无效输入设为 0，
    保证两个指数都不溢出，并在 x=0 显式选中正分支，保留 σ'(0)=1/4。
    不能直接使用 exp(-abs(x))：abs 在 0 的梯度会错误地变成 0。
    """
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    nonnegative = work >= 0
    zeros = torch.zeros_like(work)
    positive_exp = torch.exp(-torch.where(nonnegative, work, zeros))
    negative_exp = torch.exp(torch.where(nonnegative, zeros, work))
    result = torch.where(nonnegative, 1 / (1 + positive_exp), negative_exp / (1 + negative_exp))
    return result.to(x.dtype)


def silu(x):
    """SiLU(x)=x·σ(x)，输入为浮点 Tensor；又称 Swish。"""
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    return (work * sigmoid(work)).to(x.dtype)


if __name__ == "__main__":
    values = torch.tensor([-1000., -1., 0., 1., 1000.])
    print("sigmoid:", sigmoid(values))
    print("silu:", silu(values))
