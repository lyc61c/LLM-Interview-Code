"""手写 Greedy、Temperature、Top-k 和 Top-p 采样。"""

import torch


def filter_logits(logits, temperature=1.0, top_k=None, top_p=1.0):
    """依次温度缩放、Top-k、Top-p，不修改输入。

    p_i=softmax(logits/T)。Top-k 精确保留 k 个候选；Top-p 按概率降序，
    保留累计概率首次达到阈值的 token，至少保留一个。两者可组合。
    前提：logits 为 [V] 或 [B,V] 浮点 Tensor，每行至少一个有限候选，
    无 NaN/+inf；temperature>0，1<=top_k<=vocab 或 None，0<top_p<=1。
    允许 -inf 作为禁止生成的 token。
    返回相同 shape 的 logits，剔除项为 -inf。
    """
    # 步骤1: 温度缩放。
    work = logits / temperature
    if top_k is not None:
        # 步骤2: 保留 Top-k 候选，其他位置设为 -inf。
        values, indices = work.topk(top_k, dim=-1)
        work = torch.full_like(work, -torch.inf).scatter(-1, indices, values)
    if top_p < 1:
        # 步骤3: 按概率排序，保留累计概率达到 top_p 的候选。
        sorted_logits, indices = work.sort(dim=-1, descending=True)
        cumulative = sorted_logits.softmax(-1).cumsum(-1)
        remove = cumulative >= top_p
        # 右移一位：跨越阈值的那个 token 仍被保留。
        remove = torch.cat((torch.zeros_like(remove[..., :1]), remove[..., :-1]), dim=-1)
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        work = torch.full_like(work, -torch.inf).scatter(-1, indices, sorted_logits)
    return work


def sample_logits(logits, temperature=1.0, top_k=None, top_p=1.0, *, greedy=False):
    """返回 long Tensor，shape=logits.shape[:-1]（1D 输入返回标量）。

    greedy=True 返回 argmax；否则从过滤后概率中 multinomial 抽样。
    """
    if greedy:
        # 贪婪解码直接选最大 logit。
        return logits.argmax(dim=-1)
    filtered = filter_logits(logits, temperature, top_k, top_p)
    probabilities = filtered.softmax(dim=-1)
    return torch.multinomial(probabilities, 1).squeeze(-1)


if __name__ == "__main__":
    scores = torch.tensor([[2., 1., 0.], [0., 1., 2.]])
    torch.manual_seed(42)
    print("greedy:", sample_logits(scores, greedy=True))
    print("top-p:", sample_logits(scores, top_p=.9))
