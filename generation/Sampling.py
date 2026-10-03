"""从 logits [..., vocab] 生成 token IDs [...]，支持单条或多维批量输入。"""

import torch


def filter_logits(logits, temperature=1.0, top_k=None, top_p=1.0):
    """依次温度缩放、Top-k、Top-p，不修改输入。

    p_i=softmax(logits/T)。Top-k 精确保留 k 个候选；Top-p 按概率降序，
    保留累计概率首次达到阈值的 token，至少保留一个。两者可组合。
    前提：logits 为 [..., vocab] 浮点 Tensor，每行至少一个有限候选，
    无 NaN/+inf；temperature>0，1<=top_k<=vocab 或 None，0<top_p<=1。
    允许 -inf 作为禁止生成的 token。
    返回相同 shape 的浮点 logits，剔除项为 -inf，半精度输入在 FP32 计算。
    """
    # 温度缩放后减最大值，保持 softmax 不变并避免指数溢出。
    work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    work = work / temperature
    work = work - work.amax(dim=-1, keepdim=True)
    if top_k is not None:
        values, indices = work.topk(top_k, dim=-1)
        work = torch.full_like(work, -torch.inf).scatter(-1, indices, values)
    if top_p < 1:
        sorted_logits, indices = work.sort(dim=-1, descending=True)
        cumulative = sorted_logits.softmax(-1).cumsum(-1)
        remove = cumulative >= top_p
        # 右移一位：跨越阈值的那个 token 仍被保留。
        remove = torch.cat((torch.zeros_like(remove[..., :1]), remove[..., :-1]), dim=-1)
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        work = torch.full_like(work, -torch.inf).scatter(-1, indices, sorted_logits)
    return work


def sample_logits(logits, temperature=1.0, top_k=None, top_p=1.0, *, greedy=False, generator=None):
    """返回 long Tensor，shape=logits.shape[:-1]（1D 输入返回标量）。

    greedy=True 返回 argmax；否则从过滤后概率中 multinomial 抽样。
    传入与 logits 所在设备一致的 torch.Generator 可获得可复现结果。
    """
    if greedy:
        # argmax 对同分 token 选择第一个；Top-k 的同分截断不会改变 greedy 规则。
        return logits.argmax(dim=-1)
    filtered = filter_logits(logits, temperature, top_k, top_p)
    probabilities = filtered.softmax(dim=-1)
    ids = torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1, generator=generator)
    return ids.reshape(logits.shape[:-1])


if __name__ == "__main__":
    scores = torch.tensor([[2., 1., 0.], [0., 1., 2.]])
    rng = torch.Generator().manual_seed(42)
    print("greedy:", sample_logits(scores, greedy=True))
    print("top-p:", sample_logits(scores, top_p=.9, generator=rng))
