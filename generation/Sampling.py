"""从 logits [..., vocab] 生成 token IDs [...]，支持单条或多维批量输入。"""

import math

import torch


def filter_logits(logits, temperature=1.0, top_k=None, top_p=1.0):
    """依次温度缩放、Top-k、Top-p，不修改输入。

    p_i=softmax(logits/T)。Top-k 精确保留 k 个候选；Top-p 按概率降序，
    保留累计概率首次达到阈值的 token，至少保留一个。两者可组合。
    允许 -inf 作为禁止生成的 token；NaN/+inf/整行 -inf 则报错。
    返回相同 shape 的浮点 logits，剔除项为 -inf，半精度输入在 FP32 计算。
    """
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise TypeError("logits 必须是浮点 Tensor")
    if logits.ndim < 1 or logits.numel() == 0:
        raise ValueError("logits 必须非空，最后一维为词表")
    vocab_size = logits.shape[-1]
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature 必须是有限正数")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int)
                              or not 1 <= top_k <= vocab_size):
        raise ValueError("top_k 必须是 1 到 vocab_size 之间的整数或 None")
    if isinstance(top_p, bool) or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p 必须在 (0, 1] 内")
    if torch.isnan(logits).any() or torch.isposinf(logits).any() or not torch.isfinite(logits).any(-1).all():
        raise ValueError("logits 不能包含 NaN/+inf，且每行至少有一个有限候选")
    work = logits if logits.dtype == torch.float64 else logits.float()
    if not torch.finfo(work.dtype).tiny <= temperature <= torch.finfo(work.dtype).max:
        # Python 浮点温度可能无法表示为 FP32；在 FP64 中缩放避免 0/0。
        work = work.double()
    if temperature >= 1:
        # 高温先缩小数值，避免 [-max,+max] 先相减溢出，丢失可恢复的概率。
        work = work / temperature
        work = work - work.amax(dim=-1, keepdim=True)
    else:
        # 低温先减最大值，避免最大有限 logit 除以小 T 后成为 +inf。
        work = (work - work.amax(dim=-1, keepdim=True)) / temperature
    if top_k is not None and top_k < vocab_size:
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
    filtered = filter_logits(logits, temperature, top_k, top_p)
    if greedy:
        # argmax 对同分 token 选择第一个；Top-k 的同分截断不会改变 greedy 规则。
        return logits.argmax(dim=-1)
    probabilities = filtered.softmax(dim=-1)
    ids = torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1, generator=generator)
    return ids.reshape(logits.shape[:-1])


if __name__ == "__main__":
    scores = torch.tensor([[2., 1., 0.], [0., 1., 2.]])
    rng = torch.Generator().manual_seed(42)
    print("greedy:", sample_logits(scores, greedy=True))
    print("top-p:", sample_logits(scores, top_p=.9, generator=rng))
