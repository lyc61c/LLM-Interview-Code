"""仓库根目录运行：python -m examples.quickstart。"""

import torch

from attention.AttentionMask import create_attention_mask
from attention.MultiHeadAttentionWithKVCache import MultiHeadAttentionWithKVCache
from components.Activation import sigmoid, silu
from components.Linear import Linear
from components.Quantization import asymmetric_quantize, dequantize, symmetric_quantize
from generation.Sampling import sample_logits
from loss.DAPOLoss import dapo_loss
from loss.EntropyLoss import cross_entropy_loss
from loss.GSPOLoss import gspo_loss
from loss.InfoNCELoss import info_nce_loss
from loss.KLDivergence import sampled_kl_divergence
from tokenizer.BPE import ByteBPETokenizer
from tools.ToolCallParser import parse_tool_calls


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    linear = Linear(4, 3)
    print("Linear:", tuple(linear(torch.ones(2, 4)).shape))
    values = torch.tensor([-1000., 0., 1000.])
    print("Sigmoid / SiLU:", sigmoid(values).tolist(), silu(values).tolist())

    ids = torch.tensor([[11, 12, 13, 0]])
    print("Padding + causal mask:", create_attention_mask(ids)[0, 0].int().tolist())
    attention = MultiHeadAttentionWithKVCache(8, 2).eval()
    x = torch.randn(1, 5, 8)
    with torch.no_grad():
        full, _ = attention(x)
        cache = None
        outputs = []
        for index in range(x.size(1)):
            output, cache = attention(x[:, index:index + 1], past_key_value=cache)
            outputs.append(output)
        incremental = torch.cat(outputs, dim=1)
        torch.testing.assert_close(incremental, full)
    print("KV Cache 与完整 causal attention 一致，缓存形状:", tuple(cache[0].shape))

    tokenizer = ByteBPETokenizer().fit(["你好，世界！", "你好，面试！"], num_merges=16)
    text = "你好 新词🙂\n"
    encoded = tokenizer.encode(text)
    assert tokenizer.decode(encoded) == text
    print("BPE 中文 / 空格 / emoji 往返成功，token 数:", len(encoded))

    values = torch.tensor([-2., -.3, 0., 1.2, 3.])
    for quantize in (symmetric_quantize, asymmetric_quantize):
        quantized = quantize(values)
        error = (dequantize(quantized) - values).abs().max().item()
        print(quantize.__name__, "dtype:", quantized.values.dtype, "最大误差:", round(error, 6))
    rng = torch.Generator().manual_seed(42)
    logits = torch.tensor([[2., 1., 0.]])
    print("Top-p 抽样:", sample_logits(logits, top_p=.9, generator=rng).tolist())

    queries = torch.randn(4, 6)
    print("InfoNCE:", round(info_nce_loss(queries, queries).item(), 6))
    soft_targets = torch.tensor([[.7, .2, .1]])
    print("软标签 CE:", round(cross_entropy_loss(logits, soft_targets).item(), 6))
    old = torch.zeros(2, 3)
    new = torch.zeros(2, 3)
    advantages = torch.tensor([1., -1.])
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    print("不同长度、相同比率的 DAPO / GSPO:",
          dapo_loss(old, new, advantages, mask).item(),
          gspo_loss(old, new, advantages, mask).item())
    print("两策略相同的 k3 KL:", sampled_kl_divergence(old, new).item())

    events = [
        {"type": "response.output_item.added", "output_index": 0,
         "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "weather"}},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"city":"上'},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '海"}'},
        {"type": "response.function_call_arguments.done", "output_index": 0, "arguments": '{"city":"上海"}'},
    ]
    calls = parse_tool_calls(events)
    print("工具参数解析:", calls[0].name, calls[0].arguments)


if __name__ == "__main__":
    main()
