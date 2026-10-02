# 手撕题学习清单

本清单沿用本仓库的 PyTorch 教学风格，把题目按依赖关系排列。
每道题先解释公式和输入输出，再写核心实现，最后检查数值、梯度和边界。
参考题目范围来自 [AIR-hl/llm-interview-code](https://github.com/AIR-hl/llm-interview-code)；
实现与边界处理以本地源文件为准。

## 1. 基础算子

| 题目 | 文件 | 验证重点 |
|---|---|---|
| Linear | [Linear.py](../components/Linear.py) | `x @ weight.T + bias`；任意前导维度、可选 bias、参数梯度 |
| Sigmoid / SiLU | [Activation.py](../components/Activation.py) | 大幅正负输入、零点导数、半精度 |
| Softmax / LogSoftmax / CE | [EntropyLoss.py](../loss/EntropyLoss.py) | 减最大值；硬标签、软标签；类别轴位于最后一维 |
| LayerNorm / RMSNorm | [normalization](../normalization) | 总体方差；半精度提升到 FP32；保留 FP64 精度 |

### 必须能解释的公式

$$y=xW^T+b,\quad \sigma(x)=\frac{1}{1+e^{-x}},\quad \mathrm{SiLU}(x)=x\sigma(x)$$

Softmax 先减去最大值；LogSoftmax 用 Log-Sum-Exp，避免先计算概率再取 log。
硬标签 CE 取目标类的负 log 概率；软标签 CE 是整个目标分布的加权和：

$$\mathrm{CE}(z,q)=-\sum_j q_j\log\mathrm{softmax}(z)_j$$

面试时说明 `targets` 的两种形状：类别索引 `[...]`，或概率分布 `[..., C]`。
全忽略硬标签返回可反传的零损失，避免除以零。

## 2. 注意力、掩码与位置编码

| 题目 | 文件 | 验证重点 |
|---|---|---|
| SDPA | [ScaledDotProductAttention.py](../attention/ScaledDotProductAttention.py) | 缩放因子、softmax 轴、全屏蔽行输出为零 |
| MHA / GQA | [attention](../attention) | 分头、合头、KV 共享；GQA 中 Q 头数可被 KV 头数整除 |
| RoPE | [RotaryEmbedding.py](../position/RotaryEmbedding.py) | 成对旋转、head_dim 为偶数、位置偏移 |
| Padding + Causal Mask | [AttentionMask.py](../attention/AttentionMask.py) | 广播形状、允许 / 屏蔽语义、缓存位置 |
| MHA + KV Cache | [MultiHeadAttentionWithKVCache.py](../attention/MultiHeadAttentionWithKVCache.py) | 完整序列与逐 token / 分块解码等价 |
| MLA | [MultiLatentAttention.py](../attention/MultiLatentAttention.py) | 低秩投影与形状变换；教学简化版的边界 |

本仓库统一 **True / 1 = 允许注意，False / 0 = 屏蔽**。
这与部分教程中 `True = 屏蔽` 的布尔 mask 约定相反，组合代码前要对齐语义。

```python
import torch
from attention.AttentionMask import create_attention_mask

input_ids = torch.tensor([[11, 12, 13, 0]])
allowed = create_attention_mask(input_ids, pad_token_id=0)
# allowed: [B, 1, Q, K]，限制未来 token、padding key 与 padding query。
```

### 缓存为什么不能直接用普通下三角？

假设已缓存 3 个 token，本轮输入 2 个 token。Query 的绝对位置是 3、4，
而 Key 的位置是 0、1、2、3、4。允许条件是：

$$k\_position \le past\_len + q\_position$$

因此 mask 是 `Q=2, K=5`，首行允许前 4 个 key，第二行允许全部 5 个 key。
缓存中的 K 已应用对应位置的 RoPE，追加时只旋转本轮新 K；旧 K 不再旋转。

```python
from attention.MultiHeadAttentionWithKVCache import MultiHeadAttentionWithKVCache

model = MultiHeadAttentionWithKVCache(model_dim=8, num_heads=2).eval()
x = torch.randn(1, 4, 8)
with torch.no_grad():
    full, _ = model(x)
    first, cache = model(x[:, :3])
    last, cache = model(x[:, 3:], past_key_value=cache)
    torch.testing.assert_close(torch.cat([first, last], dim=1), full)
```

MLA 继续保留原仓库简化版定位；MHA 的 KV Cache 示例不等于完整 DeepSeek MLA 缓存实现。

## 3. 分词、量化与生成

### Byte-level BPE

入口：[BPE.py](../tokenizer/BPE.py)。从 256 个 UTF-8 byte token 起步，统计相邻对，
选择高频对合并，记录学习顺序，编码时优先应用最早学到的规则。

```python
from tokenizer.BPE import ByteBPETokenizer

tokenizer = ByteBPETokenizer().fit(["你好，世界！", "你好，面试！"], num_merges=16)
text = "你好 新词🙂\n"
assert tokenizer.decode(tokenizer.encode(text)) == text
```

Byte-level 的基础词表能够表示未见过的中文、英文、空格和 emoji。
这里不实现 GPT-2 的正则预分词、特殊 token 协议或持久化模型格式。
测试关注可逆性、重叠对合并、频率并列的确定性、训练样本之间的边界。

### INT8 量化

入口：[Quantization.py](../components/Quantization.py)。

$$q=\mathrm{clamp}(\mathrm{round}(x/s)+z),\qquad \hat{x}=(q-z)s$$

对称量化取 `z=0`，使用 `[-127,127]`；非对称量化使用 `[-128,127]`，
将校准范围扩展到包含零，计算整数 zero point。
返回 `Int8Quantized(values, scale, zero_point)`，其中 `values.dtype == torch.int8`。
全零张量取合法非零 scale；正 / 负常数同样能处理。

解释量化误差时要考虑舍入误差与饱和误差；这里是逐张量量化，
不包含校准算法、QAT、逐通道量化或 INT8 矩阵乘 kernel。

### 采样

入口：[Sampling.py](../generation/Sampling.py)。

1. `temperature` 缩放 logits；必须是有限正数。
2. `top_k` 保留指定数量的候选，`None` 表示不限制。
3. `top_p` 保留按概率降序排列后、累计概率首次达到阈值的最小前缀。
4. 对过滤结果做 softmax，再用 multinomial 抽样。

跨过 top-p 阈值的 token 要保留，至少留下一个候选。
输入 logits 不被修改；允许 `-inf` 表示禁止生成，整行无候选则报错。
`sample_logits` 返回 `logits.shape[:-1]` 的 token IDs。
贪婪解码使用 `greedy=True`；可传 `torch.Generator` 复现随机结果。

## 4. 损失函数与强化学习

| 题目 | 文件 | 验证重点 |
|---|---|---|
| Pretrain / SFT | [loss](../loss) | next-token shift、prompt mask、全忽略边界 |
| InfoNCE | [InfoNCELoss.py](../loss/InfoNCELoss.py) | 正样本在对角线；分母包含正样本；批内负样本 |
| DPO | [DPOLoss.py](../loss/DPOLoss.py) | chosen / rejected 与策略 / 参考模型的四组 log 概率 |
| PPO / GRPO | [loss](../loss) | 最大化目标与最小化 loss 的符号；组内优势与 clipping |
| DAPO | [DAPOLoss.py](../loss/DAPOLoss.py) | token 级比率、非对称 clip、有效 token 等权 |
| GSPO | [GSPOLoss.py](../loss/GSPOLoss.py) | 序列级比率、序列等权、长度归一化 |
| KL | [KLDivergence.py](../loss/KLDivergence.py) | 精确 KL 与采样估计的区别、采样方向 |

### InfoNCE

`info_nce_loss(queries, keys, temperature=0.1, symmetric=False)`：
输入两组 `[B,D]` 向量，第 i 对是正样本。
归一化后计算 `queries @ keys.T / temperature`，目标类别是 `arange(B)`。
`symmetric=True` 平均两个方向的损失。

### DAPO 与 GSPO 的区别

两个函数输入 `old_log_probs, new_log_probs: [B,T]`，mask 标记有效 response token。
来自相同 prompt 的多个候选可先计算组内优势，再展平成 B 条回答。
旧策略概率与优势属于 rollout 数据，内部 detach；梯度流向新策略。

$$r_{it}=\exp(\log\pi_{new,it}-\log\pi_{old,it})$$

DAPO 对 token 级 `r` 做非对称 clip，再除以**所有有效 token 的数量**。
GSPO 先求每条序列的平均 log ratio：

$$s_i=\exp\left(\frac{\sum_t mask_{it}(\log\pi_{new,it}-\log\pi_{old,it})}{length_i}\right)$$

再对 `s_i` 做 clip，最后对**非空序列**等权平均。它不是平均 token ratio，
也不是直接连乘全部 token 的 ratio。变长回答能揭示两种归约的差异。

默认 DAPO clipping 为 `epsilon_low=0.2, epsilon_high=0.28`；
GSPO 为 `3e-4, 4e-4`。具体训练超参数需结合目标任务。
DAPO 文件仅实现损失核心，动态采样、overlong shaping 和 rollout 系统见
[DAPO 原论文](https://arxiv.org/abs/2503.14476)。GSPO 定义见
[GSPO 原论文](https://arxiv.org/abs/2507.18071)。

### KL 的方向与估计器

这里采样 `x ~ P`，`log_probs = log P(x)`，`ref_log_probs = log Q(x)`，
估计的是 `KL(P || Q)`。记 `l = log Q(x) - log P(x)`：

$$k_1=-l,\qquad k_2=\tfrac12 l^2,\qquad k_3=e^l-1-l$$

k1 的单样本值可以为负，期望是 KL；k2 在两分布接近时有较好近似，但一般有偏。
k3 是利用控制变量的估计，数值接近零时用 `expm1` 减少相消误差。
无偏性需要实际从 P 采样及相应支撑条件，不能拿任意样本套用。
全类别精确 KL 已在 `EntropyLoss.KL_divergence` 中实现。
推导见 [Schulman 的 KL 估计说明](http://joschu.net/blog/kl-approx.html)。

## 5. 工具调用流式解析

入口：[ToolCallParser.py](../tools/ToolCallParser.py)。处理三类函数调用事件：

1. `response.output_item.added` 注册调用信息。
2. `response.function_call_arguments.delta` 追加 JSON 参数片段。
3. `response.function_call_arguments.done` 完成参数、验证 JSON object。

`output_index` 区分交错调用，`item_id` 校验事件归属。
`ToolCallParser.feed(event)` 返回刚完成的调用或 None，`finish()` 检查流完整性。
`parse_tool_calls(events)` 返回按索引排序的 `ToolCall` 列表。
解析器只生成结构化调用信息，实际工具执行由调用者负责。
错误 JSON、未知索引、重复完成、流提前结束都有明确错误。

## 6. 前馈网络与 LoRA

MoE 的 Top-k 是专家路由，与生成时的 Top-k 采样具有不同含义。
路由实现支持非连续输入；验证时可用全部专家计算得到稠密参考结果。

LoRA 的基础权重冻结，但输入的梯度仍需经过基础分支传回前层。
B 初始化为零时，输出等于基础线性层；更新 B 后再检验 A / B 的梯度。
推理时 `eval()` 后使用 `merged_linear()` 得到独立合并权重：

$$W_{merged}=W_0+\frac{\alpha}{rank}BA$$

## 7. 如何检查练习完成

在仓库根目录运行 `python -m pytest -q`。

本地验证环境为 Windows、Python 3.12.14、PyTorch 2.14.1+cpu。
测试使用小张量，不需要 CUDA、模型权重或训练数据集。

测试检查：

- 与 PyTorch 的 Linear、CE、KL、SDPA 等参考计算核对数值。
- FP64 的有限差分 gradcheck 或解析梯度对照。
- 完整注意力与缓存解码的等价性。
- 变长回答、padding、全忽略、极端 logits、量化常量等边界。
- 中文 BPE 往返、采样可复现、多工具交错解析。

GPU kernel 性能、分布式训练、整模型吞吐和收敛不在这些小张量测试范围内。
