"""验证现有组件与参考计算的数值、梯度及边界行为。"""

import pytest
import torch
import torch.nn.functional as F

from ffn.MoE import MoE
from normalization.LayerNorm import LayerNorm
from normalization.RMSNorm import RMSNorm
from peft.LoRALinear import LoRALinear


def test_layernorm_matches_pytorch_and_gradcheck():
    torch.manual_seed(10)
    layer = LayerNorm(5).double()
    x = torch.randn(2, 3, 5, dtype=torch.double, requires_grad=True)
    with torch.no_grad():
        layer.gamma.uniform_(0.5, 1.5)
        layer.beta.normal_()
    expected = F.layer_norm(x, (5,), layer.gamma, layer.beta, layer.eps)
    torch.testing.assert_close(layer(x), expected)
    assert torch.autograd.gradcheck(layer, (x,))


def test_rmsnorm_keeps_double_precision_and_gradcheck():
    layer = RMSNorm(4).double()
    x = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    expected = x / torch.sqrt(x.square().mean(-1, keepdim=True) + layer.eps)
    torch.testing.assert_close(layer(x), expected)
    assert torch.autograd.gradcheck(layer, (x,))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("norm_class", [LayerNorm, RMSNorm])
def test_normalization_low_precision_is_finite(dtype, norm_class):
    layer = norm_class(4)
    # FP16 的平方会溢出；统计量应先提升到 FP32。
    x = torch.tensor([[1000.0, -1000.0, 900.0, -900.0]], dtype=dtype)
    result = layer(x)
    assert result.dtype == dtype
    assert torch.isfinite(result).all()


def test_moe_noncontiguous_input_matches_dense_reference_and_gradients():
    torch.manual_seed(12)
    moe = MoE(4, 3, 2).double()
    x = torch.randn(3, 2, 4, dtype=torch.double).transpose(0, 1).requires_grad_()
    assert not x.is_contiguous()
    flat = x.reshape(-1, 4)
    scores, indices = moe.router(flat).topk(2, dim=-1)
    weights = torch.zeros(flat.size(0), 3, dtype=x.dtype)
    weights = weights.scatter(-1, indices, scores.softmax(-1))
    all_outputs = torch.stack([expert(flat) for expert in moe.experts], dim=1)
    expected = (all_outputs * weights.unsqueeze(-1)).sum(1).reshape_as(x)
    actual = moe(x)
    torch.testing.assert_close(actual, expected)
    params = (x, moe.router.weight, moe.experts[0][0].weight)
    actual_grads = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), params)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_lora_initial_output_and_gradient_through_frozen_base():
    torch.manual_seed(13)
    layer = LoRALinear(4, 3, rank=2).double()
    x = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    torch.testing.assert_close(layer(x), layer.weight(x))
    with torch.no_grad():
        layer.lora_b.weight.normal_()
    layer(x).square().sum().backward()
    assert layer.weight.weight.grad is None
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert layer.lora_a.weight.grad.abs().sum() > 0
    assert layer.lora_b.weight.grad.abs().sum() > 0


def test_lora_merge_matches_eval_and_preserves_original():
    torch.manual_seed(14)
    layer = LoRALinear(4, 3, rank=2, dropout=0.3).double().eval()
    with torch.no_grad():
        layer.lora_b.weight.normal_()
    original_weight = layer.weight.weight.detach().clone()
    x = torch.randn(2, 5, 4, dtype=torch.double)
    merged = layer.merged_linear()
    torch.testing.assert_close(merged(x), layer(x))
    torch.testing.assert_close(layer.weight.weight, original_weight)
    assert not merged.weight.requires_grad
