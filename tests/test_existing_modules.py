"""原仓库组件在普通 FP32 输入上的基本验证。"""

import torch
import torch.nn.functional as F

from ffn.FFN import FFN
from ffn.SwiGLUFFN import SwiGLUFFN
from ffn.MoE import MoE
from normalization.LayerNorm import LayerNorm
from normalization.RMSNorm import RMSNorm
from peft.LoRALinear import LoRALinear


def test_layernorm_matches_pytorch():
    layer = LayerNorm(4)
    x = torch.randn(2, 3, 4)
    expected = F.layer_norm(x, (4,), layer.gamma, layer.beta, layer.eps)
    torch.testing.assert_close(layer(x), expected)


def test_rmsnorm_matches_formula():
    layer = RMSNorm(4)
    x = torch.randn(2, 3, 4)
    expected = x / torch.sqrt(x.square().mean(-1, keepdim=True) + layer.eps)
    torch.testing.assert_close(layer(x), expected)


def test_ffn_and_moe_output_shapes():
    x = torch.randn(2, 3, 4)
    for layer in (FFN(4, 8), SwiGLUFFN(4, 8), MoE(4, 3, 2)):
        assert layer(x).shape == x.shape


def test_lora_initial_output_and_trainable_parameters():
    layer = LoRALinear(4, 3, rank=2)
    x = torch.randn(2, 4)
    torch.testing.assert_close(layer(x), layer.weight(x))
    assert not layer.weight.weight.requires_grad
    assert layer.lora_a.weight.requires_grad
    assert layer.lora_b.weight.requires_grad
