import pytest
import torch
import torch.nn.functional as F

from components.Activation import sigmoid, silu
from components.Linear import Linear
from components.Quantization import asymmetric_quantize, dequantize, symmetric_quantize


@pytest.mark.parametrize("bias", [True, False])
def test_linear_matches_pytorch(bias):
    layer = Linear(4, 3, bias=bias)
    reference = torch.nn.Linear(4, 3, bias=bias)
    reference.load_state_dict(layer.state_dict())
    x = torch.randn(2, 5, 4, requires_grad=True)
    actual, expected = layer(x), reference(x)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.sum(), (x, layer.weight))
    expected_grad = torch.autograd.grad(expected.sum(), (x, reference.weight))
    for one, two in zip(actual_grad, expected_grad):
        torch.testing.assert_close(one, two)


@pytest.mark.parametrize("activation,reference", [(sigmoid, torch.sigmoid), (silu, F.silu)])
def test_activations_match_pytorch(activation, reference):
    x = torch.tensor([-2., -1., 0., 1., 2.], requires_grad=True)
    actual, expected = activation(x), reference(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0],
                               torch.autograd.grad(expected.sum(), x)[0])


@pytest.mark.parametrize("quantize", [symmetric_quantize, asymmetric_quantize])
def test_int8_quantization_roundtrip(quantize):
    x = torch.tensor([-2., -.3, 0., 1., 3.])
    values, scale, zero_point = quantize(x)
    assert values.dtype == torch.int8
    assert (dequantize(values, scale, zero_point) - x).abs().max() <= scale
    zero_values, zero_scale, zero_point = quantize(torch.zeros(3))
    assert torch.equal(dequantize(zero_values, zero_scale, zero_point), torch.zeros(3))
