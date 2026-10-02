import pytest
import torch

from components.Activation import sigmoid, silu
from components.Linear import Linear
from components.Quantization import asymmetric_quantize, dequantize, symmetric_quantize


@pytest.mark.parametrize("bias", [True, False])
def test_linear_forward_and_parameter_gradients_match_torch(bias):
    torch.manual_seed(12)
    manual = Linear(4, 7, bias=bias, dtype=torch.float64)
    reference = torch.nn.Linear(4, 7, bias=bias, dtype=torch.float64)
    reference.load_state_dict(manual.state_dict())
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    actual = manual(x)
    expected = reference(other)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x.grad, other.grad)
    torch.testing.assert_close(manual.weight.grad, reference.weight.grad)
    if bias:
        torch.testing.assert_close(manual.bias.grad, reference.bias.grad)
    else:
        assert "bias" not in manual.state_dict()


@pytest.mark.parametrize("in_features,out_features", [(0, 2), (2, -1), (True, 2)])
def test_linear_rejects_invalid_dimensions(in_features, out_features):
    with pytest.raises(ValueError):
        Linear(in_features, out_features)


@pytest.mark.parametrize("implementation,reference", [(sigmoid, torch.sigmoid), (silu, torch.nn.functional.silu)])
def test_activation_values_and_gradients_including_zero(implementation, reference):
    x = torch.tensor([-1000., -30., -1., 0., 1., 30., 1000.], dtype=torch.float64, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    actual, expected = implementation(x), reference(other)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, other.grad)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation", [sigmoid, silu])
def test_activation_preserves_half_dtype_and_shape(dtype, activation):
    x = torch.tensor([-1000., 0., 1000.], dtype=dtype).reshape(1, 3)
    result = activation(x)
    assert result.shape == x.shape and result.dtype == dtype
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("quantize", [symmetric_quantize, asymmetric_quantize])
@pytest.mark.parametrize("values", [
    [-3., -.5, 0., .3, 7.], [0., 0., 0.], [5., 5., 5.], [-5., -5., -5.], [1e-40, -1e-40],
])
def test_int8_quantization_roundtrip_error_bound_and_zero(values, quantize):
    x = torch.tensor(values, dtype=torch.float64).reshape(1, -1)
    saved = x.clone()
    quantized = quantize(x)
    reconstructed = dequantize(quantized)
    assert quantized.values.dtype == torch.int8
    assert quantized.scale.dtype == torch.float32 and quantized.scale.ndim == 0
    assert quantized.zero_point.dtype == torch.int64 and quantized.zero_point.ndim == 0
    assert quantized.scale > 0 and torch.isfinite(quantized.scale)
    assert reconstructed.dtype == torch.float32 and reconstructed.shape == x.shape
    # 整数零点的 rounding 可能使一个端点饱和；逐张量总误差不超过一格。
    assert ((reconstructed - x.float()).abs() <= quantized.scale * 1.001).all()
    torch.testing.assert_close(x, saved)
    if (x == 0).any():
        assert (reconstructed[x == 0] == 0).all()


@pytest.mark.parametrize("quantize", [symmetric_quantize, asymmetric_quantize])
@pytest.mark.parametrize("maximum", [3e38, torch.finfo(torch.float32).max])
def test_int8_quantization_large_finite_range_stays_finite(quantize, maximum):
    x = torch.tensor([-maximum, 0., maximum])
    q = quantize(x)
    assert torch.isfinite(q.scale)
    assert torch.isfinite(dequantize(q)).all()


@pytest.mark.parametrize("quantize", [symmetric_quantize, asymmetric_quantize])
@pytest.mark.parametrize("x", [torch.tensor([]), torch.tensor([float("inf")]), torch.tensor([float("nan")])])
def test_quantization_rejects_empty_or_nonfinite_input(quantize, x):
    with pytest.raises(ValueError):
        quantize(x)
