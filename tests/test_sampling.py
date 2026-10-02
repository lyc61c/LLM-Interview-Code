import pytest
import torch

from generation.Sampling import filter_logits, sample_logits


def test_sampling_multidimensional_shape_greedy_and_no_mutation():
    torch.manual_seed(5)
    logits = torch.randn(2, 3, 7)
    snapshot = logits.clone()
    result = sample_logits(logits, greedy=True, top_k=3, top_p=.7)
    assert result.shape == (2, 3) and result.dtype == torch.long
    assert torch.equal(result, logits.argmax(-1))
    torch.testing.assert_close(logits, snapshot)
    assert sample_logits(logits[0, 0], greedy=True).shape == ()


def test_top_p_retains_threshold_crossing_token_and_at_least_one():
    logits = torch.tensor([.5, .3, .2]).log()
    filtered = filter_logits(logits, top_p=.6)
    assert torch.isfinite(filtered).tolist() == [True, True, False]
    assert torch.isfinite(filter_logits(logits, top_p=.1)).tolist() == [True, False, False]
    assert torch.isfinite(filter_logits(logits, top_p=.5)).tolist() == [True, False, False]


def test_top_k_handles_ties_and_combines_with_top_p():
    logits = torch.zeros(2, 8)
    assert torch.isfinite(filter_logits(logits, top_k=3)).sum(-1).tolist() == [3, 3]
    assert sample_logits(logits, greedy=True, top_k=1).tolist() == [0, 0]
    logits = torch.tensor([5., 4., 3., 2.])
    assert torch.isfinite(filter_logits(logits, top_k=2, top_p=.9)).tolist() == [True, True, False, False]


def test_generator_is_reproducible_and_never_samples_filtered_ids():
    logits = torch.tensor([0., 1., 2., 3.]).expand(100, 4)
    one = torch.Generator().manual_seed(19)
    two = torch.Generator().manual_seed(19)
    first = sample_logits(logits, top_k=2, generator=one)
    second = sample_logits(logits, top_k=2, generator=two)
    assert torch.equal(first, second)
    assert (first >= 2).all()
    assert len(first.unique()) == 2


def test_temperature_distribution_and_extreme_values():
    logits = torch.tensor([1000., 1001., -torch.inf])
    actual = filter_logits(logits, temperature=2).softmax(-1)
    torch.testing.assert_close(actual, (logits / 2).softmax(-1))
    assert sample_logits(logits, temperature=1e-300).item() == 1
    huge = filter_logits(logits, temperature=1e300).softmax(-1)
    torch.testing.assert_close(huge, torch.tensor([.5, .5, 0.], dtype=huge.dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_high_temperature_scales_before_subtracting_extreme_logits(dtype):
    maximum = torch.finfo(dtype).max
    logits = torch.tensor([-maximum, maximum], dtype=dtype)
    snapshot = logits.clone()
    probabilities = filter_logits(logits, temperature=maximum).softmax(-1)
    expected = torch.tensor([-1., 1.], dtype=dtype).softmax(-1)
    torch.testing.assert_close(probabilities, expected)
    torch.testing.assert_close(logits, snapshot)


@pytest.mark.parametrize("kwargs", [
    {"temperature": 0}, {"temperature": float("nan")}, {"temperature": float("inf")},
    {"top_k": 0}, {"top_k": 4}, {"top_k": 1.5}, {"top_p": 0}, {"top_p": 1.1},
])
def test_sampling_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        sample_logits(torch.ones(3), **kwargs)


@pytest.mark.parametrize("logits", [
    torch.tensor([float("nan"), 0.]), torch.tensor([float("inf"), 0.]),
    torch.tensor([[-torch.inf, -torch.inf], [0., 0.]]), torch.tensor([]),
])
def test_sampling_rejects_invalid_logits(logits):
    with pytest.raises(ValueError):
        sample_logits(logits)
