"""损失函数的数学对照、梯度与 padding/空组回归测试。"""

import math
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from loss.DAPOLoss import dapo_loss
from loss.DPOLoss import dpo_loss
from loss.EntropyLoss import KL_divergence, cross_entropy_loss, log_softmax, softmax
from loss.GRPOLoss import compute_grpo_advantages, compute_kl_penalty, grpo_loss
from loss.GSPOLoss import gspo_loss
from loss.InfoNCELoss import InfoNCELoss, info_nce_loss
from loss.KLDivergence import kl_k1, kl_k2, kl_k3, sampled_kl_divergence
from loss.PretainLoss import PretrainLoss
from loss.SFTLoss import SFTLoss


@pytest.mark.parametrize("shape", [(5, 7), (2, 3, 7)])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_hard_and_soft_ce_match_pytorch(shape, reduction):
    torch.manual_seed(17)
    logits = torch.randn(shape, dtype=torch.double)
    hard = torch.randint(shape[-1], shape[:-1])
    probabilities = torch.randn(shape, dtype=torch.double).softmax(dim=-1)
    flat_logits = logits.reshape(-1, shape[-1])
    for targets in (hard, probabilities):
        flat_targets = targets.reshape(-1) if targets.ndim < logits.ndim else targets.reshape_as(flat_logits)
        expected = F.cross_entropy(flat_logits, flat_targets, reduction=reduction)
        if reduction == "none":
            expected = expected.reshape(shape[:-1])
        torch.testing.assert_close(cross_entropy_loss(logits, targets, reduction), expected)


def test_ce_ignore_index_and_empty_targets_have_zero_gradients():
    logits = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
    targets = torch.tensor([1, -100, 3])
    torch.testing.assert_close(cross_entropy_loss(logits, targets), F.cross_entropy(logits, targets))
    loss = cross_entropy_loss(logits, torch.full((3,), -100))
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(logits.grad, torch.zeros_like(logits))
    empty = torch.empty((0, 4), requires_grad=True)
    empty_loss = cross_entropy_loss(empty, torch.empty(0, dtype=torch.long))
    empty_loss.backward()
    assert empty_loss.item() == 0


def test_softmax_stability_and_half_precision():
    logits = torch.tensor([[1e30, 1e30, 1e30], [10000., 9999., -10000.]])
    torch.testing.assert_close(softmax(logits), F.softmax(logits, dim=-1))
    torch.testing.assert_close(log_softmax(logits), F.log_softmax(logits, dim=-1))
    half = torch.tensor([[1000., 999., 998.]], dtype=torch.float16)
    torch.testing.assert_close(softmax(half), F.softmax(half.float(), dim=-1))


def test_exact_kl_matches_pytorch_and_has_direction():
    p = torch.tensor([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]], dtype=torch.double)
    q = torch.tensor([[0.1, 0.5, 0.4], [0.5, 0.4, 0.1]], dtype=torch.double)
    expected = F.kl_div(q.log(), p, reduction="batchmean")
    torch.testing.assert_close(KL_divergence(p.log(), q.log()), expected)
    torch.testing.assert_close(KL_divergence(p.log(), p.log()), torch.zeros((), dtype=torch.double))
    assert not torch.isclose(expected, KL_divergence(q.log(), p.log()))
    torch.testing.assert_close(KL_divergence(p.log(), q.log(), "none"), (p * (p.log() - q.log())).sum(-1))


def test_sampled_kl_estimators_against_discrete_expectation():
    p = torch.tensor([0.6, 0.3, 0.1], dtype=torch.double)
    q = torch.tensor([0.2, 0.3, 0.5], dtype=torch.double)
    exact = (p * (p.log() - q.log())).sum()
    # 显式用 P 加权全部离散结果，检查采样方向与 k1/k3 的无偏性。
    for estimator in (kl_k1, kl_k3):
        values = estimator(p.log(), q.log(), reduction="none")
        torch.testing.assert_close((p * values).sum(), exact)
    k2 = kl_k2(p.log(), q.log(), reduction="none")
    torch.testing.assert_close(k2, (q.log() - p.log()).square() / 2)
    assert not torch.isclose((p * k2).sum(), exact)
    assert (kl_k3(p.log(), q.log(), reduction="none") >= 0).all()
    torch.testing.assert_close(compute_kl_penalty(p.log(), q.log()), kl_k3(p.log(), q.log()))


@pytest.mark.parametrize("estimator", ["k1", "k2", "k3"])
def test_kl_mask_and_reduction(estimator):
    p = torch.tensor([-1., -2., float("nan")], requires_grad=True)
    q = torch.tensor([-1.1, -1.7, float("inf")])
    mask = torch.tensor([True, True, False])
    values = sampled_kl_divergence(p, q, estimator, mask, "none")
    assert values[-1].item() == 0
    mean = sampled_kl_divergence(p, q, estimator, mask)
    torch.testing.assert_close(mean, values.sum() / 2)
    torch.testing.assert_close(mean, sampled_kl_divergence(p[:2], q[:2], estimator))
    mean.backward()
    assert p.grad[-1].item() == 0
    assert torch.isfinite(p.grad).all()


def test_k3_expm1_is_accurate_near_zero():
    # float32 的 exp(1e-4)-1 有明显消减误差；expm1 更接近 double 参考。
    p = torch.zeros(1, dtype=torch.float32)
    q = torch.tensor([1e-4], dtype=torch.float32)
    expected = kl_k3(p.double(), q.double()).float()
    stable = kl_k3(p, q)
    naive = (torch.exp(q - p) - (q - p) - 1).mean()
    assert abs(stable - expected) < abs(naive - expected)


@pytest.mark.parametrize("symmetric", [False, True])
def test_info_nce_matches_reference(symmetric):
    torch.manual_seed(19)
    queries, keys = torch.randn(4, 6, dtype=torch.double), torch.randn(4, 6, dtype=torch.double)
    scores = F.normalize(queries, dim=-1) @ F.normalize(keys, dim=-1).T / 0.3
    targets = torch.arange(4)
    expected = F.cross_entropy(scores, targets)
    if symmetric:
        expected = (expected + F.cross_entropy(scores.T, targets)) / 2
    torch.testing.assert_close(info_nce_loss(queries, keys, 0.3, symmetric), expected)
    torch.testing.assert_close(InfoNCELoss(0.3, symmetric)(queries, keys), expected)


def test_info_nce_pair_alignment_batch_one_and_zero_vectors():
    features = torch.eye(4, dtype=torch.double)
    assert info_nce_loss(features, features) < info_nce_loss(features, features.roll(1, 0))
    assert info_nce_loss(features[:1], features[:1]).item() == 0
    zeros = torch.zeros((3, 5), dtype=torch.double, requires_grad=True)
    loss = info_nce_loss(zeros, zeros)
    torch.testing.assert_close(loss, torch.tensor(math.log(3), dtype=torch.double))
    loss.backward()
    assert torch.isfinite(zeros.grad).all()


def test_dapo_asymmetric_clipping_for_both_advantage_signs():
    old = torch.zeros((4, 1), dtype=torch.double, requires_grad=True)
    new = torch.tensor([[1.4], [0.5], [1.4], [0.5]], dtype=torch.double).log().requires_grad_()
    advantages = torch.tensor([1., 1., -1., -1.], dtype=torch.double, requires_grad=True)
    loss = dapo_loss(old, new, advantages, epsilon_low=0.2, epsilon_high=0.3)
    torch.testing.assert_close(loss, torch.tensor(0.1, dtype=torch.double))
    loss.backward()
    torch.testing.assert_close(new.grad, torch.tensor([[0.], [-.125], [.35], [0.]], dtype=torch.double))
    assert old.grad is None and advantages.grad is None


def test_gspo_uses_geometric_sequence_ratio_and_sequence_gradients():
    old = torch.zeros((1, 2), dtype=torch.double)
    new = torch.tensor([[4., .25]], dtype=torch.double).log().requires_grad_()
    advantages = torch.ones(1, dtype=torch.double)
    loss = gspo_loss(old, new, advantages, epsilon_low=.1, epsilon_high=.1)
    torch.testing.assert_close(loss, torch.tensor(-1., dtype=torch.double))
    loss.backward()
    torch.testing.assert_close(new.grad, torch.full((1, 2), -.5, dtype=torch.double))
    token_loss = dapo_loss(old, new, advantages, epsilon_low=.1, epsilon_high=.1)
    torch.testing.assert_close(token_loss, torch.tensor(-.675, dtype=torch.double))


def test_dapo_gspo_variable_length_normalization():
    old = torch.zeros((2, 3), dtype=torch.double)
    advantages = torch.tensor([1., -1.], dtype=torch.double)
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool)
    # 同策略时 ratio=1；DAPO token等权：(1-3)/4；GSPO回答等权：(1-1)/2。
    torch.testing.assert_close(dapo_loss(old, old, advantages, mask), torch.tensor(.5, dtype=torch.double))
    torch.testing.assert_close(gspo_loss(old, old, advantages, mask), torch.tensor(0., dtype=torch.double))
    equal_lengths = torch.ones_like(mask)
    torch.testing.assert_close(dapo_loss(old, old, advantages, equal_lengths),
                               gspo_loss(old, old, advantages, equal_lengths))


@pytest.mark.parametrize("loss_fn", [dapo_loss, gspo_loss])
def test_policy_loss_padding_nan_and_empty_sequence_invariance(loss_fn):
    old = torch.tensor([[-1., -2.], [-1., -1.]], dtype=torch.double)
    new = torch.tensor([[-.99, -2.02], [-1.02, -1.01]], dtype=torch.double)
    advantages = torch.tensor([1., -1.], dtype=torch.double)
    expected = loss_fn(old, new, advantages)
    padded_old = F.pad(old, (0, 2), value=float("nan"))
    padded_new = F.pad(new, (0, 2), value=float("nan")).requires_grad_()
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)
    actual = loss_fn(padded_old, padded_new, advantages, mask)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.equal(padded_new.grad[:, 2:], torch.zeros_like(padded_new.grad[:, 2:]))
    assert torch.isfinite(padded_new.grad).all()
    # 插入全 padding 回答不会稀释已有回答损失。
    extra = torch.full((1, 4), float("nan"), dtype=torch.double)
    torch.testing.assert_close(loss_fn(torch.cat((padded_old, extra)),
                                       torch.cat((padded_new.detach(), extra)),
                                       torch.cat((advantages, torch.tensor([float("nan")], dtype=torch.double))),
                                       torch.cat((mask, torch.zeros((1, 4), dtype=torch.bool)))), expected)


@pytest.mark.parametrize("loss_fn", [dapo_loss, gspo_loss])
def test_policy_loss_all_masked_returns_zero_with_zero_gradients(loss_fn):
    values = torch.full((2, 3), float("nan"), requires_grad=True)
    mask = torch.zeros_like(values, dtype=torch.bool)
    loss = loss_fn(values.detach(), values, torch.tensor([float("nan"), float("nan")]), mask)
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(values.grad, torch.zeros_like(values))


def test_gspo_asymmetric_clipping():
    old = torch.zeros((2, 2), dtype=torch.double)
    new = torch.tensor([[1.2, 1.2], [.8, .8]], dtype=torch.double).log().requires_grad_()
    loss = gspo_loss(old, new, torch.tensor([1., -1.], dtype=torch.double),
                     epsilon_low=.05, epsilon_high=.1)
    torch.testing.assert_close(loss, torch.tensor(-.075, dtype=torch.double))
    loss.backward()
    assert torch.equal(new.grad, torch.zeros_like(new))


def test_grpo_group_edges_and_known_advantages():
    rewards = torch.tensor([[1., 1., 1.], [1., 2., 3.]], dtype=torch.double)
    result = compute_grpo_advantages(rewards)
    torch.testing.assert_close(result[0], torch.zeros(3, dtype=torch.double))
    torch.testing.assert_close(result[1], torch.tensor([-1., 0., 1.], dtype=torch.double) / (math.sqrt(2/3) + 1e-8))
    torch.testing.assert_close(compute_grpo_advantages(torch.tensor([[2.], [3.]])), torch.zeros((2, 1)))
    probabilities = torch.zeros(3)
    torch.testing.assert_close(grpo_loss(probabilities, probabilities, result[0].float()), torch.tensor(0.))


@pytest.mark.parametrize("seq_len", [0, 1, 5])
def test_causal_losses_all_ignored_and_short_sequences(seq_len):
    logits = torch.randn(2, seq_len, 7, requires_grad=True)
    labels = torch.full((2, seq_len), -100)
    for loss in (PretrainLoss()(logits, labels), SFTLoss()(logits, labels, [seq_len, seq_len])):
        assert loss.item() == 0
        gradient, = torch.autograd.grad(loss, logits)
        assert torch.equal(gradient, torch.zeros_like(logits))


def test_causal_loss_shift_prompt_mask_and_non_contiguous_logits():
    torch.manual_seed(23)
    logits = torch.randn(2, 9, 5, dtype=torch.double)[:, ::2, :].requires_grad_()
    assert not logits.is_contiguous()
    labels = torch.tensor([[1, 2, 3, -100, 0], [0, 3, 4, 1, -100]])
    original = labels.clone()
    expected = F.cross_entropy(logits[:, :-1].reshape(-1, 5), labels[:, 1:].reshape(-1))
    torch.testing.assert_close(PretrainLoss()(logits, labels), expected)
    masked = labels.clone()
    masked[0, :2] = -100
    masked[1, :4] = -100
    expected = F.cross_entropy(logits[:, :-1].reshape(-1, 5), masked[:, 1:].reshape(-1))
    torch.testing.assert_close(SFTLoss()(logits, labels, [2, 4]), expected)
    assert torch.equal(labels, original)


def test_dpo_stable_smoothing_and_reference_behavior():
    same = torch.tensor([-3., -4.], dtype=torch.double)
    for smoothing in (0., .1, .5):
        torch.testing.assert_close(dpo_loss(same, same, same, same, label_smoothing=smoothing),
                                   torch.tensor(math.log(2), dtype=torch.double))
    chosen, rejected = torch.tensor([1000.]), torch.tensor([-1000.])
    zeros = torch.zeros(1)
    loss = dpo_loss(chosen, rejected, zeros, zeros)
    assert torch.isfinite(loss)
    smoothed = dpo_loss(chosen, rejected, zeros, zeros, label_smoothing=.1)
    assert smoothed > loss


@pytest.mark.parametrize("name", ["ce", "infonce", "k3", "dapo", "gspo"])
def test_loss_gradcheck(name):
    torch.manual_seed(29)
    if name == "ce":
        values = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
        targets = torch.tensor([[.2, .3, .5], [.4, .5, .1]], dtype=torch.double)
        fn = lambda x: cross_entropy_loss(x, targets)
    elif name == "infonce":
        values = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
        keys = torch.randn(3, 4, dtype=torch.double)
        fn = lambda x: info_nce_loss(x, keys, temperature=.7, symmetric=True)
    elif name == "k3":
        values = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
        reference = torch.randn_like(values)
        fn = lambda x: kl_k3(x, reference)
    else:
        values = torch.tensor([[.03, -.02], [-.05, .01]], dtype=torch.double, requires_grad=True)
        old = torch.zeros_like(values)
        advantages = torch.tensor([1., -1.], dtype=torch.double)
        loss_fn = dapo_loss if name == "dapo" else gspo_loss
        fn = lambda x: loss_fn(old, x, advantages, epsilon_low=.2, epsilon_high=.3)
    assert torch.autograd.gradcheck(fn, (values,))


@pytest.mark.parametrize("call", [
    lambda: cross_entropy_loss(torch.randn(2, 3), torch.ones((2, 3))),
    lambda: cross_entropy_loss(torch.randn(2, 3), torch.tensor([3, 0])),
    lambda: info_nce_loss(torch.randn(2, 3), torch.randn(2, 3), temperature=0),
    lambda: dapo_loss(torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(3)),
    lambda: gspo_loss(torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2, 3)),
    lambda: kl_k3(torch.ones(2), torch.ones(3)),
    lambda: kl_k3(torch.ones(2), torch.ones(2), mask=torch.tensor([1., .5])),
    lambda: sampled_kl_divergence(torch.ones(2), torch.ones(2), "unknown"),
    lambda: compute_grpo_advantages(torch.empty(2, 0)),
    lambda: SFTLoss()(torch.randn(2, 3, 4), torch.zeros(2, 3, dtype=torch.long), [4, 2]),
    lambda: dpo_loss(torch.ones(2), torch.ones(1), torch.ones(2), torch.ones(2)),
    lambda: dpo_loss(*(torch.ones(2) for _ in range(4)), label_smoothing=-.1),
])
def test_invalid_inputs_raise_actionable_errors(call):
    with pytest.raises(ValueError):
        call()


def test_ppo_import_has_no_plotting_dependency_or_side_effect():
    # 子进程中令 matplotlib/numpy 的 import 失败，确保它们只在绘图函数中加载。
    code = """
import builtins
import torch
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.startswith('matplotlib') or name == 'numpy':
        raise RuntimeError('PPO import must not load plotting dependencies')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from loss.PPOLoss import ppo_clip_loss
assert ppo_clip_loss(torch.zeros(1), torch.zeros(1), torch.ones(1)).item() == -1
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("loss_fn,expected", [(dapo_loss, -1.28), (gspo_loss, -1.0004)])
def test_positive_policy_clipping_avoids_overflow_in_backward(loss_fn, expected):
    # exp(1000)=Inf，正优势的目标实际已经被上界 clip 到有限值。
    old = torch.tensor([[-1000.]])
    new = torch.tensor([[0.]], requires_grad=True)
    loss = loss_fn(old, new, torch.ones(1))
    torch.testing.assert_close(loss, torch.tensor(expected))
    loss.backward()
    assert torch.equal(new.grad, torch.zeros_like(new))


@pytest.mark.parametrize("loss_fn", [dapo_loss, gspo_loss])
def test_zero_policy_advantage_does_not_multiply_zero_by_infinity(loss_fn):
    old = torch.tensor([[-1000.]])
    new = torch.tensor([[0.]], requires_grad=True)
    loss = loss_fn(old, new, torch.zeros(1))
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(new.grad, torch.zeros_like(new))


@pytest.mark.parametrize("loss_fn", [dapo_loss, gspo_loss])
def test_negative_policy_advantage_rejects_real_objective_overflow(loss_fn):
    # A<0 时 min 选择未截断的大比率，不能为“稳定”把原目标偷偷上界裁掉。
    with pytest.raises(ValueError, match="可表示范围"):
        loss_fn(torch.tensor([[-1000.]]), torch.tensor([[0.]], requires_grad=True), -torch.ones(1))


@pytest.mark.parametrize("ignored_value", [float("-inf"), float("inf"), float("nan")])
def test_ce_cleans_ignored_rows_before_softmax(ignored_value):
    values = torch.tensor([[1., 2., 3.], [ignored_value] * 3], requires_grad=True)
    targets = torch.tensor([2, -100])
    loss = cross_entropy_loss(values, targets)
    torch.testing.assert_close(loss, F.cross_entropy(values[:1], targets[:1]))
    loss.backward()
    assert torch.isfinite(values.grad).all()
    assert torch.equal(values.grad[1], torch.zeros_like(values.grad[1]))
    all_ignored = torch.full((1, 3), ignored_value, requires_grad=True)
    zero = cross_entropy_loss(all_ignored, torch.tensor([-100]))
    zero.backward()
    assert zero.item() == 0
    assert torch.equal(all_ignored.grad, torch.zeros_like(all_ignored))


@pytest.mark.parametrize("invalid", [float("-inf"), float("inf"), float("nan")])
def test_effective_ce_rows_must_define_a_distribution(invalid):
    logits = torch.full((1, 3), invalid)
    for targets in (torch.tensor([1]), torch.tensor([[.2, .5, .3]])):
        with pytest.raises(ValueError):
            cross_entropy_loss(logits, targets)
    # 单个屏蔽类的 -Inf 则是合法分布；软标签在该类必须为 0 才是有限 CE。
    logits = torch.tensor([[0., float("-inf"), 0.]], requires_grad=True)
    loss = cross_entropy_loss(logits, torch.tensor([[.5, 0., .5]]))
    torch.testing.assert_close(loss, torch.tensor(math.log(2)))
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("loss_type", ["pretrain", "sft"])
@pytest.mark.parametrize("ignored_value", [float("-inf"), float("inf"), float("nan")])
def test_causal_ce_nonfinite_ignored_logits_have_zero_gradient(loss_type, ignored_value):
    # shifted label t=1 被忽略，对应预测 logits t=0；最后一个 logits 无需预测。
    logits = torch.tensor([[[ignored_value] * 3, [1., 2., 3.], [ignored_value] * 3]],
                          requires_grad=True)
    labels = torch.tensor([[0, -100, 2]])
    if loss_type == "pretrain":
        loss = PretrainLoss()(logits, labels)
    else:
        loss = SFTLoss()(logits, labels, [2])
    torch.testing.assert_close(loss, F.cross_entropy(logits[:, 1, :], labels[:, 2]))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[:, [0, 2], :], torch.zeros_like(logits.grad[:, [0, 2], :]))
    all_ignored = torch.full((1, 3, 3), ignored_value, requires_grad=True)
    if loss_type == "pretrain":
        zero = PretrainLoss()(all_ignored, torch.full((1, 3), -100))
    else:
        zero = SFTLoss()(all_ignored, torch.tensor([[0, 1, 2]]), [3])
    zero.backward()
    assert zero.item() == 0
    assert torch.equal(all_ignored.grad, torch.zeros_like(all_ignored))


@pytest.mark.parametrize("loss_fn", [dapo_loss, gspo_loss, kl_k3])
def test_complex_masks_are_rejected(loss_fn):
    old = torch.zeros((2, 3))
    mask = torch.ones((2, 3), dtype=torch.complex64)
    with pytest.raises(ValueError, match="复数"):
        if loss_fn == kl_k3:
            loss_fn(old, old, mask=mask)
        else:
            loss_fn(old, old, torch.ones(2), mask=mask)


def test_info_nce_preserves_float32_computation_under_autocast():
    torch.manual_seed(31)
    queries = torch.randn(4, 7, requires_grad=True)
    keys = torch.randn(4, 7, requires_grad=True)
    expected = info_nce_loss(queries, keys, temperature=1e-4, symmetric=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = info_nce_loss(queries, keys, temperature=1e-4, symmetric=True)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(queries.grad).all() and torch.isfinite(keys.grad).all()


def test_info_nce_rejects_unrepresentable_temperature_and_feature_norm():
    features = torch.eye(2)
    with pytest.raises(ValueError, match="temperature"):
        info_nce_loss(features, features, temperature=1e-50)
    large_features = torch.full((2, 3), 1e30)
    with pytest.raises(ValueError, match="范数"):
        info_nce_loss(large_features, large_features)
