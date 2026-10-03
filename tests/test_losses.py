"""新增损失函数在 FP32 正常输入下的公式对照。"""

import pytest
import torch

from loss.DAPOLoss import dapo_loss
from loss.GSPOLoss import gspo_loss
from loss.InfoNCELoss import info_nce_loss
from loss.KLDivergence import sampled_kl_divergence


def test_info_nce_formula():
    queries = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    keys = torch.tensor([[1., 1.], [0., 2.], [2., 1.]])
    normalized_queries = queries / queries.norm(dim=-1, keepdim=True)
    normalized_keys = keys / keys.norm(dim=-1, keepdim=True)
    similarities = normalized_queries @ normalized_keys.T / 0.3
    expected = -similarities.log_softmax(dim=-1).diagonal().mean()
    torch.testing.assert_close(info_nce_loss(queries, keys, temperature=0.3), expected)


def test_dapo_asymmetric_clipping():
    old = torch.zeros(4, 1)
    new = torch.tensor([[1.4], [0.5], [1.4], [0.5]]).log()
    advantages = torch.tensor([1., 1., -1., -1.])
    # 代理目标为 [1.3, 0.5, -1.4, -0.8]。
    expected = torch.tensor(0.1)
    torch.testing.assert_close(dapo_loss(old, new, advantages,
                                        epsilon_low=0.2, epsilon_high=0.3), expected)


def test_gspo_sequence_ratio():
    old = torch.zeros(1, 2)
    new = torch.tensor([[4., 0.25]]).log()
    advantages = torch.ones(1)
    # exp((log 4+log 0.25)/2)=1，序列损失为 -1。
    torch.testing.assert_close(gspo_loss(old, new, advantages), torch.tensor(-1.))


def test_dapo_and_gspo_length_weighting():
    log_probs = torch.zeros(2, 3)
    advantages = torch.tensor([1., -1.])
    mask = torch.tensor([[1., 0., 0.], [1., 1., 1.]])
    # DAPO 对四个有效 token 平均；GSPO 对两条回答平均。
    torch.testing.assert_close(dapo_loss(log_probs, log_probs, advantages, mask), torch.tensor(0.5))
    torch.testing.assert_close(gspo_loss(log_probs, log_probs, advantages, mask), torch.tensor(0.))


@pytest.mark.parametrize("estimator", ["k1", "k2", "k3"])
def test_sampled_kl_formula(estimator):
    p = torch.tensor([0.6, 0.3, 0.1])
    q = torch.tensor([0.2, 0.3, 0.5])
    log_ratio = (q / p).log()
    expected = {
        "k1": -log_ratio,
        "k2": 0.5 * log_ratio.square(),
        "k3": q / p - log_ratio - 1,
    }[estimator].mean()
    torch.testing.assert_close(sampled_kl_divergence(p.log(), q.log(), estimator), expected)
