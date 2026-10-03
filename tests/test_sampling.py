import torch

from generation.Sampling import filter_logits, sample_logits


def test_temperature_and_candidate_filters():
    logits = torch.tensor([.5, .3, .2]).log()
    snapshot = logits.clone()
    torch.testing.assert_close(filter_logits(logits, temperature=2).softmax(-1),
                               (logits / 2).softmax(-1))
    assert torch.isfinite(filter_logits(logits, top_k=2)).tolist() == [True, True, False]
    assert torch.isfinite(filter_logits(logits, top_p=.6)).tolist() == [True, True, False]
    torch.testing.assert_close(logits, snapshot)


def test_greedy_and_random_sampling():
    logits = torch.tensor([[0., 1., 2.], [2., 1., 0.]])
    assert sample_logits(logits, greedy=True).tolist() == [2, 0]
    assert sample_logits(logits, top_k=1).tolist() == [2, 0]
    assert sample_logits(logits[0], greedy=True).item() == 2
    torch.manual_seed(5)
    first = sample_logits(logits, top_p=.9)
    torch.manual_seed(5)
    assert torch.equal(first, sample_logits(logits, top_p=.9))
