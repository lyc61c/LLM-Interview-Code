from tokenizer.BPE import ByteBPETokenizer


def test_bpe_roundtrip_preserves_unicode_spaces_and_unseen_characters():
    tokenizer = ByteBPETokenizer().fit(["hello hello hello", "你好，你好！", "a  b\nc"], num_merges=20)
    for text in ["", "你好  世界🙂\t\nhello", " unseen words ", "𠮷\x00é"]:
        assert tokenizer.decode(tokenizer.encode(text)) == text
        assert b"".join(tokenizer.tokenize(text)) == text.encode("utf-8")


def test_bpe_deterministic_tie_break_and_compression():
    first = ByteBPETokenizer().fit(["ab", "bc"], num_merges=1, min_frequency=1)
    second = ByteBPETokenizer().fit(["bc", "ab"], num_merges=1, min_frequency=1)
    assert first.merges == second.merges == [(ord("a"), ord("b"))]
    assert first.encode("ab") == [256]
    assert first.decode([256]) == "ab"


def test_bpe_merge_rank_takes_priority_over_encoding_frequency():
    # 学到 ab（rank 0）再学 bc（rank 1）；新文本的 bc 即使更多仍先合并 ab。
    tokenizer = ByteBPETokenizer().fit(["ab"] * 5 + ["bc"] * 4, num_merges=2)
    assert tokenizer.merges == [(97, 98), (98, 99)]
    assert tokenizer.encode("abcbcbc") == [256, 99, 257, 257]


def test_bpe_does_not_merge_across_documents_and_fit_resets():
    tokenizer = ByteBPETokenizer().fit(["a", "b"], num_merges=5, min_frequency=1)
    assert tokenizer.merges == []
    tokenizer.fit("aaaa", num_merges=1)
    assert tokenizer.merges == [(97, 97)]
    assert tokenizer.encode("aaaa") == [256, 256]
    tokenizer.fit([""], num_merges=5)
    assert len(tokenizer.vocab) == 256 and tokenizer.merges == []
