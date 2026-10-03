from tokenizer.BPE import ByteBPETokenizer


def test_bpe_compression_and_roundtrip():
    tokenizer = ByteBPETokenizer().fit(["hello hello hello", "你好，你好！"], num_merges=20)
    assert len(tokenizer.encode("hello hello")) < len("hello hello".encode("utf-8"))
    text = "你好 世界🙂  hello\n"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert b"".join(tokenizer.tokenize(text)) == text.encode("utf-8")


def test_bpe_applies_merges_in_learning_order():
    tokenizer = ByteBPETokenizer().fit(["ab"] * 5 + ["bc"] * 4, num_merges=2)
    assert tokenizer.merges == [(97, 98), (98, 99)]
    assert tokenizer.encode("abcbcbc") == [256, 99, 257, 257]
