"""手写 byte-level BPE：统计相邻 token 对，合并高频对，按学习顺序编码。"""

from collections import Counter


class ByteBPETokenizer:
    """每轮合并训练语料中出现最频繁的相邻 token 对。

    编码按学习到的合并顺序使用规则，而不是重新统计待编码文本的频率。
    每条训练文本独立统计，合并不能跨文档边界；文档内部允许合并空白。
    """

    def __init__(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []

    def merge_pair(self, tokens, pair, new_id):
        """从左到右将相邻 pair 替换为 new_id，不重叠合并。"""
        result = []
        position = 0
        while position < len(tokens):
            if position + 1 < len(tokens) and (tokens[position], tokens[position + 1]) == pair:
                result.append(new_id)
                position += 2
            else:
                result.append(tokens[position])
                position += 1
        return result

    def fit(self, texts, num_merges=256, min_frequency=2):
        """训练并返回 self；再次 fit 会重置模型。

        texts 为字符串列表；num_merges>=0，min_frequency>=1。
        """
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []
        sequences = [list(text.encode("utf-8")) for text in texts]
        for _ in range(num_merges):
            counts = Counter(pair for tokens in sequences for pair in zip(tokens, tokens[1:]))
            if not counts:
                break
            pair = max(counts, key=counts.get)
            if counts[pair] < min_frequency:
                break
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merges.append(pair)
            sequences = [self.merge_pair(tokens, pair, new_id) for tokens in sequences]
        return self

    def encode(self, text):
        """text 为 str；返回 token ID 列表，空串返回 []，未见字节无 OOV。"""
        tokens = list(text.encode("utf-8"))
        for rank, pair in enumerate(self.merges):
            tokens = self.merge_pair(tokens, pair, 256 + rank)
        return tokens

    def decode(self, token_ids):
        """token_ids 来自此词表，拼接全部字节后统一解码为 UTF-8。"""
        return b"".join(self.vocab[token_id] for token_id in token_ids).decode("utf-8")

    def tokenize(self, text):
        """返回 bytes token；单个 token 未必恰好构成完整 UTF-8 字符。"""
        return [self.vocab[token_id] for token_id in self.encode(text)]


if __name__ == "__main__":
    tokenizer = ByteBPETokenizer().fit(["hello hello", "你好，你好！"], num_merges=20)
    text = "你好\nhello  世界🙂"
    ids = tokenizer.encode(text)
    print("IDs:", ids)
    print("round trip:", tokenizer.decode(ids) == text)
