"""教学版 byte-level BPE：基础词表覆盖所有 256 个 UTF-8 字节。

可无损处理中文、未见字符和空白。不添加词尾标记，也不使用 split() 丢空格。
这不是完整 GPT-2 tokenizer：没有正则预分词、特殊 token 或训练文件格式。
"""

from collections import Counter


class ByteBPETokenizer:
    """每轮合并训练语料中出现最频繁的相邻 token 对。

    同频率时按 token ID 对的字典序决定，训练结果可复现。
    编码遵守学习到的 merge rank，而不是重新按待编码文本中的频率合并。
    每条训练文本独立统计，合并不能跨文档边界；文档内部允许合并空白。
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []
        self._merge_ids = {}
        self._ranks = {}

    @staticmethod
    def _merge(tokens, pair, new_id):
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

        texts 是 str 或 str iterable；num_merges>=0，min_frequency>=1。
        """
        texts = [texts] if isinstance(texts, str) else texts
        self._reset()
        sequences = [list(text.encode("utf-8")) for text in texts]
        for _ in range(num_merges):
            counts = Counter(pair for tokens in sequences for pair in zip(tokens, tokens[1:]))
            if not counts:
                break
            pair = min(counts, key=lambda value: (-counts[value], value))
            if counts[pair] < min_frequency:
                break
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self._ranks[pair] = len(self.merges)
            self._merge_ids[pair] = new_id
            self.merges.append(pair)
            sequences = [self._merge(tokens, pair, new_id) for tokens in sequences]
        return self

    def encode(self, text):
        """text 为 str；返回 token ID 列表，空串返回 []，未见字节无 OOV。"""
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            available = {pair for pair in zip(tokens, tokens[1:]) if pair in self._ranks}
            if not available:
                break
            pair = min(available, key=self._ranks.__getitem__)
            tokens = self._merge(tokens, pair, self._merge_ids[pair])
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
