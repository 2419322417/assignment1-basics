class tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        从给定的词汇表、合并规则列表和（可选的）特殊 token 列表构建分词器
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        从序列化的词汇表和合并规则文件构建并返回一个Tokenizer实例（格式与BPE训练的输出一致），并（可选地）加载特殊token列表
        """
        # 读取vocab文件
        with open(vocab_filepath, encoding="utf-8") as vocab_f:
            vocab = {}
            for line in vocab_f:
                token, token_id = line.strip().split()
                vocab[token] = int(token_id)

        # 读取merges文件
        with open(merges_filepath, encoding="utf-8") as merges_f:
            merges = [tuple(line.strip().split()) for line in merges_f if line.strip() and not line.startswith("#")]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        将输入字符串分词为token ID列表
        """
        # 初始分词为单字节tokens
        tokens = [bytes([b]) for b in text.encode("utf-8")]
        # 应用BPE合并规则
        for merge in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == merge:
                    tokens[i] = tokens[i] + tokens[i + 1]
                    del tokens[i + 1]
                else:
                    i += 1
        # 将tokens映射到token IDs
        token_ids = [self.vocab[token] for token in tokens if token in self.vocab]
        return token_ids

    def encode_iterable(self, iterable):
        """
        给定一个可迭代的字符串（例如Python文件句柄），返回一个生成器，惰性地输出token ID。
        这是为了对无法直接载入内存的大文件进行内存高效的分词。
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        将token ID列表解码回文本字符串.
        """
        # 反转词汇表以进行ID到token的映射
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = [id_to_token[id] for id in ids if id in id_to_token]
        # 将tokens连接并解码为字符串
        text = b"".join(tokens).decode("utf-8", errors="replace")
        return text
