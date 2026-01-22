import regex as re
import os
from collections import Counter

GPT2_PATTERN =  r"""'(?i:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(
    input_file: str,
    vocab_size: int,
    special_tokens: list[str],
)-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练一个BPE模型的函数。

    参数：
        input_path (str | os.PathLike): BPE分词器训练数据的路径。
        vocab_size (int): 分词器词汇表中的总项目数（包括特殊标记）。
        special_tokens (list[str]): 要添加到分词器词汇表中的字符串特殊标记列表。
            这些字符串将永远不会被拆分成多个标记，并且始终作为单个标记保留。
            如果这些特殊标记出现在`input_path`中，它们将像其他字符串一样处理。
    返 回：
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                训练好的分词器词汇表，是一个从int（词汇表中的标记ID）
                到bytes（标记字节）的映射。
            merges:
                BPE合并。每个列表项是一个bytes元组（<token1>, <token2>），
                表示<token1>与<token2>被合并。
                合并按创建顺序排序。
    """
    # read_size = 1024 * 1024 * 1024 * 2
    with open(input_file, "r", encoding="utf-8") as f:  
        text = f.read()
    words = re.findall(GPT2_PATTERN, text)
    assert len(words) > 0, "训练数据不能为空。"

    vocab = {i: bytes([i]) for i in range(256)}
    
    vocab_counts = Counter()
    for word in words:
        word_bytes = word.encode("utf-8")
        vocab_counts[tuple(word_bytes)] += 1

    merges = []

    num_merges = vocab_size - len(vocab) - len(special_tokens)

    current_token_id = 256
    for i in range(num_merges):
        pair_counts = Counter()
        for token_ids, count in vocab_counts.items():
            if len(token_ids) < 2:
                continue
            for j in range(len(token_ids) - 1):
                pair = (token_ids[j], token_ids[j + 1])
                pair_counts[pair] += count
        
        if not pair_counts:
            break

        best_pair, _ = pair_counts.most_common(1)[0]
        p0, p1 = best_pair

        token_bytes_1 = vocab[p0]
        token_bytes_2 = vocab[p1]
        merges.append((token_bytes_1, token_bytes_2))

        vocab[current_token_id] = token_bytes_1 + token_bytes_2

        new_vocab_counts = Counter()
        for token_ids, count in vocab_counts.items():
            if len(token_ids) < 2:
                new_vocab_counts[token_ids] += count
                continue
            
            new_ids = []
            j = 0
            n = len(token_ids)
            while j < n:
                if j < n - 1 and token_ids[j] == p0 and token_ids[j+1] == p1:
                    new_ids.append(current_token_id)
                    j += 2
                else:
                    new_ids.append(token_ids[j])
                    j += 1
            new_vocab_counts[tuple(new_ids)] += count
            
        vocab_counts = new_vocab_counts
        current_token_id += 1
        
    for st in special_tokens:
        vocab[current_token_id] = st.encode("utf-8")
        current_token_id += 1
        
    return vocab, merges
