import regex as re
from collections import Counter
import os
import multiprocessing
from typing import BinaryIO

GPT2_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _process_chunk(args):
    input_file, start, end, special_tokens = args
    with open(input_file, "rb") as f:
        f.seek(start)
        content = f.read(end - start)
    text = content.decode("utf-8", errors="ignore")
    if special_tokens:
        pattern = "|".join(re.escape(tok) for tok in special_tokens)

        segments = re.split(pattern, text)
    else:
        segments = [text]
    counts = Counter()
    for segment in segments:
        if not segment:
            continue
        words = re.findall(GPT2_PATTERN, segment)
        for word in words:
            counts[tuple(word.encode("utf-8"))] += 1
    return counts


def train_bpe(
    input_file: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练一个BPE模型的函数。

    参数：
        input_path (str | os.PathLike): BPE分词器训练数据的路径。
        vocab_size (int): 分词器词汇表中的总项目数（包括特殊标记）。
        special_tokens (list[str]): 要添加到分词器词汇表中的字符串特殊标记列表。
            这些字符串将永远不会被拆分成多个标记，并且始终作为单个标记保留。
            如果这些特殊标记出现在`input_path`中，它们将像其他字符串一样处理。
    返回：
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                训练好的分词器词汇表，是一个从int（词汇表中的标记ID）
                到bytes（标记字节）的映射。
            merges:
                BPE合并。每个列表项是一个bytes元组（<token1>, <token2>），
                表示<token1>与<token2>被合并。
                合并按创建顺序排序。
    """
    num_processes = max(1, multiprocessing.cpu_count())

    if special_tokens:
        split_token = special_tokens[0].encode("utf-8")
    else:
        split_token = b"\n"

    with open(input_file, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)
    pool_args = []
    for i in range(len(boundaries) - 1):
        pool_args.append((input_file, boundaries[i], boundaries[i + 1], special_tokens))
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(_process_chunk, pool_args)

    vocab_counts = Counter()
    for c in results:
        vocab_counts.update(c)

    if sum(vocab_counts.values()) == 0:
        raise ValueError("训练数据为空")

    vocab = {i: bytes([i]) for i in range(256)}

    # vocab_counts = Counter()
    # for word in words:
    #     word_bytes = word.encode("utf-8")
    #     vocab_counts[tuple(word_bytes)] += 1
    words_list = []
    counts_list = []
    for tokens, count in vocab_counts.items():
        words_list.append(list(tokens))
        counts_list.append(count)

    stats = Counter()
    indices = {}
    for idx, word in enumerate(words_list):
        freq = counts_list[idx]
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            stats[pair] += freq
        for token in word:
            if token not in indices:
                indices[token] = set()
            indices[token].add(idx)

    merges = []

    num_merges = vocab_size - len(vocab) - len(special_tokens)

    current_token_id = 256

    for i in range(num_merges):
        if not stats:
            break

        best_pair = max(stats.keys(), key=lambda p: (stats[p], vocab[p[0]], vocab[p[1]]))
        p0, p1 = best_pair

        token_bytes_1 = vocab[p0]
        token_bytes_2 = vocab[p1]
        merges.append((token_bytes_1, token_bytes_2))

        vocab[current_token_id] = token_bytes_1 + token_bytes_2

        if p0 == p1:
            candidate_indices = list(indices.get(p0, set()))
        else:
            set_p0 = indices.get(p0, set())
            set_p1 = indices.get(p1, set())
            candidate_indices = list(set_p0.intersection(set_p1))

        for w_idx in candidate_indices:
            word = words_list[w_idx]
            freq = counts_list[w_idx]

            i = 0
            changed = False
            while i < len(word) - 1:
                if word[i] == p0 and word[i + 1] == p1:
                    if i > 0:
                        prev = word[i - 1]
                        stats[(prev, p0)] -= freq
                        if stats[(prev, p0)] == 0:
                            del stats[(prev, p0)]

                    if i < len(word) - 2:
                        next_t = word[i + 2]
                        stats[(p1, next_t)] -= freq
                        if stats[(p1, next_t)] == 0:
                            del stats[(p1, next_t)]

                    stats[(p0, p1)] -= freq
                    if stats[(p0, p1)] == 0:
                        del stats[(p0, p1)]

                    word[i] = current_token_id
                    del word[i + 1]
                    changed = True

                    if i > 0:
                        prev = word[i - 1]
                        stats[(prev, current_token_id)] += freq

                    if i < len(word) - 1:
                        next_t = word[i + 1]
                        stats[(current_token_id, next_t)] += freq
                else:
                    i += 1

            if changed:
                if current_token_id not in indices:
                    indices[current_token_id] = set()
                indices[current_token_id].add(w_idx)

                # Update indices for merged parts if they no longer exist in word
                if p0 not in word:
                    if p0 in indices and w_idx in indices[p0]:
                        indices[p0].remove(w_idx)
                        if not indices[p0]:
                            del indices[p0]
                if p1 not in word:
                    if p1 in indices and w_idx in indices[p1]:
                        indices[p1].remove(w_idx)
                        if not indices[p1]:
                            del indices[p1]

        current_token_id += 1
    # for i in range(num_merges):
    #     pair_counts = Counter()
    #     for token_ids, count in vocab_counts.items():
    #         if len(token_ids) < 2:
    #             continue
    #         for j in range(len(token_ids) - 1):
    #             pair = (token_ids[j], token_ids[j + 1])
    #             pair_counts[pair] += count

    #     if not pair_counts:
    #         break

    #     best_pair = max(pair_counts.keys(), key = lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]]))
    #     p0, p1 = best_pair

    #     token_bytes_1 = vocab[p0]
    #     token_bytes_2 = vocab[p1]
    #     merges.append((token_bytes_1, token_bytes_2))

    #     vocab[current_token_id] = token_bytes_1 + token_bytes_2

    #     new_vocab_counts = Counter()
    #     for token_ids, count in vocab_counts.items():
    #         if len(token_ids) < 2:
    #             new_vocab_counts[token_ids] += count
    #             continue

    #         new_ids = []
    #         j = 0
    #         n = len(token_ids)
    #         while j < n:
    #             if j < n - 1 and token_ids[j] == p0 and token_ids[j+1] == p1:
    #                 new_ids.append(current_token_id)
    #                 j += 2
    #             else:
    #                 new_ids.append(token_ids[j])
    #                 j += 1
    #         new_vocab_counts[tuple(new_ids)] += count

    #     vocab_counts = new_vocab_counts
    #     current_token_id += 1

    for st in special_tokens:
        vocab[current_token_id] = st.encode("utf-8")
        current_token_id += 1
    return vocab, merges
