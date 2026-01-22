import os
from typing import BinaryIO
import multiprocessing
import regex as re
from collections import defaultdict


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

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


def pretokenize_chunk(
    chunk: str,
    special_tokens: list[str],
    rx: re.Pattern,
) -> dict[tuple[bytes, ...], int]:
    """
    Pretokenize a text chunk into byte tuples and count their frequencies.
    Special tokens are removed before pretokenization.
    """
    frequency_table: dict[tuple[bytes, ...], int] = defaultdict(int)
    # Removing special tokens before pre-tokenization
    pattern: str = "|".join(re.escape(tok) for tok in special_tokens)
    texts: list[str] = re.split(pattern, chunk)
    for text in texts:
        # do pre-tokenization
        for m in rx.finditer(text):
            word: bytes = m.group(0).encode("utf-8")
            byte_tuple: tuple[bytes, ...] = tuple(
                word[i : i + 1] for i in range(len(word))
            )
            frequency_table[byte_tuple] += 1
    return frequency_table


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 8,
):
    """
    parameters:
        input_path: str Path to a text file with BPE tokenizer training data.
        vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
        initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
        otherwise affect BPE training.
        num_processes: int Number of chunking

    return:
        vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
        merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
        is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
        <token2>. The merges should be ordered by order of creation.
    """

    # read and chunk the text
    chunks: list[str] = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    # pretokenization
    frequency_table: dict[tuple[bytes, ...], int] = defaultdict(int)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    rx = re.compile(PAT)
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            pretokenize_chunk,
            [(chunk, special_tokens, rx) for chunk in chunks],
        )
    for sub_frequency_table in results:
        for k, v in sub_frequency_table.items():
            frequency_table[k] += v

    vocab: dict[int, bytes] = {}
    # add special tokens
    for idx, special_token in enumerate(special_tokens):
        vocab[idx] = special_token.encode("utf-8")
    # add bytes
    for i in range(256):
        idx: int = len(vocab)
        vocab[idx] = bytes([i])
    # merging
    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        # count byte pairs
        bytepair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
        for byte_tuple, freq in frequency_table.items():
            for i in range(len(byte_tuple) - 1):
                pair: tuple[bytes, bytes] = (byte_tuple[i], byte_tuple[i + 1])
                bytepair_counts[pair] += freq

        if not bytepair_counts:
            break
        # find the most frequent byte pair
        most_frequent_pair: tuple[bytes, bytes] = max(
            bytepair_counts.items(), key=lambda kv: (kv[1], kv[0])
        )[0]
        # create new vocabulary entry
        merges.append(most_frequent_pair)
        new_token: bytes = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[len(vocab)] = new_token
        # update frequency table
        new_frequency_table: dict[tuple[bytes, ...], int] = defaultdict(int)
        for byte_tuple, freq in frequency_table.items():
            i = 0
            new_byte_tuple: list[bytes] = []
            while i < len(byte_tuple):
                if (
                    i < len(byte_tuple) - 1
                    and byte_tuple[i] == most_frequent_pair[0]
                    and byte_tuple[i + 1] == most_frequent_pair[1]
                ):
                    new_byte_tuple.append(new_token)
                    i += 2
                else:
                    new_byte_tuple.append(byte_tuple[i])
                    i += 1
            new_frequency_table[tuple(new_byte_tuple)] += freq
        frequency_table = new_frequency_table

    return vocab, merges


if __name__ == "__main__":
    input_path = "data/sample.txt"
    vocab_size = 1 + 256 + 6
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens, num_processes=1)
    print("Vocabulary:")
    for idx, token in vocab.items():
        print(f"{idx}: {token}")
    print("\nMerges:")
    for merge in merges:
        print(merge)
