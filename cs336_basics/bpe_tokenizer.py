import os
from typing import BinaryIO, Iterable, Iterator
import multiprocessing
import regex as re
from collections import defaultdict
import pickle
from loguru import logger
import time
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)


PRETOKEN_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


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
) -> list[tuple[bytes, ...]]:
    """
    Pretokenize a text chunk into list of byte tuples.
    """
    pretokens: list[tuple[bytes, ...]] = []
    # Removing special tokens before pre-tokenization
    if special_tokens:
        pattern: str = "|".join(re.escape(tok) for tok in special_tokens)
        texts: list[str] = re.split(pattern, chunk)
    else:
        texts = [chunk]
    for text in texts:
        # do pre-tokenization
        for m in rx.finditer(text):
            word: bytes = m.group(0).encode("utf-8")
            byte_tuple: tuple[bytes, ...] = tuple(
                word[i : i + 1] for i in range(len(word))
            )
            pretokens.append(byte_tuple)
    return pretokens


def get_frequency_table(
    chunk: str, special_tokens: list[str], rx: re.Pattern
) -> dict[tuple[bytes, ...], int]:
    """
    Get frequency table of byte tuples from a text chunk.
    """
    frequency_table: dict[tuple[bytes, ...], int] = defaultdict(int)
    pretokens: list[tuple[bytes, ...]] = pretokenize_chunk(chunk, special_tokens, rx)
    for byte_tuple in pretokens:
        frequency_table[byte_tuple] += 1
    return frequency_table


def merge(
    byte_tuple: tuple[bytes, ...], merge_pair: tuple[bytes, bytes], merge_token: bytes
) -> tuple[bytes, ...]:
    i = 0
    new_byte_tuple: list[bytes] = []
    while i < len(byte_tuple):
        if (
            i < len(byte_tuple) - 1
            and byte_tuple[i] == merge_pair[0]
            and byte_tuple[i + 1] == merge_pair[1]
        ):
            new_byte_tuple.append(merge_token)
            i += 2
        else:
            new_byte_tuple.append(byte_tuple[i])
            i += 1
    return tuple(new_byte_tuple)


def get_pairs(
    byte_tuple: tuple[bytes, ...],
) -> tuple[tuple[bytes, bytes], ...]:
    return tuple((byte_tuple[i], byte_tuple[i + 1]) for i in range(len(byte_tuple) - 1))


class BpeTokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

        # appending special_tokens to the vocabulary if they aren’t already there
        if self.special_tokens is not None:
            for special_token in self.special_tokens:
                token_bytes = special_token.encode("utf-8")
                if token_bytes not in self.vocab.values():
                    self.vocab[len(self.vocab)] = token_bytes

        # map from bytes to token IDs for encoding
        self.byte_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self._special_token_to_id: dict[str, int] = {
            tok: self.byte_to_id[tok.encode("utf-8")] for tok in self.special_tokens
        }
        self._pretok_rx = re.compile(PRETOKEN_PATTERN)
        self._special_token_rx: re.Pattern | None = None
        if self.special_tokens:
            ordered_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(tok) for tok in ordered_tokens)
            self._special_token_rx = re.compile(pattern)
        self._merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }
        self._token_id_cache: dict[tuple[bytes, ...], tuple[int, ...]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary
        and list of merges (in the same format that your BPE training code output)
        and (optionally) a list of specialtokens
        """
        with open(vocab_filepath, "rb") as vf:
            vocab = pickle.load(vf)
        with open(merges_filepath, "rb") as mf:
            merges = pickle.load(mf)
        return cls(vocab, merges, special_tokens)

    def _split_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        if not self._special_token_rx or not text:
            return [(text, False)] if text else []

        segments: list[tuple[str, bool]] = []
        last_end = 0
        for match in self._special_token_rx.finditer(text):
            if match.start() > last_end:
                segments.append((text[last_end : match.start()], False))
            segments.append((match.group(0), True))
            last_end = match.end()
        if last_end < len(text):
            segments.append((text[last_end:], False))
        return segments

    def _apply_bpe(self, token: tuple[bytes, ...]) -> tuple[bytes, ...]:
        if len(token) < 2 or not self._merge_ranks:
            return token
        pairs = get_pairs(token)
        if not pairs:
            return token

        merge_ranks = self._merge_ranks
        while True:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank: int | None = None
            for pair in pairs:
                rank = merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            token = merge(token, best_pair, best_pair[0] + best_pair[1])
            if len(token) < 2:
                break
            pairs = get_pairs(token)
        return token

    def encode(self, text: str) -> list[int]:
        tokens: list[int] = []

        segments = self._split_special_tokens(text)
        token_cache = self._token_id_cache
        pretoken_rx = self._pretok_rx
        byte_to_id = self.byte_to_id
        for segment, is_special in segments:
            if is_special:
                tokens.append(self._special_token_to_id[segment])
                continue

            pretokens: list[tuple[bytes, ...]] = pretokenize_chunk(
                segment, [], pretoken_rx
            )
            for pretoken in pretokens:
                cached_ids = token_cache.get(pretoken)
                if cached_ids is None:
                    merged = self._apply_bpe(pretoken)
                    cached_ids = tuple(byte_to_id[byte] for byte in merged)
                    token_cache[pretoken] = cached_ids
                tokens.extend(cached_ids)
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        vocab = self.vocab
        decoded_text: str = b"".join(vocab[token_id] for token_id in ids).decode(
            "utf-8", errors="replace"
        )
        return decoded_text


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 8,
    save_dir: str | os.PathLike | None = None,
):
    """
    parameters:
        input_path: str Path to a text file with BPE tokenizer training data.
        vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
        initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
        otherwise affect BPE training.
        num_processes: int Number of processes to use for chunking and pretokenization.

    return:
        vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
        merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
        is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
        <token2>. The merges should be ordered by order of creation.
    """
    start_time = time.time()
    logger.info("Starting BPE training")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Number of processes: {num_processes}")
    if save_dir is not None:
        logger.info(f"vocabulary and merges will be saved to: {save_dir}")

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
    rx = re.compile(PRETOKEN_PATTERN)
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            get_frequency_table,
            [(chunk, special_tokens, rx) for chunk in chunks],
        )
    for sub_frequency_table in results:
        for k, v in sub_frequency_table.items():
            frequency_table[k] += v

    logger.info(
        "Completed pretokenization,time is {:.2f} seconds".format(
            time.time() - start_time
        )
    )

    vocab: dict[int, bytes] = {}
    # add special tokens
    for idx, special_token in enumerate(special_tokens):
        vocab[idx] = special_token.encode("utf-8")
    # add bytes
    for i in range(256):
        idx: int = len(vocab)
        vocab[idx] = bytes([i])
    # merging
    merges: list[tuple[bytes, bytes]] = []  # list of merges performed
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(
        int
    )  # frequency of each byte pair
    pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(
        set
    )  # tokens containing each byte pair
    token_pairs_cache: dict[tuple[bytes, ...], tuple[tuple[bytes, bytes], ...]] = (
        {}
    )  # cache of token to its byte pairs

    for byte_tuple, freq in frequency_table.items():
        pairs = token_pairs_cache.setdefault(byte_tuple, get_pairs(byte_tuple))
        for pair in pairs:
            pair_counts[pair] += freq
            pair_to_tokens[pair].add(byte_tuple)

    # Calculate total merges needed
    initial_vocab_size = len(vocab)
    total_merges = vocab_size - initial_vocab_size

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        merge_task = progress.add_task(
            "[cyan]Training BPE tokenizer...", total=total_merges
        )

        while len(vocab) < vocab_size and pair_counts:
            # find the most frequent byte pair
            most_frequent_pair: tuple[bytes, bytes] = max(
                pair_counts.items(), key=lambda kv: (kv[1], kv[0])
            )[0]

            # create new vocabulary entry
            merges.append(most_frequent_pair)
            new_token: bytes = most_frequent_pair[0] + most_frequent_pair[1]
            vocab[len(vocab)] = new_token

            # update frequency table and pair counts
            delta_freq: dict[tuple[bytes, ...], int] = defaultdict(int)
            affected_tokens = list(pair_to_tokens.get(most_frequent_pair, []))
            for byte_tuple in affected_tokens:
                freq = frequency_table.get(byte_tuple, 0)
                old_pairs = token_pairs_cache[byte_tuple]
                # subtract old pair counts
                for pair in old_pairs:
                    new_count = pair_counts.get(pair, 0) - freq
                    if new_count <= 0:
                        pair_counts.pop(pair, None)
                    else:
                        pair_counts[pair] = new_count

                new_byte_tuple = merge(byte_tuple, most_frequent_pair, new_token)
                # update delta frequencies
                delta_freq[byte_tuple] -= freq
                delta_freq[new_byte_tuple] += freq

                new_pairs = token_pairs_cache.get(new_byte_tuple)
                # add new pair counts
                if new_pairs is None:
                    new_pairs = get_pairs(new_byte_tuple)
                    token_pairs_cache[new_byte_tuple] = new_pairs
                for pair in new_pairs:
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq

            tokens_to_add: list[tuple[bytes, ...]] = []
            tokens_to_remove: list[tuple[bytes, ...]] = []
            # apply delta frequencies to frequency table
            for byte_tuple, delta in delta_freq.items():
                old_freq = frequency_table.get(byte_tuple, 0)
                new_freq = old_freq + delta
                if new_freq <= 0:
                    if old_freq > 0:
                        tokens_to_remove.append(byte_tuple)
                    frequency_table.pop(byte_tuple, None)
                else:
                    if old_freq <= 0:
                        tokens_to_add.append(byte_tuple)
                    frequency_table[byte_tuple] = new_freq

            # update pair_to_tokens mapping
            for byte_tuple in tokens_to_remove:
                pairs = token_pairs_cache[byte_tuple]
                for pair in set(pairs):
                    token_set = pair_to_tokens.get(pair)
                    if not token_set:
                        continue
                    token_set.discard(byte_tuple)
                    if not token_set:
                        pair_to_tokens.pop(pair, None)
            for byte_tuple in tokens_to_add:
                pairs = token_pairs_cache[byte_tuple]
                for pair in set(pairs):
                    pair_to_tokens[pair].add(byte_tuple)

            # Update progress bar
            progress.update(merge_task, advance=1)

    if save_dir is not None:
        logger.info(f"Saving vocabulary and merges to directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        vocab_path = os.path.join(save_dir, "vocab.pkl")
        merges_path = os.path.join(save_dir, "merges.pkl")
        with open(vocab_path, "wb") as vf:
            pickle.dump(vocab, vf)
        with open(merges_path, "wb") as mf:
            pickle.dump(merges, mf)

    elapsed_time = time.time() - start_time
    logger.info(f"BPE training completed in {elapsed_time:.2f} seconds")

    return vocab, merges


if __name__ == "__main__":

    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    num_processes = 8
    save_dir = "tinystories_tokenizer"
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=num_processes,
        save_dir=save_dir,
    )

    # print the longest tokens in the vocabulary
    longest_tokens = sorted(vocab.values(), key=len, reverse=True)[:10]
    print("Longest tokens in the vocabulary:")
    for token in longest_tokens:
        print(token)
