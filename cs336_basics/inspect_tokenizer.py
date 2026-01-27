import os
import json
import random
import time
from bpe_tokenizer import BPETokenizer

tokenizer_tiny = BPETokenizer.from_files(
    vocab_filepath=os.path.join("tinystories_tokenizer", "vocab.pkl"),
    merges_filepath=os.path.join("tinystories_tokenizer", "merges.pkl"),
    special_tokens=["<|endoftext|>"],
)

tokenizer_owt = BPETokenizer.from_files(
    vocab_filepath=os.path.join("owt_tokenizer", "vocab.pkl"),
    merges_filepath=os.path.join("owt_tokenizer", "merges.pkl"),
    special_tokens=[],
)


# calculate average token length in characters for both tokenizers
def average_token_length(tokenizer: BPETokenizer, vocab_size) -> float:
    total_length = 0
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])
        total_length += len(token_str)
    return total_length / vocab_size


avg_length_tiny = average_token_length(tokenizer_tiny, vocab_size=10000)
avg_length_owt = average_token_length(tokenizer_owt, vocab_size=32000)
print(f"Average token length (TinyStories tokenizer): {avg_length_tiny:.2f}")
print(f"Average token length (OpenWebText tokenizer): {avg_length_owt:.2f}")


# sample and print some tokens from both tokenizers
def sample_tokens(tokenizer: BPETokenizer, num_tokens: int, vocab_size: int):
    for idx in range(num_tokens):
        token_id = random.randint(0, vocab_size - 1)
        token_str = tokenizer.decode([token_id])
        print(f"Token ID: {token_id}, Token: '{token_str}'")
    print()


# print(f"Sampling {20} tokens from tokenizer_tiny:")
# sample_tokens(tokenizer_tiny, num_tokens=20, vocab_size=10000)
# print(f"Sampling {20} tokens from tokenizer_owt:")
# sample_tokens(tokenizer_owt, num_tokens=20, vocab_size=32000)

# compare compression ratio on a sample text
sample_text = (
    "Once upon a time in a land far, far away, there lived a young princess who dreamed "
    "of adventure and exploring the world beyond her castle walls. Every day, she would gaze "
    "longingly at the horizon, imagining the exciting journeys that awaited her."
)
tokens_tiny = tokenizer_tiny.encode(sample_text)
tokens_owt = tokenizer_owt.encode(sample_text)
compression_ratio_tiny = len(sample_text.encode()) / len(tokens_tiny)
compression_ratio_owt = len(sample_text.encode()) / len(tokens_owt)
print(
    f"Compression ratio on English text (TinyStories tokenizer): {compression_ratio_tiny:.2f}"
)
print(
    f"Compression ratio on English text (OpenWebText tokenizer): {compression_ratio_owt:.2f}"
)

sample_text = "请各单位根据本通知，及早合理安排教学、科研、管理服务等有关工作。节假日期间，各单位应做好安全、保卫等工作，根据实际需要合理安排带值班，对各类突发情况应按规定及时报告并妥善处置。"
tokens_tiny = tokenizer_tiny.encode(sample_text)
tokens_owt = tokenizer_owt.encode(sample_text)
compression_ratio_tiny = len(sample_text.encode()) / len(tokens_tiny)
compression_ratio_owt = len(sample_text.encode()) / len(tokens_owt)
print(
    f"Compression ratio on Chinese text (TinyStories tokenizer): {compression_ratio_tiny:.2f}"
)
print(
    f"Compression ratio on Chinese text (OpenWebText tokenizer): {compression_ratio_owt:.2f}"
)


def read_file(filepath: str, maxlen: int) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read(maxlen)


sample_tiny = read_file(
    os.path.join("data", "TinyStoriesV2-GPT4-valid.txt"), maxlen=100000
)
sample_owt = read_file(os.path.join("data", "owt_valid.txt"), maxlen=100000)

tokens_tiny = tokenizer_tiny.encode(sample_tiny)
tokens_owt = tokenizer_owt.encode(sample_owt)
compression_ratio_tiny = len(sample_tiny.encode()) / len(tokens_tiny)
compression_ratio_owt = len(sample_owt.encode()) / len(tokens_owt)
print(
    f"Compression ratio on TinyStoriesV2 valid set (TinyStories tokenizer): {compression_ratio_tiny:.2f}"
)
print(
    f"Compression ratio on OpenWebText valid set (OpenWebText tokenizer): {compression_ratio_owt:.2f}"
)

tokens = tokenizer_tiny.encode(sample_owt)
compression_ratio_tiny_on_owt = len(sample_owt.encode()) / len(tokens)
print(
    f"Compression ratio on OpenWebText valid set (TinyStories tokenizer): {compression_ratio_tiny_on_owt:.2f}"
)

# estimate throughput of my tokenizer
tinystories_valid = read_file(
    os.path.join("data", "TinyStoriesV2-GPT4-valid.txt"), maxlen=-1
)
start_time = time.time()
tokens = tokenizer_tiny.encode(tinystories_valid)
end_time = time.time()
elapsed_time = end_time - start_time
throughput = len(tinystories_valid.encode()) / elapsed_time
print(
    f"Throughput of TinyStories tokenizer: {throughput:.2f} bytes/second over {elapsed_time:.2f} seconds"
)
