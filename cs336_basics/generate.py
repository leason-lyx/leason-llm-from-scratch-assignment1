# generating script for the trained model
# Features:
# Generate completions for a user-provided prompt
# (i.e., take in some x1...t and sample a completionuntil you hit an <|endoftext|> token).
# Allow the user to control the maximum number of generated tokens.
# Given a desired temperature value, apply softmax temperature scaling to the predicted next-word distributions before sampling.
# Top-p sampling (Holtzman et al., 2020; also referred to as nucleus sampling), given a user-specified threshold value.

import os
import json
import torch
from torch import Tensor
from typing import Any
from pathlib import Path
from einops import rearrange
from jaxtyping import jaxtyped, Float, Int, Bool
from beartype import beartype as typechecker

from myoperator import TransformerLM
from myoperator import softmax
from bpe_tokenizer import BPETokenizer


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@jaxtyped(typechecker=typechecker)
def decode_logits(
    logits: Float[Tensor, "vocab_size"], temperature: float, top_p: float
) -> int:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs: Float[Tensor, "vocab_size"] = softmax(sorted_logits / temperature)
    cumulative_probs: Float[Tensor, "vocab_size"] = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove: Bool[Tensor, "vocab_size"] = cumulative_probs > top_p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False
    sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)
    # sample from the filtered distribution
    next_pos = torch.multinomial(sorted_probs, num_samples=1).item()
    next_token = sorted_indices[int(next_pos)].item()
    return int(next_token)


def generate(config_path: Path, prompt: str) -> str:
    # Load configuration
    config = _load_json(config_path)
    # Determine device
    if config["device"] == "cuda" and torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device("cuda")
    elif config["device"] == "mps" and torch.backends.mps.is_available():
        print("Using MPS")
        device = torch.device("mps")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    # get tokenizer config
    tokenizer_config = config["tokenizer"]
    vocab_path = tokenizer_config["vocab_path"]
    merges_path = tokenizer_config["merges_path"]
    special_tokens = tokenizer_config["special_tokens"]
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=Path(vocab_path),
        merges_filepath=Path(merges_path),
        special_tokens=special_tokens,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer does not have an eos_token_id defined.")
    # get model config
    model_config = config["model"]
    model_checkpoint_path = model_config["checkpoint_path"]
    vocab_size = model_config["vocab_size"]
    context_length = model_config["context_length"]
    d_model = model_config["d_model"]
    num_layers = model_config["num_layers"]
    num_heads = model_config["num_heads"]
    d_ff = model_config["dff"]
    rope_theta = model_config["rope_theta"]
    if model_config["dtype"] == "float32":
        model_dtype = torch.float32
    elif model_config["dtype"] == "float16":
        model_dtype = torch.float16
    elif model_config["dtype"] == "bfloat16":
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        dtype=model_dtype,
        device=device,
    )
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Tokenize prompt using BPE tokenizer
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # get generation config
    generation_config = config["generation"]
    max_new_tokens = generation_config.get("max_new_tokens")
    temperature = generation_config.get("temperature", 1.0)
    top_p = generation_config.get("top_p", 1.0)

    # Generation loop
    generated_ids = input_tensor.tolist()[0]
    while len(generated_ids) < len(input_ids) + max_new_tokens:
        input_tensor = torch.tensor(
            [generated_ids[-context_length:]], dtype=torch.long, device=device
        )
        with torch.no_grad():
            logits = model(input_tensor)
        next_token_id = decode_logits(
            logits=logits[0, -1, :],
            temperature=temperature,
            top_p=top_p,
        )
        next_token = int(next_token_id)
        generated_ids.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break
    # Generate tokens
    generated_text = tokenizer.decode(generated_ids)
    return generated_text


if __name__ == "__main__":
    config_path = Path("config.json")
    prompt = "Once upon a time"
    generated_output = generate(config_path, prompt)
    print("Generated Output:", generated_output)
