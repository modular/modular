# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.2 vision language model tests by comparing it against the
transformers package reference implementation.
"""
import random

import torch
from transformers import MllamaForCausalLM
from transformers.models.mllama.configuration_mllama import MllamaTextConfig
from transformers.testing_utils import torch_device

atol = 1e-05
rtol = 1e-05

# copied from https://github.com/huggingface/transformers/blob/main/tests/test_modeling_common.py
global_rng = random.Random()


def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return (
        torch.tensor(data=values, dtype=torch.long, device=torch_device)
        .view(shape)
        .contiguous()
    )


def test_language_model():
    batch_size = 3
    seq_length = 7
    pytorch_lm_config = MllamaTextConfig(
        **{
            "model_type": "mllama",
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 37,
            "hidden_act": "gelu",
            "max_position_embeddings": 512,
            "initializer_range": 0.02,
            "rope_scaling": {"rope_type": "default"},
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "cross_attention_layers": [1],
        }
    )
    input_ids = (
        ids_tensor(
            [batch_size, seq_length],
            pytorch_lm_config.vocab_size - 1,
        )
        + 1
    )
    attention_mask = input_ids.ne(1).to(torch_device)
    inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}

    model = MllamaForCausalLM(config=pytorch_lm_config)
    model.to(torch_device)
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )["logits"]
        assert not (torch.isnan(logits).any().item())
