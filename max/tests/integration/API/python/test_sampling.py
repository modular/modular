# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import json

import numpy as np
import torch
import xgrammar as xgr
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.pipelines.lib import SamplingConfig, rejection_sampler, token_sampler
from transformers import AutoConfig, AutoTokenizer


def test_bitmask_sampling_vs_xgrammar(session: InferenceSession):
    # Get Tokenizer and Model Info
    model_id = "modularai/llama-3.1"
    config = AutoConfig.from_pretrained(model_id)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        AutoTokenizer.from_pretrained(model_id), vocab_size=config.vocab_size
    )

    # Create a grammar compiler for the tokenizer
    compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)

    # Compile the grammar for a sample schema.
    person_schema = {
        "title": "Person",
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {
                "type": "integer",
            },
        },
        "required": ["name", "age"],
    }
    compiled_grammar = compiler.compile_json_schema(json.dumps(person_schema))

    # Instantiate grammar matcher
    matcher = xgr.GrammarMatcher(compiled_grammar)

    # Create one op sampling graph
    graph = token_sampler(
        SamplingConfig(
            top_k=5,
            enable_structured_output=True,
            in_dtype=DType.float32,
            out_dtype=DType.float32,
        ),
        device=DeviceRef.CPU(),
    )

    device = session.devices[0]
    sampler = session.load(graph)

    # Variables
    batch_size = 1
    vocab_size = tokenizer_info.vocab_size
    n_trials = 1

    generated_tokens = Tensor(
        shape=(batch_size, 0),
        dtype=DType.int64,
        device=device,
    )

    for i in range(n_trials):
        # Allocate a bitmask
        token_bitmask = torch.ones(
            xgr.get_bitmask_shape(1, vocab_size),
            dtype=torch.int32,
        )

        # Update the bitmask
        matcher.fill_next_token_bitmask(token_bitmask, 0)

        # Generate Random Logits
        logits = np.random.default_rng().random(
            size=(batch_size, vocab_size), dtype=np.float32
        )

        # Unpack bitmask
        bits = 2 ** torch.arange(32, dtype=torch.int32)
        bitmask = (token_bitmask.unsqueeze(-1) & bits) != 0
        bitmask = bitmask.reshape(
            batch_size,
            vocab_size,
        ).to(torch.bool)

        # Run through Sampler
        _, new_tokens = sampler(
            Tensor.from_dlpack(logits).to(device),
            generated_tokens,  # This isnt used by the sampler, so we can safely ignore it.
            Tensor.from_dlpack(bitmask).to(device),
        )[:2]
        assert isinstance(new_tokens, Tensor)
        for token in new_tokens.to_numpy():
            assert matcher.accept_token(token[0], debug_print=True)


def test_sampling_return_logits(session: InferenceSession):
    # Create one op sampling graph
    graph = token_sampler(
        SamplingConfig(
            top_k=5,
            enable_structured_output=False,
            in_dtype=DType.float32,
            out_dtype=DType.float32,
        ),
        return_logits=True,
        device=DeviceRef.CPU(),
    )

    device = session.devices[0]
    sampler = session.load(graph)

    # Variables
    batch_size = 3
    vocab_size = 10
    generated_tokens = Tensor(
        shape=(batch_size, 0),
        dtype=DType.int64,
        device=device,
    )
    generated_logits = Tensor(
        shape=(batch_size, 0), dtype=DType.float32, device=device
    )

    # Generate Random Logits
    for j in range(3):
        logits = np.random.default_rng().random(
            size=(batch_size, vocab_size), dtype=np.float32
        )

        # Run through Sampler
        new_tokens, generated_tokens_max, generated_logits_max = sampler(
            Tensor.from_dlpack(logits).to(device),
            generated_tokens,  # This isnt used by the sampler, so we can safely ignore it.
            generated_logits,
        )[:3]
        assert isinstance(generated_tokens_max, Tensor)
        assert isinstance(generated_logits_max, Tensor)
        assert isinstance(new_tokens, Tensor)
        generated_tokens = generated_tokens_max
        generated_logits = generated_logits_max

        # Ensure that the tokens generated, match the correct logits expected.
        numpy_logits = generated_logits.to_numpy()
        for i, token_idx in enumerate(new_tokens.to_numpy()):
            assert numpy_logits[i, j] == logits[i, token_idx]


def test_rejection_sampler(session: InferenceSession):
    device = session.devices[0]
    graph = rejection_sampler(
        top_k=1,
        device=DeviceRef.from_device(device),
    )

    sampler = session.load(graph)

    # Variables
    batch_size = 3
    num_steps = 5
    vocab_size = 10

    # Generate Random Logits and Pass Through
    draft_logits = np.random.default_rng().random(
        size=(batch_size, num_steps), dtype=np.float32
    )
    draft_tokens = np.random.randint(
        0, vocab_size, size=(batch_size, num_steps)
    )
    target_logits = np.random.default_rng().random(
        size=(batch_size * (num_steps + 1), vocab_size),
        dtype=np.float32,
    )
    target_logit_offsets = np.arange(
        0, (batch_size + 1) * (num_steps + 1), num_steps + 1
    )

    first_rejected_token, sampled_tokens = sampler(
        Tensor.from_dlpack(draft_tokens).to(device),
        Tensor.from_dlpack(draft_logits).to(device),
        Tensor.from_dlpack(target_logits).to(device),
        Tensor.from_dlpack(target_logit_offsets).to(device),
    )
    assert isinstance(first_rejected_token, Tensor)
    assert isinstance(sampled_tokens, Tensor)

    # Bring these back to CPU
    first_rejected_token_np = first_rejected_token.to_numpy()
    sampled_tokens_np = sampled_tokens.to_numpy()

    # Basic Rejection Sampler Impl in Python
    for x in range(batch_size):
        for i in range(num_steps):
            target_idx = (x * (num_steps + 1)) + i
            draft_logit = draft_logits[x][i]
            token_idx = draft_tokens[x][i]
            target_logit = target_logits[target_idx][token_idx]

            if draft_logit > target_logit:
                assert first_rejected_token_np[x][0] == i, f"x: {x}, i: {i}"
                assert (
                    np.argmax(target_logits[target_idx])
                    == sampled_tokens_np[x][0]
                ), (
                    f"target_logits: {target_logits[target_idx]}, sampled: {sampled_tokens[x][0]}"
                )

                break
