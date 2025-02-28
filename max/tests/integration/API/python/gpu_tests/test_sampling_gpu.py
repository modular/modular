# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import json

import numpy as np
import torch
import xgrammar as xgr
from max.driver import CPU, Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.pipelines import SamplingConfig
from max.pipelines.sampling import token_sampler
from transformers import AutoConfig, AutoTokenizer


def test_bitmask_sampling_vs_xgrammar():
    # Get Tokenizer and Model Info
    model_id = "modularai/llama-3.1"
    config = AutoConfig.from_pretrained(model_id)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        AutoTokenizer.from_pretrained(model_id), vocab_size=config.vocab_size
    )

    # Create a grammar compiler for the tokenizer
    compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)

    # Initialize Device
    device = Accelerator()

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
        )
    )

    session = InferenceSession(devices=[device])
    sampler = session.load(graph)

    # Variables
    batch_size = 1
    vocab_size = tokenizer_info.vocab_size
    n_trials = 1

    generated_tokens = Tensor.zeros(
        (batch_size, 0),
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
        for token in new_tokens.to(CPU()).to_numpy():
            assert matcher.accept_token(token[0], debug_print=True)
