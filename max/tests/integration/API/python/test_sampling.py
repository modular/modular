# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import json

import numpy as np
import pytest
import torch
import xgrammar as xgr
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.pipelines.lib import SamplingConfig, rejection_sampler, token_sampler
from transformers import AutoConfig, AutoTokenizer


def test_bitmask_sampling_vs_xgrammar(
    session: InferenceSession, modular_ai_llama_3_1_local_path
):
    # Get Tokenizer and Model Info
    config = AutoConfig.from_pretrained(modular_ai_llama_3_1_local_path)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        AutoTokenizer.from_pretrained(modular_ai_llama_3_1_local_path),
        vocab_size=config.vocab_size,
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
        device=DeviceRef.GPU(),
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


@pytest.mark.skip("TODO(AITLIB-348): Fix this test")
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
        device=DeviceRef.GPU(),
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


def test_apply_penalties_to_logits(session: InferenceSession):
    BATCH_SIZE = 14
    VOCAB_SIZE = 1024
    FREQ_PENALTY_SCALAR = 0.5
    PRESENCE_PENALTY_SCALAR = 1.2
    REPETITION_PENALTY_SCALAR = 1.1

    device = session.devices[0]
    device_ref = DeviceRef.from_device(device)
    logits_in_type = BufferType(
        DType.float32,
        ["total_output_len", "vocab_size"],
        device=device_ref,
    )
    compressed_frequency_data_type = TensorType(
        DType.int32,
        ["unique_tokens", 2],
        device=device_ref,
    )
    frequency_offsets_type = TensorType(
        DType.uint32,
        ["batch_size_plus_1"],
        device=device_ref,
    )

    prompt_lens = np.random.randint(10, 20, [BATCH_SIZE])
    prompt_tokens = [
        np.random.randint(0, VOCAB_SIZE, [prompt_len])
        for prompt_len in prompt_lens
    ]

    frequency_offsets_np = np.zeros(BATCH_SIZE + 1, dtype=np.uint32)
    compressed_frequency_data_np = np.zeros(
        [np.sum(prompt_lens), 2], dtype=np.int32
    )

    for i in range(BATCH_SIZE):
        unique_tokens, counts = np.unique(prompt_tokens[i], return_counts=True)

        start_idx = frequency_offsets_np[i]
        end_idx = start_idx + len(unique_tokens)
        frequency_offsets_np[i + 1] = end_idx

        compressed_frequency_data_np[start_idx:end_idx, 0] = unique_tokens
        compressed_frequency_data_np[start_idx:end_idx, 1] = counts

    # resize compressed_frequency_data to the correct size
    compressed_frequency_data_np = compressed_frequency_data_np[
        : frequency_offsets_np[BATCH_SIZE], :
    ]

    logits_np = torch.randn([BATCH_SIZE, VOCAB_SIZE], dtype=torch.float32)

    with Graph(
        "apply_penalties_to_logits",
        input_types=(
            logits_in_type,
            compressed_frequency_data_type,
            frequency_offsets_type,
        ),
    ) as graph:
        logits = graph.inputs[0].buffer
        compressed_frequency_data = graph.inputs[1].tensor
        frequency_offsets = graph.inputs[2].tensor

        ops.inplace_custom(
            "sampler.apply_penalties",
            values=[
                logits,
                compressed_frequency_data,
                frequency_offsets,
                ops.constant(
                    FREQ_PENALTY_SCALAR, DType.float32, device=DeviceRef.CPU()
                ),
                ops.constant(
                    PRESENCE_PENALTY_SCALAR,
                    DType.float32,
                    device=DeviceRef.CPU(),
                ),
                ops.constant(
                    REPETITION_PENALTY_SCALAR,
                    DType.float32,
                    device=DeviceRef.CPU(),
                ),
            ],
            device=device_ref,
        )

        graph.output(logits)

    model = session.load(graph)

    logits_out = model(
        Tensor.from_dlpack(logits_np).to(device),
        Tensor.from_dlpack(compressed_frequency_data_np).to(device),
        Tensor.from_dlpack(frequency_offsets_np).to(device),
    )[0]

    max_result = torch.from_dlpack(logits_out)

    # create reference result
    ref_result = logits_np.clone()
    for i in range(BATCH_SIZE):
        unique_tokens, counts = np.unique(prompt_tokens[i], return_counts=True)
        for token, count in zip(unique_tokens, counts):
            if ref_result[i][token] > 0:
                ref_result[i][token] /= REPETITION_PENALTY_SCALAR
            else:
                ref_result[i][token] *= REPETITION_PENALTY_SCALAR
            ref_result[i][token] -= FREQ_PENALTY_SCALAR * count
            ref_result[i][token] -= PRESENCE_PENALTY_SCALAR

    torch.testing.assert_close(max_result.to("cpu"), ref_result)


def test_update_frequency_data(session: InferenceSession):
    device = session.devices[0]
    device_ref = DeviceRef.from_device(device)
    compressed_frequency_data_type = BufferType(
        DType.int32,
        ["unique_tokens", 2],
        device=device_ref,
    )
    frequency_offsets_type = TensorType(
        DType.uint32,
        ["batch_size_plus_1"],
        device=device_ref,
    )
    new_tokens_type = TensorType(
        DType.int64,
        ["batch_size"],
        device=device_ref,
    )

    PADDING_TOKEN = -1

    frequency_offsets_np = np.array([0, 6, 10], dtype=np.uint32)
    compressed_frequency_data_np = np.array(
        [
            [0, 1],
            [1, 1],
            [2, 1],
            [3, 1],
            [4, 1],
            [PADDING_TOKEN, 0],
            [0, 1],
            [1, 1],
            [2, 1],
            [PADDING_TOKEN, 0],
        ],
        dtype=np.int32,
    )
    new_tokens_np = np.array([3, 6], dtype=np.int64)

    with Graph(
        "update_frequency_data",
        input_types=(
            compressed_frequency_data_type,
            frequency_offsets_type,
            new_tokens_type,
        ),
    ) as graph:
        compressed_frequency_data = graph.inputs[0].buffer
        frequency_offsets = graph.inputs[1].tensor
        new_tokens = graph.inputs[2].tensor

        ops.inplace_custom(
            "sampler.update_frequency_data",
            values=[
                compressed_frequency_data,
                frequency_offsets,
                new_tokens,
            ],
            device=device_ref,
        )

        graph.output(compressed_frequency_data)

    model = session.load(graph)

    compressed_frequency_data_out = model(
        Tensor.from_dlpack(compressed_frequency_data_np).to(device),
        Tensor.from_dlpack(frequency_offsets_np).to(device),
        Tensor.from_dlpack(new_tokens_np).to(device),
    )[0]

    assert isinstance(compressed_frequency_data_out, Tensor)
    compressed_frequency_data_out = compressed_frequency_data_out.to(CPU())

    ref_result = np.array(
        [
            [0, 1],
            [1, 1],
            [2, 1],
            [3, 2],  # incremented
            [4, 1],
            [PADDING_TOKEN, 0],
            [0, 1],
            [1, 1],
            [2, 1],
            [6, 1],  # added
        ],
        dtype=np.int32,
    )

    assert np.all(ref_result == np.from_dlpack(compressed_frequency_data_out))
