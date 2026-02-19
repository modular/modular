# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import json

import llguidance.hf
import llguidance.numpy
import numpy as np
import pytest
import torch
from llguidance import LLMatcher
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.interfaces import SamplingParams
from max.pipelines.lib import (
    SamplingConfig,
    rejection_sampler,
    rejection_sampler_with_residuals,
    token_sampler,
)
from transformers import AutoConfig, AutoTokenizer


def test_sampling_top_k() -> None:
    with pytest.raises(ValueError):
        SamplingParams(top_k=257)


def test_temperature_zero_sets_top_k_to_one() -> None:
    sampling_params = SamplingParams(temperature=0.0)
    assert sampling_params.top_k == 1


def test_llguidance_sampling(
    session: InferenceSession, modular_ai_llama_3_1_local_path: str
) -> None:
    config = AutoConfig.from_pretrained(modular_ai_llama_3_1_local_path)
    hf_tokenizer = AutoTokenizer.from_pretrained(
        modular_ai_llama_3_1_local_path
    )
    tokenizer = llguidance.hf.from_tokenizer(
        hf_tokenizer, n_vocab=config.vocab_size
    )

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

    matcher = LLMatcher(tokenizer, json.dumps(person_schema))
    sampling_config = SamplingConfig(
        enable_structured_output=True,
        in_dtype=DType.float32,
        out_dtype=DType.float32,
    )
    # Create one op sampling graph
    graph = token_sampler(
        sampling_config,
        device=DeviceRef.GPU(),
    )

    device = session.devices[0]
    sampler = session.load(graph)

    # Variables
    batch_size = 1
    vocab_size = tokenizer.vocab_size
    n_trials = 1

    sampling_params = SamplingParams(top_k=5)

    generated_tokens = Buffer(
        shape=(batch_size, 0),
        dtype=DType.int64,
        device=device,
    )

    temperature = Buffer.from_numpy(
        np.array([sampling_params.temperature] * batch_size, dtype=np.float32)
    ).to(device)
    top_k_np = np.array([sampling_params.top_k] * batch_size, dtype=np.int64)
    top_k = Buffer.from_numpy(top_k_np).to(device)
    max_k = Buffer.from_numpy(np.array(np.max(top_k_np), dtype=np.int64))
    top_p = Buffer.from_numpy(
        np.array([sampling_params.top_p] * batch_size, dtype=np.float32)
    ).to(device)
    min_top_p = Buffer.from_numpy(
        np.array(sampling_params.top_p, dtype=np.float32)
    )
    seed = Buffer.from_numpy(
        np.array([sampling_params.seed] * batch_size, dtype=np.uint64)
    ).to(device)
    for _ in range(n_trials):
        token_bitmask = llguidance.numpy.allocate_token_bitmask(
            batch_size, vocab_size
        )
        llguidance.numpy.fill_next_token_bitmask(matcher, token_bitmask)

        # Generate Random Logits
        logits = np.random.default_rng().random(
            size=(batch_size, vocab_size), dtype=np.float32
        )

        bits = 2 ** np.arange(32, dtype=np.int32)
        bitmask = (np.expand_dims(token_bitmask, axis=-1) & bits) != 0
        bitmask = bitmask.reshape(
            batch_size,
            -1,  # This will automatically calculate the correct size based on the other dimension
        ).astype(bool)

        # Run through Sampler
        _, new_tokens = sampler(
            Buffer.from_dlpack(logits).to(device),
            generated_tokens,  # This isn't used by the sampler, so we can safely ignore it.
            top_k,
            max_k,
            temperature,
            top_p,
            min_top_p,
            seed,
            Buffer.from_dlpack(bitmask).to(device),
        )[:2]
        assert isinstance(new_tokens, Buffer)
        for token in new_tokens.to_numpy():
            assert matcher.validate_tokens(token) == len(token)


@pytest.mark.skip("TODO(AITLIB-348): Fix this test")
def test_sampling_return_logits(session: InferenceSession) -> None:
    # Create one op sampling graph
    sampling_config = SamplingConfig(
        enable_structured_output=False,
        in_dtype=DType.float32,
        out_dtype=DType.float32,
    )
    sampling_params = SamplingParams(top_k=5)
    graph = token_sampler(
        sampling_config,
        return_logits=True,
        device=DeviceRef.GPU(),
    )

    device = session.devices[0]
    sampler = session.load(graph)

    # Variables
    batch_size = 3
    vocab_size = 10
    generated_tokens = Buffer(
        shape=(batch_size, 0),
        dtype=DType.int64,
        device=device,
    )
    temperature = Buffer.from_numpy(
        np.array([sampling_params.temperature] * batch_size, dtype=np.float32)
    ).to(device)
    generated_logits = Buffer(
        shape=(batch_size, 0), dtype=DType.float32, device=device
    )
    top_k_np = np.array([sampling_params.top_k] * batch_size, dtype=np.int64)
    top_k = Buffer.from_numpy(top_k_np).to(device)
    max_k = Buffer.from_numpy(np.array(np.max(top_k_np), dtype=np.int64))
    top_p = Buffer.from_numpy(
        np.array([sampling_params.top_p] * batch_size, dtype=np.float32)
    ).to(device)
    min_top_p = Buffer.from_numpy(
        np.array(sampling_params.top_p, dtype=np.float32)
    )
    seed = Buffer.from_numpy(
        np.array([sampling_params.seed] * batch_size, dtype=np.uint64)
    ).to(device)

    # Generate Random Logits
    for j in range(3):
        logits = np.random.default_rng().random(
            size=(batch_size, vocab_size), dtype=np.float32
        )

        # Run through Sampler
        new_tokens, generated_tokens_max, generated_logits_max = sampler(
            Buffer.from_dlpack(logits).to(device),
            generated_tokens,
            top_k,
            max_k,
            temperature,
            top_p,
            min_top_p,
            seed,
            generated_logits,
        )[:3]
        assert isinstance(generated_tokens_max, Buffer)
        assert isinstance(generated_logits_max, Buffer)
        assert isinstance(new_tokens, Buffer)
        generated_tokens = generated_tokens_max
        generated_logits = generated_logits_max

        # Ensure that the tokens generated, match the correct logits expected.
        numpy_logits = generated_logits.to_numpy()
        for i, token_idx in enumerate(new_tokens.to_numpy()):
            assert numpy_logits[i, j] == logits[i, token_idx]


def test_rejection_sampler(session: InferenceSession) -> None:
    device = session.devices[0]
    sampling_config = SamplingConfig(
        in_dtype=DType.float32,
        out_dtype=DType.float32,
    )
    sampling_params = SamplingParams(top_k=1, temperature=1.0)
    graph = rejection_sampler(
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

    top_k_np = np.array([sampling_params.top_k] * batch_size, dtype=np.int64)
    top_k_tensor = Buffer.from_numpy(top_k_np).to(device)
    max_k_tensor = Buffer.from_numpy(
        np.array([np.max(top_k_np)], dtype=np.int64)
    )
    temperature_tensor = Buffer.from_numpy(
        np.array([sampling_params.temperature] * batch_size, dtype=np.float32)
    ).to(device)

    first_rejected_token, sampled_tokens = sampler(
        Buffer.from_dlpack(draft_tokens).to(device),
        Buffer.from_dlpack(draft_logits).to(device),
        Buffer.from_dlpack(target_logits).to(device),
        Buffer.from_dlpack(target_logit_offsets).to(device),
    )
    assert isinstance(first_rejected_token, Buffer)
    assert isinstance(sampled_tokens, Buffer)

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
                    f"target_logits: {target_logits[target_idx]}, sampled: {sampled_tokens_np[x][0]}"
                )

                break


def test_apply_penalties_to_logits(session: InferenceSession) -> None:
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

    penalty_type = TensorType(DType.float32, ["batch_size"], device=device_ref)
    freq_penalty_type = penalty_type
    presence_penalty_type = penalty_type
    repetition_penalty_type = penalty_type

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
            freq_penalty_type,
            presence_penalty_type,
            repetition_penalty_type,
        ),
    ) as graph:
        logits = graph.inputs[0].buffer
        compressed_frequency_data = graph.inputs[1].tensor
        frequency_offsets = graph.inputs[2].tensor
        freq_penalty = graph.inputs[3].tensor
        presence_penalty = graph.inputs[4].tensor
        repetition_penalty = graph.inputs[5].tensor

        ops.inplace_custom(
            "sampler.apply_penalties",
            values=[
                logits,
                compressed_frequency_data,
                frequency_offsets,
                freq_penalty,
                presence_penalty,
                repetition_penalty,
            ],
            device=device_ref,
        )

        graph.output(logits)

    model = session.load(graph)

    logits_out = model(
        Buffer.from_dlpack(logits_np).to(device),
        Buffer.from_dlpack(compressed_frequency_data_np).to(device),
        Buffer.from_dlpack(frequency_offsets_np).to(device),
        Buffer.from_numpy(
            np.array([FREQ_PENALTY_SCALAR] * BATCH_SIZE, dtype=np.float32)
        ).to(device),
        Buffer.from_numpy(
            np.array([PRESENCE_PENALTY_SCALAR] * BATCH_SIZE, dtype=np.float32)
        ).to(device),
        Buffer.from_numpy(
            np.array([REPETITION_PENALTY_SCALAR] * BATCH_SIZE, dtype=np.float32)
        ).to(device),
    )[0]

    max_result = torch.from_dlpack(logits_out)

    # create reference result
    ref_result = logits_np.clone()
    for i in range(BATCH_SIZE):
        unique_tokens, counts = np.unique(prompt_tokens[i], return_counts=True)
        for token, count in zip(unique_tokens, counts, strict=True):
            if ref_result[i][token] > 0:
                ref_result[i][token] /= REPETITION_PENALTY_SCALAR
            else:
                ref_result[i][token] *= REPETITION_PENALTY_SCALAR
            ref_result[i][token] -= FREQ_PENALTY_SCALAR * count
            ref_result[i][token] -= PRESENCE_PENALTY_SCALAR

    torch.testing.assert_close(max_result.to("cpu"), ref_result)


def test_update_frequency_data(session: InferenceSession) -> None:
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
        Buffer.from_dlpack(compressed_frequency_data_np).to(device),
        Buffer.from_dlpack(frequency_offsets_np).to(device),
        Buffer.from_dlpack(new_tokens_np).to(device),
    )[0]

    assert isinstance(compressed_frequency_data_out, Buffer)
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


def test_sampling_with_seed(session: InferenceSession) -> None:
    """Test that sampling with the same seed produces deterministic results."""
    device = session.devices[0]
    device_ref = DeviceRef.from_device(device)

    # Test parameters
    batch_size = 1
    vocab_size = 128256  # Large enough vocab size for testing
    top_k = 100
    seed_1 = 42
    seed_2 = 41
    num_trials = 10  # Run multiple times to ensure determinism

    # Create sampling configuration
    sampling_config = SamplingConfig(
        in_dtype=DType.float32,
        out_dtype=DType.float32,
    )

    sampling_params = SamplingParams(
        top_k=top_k,
        seed=seed_1,
    )

    # Create sampling graph
    graph = token_sampler(sampling_config, device=device_ref)
    temperature = Buffer.from_numpy(
        np.array([sampling_params.temperature] * batch_size, dtype=np.float32)
    ).to(device)
    top_k_np = np.array([sampling_params.top_k] * batch_size, dtype=np.int64)
    top_k_tensor = Buffer.from_numpy(top_k_np).to(device)
    max_k = Buffer.from_numpy(np.array(np.max(top_k_np), dtype=np.int64))
    top_p = Buffer.from_numpy(
        np.array([sampling_params.top_p] * batch_size, dtype=np.float32)
    ).to(device)
    min_top_p = Buffer.from_numpy(
        np.array(sampling_params.top_p, dtype=np.float32)
    )
    seed = Buffer.from_numpy(
        np.array([sampling_params.seed] * batch_size, dtype=np.uint64)
    ).to(device)

    sampler = session.load(graph)

    # Create a random logits vector [1, vocab_size]
    np.random.seed(123)  # Fix seed for generating the same logits across runs
    logits_np = np.random.randn(batch_size, vocab_size).astype(np.float32)

    # Update sampling config with same seed
    sampling_config_42 = SamplingConfig(
        in_dtype=DType.float32,
        out_dtype=DType.float32,
    )

    graph_42 = token_sampler(sampling_config_42, device=device_ref)
    sampler_42 = session.load(graph_42)

    # Create dummy previous tokens
    prev_tokens = Buffer(
        shape=(batch_size, 0),
        dtype=DType.int64,
        device=device,
    )
    logits_tensor = Buffer.from_dlpack(logits_np).to(device)

    # Test 1: Sample multiple times with seed=42, results should be identical
    for _ in range(num_trials):
        # Sample with the same logits and seed
        _tokens, new_prev_tokens = sampler_42(
            logits_tensor,
            prev_tokens,
            top_k_tensor,
            max_k,
            temperature,
            top_p,
            min_top_p,
            seed,
        )[:2]

        assert isinstance(new_prev_tokens, Buffer)
        prev_tokens = new_prev_tokens

    prev_tokens_np = prev_tokens.to_numpy()
    # Verify all results with seed=42 are identical
    for i in range(1, num_trials):
        np.testing.assert_array_equal(
            prev_tokens_np[:, 0],
            prev_tokens_np[:, i],
            err_msg=f"Sampling with seed={seed_1} should be deterministic across runs",
        )

    # Test 2: Sample with seed=41, result should be different from seed=42
    sampling_config_41 = SamplingConfig(
        in_dtype=DType.float32,
        out_dtype=DType.float32,
    )
    sampling_params_41 = SamplingParams(
        top_k=top_k,
        seed=seed_2,  # Different seed
    )

    graph_41 = token_sampler(sampling_config_41, device=device_ref)
    sampler_41 = session.load(graph_41)
    seed_41 = Buffer.from_numpy(
        np.array([sampling_params_41.seed] * batch_size, dtype=np.uint64)
    ).to(device)
    tokens_41, _ = sampler_41(
        logits_tensor,
        prev_tokens,
        top_k_tensor,
        max_k,
        temperature,
        top_p,
        min_top_p,
        seed_41,
    )[:2]

    assert isinstance(tokens_41, Buffer)
    result_seed_41 = tokens_41.to_numpy()

    # Verify result with seed=41 is different from seed=42
    # Note: There's a very small chance they could be the same by coincidence,
    # but with a large vocab_size and top_k=100, this is extremely unlikely
    assert not np.array_equal(prev_tokens_np[:, 0], result_seed_41), (
        f"Sampling with different seeds ({seed_1} vs {seed_2}) should produce different results"
    )


def test_top_p_sampling(session: InferenceSession) -> None:
    """Test that top_p sampling produces the correct results."""
    device = session.devices[0]
    device_ref = DeviceRef.from_device(device)

    # Test parameters
    vocab_size = 8
    top_k = vocab_size
    temp = 1.0
    num_trials = 100

    # After softmax, the first 4 tokens have a probability of 0.501
    logits_np = np.array(
        [[1.0, 0.999, 0.998, 0.997, 0.996, 0.995, 0.994, 0.993]],
        dtype=np.float32,
    )

    # Create graph with logits and seed as inputs
    logits_type = TensorType(DType.float32, [1, vocab_size], device=device_ref)
    seed_type = TensorType(DType.uint64, [1], device=device_ref)
    top_p_type = TensorType(DType.float32, [1], device=device_ref)
    top_k_type = TensorType(DType.int64, [1], device=device_ref)
    max_k_type = TensorType(DType.int64, [], device=DeviceRef.CPU())
    temperature_type = TensorType(DType.float32, [1], device=device_ref)

    def create_sampling_graph():  # noqa: ANN202
        with Graph(
            "top_p_sampling",
            input_types=(
                logits_type,
                seed_type,
                top_p_type,
                top_k_type,
                max_k_type,
                temperature_type,
            ),
        ) as graph:
            logits_input = graph.inputs[0].tensor
            seed_input = graph.inputs[1].tensor
            top_p_input = graph.inputs[2].tensor
            top_k_input = graph.inputs[3].tensor
            max_k_input = graph.inputs[4].tensor
            temperature_input = graph.inputs[5].tensor

            # We need to manually create the custom op since topk_fused_sampling
            # doesn't accept seed as tensor
            sampled_tokens = ops.custom(
                "sampler.fused_token_sampling",
                device=device_ref,
                values=[
                    top_k_input,
                    max_k_input,
                    temperature_input,
                    top_p_input,
                    # min_top_p must be a scalar; set to 1.0 to match default behavior
                    ops.constant(
                        1.0, dtype=DType.float32, device=DeviceRef.CPU()
                    ),
                    seed_input,
                    logits_input,
                ],
                out_types=[
                    TensorType(
                        dtype=DType.int64, shape=[1, 1], device=device_ref
                    )
                ],
            )[0].tensor

            graph.output(sampled_tokens)
        return graph

    # Test 0: top_p = 0.5, should only sample from the first 4 tokens
    graph_0 = create_sampling_graph()
    sampler_0 = session.load(graph_0)

    logits_tensor = Buffer.from_dlpack(logits_np).to(device)
    sampled_tokens_0 = []

    temperature = Buffer.from_numpy(np.array([temp], dtype=np.float32)).to(
        device
    )
    top_k_np = np.array([top_k], dtype=np.int64)
    top_k_tensor = Buffer.from_numpy(top_k_np).to(device)
    max_k = Buffer.from_numpy(np.array(np.max(top_k_np), dtype=np.int64))
    top_p_tensor = Buffer.from_numpy(np.array([0.5], dtype=np.float32)).to(
        device
    )
    for seed in range(num_trials):
        seed_tensor = Buffer.from_dlpack(np.array([seed], dtype=np.uint64)).to(
            device
        )
        tokens = sampler_0(
            logits_tensor,
            seed_tensor,
            top_p_tensor,
            top_k_tensor,
            max_k,
            temperature,
        )[0]
        assert isinstance(tokens, Buffer)
        token_idx = tokens.to_numpy()[0, 0]
        sampled_tokens_0.append(token_idx.item())

    # Verify all sampled indices are in [0, 1, 2, 3]
    expected_tokens_0 = {0, 1, 2, 3}
    actual_tokens_0 = set(sampled_tokens_0)
    assert actual_tokens_0 == expected_tokens_0

    # Test 1: top_p = 0.51, should only sample from indices [0, 1, 2, 3, 4]
    graph_1 = create_sampling_graph()
    sampler_1 = session.load(graph_1)

    sampled_tokens_1 = []
    top_p_tensor = Buffer.from_numpy(np.array([0.51], dtype=np.float32)).to(
        device
    )
    for seed in range(num_trials):
        seed_tensor = Buffer.from_dlpack(np.array([seed], dtype=np.uint64)).to(
            device
        )
        tokens = sampler_1(
            logits_tensor,
            seed_tensor,
            top_p_tensor,
            top_k_tensor,
            max_k,
            temperature,
        )[0]
        assert isinstance(tokens, Buffer)
        token_idx = tokens.to_numpy()[0, 0]
        sampled_tokens_1.append(token_idx.item())

    # Verify all sampled indices are in [0, 1, 2, 3, 4]
    expected_tokens_1 = {0, 1, 2, 3, 4}
    actual_tokens_1 = set(sampled_tokens_1)
    assert actual_tokens_1 == expected_tokens_1


def test_batch_sampling_arguments(session: InferenceSession) -> None:
    device = session.devices[0]
    device_ref = DeviceRef.from_device(device)

    # Test parameters
    batch_size = 4
    vocab_size = 8

    sampling_config = SamplingConfig()

    graph = token_sampler(sampling_config, device=device_ref)
    sampler = session.load(graph)

    prev_tokens = Buffer(
        shape=(batch_size, 0),
        dtype=DType.int64,
        device=device,
    )

    num_trials = 100

    def test_top_p_sampling() -> None:
        """Test that different top_p values affect sampling correctly."""
        k = np.array([vocab_size] * batch_size, dtype=np.int64)
        temperature = np.array([1.0] * batch_size, dtype=np.float32)
        logits_np = np.array(
            [[1.0, 0.999, 0.998, 0.997, 0.996, 0.995, 0.994, 0.993]],
            dtype=np.float32,
        )
        batch_logits_np = np.repeat(logits_np, repeats=batch_size, axis=0)
        logits = Buffer.from_dlpack(batch_logits_np).to(device)
        top_k = Buffer.from_numpy(k).to(device)
        max_k = Buffer.from_numpy(np.array(np.max(k), dtype=np.int64))
        temperature_tensor = Buffer.from_numpy(temperature).to(device)

        top_p = np.array([0.51, 0.5, 0.5, 0.5], dtype=np.float32)
        top_p_tensor = Buffer.from_numpy(top_p).to(device)
        min_top_p_tensor = Buffer.from_numpy(
            np.array(np.min(top_p), dtype=np.float32)
        )
        batch_sampled_tokens: list[list[int]] = [[] for _ in range(batch_size)]
        for seed_val in range(num_trials):
            seed_array = np.array(
                [seed_val, seed_val, seed_val, 0], dtype=np.uint64
            )
            seed_tensor = Buffer.from_numpy(seed_array).to(device)
            tokens = sampler(
                logits,
                prev_tokens,
                top_k,
                max_k,
                temperature_tensor,
                top_p_tensor,
                min_top_p_tensor,
                seed_tensor,
            )[0]
            assert isinstance(tokens, Buffer)
            tokens_np = tokens.to_numpy()
            for i in range(batch_size):
                batch_sampled_tokens[i].append(tokens_np[i].item())

        # The 0th batch index has a p of 0.51 so it samples from 5 tokens
        expected_tokens = [{0, 1, 2, 3, 4}, {0, 1, 2, 3}, {0, 1, 2, 3}]
        for i in range(batch_size - 1):
            assert set(batch_sampled_tokens[i]) == expected_tokens[i]

        # Seed doesn't change, so it only samples 1 token
        assert len(set(batch_sampled_tokens[batch_size - 1])) == 1

    def test_top_k_sampling() -> None:
        """Test that different top_k values affect sampling correctly."""
        logits_np = np.array(
            [[10.0, 8.0, 2.0, -1.0, -1.5, -2.0, -2.5, -3.0]],
            dtype=np.float32,
        )
        batch_logits_np = np.repeat(logits_np, repeats=batch_size, axis=0)
        logits = Buffer.from_dlpack(batch_logits_np).to(device)
        top_p = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        top_p_tensor = Buffer.from_numpy(top_p).to(device)
        min_top_p_tensor = Buffer.from_numpy(
            np.array(np.min(top_p), dtype=np.float32)
        )

        temperature = np.array([1.0] * batch_size, dtype=np.float32)
        temperature_tensor = Buffer.from_numpy(temperature).to(device)
        batch_sampled_tokens: list[list[int]] = [[] for _ in range(batch_size)]
        for seed_val in range(num_trials):
            seed_array = np.array([seed_val] * batch_size, dtype=np.uint64)
            seed_tensor = Buffer.from_numpy(seed_array).to(device)
            k = np.array([1, 2, 3, 4], dtype=np.int64)
            top_k = Buffer.from_numpy(k).to(device)
            max_k = Buffer.from_numpy(np.array(np.max(k), dtype=np.int64))
            tokens = sampler(
                logits,
                prev_tokens,
                top_k,
                max_k,
                temperature_tensor,
                top_p_tensor,
                min_top_p_tensor,
                seed_tensor,
            )[0]
            assert isinstance(tokens, Buffer)
            tokens_np = tokens.to_numpy()
            for i in range(batch_size):
                batch_sampled_tokens[i].append(tokens_np[i].item())

        assert set(batch_sampled_tokens[0]) == {0}
        for i in range(1, batch_size):
            assert len(set(batch_sampled_tokens[i])) > 1

    def test_temperature_sampling() -> None:
        """Test that different temperature values affect sampling correctly."""
        logits_np = np.array(
            [[10.0, 8.0, 2.0, -1.0, -1.5, -2.0, -2.5, -3.0]],
            dtype=np.float32,
        )
        batch_logits_np = np.repeat(logits_np, repeats=batch_size, axis=0)
        logits = Buffer.from_dlpack(batch_logits_np).to(device)
        top_p = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        top_p_tensor = Buffer.from_numpy(top_p).to(device)
        min_top_p_tensor = Buffer.from_numpy(
            np.array(np.min(top_p), dtype=np.float32)
        )
        k = np.array([vocab_size] * batch_size, dtype=np.int64)
        top_k = Buffer.from_numpy(k).to(device)
        max_k = Buffer.from_numpy(np.array(np.max(k), dtype=np.int64))

        batch_sampled_tokens: list[list[int]] = [[] for _ in range(batch_size)]
        for seed_val in range(num_trials):
            seed_array = np.array([seed_val] * batch_size, dtype=np.uint64)
            seed_tensor = Buffer.from_numpy(seed_array).to(device)

            temperature = np.array([0.01, 5.0, 5.0, 5.0], dtype=np.float32)
            temperature_tensor = Buffer.from_numpy(temperature).to(device)

            tokens = sampler(
                logits,
                prev_tokens,
                top_k,
                max_k,
                temperature_tensor,
                top_p_tensor,
                min_top_p_tensor,
                seed_tensor,
            )[0]
            assert isinstance(tokens, Buffer)
            tokens_np = tokens.to_numpy()
            for i in range(batch_size):
                batch_sampled_tokens[i].append(tokens_np[i].item())

        # low temperature is more or less deterministic, high temperature is more random
        assert set(batch_sampled_tokens[0]) == {0}
        for i in range(1, batch_size):
            assert len(set(batch_sampled_tokens[i])) > vocab_size // 2

    # Run all tests
    test_top_p_sampling()
    test_top_k_sampling()
    test_temperature_sampling()


def rejection_sampler_reference(  # noqa: ANN201
    target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
    draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
    draft_token_ids: torch.Tensor,  # [batch_size, k]
    rejection_rand: torch.Tensor,  # [batch_size, k]
    residual_rand: torch.Tensor | None = None,  # [batch_size * k, vocab_size]
):
    """
    Rejection sampler reference implementation.
    """

    def _get_first_rejected_token_idx(
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
    ) -> torch.Tensor:
        batch_size, k, _ = draft_probs.shape
        batch_indices = torch.arange(batch_size, device=target_probs.device)[
            :, None
        ]
        probs_indices = torch.arange(k, device=target_probs.device)

        # shape [batch_size, k]
        selected_draft_probs = draft_probs[
            batch_indices, probs_indices, draft_token_ids
        ]

        # shape [batch_size, k]
        selected_target_probs = target_probs[
            batch_indices, probs_indices, draft_token_ids
        ]

        ratio = selected_target_probs / selected_draft_probs
        capped_ratio = torch.minimum(
            ratio,
            torch.full((1,), 1, device=target_probs.device),
        )
        rejected = (rejection_rand >= capped_ratio).long()
        rejected_with_sentinel = torch.concat(
            [rejected, torch.ones((batch_size, 1))],
            dim=-1,
        )

        first_rejected_token_idx = torch.argmax(rejected_with_sentinel, dim=-1)
        return first_rejected_token_idx

    def _get_recovered_probs(
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
    ) -> torch.Tensor:
        _, k, _ = draft_probs.shape

        # shape [batch_size, k, vocab_size]
        difference = target_probs - draft_probs

        # TODO(cade): Can we use logprobs instead of probs, and avoid the
        # division-by-zero errors without introducing distribution drift?

        # shape [batch_size, k, vocab_size]
        f = torch.clamp(difference, min=torch.finfo(difference.dtype).tiny)

        # shape [batch_size, k, vocab_size]
        recovered_probs = f / torch.sum(f, dim=-1).reshape(-1, k, 1)

        return recovered_probs

    def _multinomial(
        probs: torch.Tensor,
        residual_rand: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_samples = 1
        if residual_rand is not None:
            # Use provided uniform random numbers
            eps = torch.finfo(probs.dtype).eps
            clamped_uniform = torch.clamp(residual_rand, min=eps, max=1.0 - eps)
            q = -torch.log(clamped_uniform)
        else:
            # Generate random exponential numbers
            q = torch.empty_like(probs)
            q.exponential_(1.0)
        return probs.div_(q).argmax(dim=1).view(-1, num_samples)

    batch_size, k, vocab_size = draft_probs.shape
    # shape [batch_size, k]
    first_rejected_token_idx = _get_first_rejected_token_idx(
        target_probs, draft_probs, draft_token_ids
    )

    recovered_probs = _get_recovered_probs(target_probs, draft_probs).reshape(
        batch_size * k, vocab_size
    )

    # NOTE: the recovered_probs are overwritten by this method.
    recovered_token_ids = _multinomial(
        recovered_probs,
        residual_rand,
    ).reshape(batch_size, k)

    return first_rejected_token_idx, recovered_token_ids


def test_rejection_sampler_with_residuals(session: InferenceSession) -> None:
    batch_size = 3
    num_steps = 4
    vocab_size = 5
    torch.manual_seed(0)

    # num_steps +1 for bonus token
    target_logits = 3 * torch.randn(batch_size, num_steps + 1, vocab_size)

    draft_logits = target_logits[:, :-1] + 0.7 * torch.randn(
        batch_size, num_steps, vocab_size
    )
    target_probs = torch.softmax(target_logits, dim=-1)
    draft_probs = torch.softmax(draft_logits, dim=-1)
    draft_token_ids = torch.argmax(draft_probs, dim=-1)

    # Generate controlled uniform random numbers for both rejection and multinomial sampling
    rejection_rand = torch.rand(batch_size, num_steps)
    residual_rand = torch.rand(batch_size * num_steps, vocab_size)

    first_rejected_token_idx, recovered_token_ids = rejection_sampler_reference(
        target_probs[:, :-1],
        draft_probs,
        draft_token_ids,
        rejection_rand,
        residual_rand,
    )

    device = session.devices[0]
    graph = rejection_sampler_with_residuals(
        device=DeviceRef.from_device(device), debug=True
    )
    sampler = session.load(graph)
    draft_logits_for_sampled_tokens = torch.gather(
        draft_logits, dim=-1, index=draft_token_ids.unsqueeze(-1)
    ).squeeze(-1)
    target_logit_offsets = np.arange(
        0, (batch_size + 1) * (num_steps + 1), num_steps + 1
    )
    target_logits_tensor = target_logits.reshape(
        batch_size * (num_steps + 1), vocab_size
    )

    draft_logits_tensor = draft_logits.permute(1, 0, 2).contiguous()

    first_rejected_token, recovered_tokens, _bonus_tokens = sampler(
        Buffer.from_dlpack(draft_token_ids).to(device),
        Buffer.from_dlpack(draft_logits_for_sampled_tokens).to(device),
        Buffer.from_dlpack(target_logits_tensor).to(device),
        Buffer.from_dlpack(target_logit_offsets).to(device),
        Buffer.from_dlpack(draft_logits_tensor).to(device),
        Buffer.from_dlpack(rejection_rand).to(device),
        Buffer.from_dlpack(
            residual_rand.reshape(batch_size, num_steps, vocab_size)
        ).to(device),
    )

    # Now we can compare the results deterministically
    assert isinstance(first_rejected_token, Buffer)
    assert isinstance(recovered_tokens, Buffer)

    first_rejected_token_np = first_rejected_token.to_numpy()
    recovered_tokens_np = recovered_tokens.to_numpy()

    # Compare first rejected token indices
    np.testing.assert_array_equal(
        first_rejected_token_np,
        first_rejected_token_idx.numpy(),
        err_msg="First rejected token indices should match",
    )

    # Compare recovered token IDs
    np.testing.assert_array_equal(
        recovered_tokens_np,
        recovered_token_ids.numpy(),
        err_msg="Recovered token IDs should match",
    )
