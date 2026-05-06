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
"""Tests for sampler penalty ops (apply_penalties, update_frequency_data)."""

import numpy as np
import torch
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops


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
