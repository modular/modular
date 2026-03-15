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
"""Tests for max.nn.sampling"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import CPU, Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.sampling import MinPSampler
from max.pipelines.lib.sampling import (
    build_greedy_acceptance_sampler_graph,
    build_stochastic_acceptance_sampler_graph,
)


# NOTE THAT ONLY RANK 2 TENSORS
# ARE CURRENTLY SUPPORTED
@pytest.mark.parametrize(
    ("input_shape", "min_p", "temperature"),
    [
        ((5, 5), 0.0, 0.0),
        ((3, 5), 0.1, 0.3),
    ],
)
def test_min_p_execution(
    session: InferenceSession,
    input_shape: tuple[int, ...],
    min_p: float,
    temperature: float,
) -> None:
    """Tests end-to-end MinPSampling lowering and execution."""
    with Graph(
        "min_p_test",
        input_types=[
            TensorType(DType.float32, shape=input_shape, device=DeviceRef.CPU())
        ],
    ) as graph:
        inputs, *_ = graph.inputs
        sampler = MinPSampler(DType.float32, input_shape, temperature)
        out = sampler(inputs.tensor, min_p)
        graph.output(out)

        # Compile and execute the graph.
        model = session.load(graph)

        # Generate random input data.
        np_input = np.random.randn(*input_shape).astype(np.float32)

        # Execute MAX model.
        min_p_output, *_ = model.execute(np_input)
        cast(Buffer, min_p_output).to_numpy()


def test_min_p_known_inputs_outputs(session: InferenceSession) -> None:
    """Tests MinP sampling with known inputs and expected behavior."""
    batch_size = 5
    vocab_size = 4
    input_shape = (batch_size, vocab_size)
    temperature = 1.0  # Set to 1 for predictable softmax behavior

    with Graph(
        "min_p_known_test",
        input_types=[
            TensorType(
                DType.float32, shape=input_shape, device=DeviceRef.CPU()
            ),
            TensorType(
                DType.float32, shape=(batch_size,), device=DeviceRef.CPU()
            ),
        ],
    ) as graph:
        inputs, min_p_tensor, *_ = graph.inputs
        sampler = MinPSampler(DType.float32, input_shape, temperature)
        out = sampler(inputs.tensor, min_p_tensor.tensor)
        graph.output(out)
        model = session.load(graph)

        np_input = np.array(
            [
                [-1.0, 1.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 3.0],
                [1.0, 1.0, 1.1, 1.0],
                [0.0, 2.0, 4.0, 1.0],
                [0.0, 0.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        )

        min_p_array = np.array([0.1, 0.1, 0.26, 0.1, 0.1], dtype=np.float32)
        min_p_output, *_ = model.execute(np_input, min_p_array)
        result = cast(Buffer, min_p_output).to_numpy()
        assert (
            result[2, 0] == 2
        )  # 1.1 is the only logit that is greater than 0.26 after softmax


def test_greedy_acceptance_sampler(session: InferenceSession) -> None:
    """Tests greedy_acceptance_sampler (accept iff draft == argmax)."""
    device = session.devices[0]
    graph = build_greedy_acceptance_sampler_graph(
        device=DeviceRef.from_device(device)
    )
    model = session.load(graph)

    batch_size = 2
    num_steps = 4
    vocab_size = 6

    # Batch 0: target argmax = [1, 1, 3, 0, bonus=2], draft = [1, 1, 0, 0] → reject at 2
    # Batch 1: target argmax = [2, 4, 4, 5, bonus=1], draft = [2, 4, 4, 5] → all accepted
    target_logits = np.zeros(
        (batch_size * (num_steps + 1), vocab_size), dtype=np.float32
    )
    target_argmax = [[1, 1, 3, 0, 2], [2, 4, 4, 5, 1]]
    for b in range(batch_size):
        for s in range(num_steps + 1):
            target_logits[b * (num_steps + 1) + s, target_argmax[b][s]] = 10.0

    draft_tokens_np = np.array([[1, 1, 0, 0], [2, 4, 4, 5]], dtype=np.int64)

    first_rejected, target_tokens, bonus_tokens = model.execute(
        Buffer.from_numpy(draft_tokens_np).to(device),
        Buffer.from_numpy(target_logits).to(device),
    )
    first_rejected_np = cast(Buffer, first_rejected).to_numpy()
    target_tokens_np = cast(Buffer, target_tokens).to_numpy()
    bonus_tokens_np = cast(Buffer, bonus_tokens).to_numpy()

    assert first_rejected_np.shape == (batch_size,)
    assert target_tokens_np.shape == (batch_size, num_steps)
    assert bonus_tokens_np.shape == (batch_size, 1)
    np.testing.assert_array_equal(first_rejected_np, [2, num_steps])
    np.testing.assert_array_equal(
        target_tokens_np, [[1, 1, 3, 0], [2, 4, 4, 5]]
    )
    np.testing.assert_array_equal(bonus_tokens_np, [[2], [1]])


@pytest.mark.parametrize("device", [Accelerator(), CPU()])
def test_typical_acceptance_sampler(device: Device) -> None:
    """Tests stochastic mode (accept with prob p_target)."""
    session = InferenceSession(devices=[device])
    graph = build_stochastic_acceptance_sampler_graph(
        device=DeviceRef.from_device(device)
    )
    model = session.load(graph)

    def _make_inputs(
        batch_size: int,
        vocab_size: int,
        draft_tokens_np: npt.NDArray[np.int64],
        logits: npt.NDArray[np.float32],
    ) -> list[Buffer]:
        return [
            Buffer.from_numpy(draft_tokens_np).to(device),
            Buffer.from_numpy(logits).to(device),
            Buffer.from_numpy(np.ones(batch_size, dtype=np.float32)).to(device),
            Buffer.from_numpy(
                np.full(batch_size, vocab_size, dtype=np.int64)
            ).to(device),
            Buffer.from_numpy(np.array(vocab_size, dtype=np.int64)),
            Buffer.from_numpy(np.ones(batch_size, dtype=np.float32)).to(device),
            Buffer.from_numpy(np.array(1.0, dtype=np.float32)),
        ]

    vocab_size = 5

    # Test both num_steps = 0 and num_steps = 3
    # With num_steps = 0, some tensors will be empty.
    # The kernels should handle this gracefully and not crash.
    for num_steps in [0, 3]:
        # Case 1: draft token has ~1.0 probability → all accepted
        logits_high = np.full(
            (num_steps + 1, vocab_size), -100.0, dtype=np.float32
        )
        logits_high[:, 2] = 100.0
        draft_high = np.full((1, num_steps), 2, dtype=np.int64)

        first_rejected, recovered, bonus = model.execute(
            *_make_inputs(1, vocab_size, draft_high, logits_high)
        )
        assert cast(Buffer, first_rejected).to_numpy()[0] == num_steps
        assert cast(Buffer, recovered).to_numpy().shape == (1, num_steps)
        assert cast(Buffer, bonus).to_numpy().shape == (1, 1)

        # Case 2: draft token has ~0.0 probability → rejected at position 0
        logits_low = np.full(
            (num_steps + 1, vocab_size), -100.0, dtype=np.float32
        )
        logits_low[:, 0] = 100.0
        draft_low = np.full((1, num_steps), 4, dtype=np.int64)

        first_rejected, _, _ = model.execute(
            *_make_inputs(1, vocab_size, draft_low, logits_low)
        )
        assert cast(Buffer, first_rejected).to_numpy()[0] == 0
