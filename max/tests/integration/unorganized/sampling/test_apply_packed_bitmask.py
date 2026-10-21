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
"""End-to-end graph tests for the fused ``apply_packed_bitmask`` op.

Builds a tiny graph that calls :func:`apply_packed_bitmask` and runs it on the
GPU, comparing against a numpy unpack + ``where`` reference. Covers both the
rank-2 (token sampler) and rank-3 (speculative-decode acceptance sampler) paths.
"""

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import (
    apply_packed_bitmask,
    apply_packed_bitmask_with_penalties,
)
from max.support.math import ceildiv

_FILL = -10000.0  # Value used for masked-out logits


def _packed_to_bool(packed: np.ndarray, vocab_size: int) -> np.ndarray:
    """Reference unpack of a packed int32 bitmask to a bool mask."""
    bits = 2 ** np.arange(32, dtype=np.int32)
    unpacked = (packed[..., np.newaxis] & bits) != 0
    unpacked = unpacked.reshape(*packed.shape[:-1], -1)
    return unpacked[..., :vocab_size]


def _random_inputs(
    shape: tuple[int, ...], vocab_size: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(shape, dtype=np.float32)
    packed_vocab = ceildiv(vocab_size, 32)
    packed = rng.integers(
        np.iinfo(np.int32).min,
        np.iinfo(np.int32).max,
        size=(*shape[:-1], packed_vocab),
        dtype=np.int32,
    )
    return logits, packed


@pytest.mark.parametrize(
    "shape",
    [
        (3, 40),  # rank-2: token sampler ([batch, vocab])
        (2, 3, 40),  # rank-3: acceptance sampler ([batch, num_pos, vocab])
        (2, 1, 64),  # vocab a multiple of 32 (no trailing padding)
    ],
)
def test_apply_packed_bitmask(
    session: InferenceSession, shape: tuple[int, ...]
) -> None:
    device = session.devices[0]
    gpu = DeviceRef.GPU()
    vocab_size = shape[-1]
    packed_vocab = ceildiv(vocab_size, 32)

    logits_np, packed_np = _random_inputs(shape, vocab_size)

    logits_type = TensorType(DType.float32, list(shape), device=gpu)
    packed_type = TensorType(
        DType.int32, [*shape[:-1], packed_vocab], device=gpu
    )
    with Graph(
        "apply_packed_bitmask", input_types=[logits_type, packed_type]
    ) as graph:
        logits, packed = (v.tensor for v in graph.inputs)
        graph.output(apply_packed_bitmask(logits, packed, fill_val=_FILL))

    model = session.load(graph)
    out = model(
        Buffer.from_numpy(logits_np).to(device),
        Buffer.from_numpy(packed_np).to(device),
    )[0]
    assert isinstance(out, Buffer)
    actual = out.to_numpy()

    keep = _packed_to_bool(packed_np, vocab_size)
    expected = np.where(keep, logits_np, np.float32(_FILL))
    np.testing.assert_array_equal(actual, expected)


def _run_with_penalties(
    session: InferenceSession,
    logits_np: np.ndarray,
    packed_np: np.ndarray,
    data_np: np.ndarray,
    offsets_np: np.ndarray,
    frequency_np: np.ndarray,
    presence_np: np.ndarray,
) -> np.ndarray:
    device = session.devices[0]
    gpu = DeviceRef.GPU()
    arrays = [
        logits_np,
        packed_np,
        data_np,
        offsets_np,
        frequency_np,
        presence_np,
    ]
    dtypes = [
        DType.float32,
        DType.int32,
        DType.int32,
        DType.uint32,
        DType.float32,
        DType.float32,
    ]
    input_types = [
        TensorType(dtype, list(a.shape), device=gpu)
        for a, dtype in zip(arrays, dtypes, strict=True)
    ]
    with Graph(
        "apply_packed_bitmask_with_penalties", input_types=input_types
    ) as graph:
        logits, packed, data, offsets, frequency, presence = (
            v.tensor for v in graph.inputs
        )
        graph.output(
            apply_packed_bitmask_with_penalties(
                logits, packed, _FILL, data, offsets, frequency, presence
            )
        )
    model = session.load(graph)
    out = model(*(Buffer.from_numpy(a).to(device) for a in arrays))[0]
    assert isinstance(out, Buffer)
    return out.to_numpy()


def test_penalties_ride_the_mask_and_skip_masked_tokens(
    session: InferenceSession,
) -> None:
    shape = (2, 2, 40)
    vocab_size = shape[-1]
    logits_np, packed_np = _random_inputs(shape, vocab_size)
    logits_np[1, 0, 9] = -np.inf
    keep = _packed_to_bool(packed_np, vocab_size)
    kept = np.flatnonzero(keep[0, 0])
    masked = np.flatnonzero(~keep[0, 0])

    # Logit rows 0..3; row 2 has no entries; padding sits in the last row.
    rows = [
        [[int(kept[0]), 3], [int(masked[0]), 5]],
        [[int(kept[1]), 1]],
        [],
        [[9, 2], [int(kept[0]), 1]],
    ]
    data, offsets = [], [0]
    for row in rows:
        data.extend(row)
        offsets.append(len(data))
    data.extend([[-1, 0]] * 3)
    offsets[-1] = len(data)
    data_np = np.array(data, dtype=np.int32)
    offsets_np = np.array(offsets, dtype=np.uint32)
    frequency_np = np.array([0.5, -2.0, 1.0, 1.5], dtype=np.float32)
    presence_np = np.array([-1.0, 0.25, 1.0, 2.0], dtype=np.float32)

    actual = _run_with_penalties(
        session,
        logits_np,
        packed_np,
        data_np,
        offsets_np,
        frequency_np,
        presence_np,
    )

    expected = np.where(keep, logits_np, np.float32(_FILL)).reshape(4, -1)
    expected[np.isneginf(logits_np.reshape(4, -1))] = -np.inf
    keep_2d = keep.reshape(4, -1)
    for r, row in enumerate(rows):
        for token, count in row:
            if keep_2d[r, token] and np.isfinite(expected[r, token]):
                expected[r, token] -= np.float32(
                    frequency_np[r] * count + presence_np[r]
                )
    np.testing.assert_allclose(actual.reshape(4, -1), expected, rtol=1e-6)
    # The masked token kept its fill despite a negative presence penalty.
    assert actual[0, 0, masked[0]] == np.float32(_FILL)


def test_zero_penalties_match_the_plain_mask_bit_for_bit(
    session: InferenceSession,
) -> None:
    shape = (3, 2, 64)
    vocab_size = shape[-1]
    logits_np, packed_np = _random_inputs(shape, vocab_size)
    rows = shape[0] * shape[1]
    data_np = np.array([[t, 4] for t in range(rows)], dtype=np.int32)
    offsets_np = np.arange(rows + 1, dtype=np.uint32)
    zeros = np.zeros(rows, dtype=np.float32)

    actual = _run_with_penalties(
        session, logits_np, packed_np, data_np, offsets_np, zeros, zeros
    )

    keep = _packed_to_bool(packed_np, vocab_size)
    expected = np.where(keep, logits_np, np.float32(_FILL))
    np.testing.assert_array_equal(actual, expected)
