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
"""``apply_rope_tail``'s kernel path against its graph-op path and NumPy.

The kernel path runs when the table holds one row per token of ``x``. Each
case checks both paths against a float64 reference rounded to bfloat16: the
leading columns pass through bit for bit and the rotated tail is within one
bfloat16 rounding. A case whose table broadcasts over ``x``'s batch takes the
graph-op path and is checked the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.pipelines.architectures.deepseekV4.layers.rope import (
    _rope_tail_kernel,
    _rope_tail_ops,
    apply_rope_tail,
)
from numpy.typing import NDArray

pytestmark = pytest.mark.skipif(accelerator_count() == 0, reason="needs a GPU")

ROPE = 64


def _bf16(a: NDArray[np.float64]) -> NDArray[np.float32]:
    """Rounds to the nearest bfloat16, ties to even, as float32."""
    bits = a.astype(np.float32).view(np.uint32)
    bits = bits + 0x7FFF + ((bits >> 16) & 1)
    return (bits & 0xFFFF0000).view(np.float32)


def _reference(
    x: NDArray[np.float32], table: NDArray[np.float32], inverse: bool
) -> NDArray[np.float32]:
    """``x`` with its trailing ``ROPE`` columns rotated, in float64."""
    out = x.astype(np.float64)
    tail = out[..., -ROPE:].reshape(*x.shape[:-1], ROPE // 2, 2)
    cos, sin = table[..., 0].astype(np.float64), table[..., 1]
    sin = -sin if inverse else sin
    # The table's leading axes are x's (batch, sequence), or its sequence
    # alone; it broadcasts over the axes between those and the pairs.
    lead = list(table.shape[:-2]) if table.ndim == 4 else [1, table.shape[0]]
    shape = lead + [1] * (x.ndim - 3) + [ROPE // 2]
    cos, sin = cos.reshape(shape), sin.reshape(shape)
    re, im = tail[..., 0], tail[..., 1]
    rot = np.stack([re * cos - im * sin, re * sin + im * cos], axis=-1)
    out[..., -ROPE:] = rot.reshape(*x.shape[:-1], ROPE)
    return _bf16(out)


# (x shape, table leading axes). A two-axis table gives each request its own
# positions; a one-axis table over a batch of x takes the graph-op path.
_CASES = [
    ((1, 7, 64, 512), (7,)),  # queries, one row per token
    ((1, 13, 512), (13,)),  # KV, no head axis
    ((1, 5, 32, 128), (5,)),  # indexer queries
    ((3, 2, 64, 512), (3, 2)),  # each request at its own position
    ((2, 6, 512), (6,)),  # table shared over a batch: graph ops
]
_INVERSE = [False, True]


@dataclass
class _Case:
    x: NDArray[np.float32]
    table: NDArray[np.float32]
    fast: NDArray[np.float32]
    slow: NDArray[np.float32]
    took: bool


@pytest.fixture(scope="module")
def cases() -> dict[tuple[int, bool], _Case]:
    """Runs every case and both paths in one graph, so the module compiles
    once."""
    device = Accelerator()
    ref = DeviceRef.from_device(device)
    inputs: list[NDArray[np.float32]] = []
    types: list[TensorType] = []
    for i, (x_shape, table_lead) in enumerate(_CASES):
        rng = np.random.default_rng(i)
        inputs.append(_bf16(rng.normal(0.0, 1.0, x_shape)))
        angles = rng.uniform(-np.pi, np.pi, (*table_lead, ROPE // 2))
        inputs.append(
            np.stack([np.cos(angles), np.sin(angles)], -1).astype(np.float32)
        )
        # Symbolic batch and sequence, named per case.
        x_dims: list[int | str] = [1 if x_shape[0] == 1 else f"b{i}", f"s{i}"]
        x_dims += list(x_shape[2:])
        t_dims = x_dims[:2] if len(table_lead) == 2 else [f"s{i}"]
        types.append(TensorType(DType.float32, x_dims, ref))
        types.append(TensorType(DType.float32, [*t_dims, ROPE // 2, 2], ref))
    took: dict[tuple[int, bool], bool] = {}
    with Graph("rope_tail", input_types=types) as graph:
        outputs = []
        for i in range(len(_CASES)):
            xv = graph.inputs[2 * i].tensor
            tv = graph.inputs[2 * i + 1].tensor
            xb = ops.cast(xv, DType.bfloat16)
            for inverse in _INVERSE:
                kernel = _rope_tail_kernel(xb, tv, ROPE, inverse)
                took[i, inverse] = kernel is not None
                fast = apply_rope_tail(xb, tv, ROPE, inverse=inverse)
                slow = _rope_tail_ops(xb, tv, ROPE, inverse)
                outputs += [
                    ops.cast(fast, DType.float32),
                    ops.cast(slow, DType.float32),
                ]
        graph.output(*outputs)
    model = InferenceSession(devices=[device]).load(graph)
    results = model.execute(*(Buffer.from_numpy(a).to(device) for a in inputs))
    host = [r.to(CPU()).to_numpy() for r in results]
    out: dict[tuple[int, bool], _Case] = {}
    for i in range(len(_CASES)):
        for j, inverse in enumerate(_INVERSE):
            k = 2 * (i * len(_INVERSE) + j)
            out[i, inverse] = _Case(
                inputs[2 * i],
                inputs[2 * i + 1],
                host[k],
                host[k + 1],
                took[i, inverse],
            )
    return out


def _check(
    got: NDArray[np.float32],
    want: NDArray[np.float32],
    x: NDArray[np.float32],
) -> None:
    np.testing.assert_array_equal(got[..., :-ROPE], x[..., :-ROPE])
    tol = 2.0**-7 * np.maximum(np.abs(got), np.abs(want)) + 2.0**-14
    bad = np.abs(got - want) > tol
    assert not bad.any(), f"{bad.sum()} tail values off by more than an ulp"


@pytest.mark.parametrize("inverse", _INVERSE)
@pytest.mark.parametrize("case", range(len(_CASES)))
def test_rope_tail(
    cases: dict[tuple[int, bool], _Case], case: int, inverse: bool
) -> None:
    c = cases[case, inverse]
    want = _reference(c.x, c.table, inverse)
    # The rotation moves the tail; an identity kernel would pass the head
    # check alone.
    assert np.abs(want[..., -ROPE:] - c.x[..., -ROPE:]).mean() > 0.1
    x_shape, table_lead = _CASES[case]
    assert c.took == (len(table_lead) == 2 or x_shape[0] == 1)
    _check(c.fast, want, c.x)
    _check(c.slow, want, c.x)
