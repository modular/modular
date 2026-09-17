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
"""End-to-end tests for `mo.batch_matmul` with a fused elementwise epilogue.

The GPU batched matmul collapses every leading batch dimension into one and
runs a rank-3 kernel, but a fused epilogue stores through the graph tensor's
original rank. Handing the epilogue a rank-3 coordinate for a rank-4 output
failed to compile (GEX-4200) on the tiled dispatch path -- static N and K,
`N % 128 == 0`, `K % 32 == 0`, `K >= 128`, batch > 1 -- which is the shape an
attention score matmul takes.

Each rank-4 case is paired with its rank-3 equivalent, and every case checks
the result against numpy, so a coordinate that compiles but lands in the
wrong place fails here too.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import max.driver as md
import numpy as np
import numpy.typing as npt
import pytest
from max.driver import accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops

_Array = npt.NDArray[np.float32]

# An awkward sequence length: not a multiple of any tile size, so the
# collapsed batch index has to be decomposed rather than guessed.
_SEQ_LEN = 198


def _scaled(operands: list[TensorValue]) -> TensorValue:
    a, b = operands
    return ops.matmul(a, b) * 0.5


def _plain(operands: list[TensorValue]) -> TensorValue:
    a, b = operands
    return ops.matmul(a, b)


def _divided(operands: list[TensorValue]) -> TensorValue:
    a, b, divisor = operands
    return ops.matmul(a, b) / divisor


def _ref_scaled(arrays: list[_Array]) -> _Array:
    a, b = arrays
    return (a @ b) * np.float32(0.5)


def _ref_plain(arrays: list[_Array]) -> _Array:
    a, b = arrays
    return a @ b


def _ref_divided(arrays: list[_Array]) -> _Array:
    a, b, divisor = arrays
    return (a @ b) / divisor


@dataclass(frozen=True)
class _Case:
    """One batched-matmul graph, its numpy reference, and its input shapes."""

    name: str
    shapes: list[list[int]]
    build: Callable[[list[TensorValue]], TensorValue]
    reference: Callable[[list[_Array]], _Array]
    divisor_index: int | None = None


_CASES = [
    # N=128 and K=512 put these two on the tiled dispatch, where the rank-4
    # epilogue coordinate regressed.
    _Case(
        "r4_scores_epi",
        [[1, _SEQ_LEN, 64, 512], [1, _SEQ_LEN, 512, 128]],
        _scaled,
        _ref_scaled,
    ),
    _Case(
        "r4_scores_plain",
        [[1, _SEQ_LEN, 64, 512], [1, _SEQ_LEN, 512, 128]],
        _plain,
        _ref_plain,
    ),
    _Case(
        "r3_scores_epi",
        [[_SEQ_LEN, 64, 512], [_SEQ_LEN, 512, 128]],
        _scaled,
        _ref_scaled,
    ),
    _Case(
        "r4_out_div",
        [[1, _SEQ_LEN, 64, 128], [1, _SEQ_LEN, 128, 512], [1, _SEQ_LEN, 64, 1]],
        _divided,
        _ref_divided,
        divisor_index=2,
    ),
    _Case(
        "r3_out_div",
        [[_SEQ_LEN, 64, 128], [_SEQ_LEN, 128, 512], [_SEQ_LEN, 64, 1]],
        _divided,
        _ref_divided,
        divisor_index=2,
    ),
    # N=52 is not a multiple of 128, so this one falls to the naive kernel.
    _Case(
        "r4_small_n_epi",
        [[1, _SEQ_LEN, 64, 512], [1, _SEQ_LEN, 512, 52]],
        _scaled,
        _ref_scaled,
    ),
]


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
@pytest.mark.parametrize("case", _CASES, ids=[case.name for case in _CASES])
def test_batched_matmul_epilogue(
    session: InferenceSession, case: _Case
) -> None:
    gpu = DeviceRef.GPU()
    with Graph(
        case.name,
        input_types=[
            TensorType(DType.float32, shape, device=gpu)
            for shape in case.shapes
        ],
    ) as graph:
        graph.output(case.build([value.tensor for value in graph.inputs]))

    model = session.load(graph)
    device = model.input_devices[0]

    rng = np.random.default_rng(0)
    # NVIDIA multiplies fp32 through tf32 tensor cores, whose absolute error
    # scales with the result, not with the operands. Scaling the LHS by
    # 1/sqrt(K) puts the product near unit magnitude so the absolute
    # tolerance below means the same thing on every case.
    lhs_scale = 1.0 / np.sqrt(case.shapes[0][-1])
    arrays: list[_Array] = []
    for index, shape in enumerate(case.shapes):
        if index == case.divisor_index:
            # Bounded away from zero, so the division can't amplify the
            # matmul's own error past the tolerance below.
            values = rng.uniform(0.5, 1.5, shape)
        elif index == 0:
            values = rng.standard_normal(shape) * lhs_scale
        else:
            values = rng.standard_normal(shape)
        arrays.append(values.astype(np.float32))

    outputs = model.execute(
        *[md.Buffer.from_numpy(array).to(device) for array in arrays]
    )
    assert len(outputs) == 1
    output = outputs[0]
    assert isinstance(output, md.Buffer)

    np.testing.assert_allclose(
        output.to(md.CPU()).to_numpy(),
        case.reference(arrays),
        rtol=1e-2,
        atol=1e-2,
    )
