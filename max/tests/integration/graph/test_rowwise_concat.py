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
"""A concat fused into a reduction reads its inputs in place.

The concat's select sits in the fold, or in the map phase after it. Over the
innermost dim, the kernel's width divides every input's extent, and every
vector starts on a multiple of it, so no vector reads across two inputs.
Distinct input values catch one that does.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from fusion_utils import run_and_verify_fusion
from max.driver import accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops

_DEVICES = [
    DeviceRef.CPU(),
    pytest.param(
        DeviceRef.GPU(),
        marks=pytest.mark.skipif(accelerator_count() == 0, reason="no GPU"),
    ),
]

_REDUCTIONS: dict[
    str,
    tuple[
        Callable[[TensorValue, int], TensorValue],
        Callable[..., np.ndarray],
    ],
] = {
    "sum": (lambda x, axis: ops.sum(x, axis=axis), np.sum),
    "max": (lambda x, axis: ops.max(x, axis=axis), np.max),
    "min": (lambda x, axis: ops.min(x, axis=axis), np.min),
    "mean": (lambda x, axis: ops.mean(x, axis=axis), np.mean),
}


def _check(
    session: InferenceSession,
    device: DeviceRef,
    shapes: tuple[list[int], list[int]],
    concat_axis: int,
    reduce_axis: int,
    reduction: str,
) -> None:
    build, reference = _REDUCTIONS[reduction]
    with Graph(
        "rowwise_concat",
        input_types=[
            TensorType(DType.float32, s, device=device) for s in shapes
        ],
    ) as graph:
        a, b = (v.tensor for v in graph.inputs)
        graph.output(build(ops.concat([a, b], axis=concat_axis), reduce_axis))

    rng = np.random.default_rng(0)
    a_np = rng.standard_normal(shapes[0]).astype(np.float32)
    b_np = rng.standard_normal(shapes[1]).astype(np.float32) + 10
    (out,) = run_and_verify_fusion(
        session, graph, a_np, b_np, fused=r"mo\.concat.*mo\.reduce"
    )
    expected = reference(
        np.concatenate([a_np, b_np], axis=concat_axis),
        axis=reduce_axis,
        keepdims=True,
    )
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", _DEVICES)
# Each layout with one reduction, the four in turn: they share the fold path,
# so a full cross product only multiplies compiles.
@pytest.mark.parametrize(
    ("shapes", "concat_axis", "reduce_axis", "reduction"),
    [
        # Over the concat axis, innermost: width limit gcd(6, 10) = 2.
        (([4, 6], [4, 10]), 1, 1, "sum"),
        (([3, 5], [3, 7]), 1, 1, "max"),
        # Over an outer concat axis, across an innermost one.
        (([3, 8], [5, 8]), 0, 0, "min"),
        (([3, 8], [5, 8]), 0, 1, "mean"),
        (([4, 6], [4, 10]), 1, 0, "sum"),
        # A 3-D concat over the middle axis, reduced over the last.
        (([2, 3, 12], [2, 5, 12]), 1, 2, "max"),
    ],
    ids=["inner", "inner_odd", "outer", "across", "outer_of_inner", "middle"],
)
def test_reduction_of_concat(
    session: InferenceSession,
    device: DeviceRef,
    reduction: str,
    shapes: tuple[list[int], list[int]],
    concat_axis: int,
    reduce_axis: int,
) -> None:
    _check(session, device, shapes, concat_axis, reduce_axis, reduction)


@pytest.mark.parametrize("device", _DEVICES)
def test_long_row_of_concat(
    session: InferenceSession, device: DeviceRef
) -> None:
    """Two long rows of 16386 + 16384 at width 2. On CPU they run the
    cooperative tier: the split-axis tier needs rows of 262,144 or more, so
    its stripes are tested directly in test_elementwise_cpu_chunk_alignment."""
    _check(session, device, ([2, 16386], [2, 16384]), 1, 1, "sum")


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "shape",
    # Width limits 8 (the half), 1 (an odd half) and 4 on a 3-D row.
    [[4, 16], [3, 10], [2, 3, 8]],
    ids=["even_half", "odd_half", "3d"],
)
def test_rotate_half_after_reduction(
    session: InferenceSession, device: DeviceRef, shape: list[int]
) -> None:
    """A concat over the reduced axis after the reduction joins its map
    phase, where each element reads its partner in the other half of the
    row: rotate-half, as in RoPE."""
    half = shape[-1] // 2
    with Graph(
        "rotate_half_after_reduction",
        input_types=[TensorType(DType.float32, shape, device=device)] * 3,
    ) as graph:
        x, cos, sin = (v.tensor for v in graph.inputs)
        y = x - ops.max(x, axis=-1)
        y1, y2 = ops.split(y, [half, half], axis=-1)
        graph.output(y * cos + ops.concat([-y2, y1], axis=-1) * sin)

    rng = np.random.default_rng(0)
    x_np, cos_np, sin_np = (
        rng.standard_normal(shape).astype(np.float32) for _ in range(3)
    )
    (out,) = run_and_verify_fusion(
        session,
        graph,
        x_np,
        cos_np,
        sin_np,
        fused=r"mo\.reduce\.max.*mo\.concat",
    )
    y_np = x_np - x_np.max(axis=-1, keepdims=True)
    rotated = np.concatenate([-y_np[..., half:], y_np[..., :half]], axis=-1)
    expected = y_np * cos_np + rotated * sin_np
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
