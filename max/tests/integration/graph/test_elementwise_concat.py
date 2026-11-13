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
"""A concat fused into an elementwise loop reads its inputs in place.

Over the innermost dim, the loop's width divides every input's extent, so no
vector reads across two inputs. Distinct input values catch one that does.
"""

from __future__ import annotations

import numpy as np
import pytest
from fusion_utils import run_and_verify_fusion
from max.driver import accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

_DEVICES = [
    DeviceRef.CPU(),
    pytest.param(
        DeviceRef.GPU(),
        marks=pytest.mark.skipif(accelerator_count() == 0, reason="no GPU"),
    ),
]

_CONCAT_EXP = r"^(?=.*mo\.concat)(?=.*mo\.exp)"


def _inputs(*shapes: list[int]) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [
        rng.standard_normal(s).astype(np.float32) * 0.1 + i
        for i, s in enumerate(shapes)
    ]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    ("shapes", "axis"),
    [
        # Innermost axis, width limit gcd(6, 10) = 2.
        ([[4, 6], [4, 10]], 1),
        # Innermost axis, width limit 1.
        ([[4, 5], [4, 6]], 1),
        # Innermost axis, wide inputs: no effective limit.
        ([[8, 512], [8, 768]], 1),
        # Outer axis: a vector stays in one input.
        ([[3, 16], [5, 16]], 0),
    ],
)
def test_exp_of_concat(
    session: InferenceSession,
    device: DeviceRef,
    shapes: list[list[int]],
    axis: int,
) -> None:
    with Graph(
        "exp_of_concat",
        input_types=[
            TensorType(DType.float32, s, device=device) for s in shapes
        ],
    ) as graph:
        a, b = (v.tensor for v in graph.inputs)
        graph.output(ops.exp(ops.concat([a, b], axis=axis)))

    a_np, b_np = _inputs(*shapes)
    (out,) = run_and_verify_fusion(
        session, graph, a_np, b_np, fused=_CONCAT_EXP
    )
    expected = np.exp(np.concatenate([a_np, b_np], axis=axis))
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "extents",
    [
        # The limit is picked at launch: the preferred width, 2, then 1.
        (64, 128),
        (6, 10),
        (5, 8),
    ],
)
def test_exp_of_dynamic_concat(
    session: InferenceSession,
    device: DeviceRef,
    extents: tuple[int, int],
) -> None:
    with Graph(
        "exp_of_dynamic_concat",
        input_types=[
            TensorType(DType.float32, [4, "n0"], device=device),
            TensorType(DType.float32, [4, "n1"], device=device),
        ],
    ) as graph:
        a, b = (v.tensor for v in graph.inputs)
        graph.output(ops.exp(ops.concat([a, b], axis=1)))

    a_np, b_np = _inputs([4, extents[0]], [4, extents[1]])
    (out,) = run_and_verify_fusion(
        session, graph, a_np, b_np, fused=_CONCAT_EXP
    )
    expected = np.exp(np.concatenate([a_np, b_np], axis=1))
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", _DEVICES)
def test_exp_of_nested_concat(
    session: InferenceSession, device: DeviceRef
) -> None:
    """An outer concat of two innermost ones: the inner extents limit the
    width, gcd(4, 12, 8, 8) = 4."""
    shapes = [[2, 4], [2, 12], [2, 8], [2, 8]]
    with Graph(
        "exp_of_nested_concat",
        input_types=[
            TensorType(DType.float32, s, device=device) for s in shapes
        ],
    ) as graph:
        a, b, c, d = (v.tensor for v in graph.inputs)
        ab = ops.concat([a, b], axis=1)
        cd = ops.concat([c, d], axis=1)
        graph.output(ops.exp(ops.concat([ab, cd], axis=0)))

    arrays = _inputs(*shapes)
    (out,) = run_and_verify_fusion(session, graph, *arrays, fused=_CONCAT_EXP)
    expected = np.exp(
        np.concatenate(
            [
                np.concatenate(arrays[:2], axis=1),
                np.concatenate(arrays[2:], axis=1),
            ],
            axis=0,
        )
    )
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
