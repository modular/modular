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
"""Test the max.graph Python bindings for allreduce_post_norm_residual_rms_norm.

The composite op has no MLIR verifier and its kernel reads both mix weights
and one 8-column pack per thread of one block, so this builder is where a
short mix or a width the kernel cannot hold is rejected.
"""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    DimLike,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.legacy import Signals

NUM_GPUS = 2
DEVICES = [DeviceRef.GPU(id=i) for i in range(NUM_GPUS)]


def _build(
    cols: DimLike = 128, mix_shape: list[DimLike] | None = None
) -> tuple[list[TensorValue], list[TensorValue]]:
    """Builds the op over ``[rows, cols]`` inputs on two GPUs."""
    mix_shape = [2] if mix_shape is None else mix_shape
    row = [
        TensorType(DType.bfloat16, ["rows", cols], device=d) for d in DEVICES
    ]
    col = [TensorType(DType.bfloat16, [cols], device=d) for d in DEVICES]
    mix = [TensorType(DType.float32, mix_shape, device=d) for d in DEVICES]
    input_types: list[TensorType | BufferType] = [
        *row,
        *row,
        *col,
        *col,
        *mix,
        *Signals(DEVICES).input_types(),
    ]
    with Graph(
        "allreduce_post_norm_residual_rms_norm", input_types=input_types
    ) as graph:
        n = NUM_GPUS
        groups = [
            [v.tensor for v in graph.inputs[i * n : (i + 1) * n]]
            for i in range(5)
        ]
        joined, normed = ops.allreduce_post_norm_residual_rms_norm(
            groups[0],
            [v.buffer for v in graph.inputs[5 * n :]],
            groups[1],
            groups[2],
            groups[3],
            groups[4],
            1e-6,
        )
        graph.output(*joined, *normed)
        assert (
            "mo.composite.distributed.allreduce_post_norm_residual_rms_norm"
            in str(graph._mlir_op)
        )
        return joined, normed


@pytest.mark.parametrize("cols", [8, 5120, 8192, "cols"])
def test_allreduce_post_norm_residual_rms_norm_builds(cols: DimLike) -> None:
    """A width the kernel holds, or a symbolic one, builds the op."""
    joined, normed = _build(cols)
    for outputs in (joined, normed):
        for device, out in zip(DEVICES, outputs, strict=True):
            assert out.device == device
            assert out.dtype == DType.bfloat16
            assert out.shape == ["rows", cols]


@pytest.mark.parametrize("mix_shape", [[0], [1], [3], ["n"]])
def test_allreduce_post_norm_residual_rms_norm_mix_must_be_two(
    mix_shape: list[DimLike],
) -> None:
    """The kernel reads both mix weights, so a shorter mix reads past it."""
    with pytest.raises(ValueError, match=r"mix on device 0 must be \[2\]"):
        _build(mix_shape=mix_shape)


@pytest.mark.parametrize("cols", [12, 8200])
def test_allreduce_post_norm_residual_rms_norm_cols_must_fit_one_block(
    cols: int,
) -> None:
    """A row is whole 8-column packs, at most one per thread of a block."""
    with pytest.raises(ValueError, match=r"one thread block"):
        _build(cols)
