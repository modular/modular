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
"""Test the max.graph Python bindings for reduce_scatter_rms_norm."""

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn import Signals

H = 128


def _graph_inputs(
    shapes: list[list[int]], devices: list[DeviceRef], signals: Signals
) -> list[TensorType]:
    """Per-device activations, then per-device gammas, then the signal buffers."""
    return [
        TensorType(dtype=DType.bfloat16, shape=shape, device=dev)
        for shape, dev in zip(shapes, devices, strict=True)
    ] + [
        TensorType(dtype=DType.bfloat16, shape=[H], device=dev)
        for dev in devices
    ]


def _build(
    shapes: list[list[int]],
    devices: list[DeviceRef],
    group_size: int | None = None,
) -> tuple[list[TensorValue], list[TensorValue]]:
    num_gpus = len(devices)
    signals = Signals(devices)
    with Graph(
        "reduce_scatter_rms_norm",
        input_types=_graph_inputs(shapes, devices, signals)
        + list(signals.input_types()),
    ) as graph:
        normed, residual = ops.reduce_scatter_rms_norm(
            inputs=[v.tensor for v in graph.inputs[:num_gpus]],
            signal_buffers=[v.buffer for v in graph.inputs[2 * num_gpus :]],
            gammas=[v.tensor for v in graph.inputs[num_gpus : 2 * num_gpus]],
            epsilon=1e-6,
            group_size=group_size,
        )
        graph.output(*normed, *residual)
        return normed, residual


def test_reduce_scatter_rms_norm_full_world_shapes() -> None:
    """Default group_size is the whole world; rows split across all devices."""
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    normed, residual = _build([[9, H]] * num_gpus, devices)

    # 9 rows over 4 devices -> 3,2,2,2 (remainder to the low ranks).
    expected_rows = [3, 2, 2, 2]
    for dev_idx, device in enumerate(devices):
        for out in (normed[dev_idx], residual[dev_idx]):
            assert out.device == device
            assert out.shape == [expected_rows[dev_idx], H]


def test_reduce_scatter_rms_norm_grouped_ragged() -> None:
    """Grouped ragged binning keys off the GROUP, not the device count.

    Rows are hard-coded rather than recomputed from the implementation's own
    formula: 8 devices as 2 groups of 4, with different per-group row counts, so
    a global-rank or world-sized divisor gives visibly wrong shapes.
    """
    num_gpus = 8
    group_size = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    shapes = [[5, H]] * group_size + [[3, H]] * group_size
    # group 0 (5 rows over 4) -> 2,1,1,1 ; group 1 (3 rows over 4) -> 1,1,1,0
    expected_rows = [2, 1, 1, 1, 1, 1, 1, 0]

    normed, residual = _build(shapes, devices, group_size=group_size)
    for dev_idx, device in enumerate(devices):
        for out in (normed[dev_idx], residual[dev_idx]):
            assert out.device == device
            assert out.shape == [expected_rows[dev_idx], H]


def test_reduce_scatter_rms_norm_group_size_must_divide_inputs() -> None:
    """A group that does not tile the device list is rejected."""
    num_gpus = 6
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    with pytest.raises(
        ValueError,
        match=r"group_size to evenly divide the number of input tensors",
    ):
        _build([[8, H]] * num_gpus, devices, group_size=4)


def test_reduce_scatter_rms_norm_group_size_one_rejected() -> None:
    """group_size=1 is a no-op reduction; the kernel asserts ngpus >= 2."""
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    with pytest.raises(ValueError, match=r"group_size to be at least 2"):
        _build([[8, H]] * num_gpus, devices, group_size=1)


def test_reduce_scatter_rms_norm_shape_mismatch_within_group() -> None:
    """Shapes must agree WITHIN a group even though groups may differ."""
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    shapes = [[8, H], [7, H], [8, H], [8, H]]
    with pytest.raises(
        ValueError,
        match=r"same shape across all input tensors in each group",
    ):
        _build(shapes, devices, group_size=2)


def test_reduce_scatter_rms_norm_differing_shapes_across_groups_ok() -> None:
    """DP replicas are independent collectives, so their shapes may differ.

    The pre-grouping builder compared every input against ``inputs[0]``, which
    rejects this outright.
    """
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    shapes = [[8, H], [8, H], [4, H], [4, H]]
    normed, residual = _build(shapes, devices, group_size=2)

    expected_rows = [4, 4, 2, 2]
    for dev_idx in range(num_gpus):
        for out in (normed[dev_idx], residual[dev_idx]):
            assert out.shape == [expected_rows[dev_idx], H]


def test_reduce_scatter_rms_norm_rank_mismatch_rejected() -> None:
    """Rank must match across the whole world, not just within a group."""
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    shapes = [[8, H], [8, H], [4, H, 1], [4, H, 1]]
    with pytest.raises(ValueError, match=r"same rank across all input tensors"):
        _build(shapes, devices, group_size=2)
