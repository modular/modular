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
"""Test the max.graph Python bindings for allreduce_rms_norm.

The composite op has no MLIR verifier, so this builder is the only place a
bad dtype, an input-shape mismatch or a wrong-length gamma is rejected with
a diagnostic rather than a Mojo comptime assert.
"""

from collections.abc import Sequence

import pytest
from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.legacy import Signals

H = 128


def _graph_inputs(
    shapes: list[list[int]], devices: list[DeviceRef]
) -> list[TensorType]:
    """Per-device inputs, then per-device gammas (length = cols)."""
    return [
        TensorType(dtype=DType.bfloat16, shape=shape, device=dev)
        for shape, dev in zip(shapes, devices, strict=True)
    ] + [
        TensorType(dtype=DType.bfloat16, shape=[shape[-1]], device=dev)
        for shape, dev in zip(shapes, devices, strict=True)
    ]


def _build(
    shapes: list[list[int]], devices: list[DeviceRef]
) -> tuple[list[TensorValue], list[TensorValue]]:
    num_gpus = len(devices)
    signals = Signals(devices)
    with Graph(
        "allreduce_rms_norm",
        input_types=_graph_inputs(shapes, devices)
        + list(signals.input_types()),
    ) as graph:
        normed, summed = ops.allreduce_rms_norm(
            inputs=[v.tensor for v in graph.inputs[:num_gpus]],
            signal_buffers=[v.buffer for v in graph.inputs[2 * num_gpus :]],
            gammas=[v.tensor for v in graph.inputs[num_gpus : 2 * num_gpus]],
            epsilon=1e-5,
        )
        graph.output(*normed, *summed)
        return normed, summed


def _expect_rejected(
    input_types: Sequence[TensorType | BufferType], match: str
) -> None:
    """Builds the op over these inputs and asserts it rejects them."""
    with Graph("allreduce_rms_norm", input_types=input_types) as graph:
        with pytest.raises(ValueError, match=match):
            ops.allreduce_rms_norm(
                inputs=[v.tensor for v in graph.inputs[:2]],
                signal_buffers=[v.buffer for v in graph.inputs[4:]],
                gammas=[v.tensor for v in graph.inputs[2:4]],
                epsilon=1e-5,
            )


def test_allreduce_rms_norm_full_world_shapes() -> None:
    """Every device holds the sum's shape: the outputs equal the input."""
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    normed, summed = _build([[3, H]] * num_gpus, devices)

    for dev_idx, device in enumerate(devices):
        assert normed[dev_idx].device == device
        assert normed[dev_idx].shape == [3, H]
        assert normed[dev_idx].dtype == DType.bfloat16
        assert summed[dev_idx].device == device
        assert summed[dev_idx].shape == [3, H]
        assert summed[dev_idx].dtype == DType.bfloat16


@pytest.mark.parametrize("multiply_before_cast", [False, True])
def test_allreduce_rms_norm_carries_the_norm_attrs(
    multiply_before_cast: bool,
) -> None:
    """multiply_before_cast is carried into the op's MLIR verbatim."""
    num_gpus = 2
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    signals = Signals(devices)
    with Graph(
        "allreduce_rms_norm_attrs",
        input_types=_graph_inputs([[5, H]] * num_gpus, devices)
        + list(signals.input_types()),
    ) as graph:
        normed, _summed = ops.allreduce_rms_norm(
            inputs=[v.tensor for v in graph.inputs[:num_gpus]],
            signal_buffers=[v.buffer for v in graph.inputs[2 * num_gpus :]],
            gammas=[v.tensor for v in graph.inputs[num_gpus : 2 * num_gpus]],
            epsilon=1e-5,
            weight_offset=1.0,
            multiply_before_cast=multiply_before_cast,
        )
        graph.output(*normed)

    ir = str(graph._mlir_op)
    assert "mo.composite.distributed.allreduce_rms_norm" in ir
    # The op prints the attribute in MLIR's lowercase boolean form.
    assert f"multiply_before_cast = {str(multiply_before_cast).lower()}" in ir


def test_allreduce_rms_norm_single_input_rejected() -> None:
    """The fused kernel asserts ngpus >= 2; one input is a no-op reduce."""
    devices = [DeviceRef.GPU(id=0)]
    with pytest.raises(ValueError, match=r"at least two inputs"):
        _build([[8, H]], devices)


def test_allreduce_rms_norm_signal_buffer_count_must_match() -> None:
    devices = [DeviceRef.GPU(id=i) for i in range(2)]
    signals = Signals(devices)
    with Graph(
        "allreduce_rms_norm",
        input_types=_graph_inputs([[8, H]] * 2, devices)
        + list(signals.input_types()),
    ) as graph:
        with pytest.raises(
            ValueError,
            match=r"number of signal buffers",
        ):
            ops.allreduce_rms_norm(
                inputs=[v.tensor for v in graph.inputs[:2]],
                signal_buffers=[graph.inputs[4].buffer],
                gammas=[v.tensor for v in graph.inputs[2:4]],
                epsilon=1e-5,
            )


def test_allreduce_rms_norm_gamma_count_must_match() -> None:
    devices = [DeviceRef.GPU(id=i) for i in range(2)]
    with Graph(
        "allreduce_rms_norm",
        input_types=_graph_inputs([[8, H]] * 2, devices)
        + list(Signals(devices).input_types()),
    ) as graph:
        with pytest.raises(ValueError, match=r"number of gammas"):
            ops.allreduce_rms_norm(
                inputs=[v.tensor for v in graph.inputs[:2]],
                signal_buffers=[v.buffer for v in graph.inputs[4:]],
                gammas=[graph.inputs[2].tensor],
                epsilon=1e-5,
            )


def test_allreduce_rms_norm_bf16_only() -> None:
    """The fused kernel assumes bfloat16 operands."""
    devices = [DeviceRef.GPU(id=i) for i in range(2)]
    _expect_rejected(
        [
            TensorType(dtype=DType.float32, shape=[8, H], device=dev)
            for dev in devices
        ]
        + [
            TensorType(dtype=DType.float32, shape=[H], device=dev)
            for dev in devices
        ]
        + list(Signals(devices).input_types()),
        r"bfloat16-only",
    )


def test_allreduce_rms_norm_dtype_mismatch_rejected() -> None:
    devices = [DeviceRef.GPU(id=i) for i in range(2)]
    _expect_rejected(
        [
            TensorType(dtype=DType.bfloat16, shape=[8, H], device=devices[0]),
            TensorType(dtype=DType.float32, shape=[8, H], device=devices[1]),
        ]
        + [
            TensorType(dtype=DType.bfloat16, shape=[H], device=dev)
            for dev in devices
        ]
        + list(Signals(devices).input_types()),
        r"same dtype",
    )


def test_allreduce_rms_norm_shape_mismatch_rejected() -> None:
    """Every device contributes the same-sized shard to the sum."""
    devices = [DeviceRef.GPU(id=i) for i in range(2)]
    with pytest.raises(ValueError, match=r"same shape across all input"):
        _build([[8, H], [9, H]], devices)


def test_allreduce_rms_norm_rank_three_rejected() -> None:
    """The fused kernel indexes rows/cols directly, so 2D only."""
    devices = [DeviceRef.GPU(id=i) for i in range(2)]
    with pytest.raises(ValueError, match=r"2D-only"):
        _build([[4, H, 1]] * 2, devices)


def test_allreduce_rms_norm_gamma_length_must_match_cols() -> None:
    devices = [DeviceRef.GPU(id=i) for i in range(2)]
    _expect_rejected(
        [
            TensorType(dtype=DType.bfloat16, shape=[8, H], device=dev)
            for dev in devices
        ]
        + [
            TensorType(dtype=DType.bfloat16, shape=[H + 1], device=dev)
            for dev in devices
        ]
        + list(Signals(devices).input_types()),
        r"gamma to match its input's last dimension",
    )


def test_allreduce_rms_norm_gamma_dtype_rejected() -> None:
    """The kernel loads gamma in the input dtype; anything else fails late."""
    devices = [DeviceRef.GPU(id=i) for i in range(2)]
    _expect_rejected(
        [
            TensorType(dtype=DType.bfloat16, shape=[8, H], device=dev)
            for dev in devices
        ]
        + [
            TensorType(dtype=DType.float32, shape=[H], device=dev)
            for dev in devices
        ]
        + list(Signals(devices).input_types()),
        r"gamma to match its input's dtype",
    )


def test_allreduce_rms_norm_gamma_device_rejected() -> None:
    """gammas[i] is read on device i; a gamma parked elsewhere is a bug."""
    devices = [DeviceRef.GPU(id=i) for i in range(2)]
    _expect_rejected(
        [
            TensorType(dtype=DType.bfloat16, shape=[8, H], device=dev)
            for dev in devices
        ]
        + [
            TensorType(dtype=DType.bfloat16, shape=[H], device=devices[0]),
            TensorType(dtype=DType.bfloat16, shape=[H], device=DeviceRef.CPU()),
        ]
        + list(Signals(devices).input_types()),
        r"gamma to sit on its input's device",
    )


def test_allreduce_rms_norm_repeated_device_rejected() -> None:
    """A collective over the same device twice has no peer to reduce with."""
    devices = [DeviceRef.GPU(id=0), DeviceRef.GPU(id=0)]
    with pytest.raises(ValueError, match=r"unique devices"):
        _build([[8, H]] * 2, devices)
