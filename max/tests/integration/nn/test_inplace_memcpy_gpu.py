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
"""Integration tests for the ``mo.inplace_memcpy`` custom op.

The op copies a read-only ``src`` tensor into a mutable ``dst`` buffer in
place, similar to ``Buffer.inplace_copy_from`` but usable from within a
compiled MAX graph. The op supports the four direction combinations that
can be expressed with a single ``DeviceContext`` (D2D, H2D, D2H, H2H);
cross-GPU memcpy is rejected at graph build time.
"""

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, ops
from max.nn import kernels


def _build_fill_with_42_graph(
    *, dst_device: DeviceRef, src_device: DeviceRef, shape: list[int]
) -> Graph:
    """Builds a one-op graph that fills a caller-provided int32 ``dst``
    buffer with 42s using ``kernels.inplace_memcpy``. The 42-valued src
    constant is materialized on ``src_device``.
    """
    with Graph(
        "inplace_memcpy_fill_42",
        input_types=[BufferType(DType.int32, shape, device=dst_device)],
    ) as graph:
        dst = graph.inputs[0].buffer
        src = ops.constant(
            np.full(shape, 42, dtype=np.int32),
            dtype=DType.int32,
            device=src_device,
        )
        kernels.inplace_memcpy(dst=dst, src=src)
        graph.output()
    return graph


def _run_and_assert_fill_with_42(
    *,
    compute_device: Device,
    dst_device_ref: DeviceRef,
    src_device_ref: DeviceRef,
) -> None:
    shape = [8]
    graph = _build_fill_with_42_graph(
        dst_device=dst_device_ref, src_device=src_device_ref, shape=shape
    )

    devices: list[Device] = [compute_device]
    if isinstance(compute_device, Accelerator):
        devices.append(CPU())
    session = InferenceSession(devices=devices)
    model = session.load(graph)

    dst_target = compute_device if dst_device_ref.is_gpu() else CPU()
    initial = np.zeros(shape, dtype=np.int32)
    dst_buffer = Buffer.from_numpy(initial).to(dst_target)

    model.execute(dst_buffer)
    if isinstance(compute_device, Accelerator):
        compute_device.synchronize()

    result = dst_buffer.to(CPU()).to_numpy()
    np.testing.assert_array_equal(result, np.full(shape, 42, dtype=np.int32))


def test_inplace_memcpy_d2d() -> None:
    """GPU dst, GPU src: same-device async memcpy."""
    accelerator = Accelerator()
    gpu_ref = DeviceRef.from_device(accelerator)
    _run_and_assert_fill_with_42(
        compute_device=accelerator,
        dst_device_ref=gpu_ref,
        src_device_ref=gpu_ref,
    )


def test_inplace_memcpy_h2d() -> None:
    """CPU src, GPU dst: host-to-device async memcpy."""
    accelerator = Accelerator()
    gpu_ref = DeviceRef.from_device(accelerator)
    _run_and_assert_fill_with_42(
        compute_device=accelerator,
        dst_device_ref=gpu_ref,
        src_device_ref=DeviceRef.CPU(),
    )


def test_inplace_memcpy_d2h() -> None:
    """GPU src, CPU dst: device-to-host async memcpy."""
    accelerator = Accelerator()
    gpu_ref = DeviceRef.from_device(accelerator)
    _run_and_assert_fill_with_42(
        compute_device=accelerator,
        dst_device_ref=DeviceRef.CPU(),
        src_device_ref=gpu_ref,
    )


def test_inplace_memcpy_h2h() -> None:
    """CPU dst, CPU src: synchronous host memcpy."""
    cpu = CPU()
    cpu_ref = DeviceRef.CPU()
    _run_and_assert_fill_with_42(
        compute_device=cpu,
        dst_device_ref=cpu_ref,
        src_device_ref=cpu_ref,
    )


def test_inplace_memcpy_dtype_mismatch_raises() -> None:
    """`inplace_memcpy` rejects operands with mismatched dtype."""
    device_ref = DeviceRef.GPU()
    with pytest.raises(
        ValueError, match=r"expected dst and src to have the same dtype"
    ):
        with Graph(
            "dtype_mismatch",
            input_types=[BufferType(DType.int32, [4], device=device_ref)],
        ) as graph:
            dst = graph.inputs[0].buffer
            src = ops.constant(
                np.zeros([4], dtype=np.int64),
                dtype=DType.int64,
                device=device_ref,
            )
            kernels.inplace_memcpy(dst=dst, src=src)
            graph.output()


def test_inplace_memcpy_shape_mismatch_raises() -> None:
    """`inplace_memcpy` rejects operands with mismatched shape."""
    device_ref = DeviceRef.GPU()
    with pytest.raises(
        ValueError, match=r"Expected dst and src to have the same shape"
    ):
        with Graph(
            "shape_mismatch",
            input_types=[BufferType(DType.int32, [4], device=device_ref)],
        ) as graph:
            dst = graph.inputs[0].buffer
            src = ops.constant(
                np.zeros([8], dtype=np.int32),
                dtype=DType.int32,
                device=device_ref,
            )
            kernels.inplace_memcpy(dst=dst, src=src)
            graph.output()


def test_inplace_memcpy_cross_gpu_raises() -> None:
    """`inplace_memcpy` rejects two operands on different GPUs."""
    gpu0 = DeviceRef.GPU(id=0)
    gpu1 = DeviceRef.GPU(id=1)
    with pytest.raises(ValueError, match=r"Cross-GPU memcpy is not supported"):
        with Graph(
            "cross_gpu",
            input_types=[BufferType(DType.int32, [4], device=gpu0)],
        ) as graph:
            dst = graph.inputs[0].buffer
            src = ops.constant(
                np.zeros([4], dtype=np.int32),
                dtype=DType.int32,
                device=gpu1,
            )
            kernels.inplace_memcpy(dst=dst, src=src)
            graph.output()
