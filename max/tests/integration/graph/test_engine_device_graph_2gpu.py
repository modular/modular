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

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def test_capture_with_a_zero_element_device_copy() -> None:
    """Captures a graph that moves a zero-element tensor between two GPUs.

    Real graphs produce zero-byte ``mgp.buffer.device_to_device`` copies, e.g.
    a rank padding its attention output with a zero-element ``F.full``. HIP
    rejects such a copy while a stream is capturing
    (``hipErrorInvalidValue``) although it is a no-op eagerly, so the copy
    must both execute and capture here.
    """
    if accelerator_count() < 2:
        pytest.skip("Test requires at least 2 GPUs")

    session = InferenceSession(devices=[Accelerator(0), Accelerator(1)])

    with Graph(
        "zero_element_device_copy",
        input_types=[
            TensorType(DType.float32, [1, "rows", 4], device=DeviceRef.GPU(0)),
            TensorType(DType.float32, [0, 4], device=DeviceRef.GPU(0)),
        ],
    ) as graph:
        rows = graph.inputs[0].tensor
        empty = graph.inputs[1].tensor
        # A plain transfer, not a broadcast or concat over the zero-size dim:
        # those reach a rank-collapsing view no layout describes, a separate
        # limitation this test does not exercise.
        graph.output(
            rows,
            ops.transfer_to(empty, DeviceRef.GPU(1)),
        )

    model = session.load(graph)

    values = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    rows_buf = Buffer.from_numpy(values).to(model.input_devices[0])
    empty_buf = Buffer.zeros((0, 4), DType.float32, device=Accelerator(0))

    eager_rows, eager_moved = model.execute(rows_buf, empty_buf)
    np.testing.assert_array_equal(eager_rows.to_numpy(), values)
    assert eager_moved.shape == (0, 4)

    captured_rows, captured_moved = model.capture(1, rows_buf, empty_buf)
    model.replay(1, rows_buf, empty_buf)
    np.testing.assert_array_equal(captured_rows.to_numpy(), values)
    assert captured_moved.shape == (0, 4)
