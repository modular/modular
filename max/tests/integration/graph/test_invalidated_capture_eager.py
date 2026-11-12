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
"""Eager execution keeps working after the driver invalidates a capture.

A GPU mean over a dynamic axis copies the axis length from the host and syncs
the stream, which a capturing stream rejects, so ``Model.capture`` fails.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

ROWS = 6
COLS = 8


def test_eager_execute_after_invalidated_capture() -> None:
    accelerator = Accelerator()
    session = InferenceSession(devices=[accelerator])
    with Graph(
        "invalidated_capture_mean",
        input_types=[
            TensorType(DType.float32, ["rows", COLS], device=DeviceRef.GPU(0))
        ],
    ) as graph:
        graph.output(ops.mean(graph.inputs[0].tensor, axis=0))
    model = session.load(graph)

    host = np.arange(ROWS * COLS, dtype=np.float32).reshape(ROWS, COLS)
    expected = host.mean(axis=0, keepdims=True)
    x = Buffer.from_numpy(host).to(accelerator)

    (before,) = model.execute(x)
    np.testing.assert_allclose(before.to_numpy(), expected, rtol=1e-6)

    with pytest.raises(Exception, match=r"[Ii]nvalidat|STREAM_CAPTURE"):
        model.capture(1, x)

    (after,) = model.execute(x)
    np.testing.assert_allclose(after.to_numpy(), expected, rtol=1e-6)
