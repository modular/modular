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
"""Executes ops.range for literal bounds that step doesn't divide evenly."""

import numpy as np
import pytest
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, ops


@pytest.mark.parametrize(
    "start, stop, step, dtype",
    [
        (0, 5, 2, DType.int32),
        (5, 0, -2, DType.int32),
        (0.0, 5.0, 2.0, DType.float32),
        (0.0, 1.0, 0.3, DType.float32),
    ],
)
def test_range_uneven_step(
    session: InferenceSession,
    start: float,
    stop: float,
    step: float,
    dtype: DType,
) -> None:
    with Graph("range_uneven", input_types=()) as graph:
        graph.output(
            ops.range(start, stop, step, dtype=dtype, device=DeviceRef.CPU())
        )
    result = session.load(graph).execute()[0]
    np.testing.assert_allclose(
        result.to_numpy(),
        np.arange(start, stop, step, dtype=dtype.to_numpy()),
        rtol=1e-6,
    )
