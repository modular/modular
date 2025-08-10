# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Test the max.graph Python bindings."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Tensor, accelerator_count
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, ops


# For masked scatter specifically, I am adding execution tests. Generally, we
# avoid execution tests in the graph api except for models. masked_scatter is a
# complex enough aggregate op, that it feels required to actually run.
@pytest.mark.skipif(
    accelerator_count() > 0, reason="TODO(GEX-2133): Bad results on gpu"
)
@pytest.mark.parametrize(
    "input,mask,updates,expected",
    [
        (
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]],
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            [[0, 0, 0, 1, 2], [3, 4, 0, 5, 6]],
        ),
        (
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [0, 0, 1, 1, 1],
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            [[0, 0, 1, 2, 3], [0, 0, 4, 5, 6]],
        ),
        (
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            [[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [[1, 2, 3, 0, 0], [0, 0, 8, 0, 0]],
        ),
        (
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            [[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]],
            [2],
            [[1, 2, 3, 2, 2], [2, 2, 8, 2, 2]],
        ),
    ],
)
def test_masked_scatter(
    session: InferenceSession,
    input: list[list[int]],
    mask: list[int] | list[list[int]],
    updates: list[int] | list[list[int]],
    expected: list[list[int]],
) -> None:
    with Graph("masked_scatter", input_types=[]) as graph:
        input_val = ops.constant(
            np.array(input), DType.int32, device=DeviceRef.CPU()
        )
        mask_val = ops.constant(
            np.array(mask), DType.bool, device=DeviceRef.CPU()
        )
        updates_val = ops.constant(
            np.array(updates), DType.int32, device=DeviceRef.CPU()
        )
        out = ops.masked_scatter(
            input_val, mask_val, updates_val, out_dim="masked"
        )
        graph.output(out)

    model = session.load(graph)
    result = model.execute()[0]
    assert isinstance(result, Tensor)
    np.testing.assert_equal(result.to_numpy(), expected)
