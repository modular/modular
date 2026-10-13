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
"""Numeric equivalence for the indexer's device-resident exclusive prefix sum.

``exclusive_prefix_sum`` stands in for ``concat([0, cumsum(x)])``, which the
pooled-key cache used to derive its compaction ranks and its per-request pool
row offsets from. Both feed cache addresses, so a value that disagrees
misaligns the pooled-key cache rather than failing loudly. The reference here
is ``numpy.cumsum``, the expression the graph ops replaced.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.deepseekV3_2.layers.indexer import (
    exclusive_prefix_sum,
)

DEVICES = [
    "cpu",
    pytest.param(
        "gpu",
        marks=pytest.mark.skipif(
            accelerator_count() == 0, reason="no accelerator"
        ),
    ),
]


def _run(values: np.ndarray, dtype: DType, device_kind: str) -> np.ndarray:
    driver_device = CPU() if device_kind == "cpu" else Accelerator()
    device = DeviceRef.from_device(driver_device)
    with Graph(
        "exclusive_prefix_sum",
        input_types=[TensorType(dtype, shape=["n"], device=device)],
    ) as graph:
        (x,) = graph.inputs
        graph.output(exclusive_prefix_sum(x.tensor))

    model = InferenceSession(devices=[driver_device]).load(graph)
    (result,) = model.execute(Buffer.from_numpy(values).to(driver_device))
    assert isinstance(result, Buffer)
    return result.to(CPU()).to_numpy()


@pytest.mark.parametrize("device_kind", DEVICES)
@pytest.mark.parametrize("dtype", [DType.int32, DType.int64])
@pytest.mark.parametrize(
    "values",
    [
        [1],
        [0, 0, 0, 0],
        [1, 0, 1, 1, 0, 1],
        [3, 0, 7, 2, 11, 0, 5],
        list(range(32)),
        # Long enough for the GPU kernel to split the scan into chunks.
        list(range(200)),
    ],
)
def test_matches_numpy_cumsum(
    values: list[int], dtype: DType, device_kind: str
) -> None:
    """The result is ``[0, *cumsum(x)]`` -- one entry longer than the input."""
    np_dtype = np.int32 if dtype == DType.int32 else np.int64
    array = np.asarray(values, dtype=np_dtype)
    expected = np.concatenate([[0], np.cumsum(array)]).astype(np_dtype)

    actual = _run(array, dtype, device_kind)

    assert actual.shape == (len(values) + 1,)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("device_kind", DEVICES)
def test_single_request_batch_is_zero_then_total(device_kind: str) -> None:
    """A decode step at batch 1 closes at most one pool, the degenerate case.

    ``[0, x[0]]`` is what the pooled-key writer reads as "nothing before this
    request, this many after it".
    """
    actual = _run(np.asarray([1], dtype=np.int32), DType.int32, device_kind)
    np.testing.assert_array_equal(actual, np.asarray([0, 1], dtype=np.int32))
