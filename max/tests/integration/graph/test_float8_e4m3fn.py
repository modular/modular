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
"""Test float8_e4m3fn dtype support."""

import asyncio

import pytest
from max.dtype import DType
from max.engine.api import InferenceSession
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import Graph
from test_common.graph_utils import is_h100_h200


@pytest.mark.skipif(not is_h100_h200(), reason="float8 requires H100 or H200")
@pytest.mark.parametrize("cast_dtype", [DType.float32, DType.bfloat16])
def test_f8_upcast(cast_dtype: DType) -> None:
    x = Tensor.constant([[1.0, 2.0], [3.0, 4.0]], dtype=DType.float8_e4m3fn)
    result = x.cast(cast_dtype)
    asyncio.run(result.realize)
    assert list(result._values()) == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.skipif(not is_h100_h200(), reason="float8 requires H100 or H200")
@pytest.mark.parametrize("cast_dtype", [DType.float32, DType.bfloat16])
def test_f8_downcast(cast_dtype: DType) -> None:
    x = Tensor.constant([[1.0, 2.0], [3.0, 4.0]], dtype=cast_dtype)
    result = x.cast(DType.float8_e4m3fn)
    asyncio.run(result.realize)
    assert list(result._values()) == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.skipif(not is_h100_h200(), reason="float8 requires H100 or H200")
def test_f8_matmul(session: InferenceSession) -> None:
    input = Tensor.constant([[1.0, 2.0], [3.0, 4.0]], dtype=DType.float8_e4m3fn)
    with Graph("f8", input_types=[input.type]) as graph:
        x = graph.inputs[0].tensor
        y = Tensor.constant([[1.0, 2.0], [3.0, 4.0]], dtype=DType.float8_e4m3fn)
        graph.output(x @ y)

    print(graph)
    model = F.functional(session.load(graph))

    result = model(input)[0]
    # float8 math amirite
    assert list(result._values()) == [7.0, 10.0, 15.0, 22.0]
