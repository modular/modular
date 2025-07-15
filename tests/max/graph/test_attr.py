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
"""Tests attribute factories."""

import array

from max._core import graph as _graph
from max.dtype import DType
from max.graph import DeviceRef, TensorType


def test_array_attr(mlir_context) -> None:  # noqa: ANN001
    """Tests array attribute creation."""
    buffer = array.array("f", [42, 3.14])

    array_attr = _graph.array_attr(
        "foo",
        buffer,
        TensorType(DType.float32, (2,), device=DeviceRef.CPU()).to_mlir(),
    )
    assert "dense_array" in str(array_attr)
