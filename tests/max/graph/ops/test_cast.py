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
"""Tests for ops.cast."""

from conftest import GraphBuilder
from hypothesis import given
from max.dtype import DType
from max.graph import TensorType, ops


@given(base_type=..., target_dtype=...)
def test_cast__tensor(
    graph_builder: GraphBuilder,
    base_type: TensorType,
    target_dtype: DType,
) -> None:
    """Test that cast correctly converts tensor values between different data types."""
    expected_type = base_type.cast(target_dtype)
    with graph_builder(input_types=[base_type]) as graph:
        out = ops.cast(graph.inputs[0].tensor, target_dtype)
        assert out.type == expected_type
        graph.output(out)
