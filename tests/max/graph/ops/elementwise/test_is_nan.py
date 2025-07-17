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
"""test the max.graph python bindings."""

from conftest import tensor_types
from hypothesis import given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import TensorType
from max.graph.ops import is_nan


@given(
    tensor_type=tensor_types(
        dtypes=st.sampled_from([DType.float32, DType.bfloat16, DType.float16])
    ),
)
def test_is_nan_returns_bool(graph_builder, tensor_type: TensorType) -> None:  # noqa: ANN001
    with graph_builder(input_types=[tensor_type]) as graph:
        (x,) = graph.inputs
        op = is_nan(x)

        # is_nan should always return boolean tensors regardless of input dtype
        expected_type = TensorType(
            dtype=DType.bool, shape=tensor_type.shape, device=tensor_type.device
        )
        assert op.type == expected_type
        assert op.shape == x.shape
