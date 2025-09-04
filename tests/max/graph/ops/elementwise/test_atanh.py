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

from conftest import GraphBuilder, tensor_types
from hypothesis import given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import TensorType
from max.graph.ops import atanh


@given(
    tensor_type=tensor_types(dtypes=st.just(DType.float32)),
)
def test_atanh_same_type(
    graph_builder: GraphBuilder, tensor_type: TensorType
) -> None:
    with graph_builder(input_types=[tensor_type]) as graph:
        (x,) = graph.inputs
        op = atanh(x)
        assert op.type == tensor_type
