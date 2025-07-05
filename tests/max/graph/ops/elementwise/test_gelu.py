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
from max.graph import Graph, TensorType
from max.graph.ops import gelu

approximate = st.sampled_from(["none", "tanh", "quick"])


@given(
    tensor_type=tensor_types(dtypes=st.just(DType.float32)),
    approximate=approximate,
)
def test_gelu__same_type(tensor_type: TensorType, approximate: str) -> None:
    with Graph("gelu", input_types=[tensor_type]) as graph:
        (x,) = graph.inputs
        op = gelu(x, approximate)
        assert op.type == tensor_type
