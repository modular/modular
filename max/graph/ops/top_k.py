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
"""Op implementation for top_k."""

from max.mlir.dialects import rmo

from ..graph import Graph
from ..value import TensorValue, TensorValueLike


def top_k(
    input: TensorValueLike, k: int, axis: int = -1
) -> tuple[TensorValue, TensorValue]:
    """Returns tensor with only top K values along given axis.

    Args:
        input: The input tensor from which to select top k.
        k: The number of values to select from input.
        axis: The axis from which to select top k.

    Returns:
        Top K values, Top K indices
    """
    topk_weight, topk_idx = Graph.current._add_op(
        rmo.top_k, TensorValue(input), k, axis
    )

    return topk_weight.tensor, topk_idx.tensor
