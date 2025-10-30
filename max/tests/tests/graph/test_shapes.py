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
"""Hypothesis infrastructure test for shape."""

from conftest import MAX_INT64, shapes
from hypothesis import given
from max.dtype import DType
from max.graph import DeviceRef, Dim, Graph, StaticDim, TensorType, ops


@given(shape=shapes())
def test_shape_product_fits_in_int64(shape) -> None:  # noqa: ANN001
    cumulative_product = 1
    for dim in shape:
        if isinstance(dim, StaticDim):
            cumulative_product *= dim.dim
        else:
            # Currently ignore symbolic dimensions.
            cumulative_product *= 1
    assert cumulative_product <= MAX_INT64


def test_dims_fold() -> None:
    with Graph(
        "test_dims_fold",
        input_types=[
            TensorType(
                DType.int64, shape=["input_row_offsets"], device=DeviceRef.CPU()
            )
        ],
    ) as graph:
        x = graph.inputs[0].tensor
        x = x[:-1]
        tiled_x = ops.tile(x, repeats=(4 * 37919,))
        x0_shape = tiled_x[: 37919 * x.shape[0]].shape
        for i in range(4):
            assert (
                37919 * (Dim("input_row_offsets") - 1)
                == tiled_x[i * x0_shape[0] : (i + 1) * x0_shape[0]].shape[0]
            )
