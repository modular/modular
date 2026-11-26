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

"""The concatenated output shape, shared by the concat and slice registrations."""

from extensibility import Tensor, TensorTuple
from layout import coord_to_index_list
from nn.gather_scatter import normalize_neg_index

from std.utils import IndexList


def concat_shape_impl[
    Ins: TypeList[Trait=Tensor, ...], //
](axis0: Int, inputs: TensorTuple[*Ins]) raises -> IndexList[Ins[0].rank]:
    """Computes the concatenated output shape from the input tensor shapes.

    Parameters:
        Ins: Types of the input tensors, which must share a rank.

    Args:
        axis0: Axis along which to concatenate; negative values index
            from the last dimension.
        inputs: Input tensors to concatenate, all matching in shape except
            along the concat axis.

    Returns:
        The output shape with the concat axis dimension summed across
        all inputs.
    """
    comptime rank = Ins[0].rank
    var axis = normalize_neg_index(axis0, rank)

    @inline(.always)
    def shape_equal_ignore_axis(
        s1: IndexList[rank], s2: IndexList[rank]
    ) {imm} -> Bool:
        comptime for i in range(rank):
            if i != axis and s1[i] != s2[i]:
                return False
        return True

    var concat_axis_dim_sum = 0
    var output_shape = rebind[IndexList[rank]](
        coord_to_index_list(inputs[0].shape())
    )

    comptime for i in range(len(Ins)):
        comptime assert Ins[i].rank == rank, "inputs must share a rank"
        var input_shape = rebind[IndexList[rank]](
            coord_to_index_list(inputs[i].shape())
        )
        concat_axis_dim_sum += input_shape[axis]
        if not shape_equal_ignore_axis(output_shape, input_shape):
            raise Error(
                "[concat] input shapes must match except at concat axis"
            )

    # compute and return the output shape
    output_shape[axis] = concat_axis_dim_sum
    return output_shape
