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

from std.sys import align_of

from max.gpu.intrinsics import AMDBufferResource, readfirstlane
from layout import *

from std.utils import IndexList


@inline(.always)
def load_to_simd(
    tensor: TileTensor,
    out res: SIMD[
        tensor.dtype, tensor.static_shape[0] * tensor.static_shape[1]
    ],
):
    comptime assert tensor.rank == tensor.flat_rank == 2
    comptime assert tensor.element_size == 1
    comptime assert tensor.static_shape[0] > 0 and tensor.static_shape[1] > 0
    comptime size = type_of(res).length
    comptime if tensor.static_shape[0] == 1:
        comptime assert tensor.static_stride[1] == 1
        var packed = tensor.vectorize[1, size]()
        comptime assert packed.flat_rank == 2
        return packed.load[alignment=align_of[tensor.dtype]()]((0, 0))
    else:
        comptime assert tensor.static_shape[1] == 1
        comptime assert tensor.static_stride[0] == 1
        var packed = tensor.vectorize[size, 1]()
        comptime assert packed.flat_rank == 2
        return packed.load[alignment=align_of[tensor.dtype]()]((0, 0))


@inline(.always)
def _get_bounds(tensor: TileTensor) -> Int:
    """Computes buffer bounds from a rank-2 TileTensor.

    Works with MixedLayout (Scalar + ComptimeInt dimensions).
    Only dim[0] may be runtime; strides and dim[1] are typically comptime,
    so the compiler constant-folds everything except the valid_rows multiply.
    """
    var dim0 = Int(tensor.dim[0]())
    var dim1 = Int(tensor.dim[1]())
    if dim0 <= 0 or dim1 <= 0:
        return 0
    var stride0 = Int(tensor.dynamic_stride(0))
    var stride1 = Int(tensor.dynamic_stride(1))
    return (dim0 - 1) * stride0 + (dim1 - 1) * stride1 + 1


@inline(.always)
def make_amd_buffer_resource(
    tensor: TileTensor,
) -> AMDBufferResource:
    """Creates an AMD buffer resource descriptor from a TileTensor.

    Uses _get_bounds to compute the valid range. For TileTensors with
    ComptimeInt strides, only the Scalar dimension contributes
    runtime register cost.
    """
    var size = _get_bounds(tensor)
    return AMDBufferResource(readfirstlane(tensor.ptr), readfirstlane(size))


@inline(.always)
def idx2crd[layout: Layout](idx: Int) -> IndexList[layout.rank()]:
    comptime assert layout.all_dims_known(), "Layout must be known for idx2crd"
    var res = IndexList[layout.rank()]()

    comptime for i in range(layout.rank()):
        comptime stride = layout.stride[i].value()
        comptime shape = layout.shape[i].value()
        res[i] = (idx // stride) % shape
    return res
