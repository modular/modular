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
from std.sys.intrinsics import readfirstlane

from max.gpu.intrinsics import AMDBufferResource
from layout import *
from layout.layout_tensor import LayoutTensor
from layout.tile_layout import Layout as TileLayout

from std.utils import IndexList

from .int_tuple import product


def load_to_simd(
    tensor: LayoutTensor,
    out res: SIMD[tensor.dtype, product(tensor.layout.shape)],
):
    comptime assert (
        tensor.layout.all_dims_known()
    ), "load_to_simd is supported only for tensors with known layout"
    comptime size = type_of(res).length
    return rebind[type_of(res)](
        tensor.reshape[Layout(Int(size))]().vectorize[size]()[0]
    )


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
def _get_bounds(tensor: LayoutTensor) -> Int:
    comptime assert (
        tensor.element_layout.all_dims_known()
    ), "Element layout must be known for _get_bounds"
    comptime assert (
        tensor.element_layout.size() == 1
    ), "Element layout must be a scalar"

    comptime assert tensor.rank == 2 or tensor.num_strides == 2

    var strides = tensor.runtime_layout.stride.value
    # Masked views use runtime extents even when their layout shape is static.
    var native_layout = TileLayout(
        Coord(tensor.dim[0](), tensor.dim[1]()),
        Coord(strides[0], strides[1]),
    )
    var native = TileTensor[
        tensor.dtype,
        type_of(native_layout),
        tensor.origin,
        address_space=tensor.address_space,
        linear_idx_type=.int64,
    ](tensor.ptr, native_layout)
    return _get_bounds_impl[tensor.linear_idx_type, True](native)


@inline(.always)
def make_amd_buffer_resource(
    tensor: LayoutTensor,
) -> AMDBufferResource:
    var ptr = tensor.ptr
    var size = _get_bounds(tensor)
    return AMDBufferResource(readfirstlane(ptr), readfirstlane(size))


@inline(.always)
def _get_bounds_impl[
    legacy_index_type: DType, legacy_bounds: Bool
](tensor: TileTensor) -> Int:
    var dim0 = Int(tensor.dim[0]())
    var dim1 = Int(tensor.dim[1]())
    comptime if legacy_bounds:
        if dim0 == 0 or dim1 == 0:
            return 0
        var index0 = Scalar[legacy_index_type](dim0 - 1)
        var index1 = Scalar[legacy_index_type](dim1 - 1)
        var stride0 = Scalar[legacy_index_type](
            tensor.layout.stride[0]().value()
        )
        var stride1 = Scalar[legacy_index_type](
            tensor.layout.stride[1]().value()
        )
        # Legacy offsets wrap before widening; the final +1 occurs in Int.
        var offset = Scalar[legacy_index_type](0)
        offset += index0 * stride0
        offset += index1 * stride1
        return Int(offset) + 1
    else:
        if dim0 <= 0 or dim1 <= 0:
            return 0
        var stride0 = Int(tensor.layout.stride[0]().value())
        var stride1 = Int(tensor.layout.stride[1]().value())
        return (dim0 - 1) * stride0 + (dim1 - 1) * stride1 + 1


@inline(.always)
def _get_bounds(tensor: TileTensor) -> Int:
    """Computes buffer bounds from a rank-2 TileTensor.

    Works with MixedLayout (Scalar + ComptimeInt dimensions).
    Only dim[0] may be runtime; strides and dim[1] are typically comptime,
    so the compiler constant-folds everything except the valid_rows multiply.
    """
    return _get_bounds_impl[.int64, False](tensor)


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
