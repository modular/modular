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
"""Builds WGMMA descriptors from shared-memory tensor layouts."""
from std.sys import size_of

from max.gpu.compute.mma import WGMMADescriptor
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from layout import Layout, TileTensor, TensorLayout
from layout.int_tuple import coord_to_int_tuple
from layout.tensor_core_async import (
    _wgmma_descriptor,
    tile_layout_k_major,
    tile_to_descriptor,
)


def _lhs_descriptor[
    dtype: DType,
    LayoutType: TensorLayout,
    //,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](
    tensor: TileTensor[dtype, LayoutType, address_space=.SHARED, ...]
) -> WGMMADescriptor[tensor.dtype]:
    comptime assert LayoutType.all_dims_known
    comptime layout = Layout(
        coord_to_int_tuple[*LayoutType._shape_types](),
        coord_to_int_tuple[*LayoutType._stride_types](),
    )
    comptime BM = layout[0].size()
    comptime BK = layout[1].size()
    comptime canonical_K = swizzle_mode.bytes() // size_of[
        dtype
    ]() if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE else BK
    comptime canonical_layout_flat = tile_layout_k_major[
        dtype, BM, canonical_K, swizzle_mode
    ]()
    comptime canonical_layout = tile_to_descriptor[
        dtype, canonical_layout_flat, True
    ]()
    return _wgmma_descriptor[
        layout=canonical_layout, is_k_major=True, swizzle=swizzle_mode
    ](tensor.ptr)


def _rhs_descriptor[
    dtype: DType,
    LayoutType: TensorLayout,
    //,
    transposed: Bool = False,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](
    tensor: TileTensor[dtype, LayoutType, address_space=.SHARED, ...]
) -> WGMMADescriptor[tensor.dtype]:
    comptime assert LayoutType.all_dims_known
    comptime layout = Layout(
        coord_to_int_tuple[*LayoutType._shape_types](),
        coord_to_int_tuple[*LayoutType._stride_types](),
    )
    comptime BN = layout[0].size()
    comptime BK = layout[1].size()
    comptime canonical_K = swizzle_mode.bytes() // size_of[
        dtype
    ]() if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE else BK
    comptime canonical_layout_flat = tile_layout_k_major[
        dtype, BN, canonical_K, swizzle_mode
    ]() if transposed else layout
    comptime canonical_layout = tile_to_descriptor[
        dtype, canonical_layout_flat, transposed
    ]()
    return _wgmma_descriptor[
        layout=canonical_layout, is_k_major=transposed, swizzle=swizzle_mode
    ](tensor.ptr)
