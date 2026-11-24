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

from std.sys import size_of

from max.gpu.compute.mma import WGMMADescriptor
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from layout import TileTensor, TensorLayout
from layout.tensor_core_async import (
    tile_layout_k_major_typed,
    wgmma_desc_sbo_lbo,
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
    comptime BM = LayoutType.static_shape[0]
    comptime BK = LayoutType.static_shape[1]
    comptime canonical_K = swizzle_mode.bytes() // size_of[
        dtype
    ]() if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE else BK
    comptime canonical_layout = tile_layout_k_major_typed[
        dtype, BM, canonical_K, swizzle_mode
    ]
    comptime sbo_lbo = wgmma_desc_sbo_lbo[
        dtype,
        type_of(canonical_layout),
        is_k_major=True,
        swizzle_mode=swizzle_mode,
    ]()
    return WGMMADescriptor.create[sbo_lbo[0], sbo_lbo[1], swizzle_mode](
        tensor.ptr
    )


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
    comptime BN = LayoutType.static_shape[0]
    comptime BK = LayoutType.static_shape[1]
    comptime canonical_K = swizzle_mode.bytes() // size_of[
        dtype
    ]() if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE else BK
    # Two distinct layouts: select the type, then read its descriptor strides.
    comptime CanonicalLayout = type_of(
        tile_layout_k_major_typed[dtype, BN, canonical_K, swizzle_mode]
    ) if transposed else LayoutType
    comptime sbo_lbo = wgmma_desc_sbo_lbo[
        dtype, CanonicalLayout, is_k_major=transposed, swizzle_mode=swizzle_mode
    ]()
    return WGMMADescriptor.create[sbo_lbo[0], sbo_lbo[1], swizzle_mode](
        tensor.ptr
    )
