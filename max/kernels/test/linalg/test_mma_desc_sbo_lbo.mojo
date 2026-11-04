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
"""Parity of the derived MMA descriptor offsets with the legacy layout path.

`mma_desc_sbo_lbo` computes the two descriptor byte offsets from the tile
geometry so `smem_descriptor` no longer builds a legacy `Layout`. This test
keeps that derivation honest by comparing it, at compile time, against the
strides of the legacy canonical layout for representative tile shapes, both
major orders, every swizzle mode, and the page-dense variants.
"""

from std.sys import size_of
from std.testing import assert_equal

from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
)
from linalg.arch.sm100.mma import mma_desc_sbo_lbo
from max.gpu.host.nvidia.tma import TensorMapSwizzle


@inline(.always)
def _legacy_k[
    dtype: DType, BMN: Int, BK: Int, sw: TensorMapSwizzle, pd: Bool
]() -> Tuple[Int, Int]:
    """Returns the legacy descriptor offsets for a K-major tile."""
    comptime layout = tile_layout_k_major[dtype, BMN, BK, sw, page_dense=pd]()
    comptime canonical = tile_to_descriptor[dtype, layout, is_k_major=True]()
    comptime sbo = canonical[0].stride[1].value() * size_of[dtype]()
    comptime lbo = canonical[1].stride[1].value() * size_of[dtype]()
    return (sbo, lbo)


@inline(.always)
def _legacy_mn[
    dtype: DType, BMN: Int, BK: Int, sw: TensorMapSwizzle, pd: Bool
]() -> Tuple[Int, Int]:
    """Returns the legacy descriptor offsets for an MN-major tile."""
    comptime layout = tile_layout_mn_major[dtype, BMN, BK, sw, page_dense=pd]()
    comptime canonical = tile_to_descriptor[dtype, layout, is_k_major=False]()
    comptime cl = canonical.transpose()
    comptime sbo = cl[0].stride[1].value() * size_of[dtype]()
    comptime lbo = cl[1].stride[1].value() * size_of[dtype]()
    return (sbo, lbo)


@inline(.always)
def check_k[
    dtype: DType, BMN: Int, BK: Int, sw: TensorMapSwizzle, pd: Bool
]() raises:
    """Compares the derived and legacy K-major offsets."""
    assert_equal(
        mma_desc_sbo_lbo[
            dtype,
            BMN=BMN,
            BK=BK,
            swizzle_mode=sw,
            is_k_major=True,
            page_dense=pd,
        ](),
        _legacy_k[dtype, BMN, BK, sw, pd](),
        "K-major descriptor offsets diverge from the legacy layout",
    )


@inline(.always)
def check_mn[
    dtype: DType, BMN: Int, BK: Int, sw: TensorMapSwizzle, pd: Bool
]() raises:
    """Compares the derived and legacy MN-major offsets."""
    assert_equal(
        mma_desc_sbo_lbo[
            dtype,
            BMN=BMN,
            BK=BK,
            swizzle_mode=sw,
            is_k_major=False,
            page_dense=pd,
        ](),
        _legacy_mn[dtype, BMN, BK, sw, pd](),
        "MN-major descriptor offsets diverge from the legacy layout",
    )


def test_k_major_offsets() raises:
    check_k[DType.bfloat16, 64, 64, TensorMapSwizzle.SWIZZLE_128B, False]()
    check_k[DType.bfloat16, 64, 128, TensorMapSwizzle.SWIZZLE_128B, False]()
    check_k[DType.bfloat16, 128, 64, TensorMapSwizzle.SWIZZLE_128B, False]()
    check_k[DType.bfloat16, 128, 128, TensorMapSwizzle.SWIZZLE_128B, False]()
    check_k[DType.bfloat16, 8, 64, TensorMapSwizzle.SWIZZLE_128B, False]()
    check_k[DType.bfloat16, 64, 64, TensorMapSwizzle.SWIZZLE_64B, False]()
    check_k[DType.bfloat16, 64, 64, TensorMapSwizzle.SWIZZLE_32B, False]()
    check_k[DType.bfloat16, 8, 32, TensorMapSwizzle.SWIZZLE_32B, False]()
    check_k[DType.bfloat16, 64, 64, TensorMapSwizzle.SWIZZLE_NONE, False]()
    check_k[DType.bfloat16, 64, 8, TensorMapSwizzle.SWIZZLE_NONE, False]()
    check_k[DType.bfloat16, 8, 8, TensorMapSwizzle.SWIZZLE_NONE, False]()
    check_k[DType.bfloat16, 128, 128, TensorMapSwizzle.SWIZZLE_NONE, False]()
    check_k[
        DType.float8_e4m3fn, 128, 256, TensorMapSwizzle.SWIZZLE_128B, False
    ]()
    check_k[DType.float32, 128, 128, TensorMapSwizzle.SWIZZLE_128B, False]()


def test_mn_major_offsets() raises:
    check_mn[DType.bfloat16, 64, 64, TensorMapSwizzle.SWIZZLE_128B, False]()
    check_mn[DType.bfloat16, 64, 128, TensorMapSwizzle.SWIZZLE_128B, False]()
    check_mn[DType.bfloat16, 128, 64, TensorMapSwizzle.SWIZZLE_128B, False]()
    check_mn[DType.bfloat16, 128, 128, TensorMapSwizzle.SWIZZLE_128B, False]()
    check_mn[DType.bfloat16, 64, 64, TensorMapSwizzle.SWIZZLE_32B, False]()
    check_mn[DType.bfloat16, 64, 64, TensorMapSwizzle.SWIZZLE_NONE, False]()
    check_mn[DType.bfloat16, 16, 16, TensorMapSwizzle.SWIZZLE_NONE, False]()
    check_mn[
        DType.float8_e4m3fn, 128, 256, TensorMapSwizzle.SWIZZLE_128B, False
    ]()


def test_page_dense_offsets() raises:
    check_k[DType.bfloat16, 64, 128, TensorMapSwizzle.SWIZZLE_128B, True]()
    check_k[DType.bfloat16, 128, 256, TensorMapSwizzle.SWIZZLE_128B, True]()
    check_k[DType.bfloat16, 256, 128, TensorMapSwizzle.SWIZZLE_128B, True]()
    check_k[DType.bfloat16, 8, 128, TensorMapSwizzle.SWIZZLE_128B, True]()
    check_mn[DType.bfloat16, 64, 128, TensorMapSwizzle.SWIZZLE_128B, True]()
    check_mn[DType.bfloat16, 128, 128, TensorMapSwizzle.SWIZZLE_128B, True]()
    check_mn[DType.bfloat16, 256, 128, TensorMapSwizzle.SWIZZLE_128B, True]()
    check_mn[DType.bfloat16, 8, 128, TensorMapSwizzle.SWIZZLE_128B, True]()


def main() raises:
    test_k_major_offsets()
    test_mn_major_offsets()
    test_page_dense_offsets()
