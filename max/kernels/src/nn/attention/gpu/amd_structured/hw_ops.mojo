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
"""TileTensor-native AMD GPU hardware operations for MHA.

Ports of the LayoutTensor-based HW load functions from amd/utils.mojo
to TileTensor. These use new-style layouts (from tile_layout.mojo) for
thread distribution and operate on TileTensor SMEM/register tiles.

Functions:
  ds_read_tr16_b64_row    — 4×16 transposed LDS read (raw rocdl intrinsic)
  ds_read_tr16_b64_warp   — warp-level transposed LDS read
  tt_load_b_tr            — transposed B operand load (split into halves)
  tt_load_b_tile          — single MMA tile load from SMEM with swizzle
  tt_load_b               — full B operand load from SMEM warp tile
  tt_copy_dram_to_sram_lds — fully TileTensor DMA (both dst and src)
"""

from std.sys import simd_width_of, size_of
from std.gpu import lane_id_int as lane_id, WARP_SIZE
from std.gpu._utils import to_i32, to_i64
from std.gpu.intrinsics import AMDBufferResource
from std.memory import AddressSpace
from std.memory.unsafe import bitcast
from std.math.uutils import umod, ufloordiv
from std.utils import IndexList
from layout import TileTensor
from layout.tile_layout import (
    Layout as TileLayout,
    row_major as tt_row_major,
    col_major as tt_col_major,
)
from layout.swizzle import Swizzle
from std.itertools import product


comptime _alias_scope_attr = __mlir_attr.`[#llvm.alias_scope<id= "amdgpu.AsyncCopies", domain=#llvm.alias_scope_domain<id = "amdgpu.AsyncOps">>]`
comptime _no_alias_scope_attr = __mlir_attr.`[#llvm.alias_scope<id= "amdgpu.LocalLoads", domain=#llvm.alias_scope_domain<id = "amdgpu.AsyncOps">>]`


@always_inline
def ds_read_tr16_b64_row(
    tile: TileTensor[_, _, address_space=AddressSpace.SHARED, ...],
) -> SIMD[tile.dtype, 4]:
    """4×16 transposed LDS read via rocdl.ds.read.tr16.b64.

    Each 16-lane "row" loads a 4×16 tile, with per-lane exchange so each
    lane gets a column of the tile as SIMD[dtype, 4].
    """
    comptime assert size_of[tile.dtype]() == 2
    comptime assert type_of(tile).static_shape[0] == 4
    comptime assert type_of(tile).static_shape[1] == 16

    comptime thread_layout = tt_row_major[4, 4]()
    var lane_in_row = umod(lane_id(), 16)
    var dist_result = tile.vectorize[1, 4]().distribute_with_offset[
        thread_layout
    ](lane_in_row)
    var offset = dist_result[2]
    var ptr = tile.ptr + offset

    var shared_ptr3 = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type=__mlir_type.`!llvm.ptr<3>`
    ](ptr)

    var llvm_res = __mlir_op.`rocdl.ds.read.tr16.b64`[
        _type=__mlir_type.`vector<4 x bf16>`,
        noalias_scopes=_alias_scope_attr,
        alias_scopes=_no_alias_scope_attr,
    ](shared_ptr3)

    return rebind[SIMD[tile.dtype, 4]](
        __mlir_op.`pop.cast_from_builtin`[_type=SIMD[tile.dtype, 4]._mlir_type](
            llvm_res
        )
    )


@always_inline
def ds_read_tr16_b64_warp[
    mma_shape: IndexList[3],
](
    tile: TileTensor[_, _, address_space=AddressSpace.SHARED, ...],
) -> SIMD[
    tile.dtype, 4
]:
    """Warp-level transposed LDS read distributing across 16-lane rows.

    For 32×32×16 MMA: 2×2 row distribution over 8×32 tile.
    For 16×16×32 MMA: 4×1 row distribution over 16×16 tile.
    """
    # Row distribution: 2×2 for 32x32x16, 4×1 for 16x16x32
    comptime row_dim0 = 2 if mma_shape[0] == 32 else 4
    comptime row_dim1 = 2 if mma_shape[0] == 32 else 1

    comptime assert tile.dtype == DType.bfloat16
    comptime assert type_of(tile).static_shape[0] == row_dim0 * 4
    comptime assert type_of(tile).static_shape[1] == row_dim1 * 16

    var row_idx = ufloordiv(lane_id(), 16)
    var coord0 = row_idx // row_dim1
    var coord1 = row_idx % row_dim1
    var shared_b_tile = tile.tile[4, 16](coord0, coord1)
    return ds_read_tr16_b64_row(shared_b_tile)


@always_inline
def tt_load_b_tr[
    mma_shape: IndexList[3],
](
    tile: TileTensor[_, _, address_space=AddressSpace.SHARED, ...],
) -> SIMD[
    tile.dtype, 8
]:
    """Transposed B operand load for double-rate MFMA shapes.

    Splits the tile along the K dimension into two halves (using tile[]
    instead of LayoutTensor's split[2]) and concatenates the results.
    """
    comptime assert mma_shape in (
        IndexList[3](32, 32, 16),
        IndexList[3](16, 16, 32),
    )
    comptime assert tile.dtype == DType.bfloat16
    comptime MMA_K = mma_shape[2]
    comptime MMA_N = mma_shape[1]
    comptime half_k = MMA_K // 2
    comptime assert type_of(tile).static_shape[0] == MMA_K
    comptime assert type_of(tile).static_shape[1] == MMA_N

    var part_1 = ds_read_tr16_b64_warp[mma_shape](
        tile.tile[half_k, MMA_N](0, 0)
    )
    var part_2 = ds_read_tr16_b64_warp[mma_shape](
        tile.tile[half_k, MMA_N](1, 0)
    )
    return part_1.join(part_2)


@always_inline
def tt_load_b_tile[
    mma_shape: IndexList[3],
    swizzle: Optional[Swizzle],
    k_tile_idx: Int,
](
    src: TileTensor[_, _, address_space=AddressSpace.SHARED, ...],
) -> SIMD[
    src.dtype, simd_width_of[src.dtype]()
]:
    """Single MMA tile load from SMEM with optional swizzle.

    Loads one MMA_M×MMA_K sub-tile at column k_tile_idx, distributes
    across lanes, applies swizzle, and reads via llvm.load.
    """
    comptime MMA_M = mma_shape[0]
    comptime MMA_K = mma_shape[2]
    comptime assert type_of(src).static_shape[0] == MMA_M
    comptime simd_width = simd_width_of[src.dtype]()

    comptime assert (
        mma_shape[0] == 32
    ), "tt_load_b_tile only supports MMA_M=32 (depth=128 structured kernel)"
    var sub_tile = src.tile[MMA_M, MMA_K](0, k_tile_idx)
    comptime thread_layout = tt_col_major[32, 2]()

    var dist = sub_tile.vectorize[1, simd_width]().distribute[thread_layout](
        lane_id()
    )
    comptime idx_type = src.linear_idx_type
    var offset = Scalar[idx_type](Int(dist.ptr) - Int(src.ptr)) // Scalar[
        idx_type
    ](size_of[src.dtype]())

    comptime if swizzle:
        offset = swizzle.value()(
            offset // Scalar[idx_type](simd_width)
        ) * Scalar[idx_type](simd_width)

    var shared_ptr3 = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type=__mlir_type.`!llvm.ptr<3>`
    ](src.ptr + offset)

    var llvm_res = __mlir_op.`llvm.load`[
        _type=__mlir_type.`vector<8 x bf16>`,
        alignment=to_i64(16),
        noalias_scopes=_alias_scope_attr,
        alias_scopes=_no_alias_scope_attr,
    ](shared_ptr3)

    return rebind[SIMD[src.dtype, simd_width]](
        __mlir_op.`pop.cast_from_builtin`[
            _type=SIMD[src.dtype, simd_width]._mlir_type
        ](llvm_res)
    )


@always_inline
def tt_load_b[
    mma_shape: IndexList[3],
    swizzle: Optional[Swizzle],
    num_mmas: Int,
    simd_width: Int,
](
    src: TileTensor[_, _, address_space=AddressSpace.SHARED, ...],
) -> InlineArray[SIMD[src.dtype, simd_width], num_mmas]:
    """Full B operand load from a SMEM warp tile.

    Loads all MMA tiles from a WN×BK SMEM warp tile and returns them
    as an InlineArray of SIMD fragments (one per MMA tile).
    """
    comptime MMA_M = mma_shape[0]
    comptime MMA_K = mma_shape[2]
    comptime M = type_of(src).static_shape[0] // MMA_M
    comptime N = type_of(src).static_shape[1] // MMA_K

    var result = InlineArray[SIMD[src.dtype, simd_width], num_mmas](
        uninitialized=True
    )
    comptime for i in range(M):
        comptime for j in range(N):
            result[Int(i) + Int(j) * M] = rebind[SIMD[src.dtype, simd_width]](
                tt_load_b_tile[mma_shape, swizzle, Int(j)](
                    src.tile[MMA_M, type_of(src).static_shape[1]](Int(i), 0)
                )
            )
    return result


@always_inline
def tt_copy_dram_to_sram_lds[
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
](
    dst: TileTensor[_, _, address_space=AddressSpace.SHARED, ...],
    src: TileTensor,
    lds_base_ptr: UInt32,
    bc: AMDBufferResource,
):
    """DMA from DRAM to LDS with TileTensor for both dst and src.

    Scalar offsets are relative to bc's base pointer.
    """
    comptime thread_layout = tt_row_major[16, 4]()
    var worker_idx = lane_id()

    var dram_base = bc.get_base_ptr()

    comptime M = type_of(src).static_shape[0]
    comptime N = type_of(src).static_shape[1]
    comptime BM = 32
    comptime BN = 32
    comptime BM_SUB = 16

    comptime dst_stride0 = type_of(dst).static_stride[0]
    comptime dst_stride1 = type_of(dst).static_stride[1]
    comptime assert dst_stride1 == 1
    comptime assert dst_stride0 == 32

    comptime aux = 0

    var lds_ptr = lds_base_ptr

    comptime for n_tile, m_tile, m_sub_tile in product(
        range(N // BN), range(M // BM), range(BM // BM_SUB)
    ):
        var dst_partitions = dst.tile[BM, BN](m_tile, n_tile).tile[BM_SUB, BN](
            m_sub_tile, 0
        )
        var src_partitions = src.tile[BM, BN](m_tile, n_tile).tile[BM_SUB, BN](
            m_sub_tile, 0
        )
        var worker_idx_with_offset = worker_idx + m_sub_tile * WARP_SIZE
        var src_dist = src_partitions.vectorize[
            1, simd_width_of[src.dtype]()
        ]().distribute[thread_layout](
            umod(
                swizzle.value()(
                    worker_idx_with_offset
                ) if swizzle else worker_idx_with_offset,
                WARP_SIZE,
            )
        )
        comptime dtype = src.dtype
        var dst_ptr = dst_partitions.ptr.address_space_cast[
            AddressSpace.SHARED
        ]()

        var desc_ptr_ = UnsafePointer[
            Scalar[DType.bfloat16],
            MutAnyOrigin,
            address_space=AddressSpace.BUFFER_RESOURCE,
        ]()

        var ptr_to_ptr = UnsafePointer(to=desc_ptr_)
        var ptr_to_simd = UnsafePointer(to=bc.desc)
        ptr_to_ptr[0] = ptr_to_simd.bitcast[
            UnsafePointer[
                Scalar[DType.bfloat16],
                MutAnyOrigin,
                address_space=AddressSpace.BUFFER_RESOURCE,
            ]
        ]()[0]
        var desc_ptr_llvm = __mlir_op.`builtin.unrealized_conversion_cast`[
            _type=__mlir_type.`!llvm.ptr<8>`
        ](desc_ptr_)

        var shared_ptr3 = __mlir_op.`builtin.unrealized_conversion_cast`[
            _type=__mlir_type.`!llvm.ptr<3>`
        ](dst_ptr)

        comptime num_bytes_per_lane = size_of[dtype]() * simd_width_of[dtype]()
        var vector_offset_bytes = Int(src_dist.ptr) - Int(src_partitions.ptr)
        var scalar_offset_bytes = Int(src_partitions.ptr) - dram_base

        __mlir_op.`rocdl.raw.ptr.buffer.load.lds`[
            alias_scopes=_alias_scope_attr,
            _type=None,
        ](
            desc_ptr_llvm,
            shared_ptr3,
            to_i32(Int32(num_bytes_per_lane)),
            to_i32(Int32(vector_offset_bytes)),
            to_i32(Int32(scalar_offset_bytes)),
            to_i32(0),
            to_i32(aux),
        )
        comptime num_bytes_per_warp = UInt32(
            thread_layout.size() * num_bytes_per_lane
        )
        lds_ptr += num_bytes_per_warp
