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

from std.math import align_up
from std.math.uutils import umod, ufloordiv, udivmod, uceildiv
from std.sys import (
    align_of,
    current_accelerator,
    has_amd_rdna_gpu_accelerator,
    is_nvidia_gpu,
    simd_width_of,
    size_of,
)

import max.gpu.primitives.warp as warp
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from max.gpu.sync import barrier
from max.gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_group,
    external_memory,
)
from layout import (
    Coord,
    DefaultEngine,
    Idx,
    TensorEngine,
    TensorLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout.swizzle import Swizzle, make_ldmatrix_swizzle, make_swizzle
from layout.tile_iterator import TileTensorIter
from layout.tile_layout import Layout as TileLayout
from layout.tile_tensor import stack_allocation
from layout.tile_async_copy import copy_dram_to_sram_async_ldmatrix
from layout.tile_io import (
    copy_local_to_dram,
    copy_local_to_shared,
    copy_sram_to_dram,
)
from layout.tensor_core import TensorCore, get_fragment_size, get_mma_shape

from std.utils import StaticTuple
from std.utils.index import Index
from std.utils.numerics import get_accum_type

from ...utils import (
    ElementwiseComputeFn,
    ElementwiseEpilogueFn,
    no_compute_fn,
    no_epilogue_fn,
)
from ...utils_gpu import MatmulConfig, block_swizzle
from .amd import AMDMatmul


@always_inline
def _tile_is_masked[L: TensorLayout, rows: Int, cols: Int]() -> Bool:
    """Whether `rows x cols` tiles of `L` need a masked copy: a dimension is
    dynamic, or not a multiple of its tile size.
    """
    comptime if not L.all_dims_known:
        return True
    return L.static_shape[0] % rows != 0 or L.static_shape[1] % cols != 0


@inline(.always)
def _pack_c_frags_as_a_frags(
    dst: TileTensor[mut=True, address_space=.LOCAL, ...], src: TileTensor
):
    """Packs the `(2 * num_m_mmas, c_frag_size)` fp32 output fragments of a
    16x8 mma into the `(num_m_mmas, 2 * c_frag_size)` half-precision A
    fragments of a 16x16 mma, so an accumulator tile feeds the next mma as
    its A operand.
    """
    comptime assert src.address_space == .LOCAL, "src must be in registers"
    comptime assert dst.dtype.is_half_float() and src.dtype == .float32
    comptime assert dst.rank == dst.flat_rank == 2
    comptime assert src.rank == src.flat_rank == 2
    comptime assert dst.static_stride[1] == 1 and src.static_stride[1] == 1
    comptime num_frags = src.static_shape[0]
    comptime c_frag_size = src.static_shape[1]
    comptime assert num_frags % 2 == 0
    comptime assert dst.static_shape[0] == num_frags // 2
    comptime assert dst.static_shape[1] == 2 * c_frag_size

    var dst_frags = dst.vectorize[1, c_frag_size]()
    var src_frags = src.vectorize[1, c_frag_size]()
    comptime for i in range(num_frags):
        comptime m = i % (num_frags // 2)
        comptime h = i // (num_frags // 2)
        dst_frags[i // 2, i % 2] = src_frags[2 * m + h, 0].cast[dst.dtype]()


@inline(.always)
def warp_split_k_reduction[
    c_type: DType,
    c_layout: TensorLayout,
    //,
    BM: Int,
    BN: Int,
    num_threads_per_warp_k_part: Int,
    num_warp_k_partitions: Int,
](
    warp_k_part_id: Int,
    c_reg_tile: TileTensor[
        mut=True, c_type, c_layout, address_space=.LOCAL, ...
    ],
    smem: MutPointer[Scalar[c_type], _, address_space=.SHARED],
) where (c_reg_tile.rank == 2) & (c_reg_tile.flat_rank == 2):
    comptime assert c_reg_tile.rank == c_reg_tile.flat_rank == 2
    comptime assert c_reg_tile.element_size == 1
    comptime assert c_reg_tile.Engine == DefaultEngine[element_width=1]
    comptime assert c_reg_tile.static_shape[0] > 0
    comptime assert c_reg_tile.static_shape[1] > 0
    comptime assert c_reg_tile.static_stride[1] == 1
    comptime assert c_reg_tile.static_stride[0] == c_reg_tile.static_shape[1]
    comptime red_layout_native = row_major[1, num_threads_per_warp_k_part]()

    comptime num_mmas = c_layout.static_shape[0]
    comptime c_frag_size = c_layout.static_shape[1]

    var i_red = num_warp_k_partitions // 2
    var tid = thread_idx.x

    while i_red > 0:
        barrier()
        var red_tb_smem = TileTensor(
            (
                smem.unsafe_bitcast[Scalar[c_type]]().unsafe_offset(
                    (warp_k_part_id % i_red) * BM * BN
                )
            ).as_unsafe_any_origin(),
            row_major[1, BM * BN](),
        )
        if i_red <= warp_k_part_id < 2 * i_red:
            copy_local_to_shared[thread_layout=red_layout_native](
                red_tb_smem.vectorize[1, c_frag_size](),
                c_reg_tile.vectorize[1, c_frag_size](),
            )
        barrier()
        if warp_k_part_id < i_red:
            var red_tb_thread_tile = red_tb_smem.vectorize[
                1, c_frag_size
            ]().distribute[red_layout_native](tid)
            comptime for i in range(num_mmas):
                var current = c_reg_tile.load[
                    width=c_frag_size, alignment=align_of[c_type]()
                ](Coord(i, 0))
                var incoming = red_tb_thread_tile.load[
                    width=c_frag_size, alignment=align_of[c_type]()
                ](Coord(0, i))
                c_reg_tile.store[
                    width=c_frag_size, alignment=align_of[c_type]()
                ](Coord(i, 0), current + incoming)
        i_red //= 2


@inline(.always)
def warp_split_k_reduction[
    c_type: DType,
    c_layout: TensorLayout,
    //,
    BM: Int,
    BN: Int,
    num_threads_per_warp_k_part: Int,
    num_warp_k_partitions: Int,
](
    warp_k_part_id: Int,
    c_reg_tile: TileTensor[
        mut=True, c_type, c_layout, address_space=.LOCAL, ...
    ],
) where (c_reg_tile.rank == 2) & (c_reg_tile.flat_rank == 2):
    comptime c_frag_size = c_layout.static_shape[1]

    var smem = external_memory[
        Scalar[c_type],
        address_space=.SHARED,
        alignment=align_of[SIMD[c_type, c_frag_size]](),
    ]()

    warp_split_k_reduction[
        BM, BN, num_threads_per_warp_k_part, num_warp_k_partitions
    ](warp_k_part_id, c_reg_tile, smem)


@always_inline
def _static_tile_layout[
    L: TensorLayout
]() -> TileLayout[
    shape_types=L._shape_types, stride_types=L._stride_types
] where L.all_dims_known:
    """Builds the layout instance of a fully static `L`."""
    return TileLayout[shape_types=L._shape_types, stride_types=L._stride_types](
        Coord[*L._shape_types](), Coord[*L._stride_types]()
    )


@inline(.always)
def multistage_mma[
    c_type: DType,
    c_layout: TensorLayout,
    a_type: DType,
    a_layout: TensorLayout,
    a_smem_layout: TensorLayout,
    b_type: DType,
    b_layout: TensorLayout,
    b_smem_layout: TensorLayout,
    //,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
    transpose_b: Bool,
    # Hack:
    /,
    *,
    swizzle_a: Bool = True,
    static_num_iters: Int = -1,
    prefetch_init: Bool = True,
    continue_prefetch_b: Bool = False,
    transpose_b_next: Bool = False,
    b_next_gmem_layout: TensorLayout = b_layout,
    b_next_smem_layout: TensorLayout = b_smem_layout,
    next_op_b_iter_masked: Bool = False,
    k_group_size: Int = 1,
](
    c: TileTensor[mut=True, c_type, c_layout, address_space=.LOCAL, ...],
    a_iter_arg: TileTensorIter[_, a_layout, ...],
    b_iter_arg: TileTensorIter[b_type, b_layout, ...],
    a_smem_iter_arg: TileTensorIter[
        mut=True,
        a_type,
        a_smem_layout,
        address_space=.SHARED,
        ...,
    ],
    mut b_smem_iter: TileTensorIter[
        mut=True,
        b_type,
        b_smem_layout,
        address_space=.SHARED,
        ...,
    ],
    num_iters: Int,
    /,
    *,
    num_b_rows: Optional[Int] = None,
    next_op_b_iter: Optional[
        TileTensorIter[
            b_type,
            b_next_gmem_layout,
            ImmutAnyOrigin,
            masked=next_op_b_iter_masked,
        ]
    ] = None,
):
    """Software-pipelined block MMA over `TileTensorIter` operands.

    Global-memory operands are staged into the circular shared-memory
    iterators with async copies `num_pipeline_stages - 1` tiles ahead. A may
    instead already sit in shared memory, or in registers as the output
    fragments of a previous mma. A masked global-memory iterator supplies the
    valid row count of its tile through `num_valid_rows`; `num_b_rows`
    overrides it for B as the valid rows counted from B's first tile.
    """
    comptime simd_size = simd_width_of[a_type]()

    var tid = UInt32(umod(thread_idx.x, num_threads))
    var warp_id = warp.broadcast(tid // UInt32(WARP_SIZE))

    comptime num_warps_m = BM // WM
    comptime num_warps_n = BN // WN
    var warp_y, warp_x = divmod(warp_id, UInt32(num_warps_n))

    comptime assert c.rank == c.flat_rank == 2
    comptime assert c.element_size == 1
    comptime assert c.Engine == DefaultEngine[element_width=1]
    comptime assert c.static_shape[0] > 0 and c.static_shape[1] > 0
    comptime assert c.static_stride[1] == 1
    comptime assert c.static_stride[0] == c.static_shape[1]
    comptime assert a_type == b_type, "A and B must share a dtype"
    var a_iter = a_iter_arg
    var b_iter = b_iter_arg
    var a_smem_iter = a_smem_iter_arg
    var next_b_iter = next_op_b_iter

    # If there are more threads than vectors, thread layout should be based on
    # the latter so that a vector is only mapped to one thread.
    comptime a_num_vecs = BM * BK // simd_size
    comptime async_copy_a_layout = row_major[
        min(num_threads, a_num_vecs) * simd_size // BK, BK // simd_size
    ]()

    comptime b_num_vecs = BN * BK // simd_size
    comptime async_copy_b_layout = row_major[
        min(num_threads, b_num_vecs)
        * simd_size
        // b_smem_layout.static_shape[1],
        b_smem_layout.static_shape[1] // simd_size,
    ]()

    # TODO (KERN-1337): Enable swizzle for matrix B for FP8 data type and transpose_b==False
    comptime swizzle_b = (
        transpose_b or b_type.is_half_float()
    ) and is_nvidia_gpu()

    comptime a_masked = type_of(a_iter).masked
    comptime b_masked = type_of(b_iter).masked

    @inline(.always)
    def _copy_tile_to_sram[
        thread_layout: TileLayout, swizzle: Bool, masked: Bool
    ](
        dst: TileTensor[mut=True, address_space=.SHARED, ...],
        src: TileTensor,
        num_valid_rows: Int,
    ):
        copy_dram_to_sram_async_ldmatrix[
            thread_layout,
            simd_size=simd_size,
            swizzle=swizzle,
            masked=masked,
            num_threads=num_threads,
        ](dst, src, num_valid_rows)

    # Rows of the B tile to copy: the iterator's own count unless the caller
    # bounds B globally, in which case the bound shrinks with the k tile.
    @inline(.always)
    def _b_valid_rows(
        iter_valid_rows: Int, rows_are_per_tile: Bool, k_tile: Int
    ) {imm num_b_rows} -> Int:
        if case .Some(rows) = num_b_rows:
            return rows if rows_are_per_tile else max(0, rows - k_tile * BK)
        return iter_valid_rows

    # Prefetch (num_pipeline_stages - 1) stages.
    comptime if prefetch_init:
        comptime for stage in range(num_pipeline_stages - 1):
            comptime if a_iter.address_space == .GENERIC:
                var a_smem_tile = a_smem_iter.next_unsafe(stage)[]
                _copy_tile_to_sram[async_copy_a_layout, swizzle_a, a_masked](
                    a_smem_tile, a_iter[], a_iter.num_valid_rows
                )

                a_iter._incr()

            comptime if b_iter.address_space == .GENERIC:
                var b_smem_tile = b_smem_iter.next_unsafe(stage)[]
                _copy_tile_to_sram[async_copy_b_layout, swizzle_b, b_masked](
                    b_smem_tile,
                    b_iter[],
                    _b_valid_rows(b_iter.num_valid_rows, transpose_b, stage),
                )

                b_iter._incr()

            async_copy_commit_group()

        # Guard stage 0.
        async_copy_wait_group(Int32(num_pipeline_stages - 2))
        barrier()

    comptime mma_shape = get_mma_shape[a_type, get_accum_type[a_type]()]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_k_mmas = BK // MMA_K
    comptime num_k_mma_iters: Int = num_k_mmas // k_group_size
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N
    comptime assert (
        num_k_mmas % (2 * k_group_size) == 0
    ), "num_k_mmas must be an integer multiple of 2*k_group_size"

    comptime accum_type = get_accum_type[a_type]()
    comptime frag_size = get_fragment_size[mma_shape]()
    comptime a_frag_size = frag_size[0]
    comptime b_frag_size = frag_size[1]
    comptime c_frag_size = frag_size[2]

    comptime num_reg_tiles = 2 * k_group_size
    # Register tiles: `num_reg_tiles` stacked (num_*_mmas, frag_size) tiles.
    var a_reg_native = stack_allocation[dtype=a_type, address_space=.LOCAL](
        row_major[2 * k_group_size * num_m_mmas, a_frag_size]()
    )
    var b_reg_native = stack_allocation[dtype=b_type, address_space=.LOCAL](
        row_major[2 * k_group_size * num_n_mmas, b_frag_size]()
    )

    var a_warp_tile = a_smem_iter[].tile[WM, BK](Int(warp_y), 0)

    comptime b_wtile_dim0 = WN if transpose_b else BK
    comptime b_wtile_dim1 = BK if transpose_b else WN
    var b_wtile_coord0 = Int(warp_x) if transpose_b else 0
    var b_wtile_coord1 = 0 if transpose_b else Int(warp_x)
    var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
        b_wtile_coord0, b_wtile_coord1
    )

    var mma_op = TensorCore[accum_type, a_type, mma_shape, transpose_b]()

    comptime swizzle_a_pattern = make_ldmatrix_swizzle[
        a_type, type_of(a_warp_tile).static_stride[0]
    ]() if swizzle_a else Optional[Swizzle]()

    @inline(.always)
    def _load_b_frags(
        warp_tile: TileTensor[address_space=.SHARED, ...],
        reg_idx: Int,
        k_mma_idx: Int,
        warp_x: Int,
        mma_op: type_of(mma_op),
        b_reg_native: type_of(b_reg_native),
    ):
        var b_reg = b_reg_native.tile[num_n_mmas, b_frag_size](reg_idx, 0)
        mma_op.load_b(
            warp_tile.bitcast[a_type](),
            b_reg.bitcast[a_type](),
            k_mma_idx,
            warp_x,
        )

    comptime for i in range(k_group_size):
        var a_reg = a_reg_native.tile[num_m_mmas, a_frag_size](i, 0)

        comptime if a_iter.address_space == .LOCAL:
            # Assume input is the 16x8 output of 16x8x16 or 16x8x8 mma.
            _pack_c_frags_as_a_frags(a_reg, a_iter[])
            a_iter._incr()
        else:
            mma_op.load_a[swizzle_a_pattern](a_warp_tile, a_reg, i)

        _load_b_frags(b_warp_tile, i, i, Int(warp_x), mma_op, b_reg_native)

    @inline(.always)
    def _mma(
        current: Int,
        mma_op: type_of(mma_op),
        a_reg_native: type_of(a_reg_native),
        b_reg_native: type_of(b_reg_native),
        c: type_of(c),
    ):
        var a_reg = a_reg_native.tile[num_m_mmas, a_frag_size](current, 0)
        var b_reg = b_reg_native.tile[num_n_mmas, b_frag_size](current, 0)
        mma_op.mma(
            a_reg.vectorize[1, a_frag_size](),
            b_reg.vectorize[1, b_frag_size]().bitcast[a_type](),
            c.vectorize[1, c_frag_size](),
        )

    comptime if static_num_iters >= 0:
        comptime assert (
            a_iter.address_space == .SHARED or a_iter.address_space == .LOCAL
        ), (
            "Using input in registers or shared memory requires static"
            " iteration bound.\n"
        )

        comptime for k_tile_id in range(static_num_iters):
            var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
                b_wtile_coord0,
                b_wtile_coord1,
            )

            # Perform prefetch registers and mma until current shared memory tile's
            # data has all been loaded to registers.
            comptime for k_mma0 in range(num_k_mma_iters):
                comptime for k_mma1 in range(k_group_size):
                    comptime k_mma = UInt32(k_mma0 * k_group_size + k_mma1)
                    comptime k_mma_next = k_mma + UInt32(k_group_size)
                    comptime next = Int(k_mma_next % UInt32(num_reg_tiles))

                    comptime if k_mma_next == UInt32(num_k_mmas):
                        comptime prefetch_tile_id = k_tile_id + num_pipeline_stages - 1

                        # Prefetch one k tile (if valid) from global memory to current
                        # shared memory buffer.
                        comptime if b_iter.address_space == .GENERIC:
                            comptime if prefetch_tile_id < static_num_iters:
                                var b_smem_prefetch_tile = (
                                    b_smem_iter.next_unsafe(
                                        num_pipeline_stages - 1
                                    )[]
                                )
                                _copy_tile_to_sram[
                                    async_copy_b_layout, swizzle_b, b_masked
                                ](
                                    b_smem_prefetch_tile,
                                    b_iter[],
                                    _b_valid_rows(
                                        b_iter.num_valid_rows,
                                        transpose_b,
                                        prefetch_tile_id,
                                    ),
                                )

                                b_iter._incr()

                            async_copy_commit_group()

                            # Guard the next k tile's shared memory buffer.
                            async_copy_wait_group(
                                Int32(num_pipeline_stages - 2)
                            )
                            barrier()

                        comptime if a_iter.address_space == .SHARED:
                            a_smem_iter._incr()
                        b_smem_iter._incr()

                        a_warp_tile = a_smem_iter[].tile[WM, BK](Int(warp_y), 0)
                        b_warp_tile = b_smem_iter[].tile[
                            b_wtile_dim0, b_wtile_dim1
                        ](b_wtile_coord0, b_wtile_coord1)

                    comptime kidx = Int(k_mma_next % UInt32(num_k_mmas))

                    comptime if a_iter.address_space == .SHARED:
                        mma_op.load_a[swizzle_a_pattern](
                            a_warp_tile,
                            a_reg_native.tile[num_m_mmas, a_frag_size](next, 0),
                            kidx,
                        )
                    else:
                        # Assume input is the 16x8 output of 16x8x16 or 16x8x8 mma.
                        _pack_c_frags_as_a_frags(
                            a_reg_native.tile[num_m_mmas, a_frag_size](next, 0),
                            a_iter[],
                        )
                        a_iter._incr()

                    _load_b_frags(
                        b_warp_tile,
                        next,
                        kidx,
                        Int(warp_x),
                        mma_op,
                        b_reg_native,
                    )

                comptime for k_mma1 in range(k_group_size):
                    comptime k_mma = UInt32(k_mma0 * k_group_size + k_mma1)
                    comptime current = k_mma % UInt32(num_reg_tiles)
                    _mma(Int(current), mma_op, a_reg_native, b_reg_native, c)

        return

    for k_tile_id in range(num_iters):
        var a_warp_tile = a_smem_iter[].tile[WM, BK](Int(warp_y), 0)
        var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
            b_wtile_coord0,
            b_wtile_coord1,
        )

        # Perform prefetch registers and mma until current shared memory tile's
        # data has all been loaded to registers.
        comptime for k_mma0 in range(num_k_mma_iters):
            comptime for k_mma1 in range(k_group_size):
                comptime k_mma = UInt32(k_mma0 * k_group_size + k_mma1)
                comptime k_mma_next = k_mma + UInt32(k_group_size)
                comptime next = Int(k_mma_next % UInt32(num_reg_tiles))

                comptime if k_mma_next == UInt32(num_k_mmas):
                    var prefetch_tile_id = k_tile_id + num_pipeline_stages - 1

                    # Prefetch one k tile (if valid) from global memory to current
                    # shared memory buffer.
                    if prefetch_tile_id < num_iters:
                        comptime if a_iter.address_space == .GENERIC:
                            var a_smem_prefetch_tile = a_smem_iter.next_unsafe(
                                num_pipeline_stages - 1
                            )[]
                            _copy_tile_to_sram[
                                async_copy_a_layout, swizzle_a, a_masked
                            ](
                                a_smem_prefetch_tile,
                                a_iter[],
                                a_iter.num_valid_rows,
                            )

                            a_iter._incr()

                        comptime if b_iter.address_space == .GENERIC:
                            var b_smem_prefetch_tile = b_smem_iter.next_unsafe(
                                num_pipeline_stages - 1
                            )[]
                            _copy_tile_to_sram[
                                async_copy_b_layout, swizzle_b, b_masked
                            ](
                                b_smem_prefetch_tile,
                                b_iter[],
                                _b_valid_rows(
                                    b_iter.num_valid_rows,
                                    transpose_b,
                                    prefetch_tile_id,
                                ),
                            )

                            b_iter._incr()
                    else:
                        comptime if continue_prefetch_b:
                            comptime assert (
                                b_next_smem_layout.all_dims_known
                            ), "continue_prefetch_b needs a static smem layout"
                            var b_next_smem_tile = b_smem_iter.next_unsafe(
                                num_pipeline_stages - 1
                            )[]
                            # The next operand views the same shared-memory
                            # stage under its own layout.
                            var b_smem_prefetch_tile = TileTensor[
                                b_type,
                                type_of(
                                    _static_tile_layout[b_next_smem_layout]()
                                ),
                                b_next_smem_tile.origin,
                                address_space=.SHARED,
                            ](
                                ptr=b_next_smem_tile.ptr,
                                layout=_static_tile_layout[
                                    b_next_smem_layout
                                ](),
                            )

                            comptime row_size = b_next_smem_layout.static_stride[
                                0
                            ]

                            comptime b_prefetch_thread_layout = row_major[
                                num_threads * simd_size // row_size,
                                row_size // simd_size,
                            ]()
                            comptime swizzle_prefetch_b = (
                                transpose_b_next or b_type.is_half_float()
                            ) and is_nvidia_gpu()

                            ref next_iter = next_b_iter.unsafe_value()
                            _copy_tile_to_sram[
                                b_prefetch_thread_layout,
                                swizzle_prefetch_b,
                                next_op_b_iter_masked,
                            ](
                                b_smem_prefetch_tile,
                                next_iter[],
                                _b_valid_rows(
                                    next_iter.num_valid_rows,
                                    transpose_b_next,
                                    prefetch_tile_id - num_iters,
                                ),
                            )

                            next_iter._incr()

                    async_copy_commit_group()

                    # Guard the next k tile's shared memory buffer.
                    async_copy_wait_group(Int32(num_pipeline_stages - 2))
                    barrier()

                    a_smem_iter._incr()
                    b_smem_iter._incr()

                    a_warp_tile = a_smem_iter[].tile[WM, BK](Int(warp_y), 0)
                    b_warp_tile = b_smem_iter[].tile[
                        b_wtile_dim0, b_wtile_dim1
                    ](b_wtile_coord0, b_wtile_coord1)

                comptime kidx = Int(k_mma_next % UInt32(num_k_mmas))
                mma_op.load_a[swizzle_a_pattern](
                    a_warp_tile,
                    a_reg_native.tile[num_m_mmas, a_frag_size](next, 0),
                    kidx,
                )
                _load_b_frags(
                    b_warp_tile, next, kidx, Int(warp_x), mma_op, b_reg_native
                )

            comptime for k_mma1 in range(k_group_size):
                comptime k_mma = UInt32(k_mma0 * k_group_size + k_mma1)
                comptime current = k_mma % UInt32(num_reg_tiles)
                _mma(Int(current), mma_op, a_reg_native, b_reg_native, c)


@__name(
    t"multistage_gemm_kernel_{c_type}_{a_type}_{b_type}_{transpose_b}",
)
def multistage_gemm_kernel[
    c_type: DType,
    CLT: TensorLayout,
    a_type: DType,
    ALT: TensorLayout,
    b_type: DType,
    BLT: TensorLayout,
    transpose_b: Bool,
    c_linear_idx_type: DType,
    a_linear_idx_type: DType,
    b_linear_idx_type: DType,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b, ...],
](
    c_tt: TileTensor[
        c_type,
        CLT,
        MutAnyOrigin,
        Engine=c_engine,
        linear_idx_type=c_linear_idx_type,
    ],
    a_tt: TileTensor[
        a_type,
        ALT,
        ImmutAnyOrigin,
        Engine=a_engine,
        linear_idx_type=a_linear_idx_type,
    ],
    b_tt: TileTensor[
        b_type,
        BLT,
        ImmutAnyOrigin,
        Engine=b_engine,
        linear_idx_type=b_linear_idx_type,
    ],
):
    _multistage_gemm_kernel_impl[
        c_type,
        CLT,
        a_type,
        ALT,
        b_type,
        BLT,
        transpose_b,
        c_linear_idx_type,
        a_linear_idx_type,
        b_linear_idx_type,
        c_engine,
        a_engine,
        b_engine,
        config=config,
        has_epilogue_fn=False,
        has_compute_fn=False,
    ](c_tt, a_tt, b_tt, no_epilogue_fn, no_compute_fn)


@__name(
    t"multistage_gemm_kernel_epilogue_fn_{c_type}_{a_type}_{b_type}_{transpose_b}",
)
def multistage_gemm_kernel_epilogue_fn[
    c_type: DType,
    CLT: TensorLayout,
    a_type: DType,
    ALT: TensorLayout,
    b_type: DType,
    BLT: TensorLayout,
    transpose_b: Bool,
    c_linear_idx_type: DType,
    a_linear_idx_type: DType,
    b_linear_idx_type: DType,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b, ...],
    EpilogueFnType: ElementwiseEpilogueFn,
](
    c_tt: TileTensor[
        c_type,
        CLT,
        MutAnyOrigin,
        Engine=c_engine,
        linear_idx_type=c_linear_idx_type,
    ],
    a_tt: TileTensor[
        a_type,
        ALT,
        ImmutAnyOrigin,
        Engine=a_engine,
        linear_idx_type=a_linear_idx_type,
    ],
    b_tt: TileTensor[
        b_type,
        BLT,
        ImmutAnyOrigin,
        Engine=b_engine,
        linear_idx_type=b_linear_idx_type,
    ],
    epilogue_fn: EpilogueFnType,
):
    """`multistage_gemm_kernel` that writes its output through
    `epilogue_fn`.

    Parameters:
        c_type: Element type of `c`.
        CLT: Layout of `c`.
        a_type: Element type of `a`.
        ALT: Layout of `a`.
        b_type: Element type of `b`.
        BLT: Layout of `b`.
        transpose_b: Whether `b` is stored as `(N, K)`.
        c_linear_idx_type: Linear index type of `c`.
        a_linear_idx_type: Linear index type of `a`.
        b_linear_idx_type: Linear index type of `b`.
        c_engine: Engine of `c`.
        a_engine: Engine of `a`.
        b_engine: Engine of `b`.
        config: Tile, warp, MMA, and pipeline configuration.
        EpilogueFnType: Type of `epilogue_fn`.

    Args:
        c_tt: Output matrix of shape `(M, N)`. Only its shape is read;
            `epilogue_fn` stores the result.
        a_tt: Input matrix of shape `(M, K)`.
        b_tt: Input matrix of shape `(K, N)`, or `(N, K)` when
            `transpose_b` is set.
        epilogue_fn: Stores each output vector. Its `alignment` is in
            bytes.
    """
    _multistage_gemm_kernel_impl[
        c_type,
        CLT,
        a_type,
        ALT,
        b_type,
        BLT,
        transpose_b,
        c_linear_idx_type,
        a_linear_idx_type,
        b_linear_idx_type,
        c_engine,
        a_engine,
        b_engine,
        config=config,
        has_epilogue_fn=True,
        has_compute_fn=False,
    ](c_tt, a_tt, b_tt, epilogue_fn, no_compute_fn)


@__name(
    t"multistage_gemm_kernel_compute_fn_{c_type}_{a_type}_{b_type}_{transpose_b}",
)
def multistage_gemm_kernel_compute_fn[
    c_type: DType,
    CLT: TensorLayout,
    a_type: DType,
    ALT: TensorLayout,
    b_type: DType,
    BLT: TensorLayout,
    transpose_b: Bool,
    c_linear_idx_type: DType,
    a_linear_idx_type: DType,
    b_linear_idx_type: DType,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b, ...],
    ComputeFnType: ElementwiseComputeFn,
](
    c_tt: TileTensor[
        c_type,
        CLT,
        MutAnyOrigin,
        Engine=c_engine,
        linear_idx_type=c_linear_idx_type,
    ],
    a_tt: TileTensor[
        a_type,
        ALT,
        ImmutAnyOrigin,
        Engine=a_engine,
        linear_idx_type=a_linear_idx_type,
    ],
    b_tt: TileTensor[
        b_type,
        BLT,
        ImmutAnyOrigin,
        Engine=b_engine,
        linear_idx_type=b_linear_idx_type,
    ],
    compute_fn: ComputeFnType,
):
    """`multistage_gemm_kernel` that stores `compute_fn` applied to each
    output value into `c`.

    Parameters:
        c_type: Element type of `c`.
        CLT: Layout of `c`.
        a_type: Element type of `a`.
        ALT: Layout of `a`.
        b_type: Element type of `b`.
        BLT: Layout of `b`.
        transpose_b: Whether `b` is stored as `(N, K)`.
        c_linear_idx_type: Linear index type of `c`.
        a_linear_idx_type: Linear index type of `a`.
        b_linear_idx_type: Linear index type of `b`.
        c_engine: Engine of `c`.
        a_engine: Engine of `a`.
        b_engine: Engine of `b`.
        config: Tile, warp, MMA, and pipeline configuration.
        ComputeFnType: Type of `compute_fn`.

    Args:
        c_tt: Output matrix of shape `(M, N)`.
        a_tt: Input matrix of shape `(M, K)`.
        b_tt: Input matrix of shape `(K, N)`, or `(N, K)` when
            `transpose_b` is set.
        compute_fn: Maps each output index and value to the value to store.
    """
    _multistage_gemm_kernel_impl[
        c_type,
        CLT,
        a_type,
        ALT,
        b_type,
        BLT,
        transpose_b,
        c_linear_idx_type,
        a_linear_idx_type,
        b_linear_idx_type,
        c_engine,
        a_engine,
        b_engine,
        config=config,
        has_epilogue_fn=False,
        has_compute_fn=True,
    ](c_tt, a_tt, b_tt, no_epilogue_fn, compute_fn)


@inline(.always)
def _multistage_gemm_kernel_impl[
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    c_type: DType,
    CLT: TensorLayout,
    a_type: DType,
    ALT: TensorLayout,
    b_type: DType,
    BLT: TensorLayout,
    transpose_b: Bool,
    c_linear_idx_type: DType,
    a_linear_idx_type: DType,
    b_linear_idx_type: DType,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b, ...],
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
](
    c_tt: TileTensor[
        c_type,
        CLT,
        MutAnyOrigin,
        Engine=c_engine,
        linear_idx_type=c_linear_idx_type,
    ],
    a_tt: TileTensor[
        a_type,
        ALT,
        ImmutAnyOrigin,
        Engine=a_engine,
        linear_idx_type=a_linear_idx_type,
    ],
    b_tt: TileTensor[
        b_type,
        BLT,
        ImmutAnyOrigin,
        Engine=b_engine,
        linear_idx_type=b_linear_idx_type,
    ],
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
):
    comptime assert not (
        has_epilogue_fn and has_compute_fn
    ), "pass either epilogue_fn or compute_fn, not both"
    comptime assert c_tt.rank == c_tt.flat_rank == 2
    comptime assert a_tt.rank == a_tt.flat_rank == 2
    comptime assert b_tt.rank == b_tt.flat_rank == 2
    # float16 shares bf16's MMA path here; its accumulation precision can
    # differ from bf16, so it is opt-in via the float16 quantization encoding.
    comptime assert (
        a_type in (DType.float32, DType.bfloat16, DType.float16)
        and a_type == b_type
    ) or (
        a_type in (DType.float8_e4m3fn, DType.float8_e5m2)
        and a_type == b_type
        and c_type == .float32
    ), "Pipeline gemm only supports tf32, F16, BF16, E4M3, and E5M2 mma"
    comptime simd_size = simd_width_of[c_type]()

    var M = Int(c_tt.dim[0]())
    var N = Int(b_tt.dim[0 if transpose_b else 1]())
    var K = Int(b_tt.dim[1 if transpose_b else 0]())

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]
    comptime WM = config.warp_tile_shape[0]
    comptime WN = config.warp_tile_shape[1]
    comptime num_pipeline_stages = config.num_pipeline_stages

    comptime num_warps_m = config.num_warps_m()
    comptime num_warps_n = config.num_warps_n()
    comptime num_threads = config.num_threads()

    comptime num_warp_k_partitions = config.num_warp_k_partitions
    comptime num_threads_per_warp_k_part = num_threads // num_warp_k_partitions

    var tid = thread_idx.x
    var ln_id = lane_id()
    var warp_k_part_id = (
        ufloordiv(tid, num_threads_per_warp_k_part) if num_warp_k_partitions
        > 1 else 0
    )
    var warp_id = warp.broadcast(
        ufloordiv(umod(tid, num_threads_per_warp_k_part), WARP_SIZE)
    )

    # Only apply block swizzling for half precision types.
    comptime swizzle_block = a_type.is_half_float() and b_type.is_half_float() and is_nvidia_gpu()

    # NOTE: the condition ( not (N // BN & 1)) is for a temporary solution
    # for solving mismatches in some shapes
    var block_idx_swizzle = block_swizzle(
        Index[dtype=DType.uint32](block_idx.x, block_idx.y),
        Index[dtype=DType.uint32](grid_dim.x, grid_dim.y),
    ) if swizzle_block else Index[dtype=DType.uint32](block_idx.x, block_idx.y)
    var block_m = Int(block_idx_swizzle[1])
    var block_n = Int(block_idx_swizzle[0])

    # Coordinates of the current warp.
    var warp_y, warp_x = udivmod(warp_id, num_warps_n)

    # Prepare circular shared memory buffer for A and B.
    # Each pipeline stage has its own buffer.
    comptime alignment = align_of[SIMD[a_type, simd_size]]()
    var a_smem = external_memory[
        Scalar[a_type],
        address_space=.SHARED,
        alignment=alignment,
    ]()
    comptime a_smem_size: Int = num_pipeline_stages * BM * BK
    comptime a_smem_layout = row_major[BM, BK]()
    var a_smem_iter = TileTensorIter[
        a_type,
        type_of(a_smem_layout),
        _,
        address_space=.SHARED,
        circular=True,
    ](
        a_smem.unsafe_offset(Int(warp_k_part_id) * a_smem_size),
        a_smem_layout,
        bound=a_smem_size,
    )

    # There is one pre-allocated shared buffer. Explicitly offset B after at A's end.
    var b_smem = a_smem.unsafe_offset(
        num_warp_k_partitions * a_smem_size
    ).unsafe_bitcast[Scalar[b_type]]()
    comptime b_smem_size: Int = num_pipeline_stages * BK * BN
    comptime BD_0 = BN if transpose_b else BK
    comptime BD_1 = BK if transpose_b else BN
    comptime b_smem_layout = row_major[BD_0, BD_1]()
    var b_smem_iter = TileTensorIter[
        b_type,
        type_of(b_smem_layout),
        _,
        address_space=.SHARED,
        circular=True,
    ](
        b_smem.unsafe_offset(Int(warp_k_part_id) * b_smem_size),
        b_smem_layout,
        bound=b_smem_size,
    )

    # Global memory iterators over this block's A and B tiles along K. A
    # partial last row tile of A is masked by the iterator; B's valid rows
    # are counted from its first tile and, when B is not transposed, shrink
    # with every K tile.
    var bk_start: Int = (
        ufloordiv(ufloordiv(K, BK), num_warp_k_partitions) * warp_k_part_id
    )
    var a_row0 = block_m * BM
    var a_tile_layout = TileLayout(
        Coord(Idx[BM], Idx[BK]), a_tt.layout.stride_coord()
    )
    var a_gmem_iter = TileTensorIter[
        a_type,
        type_of(a_tile_layout),
        _,
        masked=_tile_is_masked[ALT, BM, BK](),
    ](
        a_tt.tile[BM, BK](Coord(block_m, bk_start)),
        a_tile_layout,
        bound=M * Int(a_tt.dynamic_stride(0)),
        stride=BK * Int(a_tt.dynamic_stride(1)),
        num_valid_rows=M - a_row0,
    )

    comptime b_tile_axis = 1 if transpose_b else 0
    var b_row0 = block_n * BN if transpose_b else bk_start * BK
    var b_tile_layout = TileLayout(
        Coord(Idx[BD_0], Idx[BD_1]), b_tt.layout.stride_coord()
    )
    var b_gmem_iter = TileTensorIter[
        b_type,
        type_of(b_tile_layout),
        _,
        masked=_tile_is_masked[BLT, BD_0, BD_1](),
    ](
        b_tt.tile[BD_0, BD_1](
            Coord(
                block_n if transpose_b else bk_start,
                bk_start if transpose_b else block_n,
            )
        ),
        b_tile_layout,
        bound=Int(b_tt.dim[0]()) * Int(b_tt.dynamic_stride(0)),
        stride=BK * Int(b_tt.dynamic_stride(b_tile_axis)),
    )
    var b_valid_rows = Int(b_tt.dim[0]()) - b_row0

    # Compute MMA config
    comptime mma_shape = get_mma_shape[a_type, get_accum_type[a_type]()]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_k_mmas = BK // MMA_K
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N

    comptime accum_type = get_accum_type[a_type]()
    comptime frag_size = get_fragment_size[mma_shape]()
    comptime c_frag_size = frag_size[2]
    var c_reg_tile = stack_allocation[dtype=accum_type, address_space=.LOCAL](
        row_major[num_m_mmas * num_n_mmas, c_frag_size]()
    ).fill(0)

    multistage_mma[
        BM,
        BN,
        BK,
        WM,
        WN,
        num_threads_per_warp_k_part,
        num_pipeline_stages,
        transpose_b,
        k_group_size=config.k_group_size,
        swizzle_a=is_nvidia_gpu(),
    ](
        c_reg_tile,
        a_gmem_iter,
        b_gmem_iter,
        a_smem_iter,
        b_smem_iter,
        uceildiv(ufloordiv(K, num_warp_k_partitions), BK),
        num_b_rows=b_valid_rows,
    )

    # reduce within the threadblock
    comptime if num_warp_k_partitions > 1:
        warp_split_k_reduction[
            BM,
            BN,
            num_threads_per_warp_k_part,
            num_warp_k_partitions,
        ](
            warp_k_part_id,
            c_reg_tile,
        )
        if warp_k_part_id > 0:
            return

    # Map global memory tile down to thread.
    var c_gmem_warp_tile = c_tt.tile[BM, BN](Coord(block_m, block_n)).tile[
        WM, WN
    ](Coord(Int(warp_y), Int(warp_x)))

    # The NVIDIA C store below writes 2 elements at a time along N, which needs
    # every row of C to be aligned to that vector. A row starts at a multiple of
    # N * size_of[c_type](), so an odd fp32 N (e.g. GPT-2's 50257 vocab
    # projection) leaves every other row 4B-misaligned and the store faults.
    # AMD vectorizes along M instead, one column at a time, so it is exempt.
    var c_rows_aligned = not is_nvidia_gpu() or (
        N * size_of[c_type]() % align_of[SIMD[c_type, 2]]() == 0
    )
    var warp_row = block_m * BM + Int(warp_y) * WM
    var warp_col = block_n * BN + Int(warp_x) * WN
    comptime store_vec_rows = 1 if is_nvidia_gpu() else 4
    var c_tile_in_range = (
        warp_row + WM <= M and warp_col + WN <= N and M % store_vec_rows == 0
    )

    comptime has_output_fn = has_epilogue_fn or has_compute_fn

    # Stores `vec` at `(m, n)` through the output closure. `alignment` is in
    # bytes.
    @inline(.always)
    def store_output[
        dtype: DType, width: SIMDLength, //, alignment: Int
    ](m: Int, n: Int, vec: SIMD[dtype, width]) {imm}:
        comptime if has_compute_fn:
            var out = compute_fn[
                c_type, width, alignment=alignment // size_of[c_type]()
            ]((m, n), vec.cast[c_type]())
            c_tt.store[alignment=alignment](Coord(m, n), out)
        else:
            epilogue_fn[dtype, width, alignment=alignment]((m, n), vec)

    @inline(.always)
    def store_c_frags[in_range: Bool]() {imm}:
        """Writes the warp's accumulator fragments to C at their true
        `(row, col)`.

        Mirrors the vectorized stores: NVIDIA distributes `row_major(8, 4)`
        over 2-element vectors along N, AMD distributes `row_major(4, 16)`
        over 4-element vectors along M. With `in_range` the warp tile lies
        inside C and NVIDIA keeps its 2-wide stores through the output
        closure. Otherwise every element is bounded by its own coordinates:
        an odd fp32 N (misaligned rows), a warp whose columns run past N, and
        on AMD a 4-row group that runs past M carry a linear offset that
        lands on the next row, so they cannot be rejected on the coordinates
        a linear offset implies.
        """
        comptime threads_m = 8 if is_nvidia_gpu() else 4
        comptime threads_n = 4 if is_nvidia_gpu() else 16
        comptime vec_width = 2 if is_nvidia_gpu() else 4
        comptime rows_per_vec = 1 if is_nvidia_gpu() else vec_width
        comptime cols_per_vec = vec_width if is_nvidia_gpu() else 1
        comptime row_step = threads_m * rows_per_vec
        comptime col_step = threads_n * cols_per_vec
        comptime frag_rows = WM // row_step
        comptime frag_cols = WN // col_step
        comptime vecs_per_frag = c_frag_size // vec_width

        # (vecs_per_frag, num_mmas): fragment `(vec, mma)` of the mma whose
        # accumulators sit at `mma = n_mma * num_m_mmas + m_mma`.
        var c_reg_frag = c_reg_tile.vectorize[1, vec_width]().transpose()
        var thread_row = warp_row + (Int(ln_id) // threads_n) * rows_per_vec
        var thread_col = warp_col + (Int(ln_id) % threads_n) * cols_per_vec

        comptime for frag_col in range(frag_cols):
            comptime for frag_row in range(frag_rows):
                # Fragments are walked leading mode fastest.
                comptime i = frag_col * frag_rows + frag_row
                var vec = c_reg_frag[i % vecs_per_frag, i // vecs_per_frag]
                var row = thread_row + frag_row * row_step
                var col = thread_col + frag_col * col_step

                comptime if in_range and is_nvidia_gpu() and has_output_fn:
                    store_output[align_of[SIMD[c_type, vec_width]]()](
                        row, col, vec
                    )
                else:
                    comptime for j in range(vec_width):
                        var row_j = row + (0 if is_nvidia_gpu() else j)
                        var col_j = col + (j if is_nvidia_gpu() else 0)

                        if in_range or (row_j < M and col_j < N):
                            comptime if has_output_fn:
                                store_output[align_of[Scalar[c_type]]()](
                                    row_j, col_j, vec[j].cast[c_type]()
                                )
                            else:
                                c_tt[row_j, col_j] = vec[j].cast[c_type]()

    # Store FP32 mma results to half precision buffer in global memory.
    # Each thread's fragment has 2x2 fp32 values. Casting to half float and
    # directly storing to global memory results in 2 4B writes. Following cutlass,
    # we stage the fragments in shared memory so that each thread can store 16B.
    comptime if c_type.is_half_float() and is_nvidia_gpu():
        comptime swizzle = make_swizzle[
            num_rows=MMA_M // 2, row_size=WN, access_size=MMA_N
        ]()

        var accum_smem_warp_tile = TileTensor(
            a_smem.unsafe_bitcast[Scalar[c_type]]().unsafe_offset(
                Int(warp_id) * WM * WN
            ),
            row_major[WM, WN](),
        )

        copy_local_to_shared[
            thread_layout=row_major[8, 4](),
            swizzle=swizzle,
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            c_reg_tile.vectorize[1, 2]().transpose(),
        )

        # Guard writing to shared memory.
        barrier()

        # The local-to-shared copy has already cast the accumulators to half
        # precision. Read the staged values in vectors for 16-byte global stores.
        comptime thread_rows = WARP_SIZE * simd_size // WN
        comptime thread_cols = WN // simd_size
        comptime warp_layout = row_major[thread_rows, thread_cols]()
        comptime if has_output_fn:
            var c_smem_frag = accum_smem_warp_tile.vectorize[
                1, simd_size
            ]().distribute[warp_layout](Int(ln_id))
            var c_smem_frag_offset = (
                Int(c_smem_frag.ptr) - Int(accum_smem_warp_tile.ptr)
            ) // size_of[c_type]()
            var thread_row = warp_row + Int(ln_id) // thread_cols
            var thread_col = warp_col + (Int(ln_id) % thread_cols) * simd_size
            comptime alignment = align_of[SIMD[c_type, simd_size]]()

            comptime for i in range(WM // thread_rows):
                # The swizzle permutes within its period only, so it applies
                # to the in-period part of the fragment's offset.
                var src_idx = Int(c_smem_frag.layout(Idx[i]))
                var src_idx_base = src_idx % swizzle.size()
                var swizzled_idx = swizzle(
                    c_smem_frag_offset + src_idx_base
                ) + (src_idx - src_idx_base)

                var m = thread_row + i * thread_rows
                var n = thread_col
                if m < M and n < N:
                    var row, col = divmod(swizzled_idx, WN)
                    store_output[alignment](
                        m,
                        n,
                        accum_smem_warp_tile.load[
                            width=simd_size, alignment=alignment
                        ]((row, col)),
                    )
        else:
            if c_tile_in_range:
                copy_sram_to_dram[thread_layout=warp_layout, swizzle=swizzle](
                    c_gmem_warp_tile.vectorize[1, simd_size](),
                    accum_smem_warp_tile.vectorize[1, simd_size](),
                )
            else:
                # The warp tile runs past the last row of C, so bound the
                # store by the rows left. Written whole, the tile lands past
                # the end of C, or in split-K in the next partition's slab.
                copy_sram_to_dram[
                    thread_layout=warp_layout, swizzle=swizzle, masked=True
                ](
                    c_gmem_warp_tile.vectorize[1, simd_size](),
                    accum_smem_warp_tile.vectorize[1, simd_size](),
                    dst_num_valid_rows=max(0, M - warp_row),
                )

    # AMD distributes each thread's fragment down a column, four rows at a
    # time. A `vectorize[4, 1]` view of C would hand `copy_local_to_dram` an
    # element whose rows are a row stride apart, which the TileTensor copy
    # writes as one contiguous run, so the fragments are stored one scalar
    # at a time on every AMD path.
    elif not is_nvidia_gpu():
        if c_tile_in_range:
            store_c_frags[in_range=True]()
        else:
            store_c_frags[in_range=False]()
    # Store FP32 results to FP32 buffer in global memory. The TileTensor
    # `copy_local_to_dram` does not convert, so a C whose dtype differs from
    # the accumulator's goes through `store_c_frags`, which casts each element.
    else:
        if c_rows_aligned and c_tile_in_range:
            comptime if has_output_fn or c_type != accum_type:
                store_c_frags[in_range=True]()
            else:
                copy_local_to_dram[thread_layout=row_major[8, 4]()](
                    c_gmem_warp_tile.vectorize[1, 2](),
                    c_reg_tile.vectorize[1, 2]().transpose(),
                )
        else:
            store_c_frags[in_range=False]()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(config.num_threads())
    )
)
@__name(
    t"multistage_gemm_split_k_kernel_{c_type}_{a_type}_{b_type}_{transpose_b}",
)
def multistage_gemm_split_k_kernel[
    c_type: DType,
    CLT: TensorLayout,
    a_type: DType,
    ALT: TensorLayout,
    b_type: DType,
    BLT: TensorLayout,
    work_space_type: DType,
    WLT: TensorLayout,
    transpose_b: Bool,
    c_linear_idx_type: DType,
    a_linear_idx_type: DType,
    b_linear_idx_type: DType,
    workspace_linear_idx_type: DType,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
](
    c: TileTensor[c_type, CLT, MutAnyOrigin, linear_idx_type=c_linear_idx_type],
    a: TileTensor[
        a_type, ALT, ImmutAnyOrigin, linear_idx_type=a_linear_idx_type
    ],
    b: TileTensor[
        b_type, BLT, ImmutAnyOrigin, linear_idx_type=b_linear_idx_type
    ],
    work_space: TileTensor[
        work_space_type,
        WLT,
        MutAnyOrigin,
        linear_idx_type=workspace_linear_idx_type,
    ],
    num_partitions: Int32,
):
    var _num_partitions = Int(num_partitions)
    comptime assert c.rank == c.flat_rank == 2
    comptime assert a.rank == a.flat_rank == 2
    comptime assert b.rank == b.flat_rank == 2
    comptime assert work_space.rank == work_space.flat_rank == 3
    var M = Int(c.dim[0]())
    comptime N = b.static_shape[0] if transpose_b else b.static_shape[1]
    comptime K = b.static_shape[1] if transpose_b else b.static_shape[0]
    comptime BK = config.block_tile_shape[2]

    var ws_tt = TileTensor(
        work_space.ptr_at_offset((block_idx.z, 0, 0)),
        row_major(c.layout.shape[0](), c.layout.shape[1]()),
    )
    comptime k_partition_config = MatmulConfig[
        a_type,
        b_type,
        work_space_type,
        transpose_b,
    ](
        block_tile_shape=config.block_tile_shape,
        warp_tile_shape=config.warp_tile_shape,
        num_pipeline_stages=config.num_pipeline_stages,
    )

    comptime if (
        current_accelerator().is_amd_gpu()
        and not has_amd_rdna_gpu_accelerator()
        and transpose_b
    ):
        # `.split` makes the K axis dynamic, but AMDMatmul derives its
        # K-loop bound from the static shape. Carve comptime K/P tiles so
        # each partition keeps a static K (parent strides preserved).
        comptime assert K % config.num_k_partitions == 0, (
            "AMD split-K carves static K/P tiles, so K must be divisible by"
            " num_k_partitions (no ragged remainder)."
        )
        comptime K_part = K // config.num_k_partitions
        # AMDMatmul's K-loop iterates K_part // BK tiles by integer division,
        # so a ragged tail would be silently dropped. The dispatch gate keeps
        # BK == 64 and K a multiple of num_k_partitions * 64.
        comptime assert (
            K_part % BK == 0
        ), "AMD split-K requires each K partition to be a multiple of BK."
        var z = block_idx.z
        var a_amd = a.tile(Coord(M, Idx[K_part]), Coord(0, z))
        var b_amd = b.tile(Coord(Idx[N], Idx[K_part]), Coord(0, z))
        AMDMatmul[
            a_type,
            b_type,
            work_space_type,
            transpose_b,
            k_partition_config,
        ].run[
            type_of(ws_tt).LayoutType,
            type_of(a_amd).LayoutType,
            type_of(b_amd).LayoutType,
            type_of(ws_tt).Engine,
            type_of(a_amd).Engine,
            type_of(b_amd).Engine,
        ](
            ws_tt, a_amd, b_amd
        )

    else:
        comptime assert (
            a.static_shape[1] != UNKNOWN_VALUE
            and a.static_stride[1] != UNKNOWN_VALUE
        ), "Shouldn't split dynamic dimension."
        comptime b_k_axis = 1 if transpose_b else 0
        comptime assert (
            b.static_shape[b_k_axis] != UNKNOWN_VALUE
            and b.static_stride[b_k_axis] != UNKNOWN_VALUE
        ), "Shouldn't split dynamic dimension."
        # Preserve the historical floor-then-align split. TileTensor.split
        # uses ceil division and clamps exhausted partitions instead.
        var partition_size = align_up(K // _num_partitions, BK)
        var k_start = block_idx.z * partition_size
        var k_size = Int64(min(partition_size, K - k_start))
        var a_part = TileTensor(
            a.ptr_at_offset((0, k_start)),
            TileLayout(
                Coord(a.layout.shape[0](), k_size), a.layout.stride_coord()
            ),
        )
        comptime if transpose_b:
            var b_part = TileTensor(
                b.ptr_at_offset((0, k_start)),
                TileLayout(
                    Coord(b.layout.shape[0](), k_size), b.layout.stride_coord()
                ),
            )
            multistage_gemm_kernel[config=k_partition_config](
                ws_tt, a_part, b_part
            )
        else:
            var b_part = TileTensor(
                b.ptr_at_offset((k_start, 0)),
                TileLayout(
                    Coord(k_size, b.layout.shape[1]()), b.layout.stride_coord()
                ),
            )
            multistage_gemm_kernel[config=k_partition_config](
                ws_tt, a_part, b_part
            )
