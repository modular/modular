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
"""Provides GPU kernels for block-wise quantized int4 matrix multiplication.

Computes `C = A @ B` on the GPU, where `B` holds 4-bit quantized integer
weights packed eight values per 32-bit word and `A` and `C` are `bfloat16`.
The weights are dequantized in shared memory and fed through a multistage,
software-pipelined tensor-core GEMM.

## Limitations

- Requires an NVIDIA GPU target.
- `B` must be repacked before the matmul: use `gpu_qint4_repack_Q4_0` for GGUF
  Q4_0 weights or `gpu_qint4_repack_GPTQ` for GPTQ weights (with optional
  activation-order permutation).
"""

from std.math import ceildiv, divmod
from std.math.uutils import umod, ufloordiv, udivmod, uceildiv

from std.sys import align_of, is_nvidia_gpu, simd_width_of, size_of

from std.bit import log2_floor
from std.collections import OptionalReg
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.info import is_gpu
from max.gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_group,
    external_memory,
)
from layout import Coord, Idx, TileTensor, TensorLayout, row_major
from layout.tile_iterator import TileTensorIter
from layout.tile_tensor import stack_allocation
from layout.swizzle import Swizzle, make_ldmatrix_swizzle, make_swizzle
from layout.tile_io import (
    GenericToSharedAsyncTileCopier,
    ThreadScope,
    copy_dram_to_sram,
    copy_local_to_dram,
    copy_local_to_shared,
    copy_sram_to_dram,
)
from layout.tensor_core import TensorCore, get_fragment_size, get_mma_shape
from linalg.matmul.gpu._multistage_gemm_gpu import warp_split_k_reduction
from linalg.utils import ElementwiseEpilogueFn, GemmShape, no_epilogue_fn
from linalg.utils_gpu import MatmulConfig, block_swizzle
from std.memory.unsafe import bitcast


from std.utils.index import Index, StaticTuple
from std.utils.numerics import get_accum_type


@inline(.always)
@doc_hidden
def _quantized_scale_stage(stage: Int, ahead: Int, count: Int) -> Int:
    """Returns a ring stage advanced by `ahead`, wrapping once at `count`."""
    var next_stage = stage + ahead
    return next_stage - count if next_stage >= count else next_stage


@inline(.always)
@doc_hidden
def _copy_quantized_scale_tile(
    dst: TileTensor[mut=True, address_space=.SHARED, ...],
    src: TileTensor[address_space=.GENERIC, ...],
):
    comptime assert dst.dtype == DType.bfloat16 and src.dtype == DType.bfloat16
    comptime assert dst.static_shape[0] == 1 and src.static_shape[0] == 1
    comptime assert dst.static_shape[1] == src.static_shape[1]
    comptime assert dst.rank == dst.flat_rank == 2
    comptime assert src.rank == src.flat_rank == 2
    comptime assert dst.element_size == src.element_size == 1
    comptime assert dst.static_shape[1] % WARP_SIZE == 0
    comptime width = dst.static_shape[1] // WARP_SIZE
    comptime assert width in (2, 4, 8)
    var copier = GenericToSharedAsyncTileCopier[
        row_major[1, WARP_SIZE](), thread_scope=ThreadScope.WARP
    ]()
    copier.copy(dst.vectorize[1, width](), src.vectorize[1, width]())


@inline(.always)
@doc_hidden
def _load_quantized_scale_registers[
    dtype: DType
](
    dst: TileTensor[mut=True, dtype, address_space=.LOCAL, ...],
    src: TileTensor[dtype, address_space=.SHARED, ...],
    lane: Int,
):
    comptime assert dst.dtype == DType.bfloat16 and src.dtype == DType.bfloat16
    comptime assert dst.rank == dst.flat_rank == 2
    comptime assert src.rank == src.flat_rank == 2
    comptime assert dst.element_size == src.element_size == 1
    comptime assert dst.static_shape[1] == 1
    comptime assert dst.static_shape[0] in (2, 4, 6, 8)
    comptime assert src.static_shape[0] == 1
    comptime assert src.static_shape[1] == 64
    # Repacking interleaves scale pairs: four lanes share eight contiguous
    # packed scales, rather than eight naturally strided matrix columns.
    comptime for i in range(dst.static_shape[0]):
        var value = src.load[alignment=align_of[dtype]()](
            (0, 8 * (lane // 4) + i)
        )
        dst.store[alignment=align_of[dtype]()]((i, 0), value)


@inline(.always)
@doc_hidden
def args_to_tuple[swap: Bool](arg_0: Int, arg_1: Int) -> Tuple[Int, Int]:
    """Returns the two integer arguments as a tuple, swapping their order when `swap` is set.

    Parameters:
        swap: Whether to swap the order of the two arguments.

    Args:
        arg_0: The first integer argument.
        arg_1: The second integer argument.

    Returns:
        A tuple of the two arguments in the requested order.
    """
    comptime if swap:
        return (arg_1, arg_0)
    else:
        return (arg_0, arg_1)


@inline(.always)
@doc_hidden
@inline(.always)
@doc_hidden
def multistage_mma_q[
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
    transpose_b: Bool,
    group_size: Int,
    pack_factor: Int,
    c_type: DType,
    c_layout: TensorLayout,
    a_type: DType,
    a_layout: TensorLayout,
    a_smem_layout: TensorLayout,
    b_type: DType,
    b_layout: TensorLayout,
    b_smem_layout: TensorLayout,
    scales_type: DType,
    scales_layout: TensorLayout,
    scales_smem_layout: TensorLayout,
    /,
    *,
    swizzle_a: Bool = True,
    prefetch_init: Bool = True,
    native_scale_stream: Bool = False,
](
    c: TileTensor[mut=True, c_type, c_layout, address_space=.LOCAL, ...],
    a_iter_arg: TileTensorIter[
        a_type, a_layout, _, address_space=.GENERIC, ...
    ],
    b_iter_arg: TileTensorIter[
        b_type, b_layout, _, address_space=.GENERIC, ...
    ],
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
    scales_smem_iter_arg: TileTensorIter[
        mut=True,
        scales_type,
        scales_smem_layout,
        address_space=.SHARED,
        ...,
    ],
    scales_iter_arg: TileTensorIter[
        scales_type, scales_layout, _, address_space=.GENERIC, ...
    ],
    scales_native_src: TileTensor[scales_type, address_space=.GENERIC, ...],
    scales_native_ring: TileTensor[
        mut=True, scales_type, address_space=.SHARED, ...
    ],
    num_iters: Int,
):
    """Performs the multi-stage tensor-core MMA loop for a quantized matrix multiplication tile.

    Iterates over the K dimension, prefetching A, B, and per-group scale
    tiles into shared memory across `num_pipeline_stages` stages while issuing
    tensor-core MMA instructions, with optional swizzling of the A-tile copy.

    Parameters:
        BM: The block tile size along the M dimension.
        BN: The block tile size along the N dimension.
        BK: The block tile size along the K dimension.
        WM: The warp tile size along the M dimension.
        WN: The warp tile size along the N dimension.
        num_threads: The number of threads per thread block.
        num_pipeline_stages: The number of software-pipelined prefetch stages.
        transpose_b: Whether the B matrix is stored transposed.
        group_size: The number of K elements sharing a single scale.
        pack_factor: The number of 4-bit elements packed into one `uint32`.
        c_type: The dtype of the output accumulator tile.
        c_layout: The layout of the output accumulator tile.
        a_type: The dtype of the A matrix elements.
        a_layout: The layout of the A matrix in global memory.
        a_smem_layout: The layout of the A tile in shared memory.
        b_type: The dtype of the packed B matrix elements.
        b_layout: The layout of the B matrix in global memory.
        b_smem_layout: The layout of the B tile in shared memory.
        scales_type: The dtype of the quantization scales.
        scales_layout: The layout of the scales in global memory.
        scales_smem_layout: The layout of the scales tile in shared memory.
        swizzle_a: Whether to apply a swizzle pattern to the A-tile copy.
        prefetch_init: Whether to prefetch the initial pipeline stages.
        native_scale_stream: Whether the scale pipeline uses the native ring
            instead of the global iterator.

    Args:
        c: The local accumulator tile receiving the MMA results.
        a_iter_arg: The global-memory iterator over A tiles.
        b_iter_arg: The global-memory iterator over B tiles.
        a_smem_iter_arg: The shared-memory iterator over A prefetch buffers.
        b_smem_iter: The shared-memory iterator over B prefetch buffers.
        scales_smem_iter_arg: The shared-memory iterator over scale prefetch buffers.
        scales_iter_arg: The global-memory iterator over scale tiles.
        scales_native_src: The first native global scale tile, retaining row stride.
        scales_native_ring: The native shared-memory scale ring for this partition.
        num_iters: The number of K-tile iterations to execute.
    """
    comptime assert c.rank == c.flat_rank == 2
    comptime assert c.element_size == 1
    comptime simd_size = simd_width_of[a_type]()
    comptime simd_b_size = simd_width_of[b_type]()
    comptime num_scales_stages = ceildiv(
        (num_pipeline_stages - 1) * BK, group_size
    ) + 1
    comptime repack_tile = (64, 16)

    var tid = UInt32(umod(thread_idx.x, num_threads))
    var warp_id, lane_id = divmod(tid, UInt32(WARP_SIZE))

    comptime num_warps_m = BM // WM
    comptime num_warps_n = BN // WN
    var warp_y, warp_x = divmod(warp_id, UInt32(num_warps_n))

    var a_iter = a_iter_arg
    var b_iter = b_iter_arg
    var scales_iter = scales_iter_arg
    var a_smem_iter = a_smem_iter_arg
    var scales_smem_iter = scales_smem_iter_arg
    var native_scale_stage = 0
    var native_scale_group = 0

    comptime async_copy_a_layout = row_major[
        num_threads * simd_size // BK, BK // simd_size
    ]()
    comptime b_smem_row_stride = b_smem_layout.static_stride[0]
    comptime async_copy_b_layout = row_major[
        ceildiv(num_threads * simd_b_size, b_smem_row_stride),
        num_threads // ceildiv(num_threads * simd_b_size, b_smem_row_stride),
    ]()

    # The ldmatrix-friendly swizzle for the A tile, whose leading-dim stride
    # is `BK`; `load_a` reads the tile back with the same pattern.
    comptime async_swizzle_a = Optional[Swizzle](
        make_ldmatrix_swizzle[a_type, BK, log2_floor(simd_size)]()
    ) if swizzle_a else Optional[Swizzle]()

    # Both `dst` (SMEM) and `src` (DRAM) must share a `linear_idx_type`.
    # Mixing `int32` SMEM offsets with `int64` DRAM offsets in a single
    # cp.async-driven distribute() emits cross-width conversions
    # (cvt.u64.u32 + mul.wide.u32 in the index computation) that measurably
    # degrade performance on B200.
    @inline(.always)
    def _async_copy_a_tile(
        dst: TileTensor[mut=True, a_type, address_space=.SHARED, ...],
        src: TileTensor[a_type, address_space=.GENERIC, ...],
    ):
        comptime assert dst.linear_idx_type == src.linear_idx_type
        comptime assert dst.rank == dst.flat_rank == 2
        comptime assert src.rank == src.flat_rank == 2
        comptime assert dst.element_size == src.element_size == 1
        GenericToSharedAsyncTileCopier[
            async_copy_a_layout,
            swizzle=async_swizzle_a,
        ]().copy(
            dst.vectorize[1, simd_size](),
            src.vectorize[1, simd_size](),
        )

    @inline(.always)
    def _async_copy_b_tile(
        dst: TileTensor[mut=True, b_type, address_space=.SHARED, ...],
        src: TileTensor[b_type, address_space=.GENERIC, ...],
    ):
        comptime assert dst.linear_idx_type == src.linear_idx_type
        comptime assert dst.rank == dst.flat_rank == 2
        comptime assert src.rank == src.flat_rank == 2
        comptime assert dst.element_size == src.element_size == 1
        GenericToSharedAsyncTileCopier[async_copy_b_layout]().copy(
            dst.vectorize[1, simd_b_size](),
            src.vectorize[1, simd_b_size](),
        )

    # Prefetch (num_pipeline_stages - 1) stages.
    comptime if prefetch_init:
        comptime for stage in range(num_pipeline_stages - 1):
            _async_copy_a_tile(a_smem_iter.next_unsafe(stage)[], a_iter[])
            a_iter._incr()

            _async_copy_b_tile(b_smem_iter.next_unsafe(stage)[], b_iter[])
            b_iter._incr()

            # Every group_size rows share a scale
            # Only load scales when necessary
            if stage % (group_size // BK) == 0:
                comptime scales_stage = stage // (group_size // BK)

                comptime if native_scale_stream:
                    if tid < UInt32(WARP_SIZE):
                        var dst = scales_native_ring.tile[1, BN](
                            (scales_stage, 0)
                        )
                        # Row `native_scale_group` of the [1, BN] tile:
                        # tiling past the one-row tile with the parent's
                        # stride, same offset the pointer form computed.
                        var src = scales_native_src.tile[1, BN](
                            (native_scale_group, 0)
                        )
                        _copy_quantized_scale_tile(dst, src)
                    native_scale_group += 1
                else:
                    # We only need one warp for copying scales...
                    if tid < UInt32(WARP_SIZE):
                        _copy_quantized_scale_tile(
                            scales_smem_iter.next_unsafe(scales_stage)[],
                            scales_iter[],
                        )
                    scales_iter._incr()

            async_copy_commit_group()

        # Guard stage 0.
        async_copy_wait_group(Int32(num_pipeline_stages - 2))
        barrier()

    comptime mma_shape = get_mma_shape[a_type, get_accum_type[a_type]()]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_k_mmas = BK // MMA_K
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N

    comptime accum_type = get_accum_type[a_type]()
    comptime frag_size = get_fragment_size[mma_shape]()
    comptime a_frag_size = frag_size[0]
    comptime b_frag_size = frag_size[1]
    comptime c_frag_size = frag_size[2]
    comptime assert c.static_shape[0] == num_m_mmas * num_n_mmas
    comptime assert c.static_shape[1] == c_frag_size

    var a_reg_buffer = stack_allocation[a_type, address_space=.LOCAL](
        row_major[2 * num_m_mmas, a_frag_size]()
    )
    var b_reg_buffer = stack_allocation[a_type, address_space=.LOCAL](
        row_major[2 * num_n_mmas, b_frag_size]()
    )
    var scales_reg_tiles = stack_allocation[scales_type, address_space=.LOCAL](
        row_major[num_n_mmas, 1]()
    )

    var a_warp_tile = a_smem_iter[].tile[WM, BK]((Int(warp_y), 0))

    comptime b_wtile_dim0 = WN // repack_tile[0] if transpose_b else (
        (BK * repack_tile[0]) // pack_factor
    )
    comptime b_wtile_dim1 = (
        (BK * repack_tile[0]) // pack_factor
    ) if transpose_b else WN // repack_tile[0]
    var b_wtile_coord0 = Int(warp_x) if transpose_b else 0
    var b_wtile_coord1 = 0 if transpose_b else Int(warp_x)
    var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
        (b_wtile_coord0, b_wtile_coord1)
    )
    var scales_warp_tile = scales_smem_iter[].tile[ceildiv(BK, group_size), WN](
        (0, Int(warp_x))
    )

    var mma_op = TensorCore[accum_type, a_type, mma_shape, transpose_b]()

    comptime swizzle_a_pattern = make_ldmatrix_swizzle[
        a_type, type_of(a_warp_tile).static_stride[0]
    ]() if swizzle_a else Optional[Swizzle]()

    mma_op.load_a[swizzle_a_pattern](
        a_warp_tile, a_reg_buffer.tile[num_m_mmas, a_frag_size]((0, 0))
    )

    comptime if native_scale_stream:
        var native_warp_scales = scales_native_ring.tile[1, BN](
            (native_scale_stage, 0)
        ).tile[1, WN]((0, Int(warp_x)))
        _load_quantized_scale_registers(
            scales_reg_tiles, native_warp_scales, Int(lane_id)
        )
    else:
        _load_quantized_scale_registers(
            scales_reg_tiles, scales_warp_tile, Int(lane_id)
        )

    mma_op.load_b(
        b_warp_tile,
        b_reg_buffer.tile[num_n_mmas, b_frag_size]((0, 0)).bitcast[
            DType.bfloat16
        ](),
        scales_reg_tiles.bitcast[DType.bfloat16](),
        0,
    )

    for k_tile_id in range(num_iters):
        var a_warp_tile = a_smem_iter[].tile[WM, BK]((Int(warp_y), 0))
        var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
            (b_wtile_coord0, b_wtile_coord1)
        )

        # Perform prefetch registers and mma until current shared memory tile's
        # data has all been loaded to registers.
        comptime for k_mma in range(num_k_mmas):
            var current = k_mma % 2
            var next = (k_mma + 1) % 2

            if k_mma == num_k_mmas - 1:
                a_smem_iter._incr()
                b_smem_iter._incr()

                a_warp_tile = a_smem_iter[].tile[WM, BK]((Int(warp_y), 0))
                b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
                    (b_wtile_coord0, b_wtile_coord1)
                )

                # prefetch scales into regs every (group_size) rows
                if (k_tile_id + 1) % (group_size // BK) == 0:
                    scales_smem_iter._incr()
                    scales_warp_tile = scales_smem_iter[].tile[
                        ceildiv(BK, group_size), WN
                    ]((0, Int(warp_x)))
                    comptime if native_scale_stream:
                        native_scale_stage = _quantized_scale_stage(
                            native_scale_stage, 1, num_scales_stages
                        )
                        var native_warp_scales = scales_native_ring.tile[1, BN](
                            (native_scale_stage, 0)
                        ).tile[1, WN]((0, Int(warp_x)))
                        _load_quantized_scale_registers(
                            scales_reg_tiles, native_warp_scales, Int(lane_id)
                        )
                    else:
                        _load_quantized_scale_registers(
                            scales_reg_tiles, scales_warp_tile, Int(lane_id)
                        )

            mma_op.load_a[swizzle_a_pattern](
                a_warp_tile,
                a_reg_buffer.tile[num_m_mmas, a_frag_size]((next, 0)),
                (k_mma + 1) % num_k_mmas,
            )
            mma_op.load_b(
                b_warp_tile,
                b_reg_buffer.tile[num_n_mmas, b_frag_size]((next, 0)).bitcast[
                    DType.bfloat16
                ](),
                scales_reg_tiles.bitcast[DType.bfloat16](),
                (k_mma + 1) % num_k_mmas,
            )

            mma_op.mma(
                a_reg_buffer.tile[num_m_mmas, a_frag_size](
                    (current, 0)
                ).vectorize[1, a_frag_size](),
                b_reg_buffer.tile[num_n_mmas, b_frag_size](
                    (current, 0)
                ).vectorize[1, b_frag_size](),
                c.vectorize[1, c_frag_size](),
            )

            if k_mma + 2 == num_k_mmas:
                var prefetch_tile_id = k_tile_id + num_pipeline_stages - 1

                # Prefetch one k tile (if valid) from global memory to current
                # shared memory buffer.
                if prefetch_tile_id < num_iters:
                    _async_copy_a_tile(
                        a_smem_iter.next_unsafe(num_pipeline_stages - 1)[],
                        a_iter[],
                    )
                    a_iter._incr()

                    _async_copy_b_tile(
                        b_smem_iter.next_unsafe(num_pipeline_stages - 1)[],
                        b_iter[],
                    )
                    b_iter._incr()

                    # Every group_size rows share a scale
                    # Only load scales when necessary
                    if (k_tile_id + num_pipeline_stages - 1) % (
                        group_size // BK
                    ) == 0:
                        comptime if native_scale_stream:
                            if tid < UInt32(WARP_SIZE):
                                var stage = _quantized_scale_stage(
                                    native_scale_stage,
                                    num_scales_stages - 1,
                                    num_scales_stages,
                                )
                                var dst = scales_native_ring.tile[1, BN](
                                    (stage, 0)
                                )
                                # Row `native_scale_group` of the [1, BN]
                                # tile: tiling past the one-row tile with
                                # the parent's stride, same offset the
                                # pointer form computed.
                                var src = scales_native_src.tile[1, BN](
                                    (native_scale_group, 0)
                                )
                                _copy_quantized_scale_tile(dst, src)
                            native_scale_group += 1
                        else:
                            # We only need one warp for copying scales...
                            if tid < UInt32(WARP_SIZE):
                                _copy_quantized_scale_tile(
                                    scales_smem_iter.next_unsafe(
                                        num_scales_stages - 1
                                    )[],
                                    scales_iter[],
                                )
                            scales_iter._incr()

                async_copy_commit_group()

                # Guard the next k tile's shared memory buffer.
                async_copy_wait_group(Int32(num_pipeline_stages - 2))
                barrier()


@__name(
    t"multistage_qgemm_{a_type}_{b_packed_type}_{c_type}_g{group_size}",
)
@doc_hidden
def multistage_qgemm_kernel[
    c_type: DType,
    c_layout: TensorLayout,
    a_type: DType,
    a_layout: TensorLayout,
    b_packed_type: DType,
    b_layout: TensorLayout,
    group_size: Int,
    pack_factor: Int,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_packed_type, c_type, transpose_b],
    EpilogueFnType: ElementwiseEpilogueFn,
    has_epilogue_fn: Bool,
](
    c_tt: TileTensor[mut=True, c_type, c_layout, MutAnyOrigin],
    a_tt: TileTensor[mut=False, a_type, a_layout, ImmutAnyOrigin],
    b_packed: TileTensor[mut=False, b_packed_type, b_layout, ImmutAnyOrigin],
    epilogue_fn: EpilogueFnType,
):
    """Implements the GPU kernel for a multi-stage quantized GEMM with per-group scales.

    Unpacks the quantized B weights and scales from the packed buffer, sets up
    the circular shared-memory prefetch buffers, drives the `multistage_mma_q`
    MMA loop, performs the warp split-K reduction, and stores the accumulator
    to global memory with an optional elementwise epilogue.

    Parameters:
        c_type: The dtype of the output matrix.
        c_layout: The layout of the output matrix.
        a_type: The dtype of the A matrix elements.
        a_layout: The layout of the A matrix.
        b_packed_type: The dtype of the packed B weight buffer.
        b_layout: The layout of the packed B weight buffer.
        group_size: The number of K elements sharing a single scale.
        pack_factor: The number of 4-bit elements packed into one `uint32`.
        transpose_b: Whether the B matrix is stored transposed.
        config: The matmul configuration describing tile and warp shapes.
        EpilogueFnType: Type of `epilogue_fn`.
        has_epilogue_fn: Whether `epilogue_fn` stores the output instead of
            the kernel.

    Args:
        c_tt: The output accumulator matrix in global memory.
        a_tt: The left-hand (activation) matrix in global memory.
        b_packed: The packed quantized weight buffer in global memory.
        epilogue_fn: Epilogue that stores each output element when
            `has_epilogue_fn` is set.
    """
    comptime assert (
        is_nvidia_gpu()
    ), "Quantized gemm only supports NVIDIA hardwares for now."
    comptime simd_size = simd_width_of[c_type]()

    comptime repack_tile = (64, 16)
    comptime group_bytes = group_size // 2 + 2

    var M = Int(c_tt.dim[0]())
    comptime N = b_layout.static_shape[0]
    comptime K = b_layout.static_shape[1] // group_bytes * group_size

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]
    comptime WM = config.warp_tile_shape[0]
    comptime WN = config.warp_tile_shape[1]
    comptime num_pipeline_stages = config.num_pipeline_stages

    comptime num_warps_m = config.num_warps_m()
    comptime num_warps_n = config.num_warps_n()
    comptime num_threads = config.num_threads()

    # Unpack quantized weights
    comptime scales_type = DType.bfloat16
    comptime b_type = DType.uint32
    var b = b_packed.bitcast[b_type]().reshape(
        row_major[N // 64, K * 64 // pack_factor]()
    )

    comptime packed_bytes = b_layout.static_product
    var scales = (
        b_packed.reshape(row_major[packed_bytes]())
        .slice[N * K // 2 : packed_bytes]()
        .bitcast[scales_type]()
        .reshape(row_major[K // group_size, N]())
    )

    comptime num_warp_k_partitions = config.num_warp_k_partitions
    comptime num_threads_per_warp_k_part = num_threads // num_warp_k_partitions

    var tid = thread_idx.x
    var ln_id = lane_id()
    var warp_k_part_id = (
        ufloordiv(tid, num_threads_per_warp_k_part) if num_warp_k_partitions
        > 1 else 0
    )
    var warp_id = ufloordiv(umod(tid, num_threads_per_warp_k_part), WARP_SIZE)

    # Only apply block swizzling for half precision types.
    comptime swizzle_block = a_type.is_half_float() and b_type.is_half_float()

    var block_idx = block_swizzle(
        (block_idx.x, block_idx.y),
        (grid_dim.x, grid_dim.y),
    ) if swizzle_block else Index(block_idx.x, block_idx.y)

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
    comptime a_smem_size = num_pipeline_stages * BM * BK

    # The shared-memory iterators take the global tensors' index types so
    # the cp.async address arithmetic stays uniform-width (see
    # `multistage_mma_q`).
    comptime a_smem_layout = row_major[BM, BK]()
    var a_smem_iter = TileTensorIter[
        a_type,
        type_of(a_smem_layout),
        _,
        address_space=.SHARED,
        circular=True,
        linear_idx_type=a_tt.linear_idx_type,
    ](
        a_smem.unsafe_offset(warp_k_part_id * a_smem_size),
        a_smem_layout,
        bound=a_smem_size,
    )

    # There is one pre-allocated shared buffer. Explicitly offset B after at A's end.
    var b_smem = a_smem.unsafe_offset(
        num_warp_k_partitions * a_smem_size
    ).unsafe_bitcast[Scalar[b_type]]()
    comptime b_smem_size = num_pipeline_stages * BK * BN // pack_factor
    comptime BD_0 = BN // repack_tile[0]
    comptime BD_1 = (BK * repack_tile[0]) // pack_factor
    comptime b_smem_layout = row_major[BD_0, BD_1]()

    var b_smem_iter = TileTensorIter[
        b_type,
        type_of(b_smem_layout),
        _,
        address_space=.SHARED,
        circular=True,
        linear_idx_type=b.linear_idx_type,
    ](
        b_smem.unsafe_offset(warp_k_part_id * b_smem_size),
        b_smem_layout,
        bound=b_smem_size,
    )

    # multiple stages may share the same scales
    comptime num_scales_stages = ceildiv(
        (num_pipeline_stages - 1) * BK, group_size
    ) + 1
    var scales_smem = (
        b_smem.unsafe_offset(num_warp_k_partitions * b_smem_size)
    ).unsafe_bitcast[Scalar[scales_type]]()
    comptime scales_smem_size = num_scales_stages * BN * ceildiv(BK, group_size)
    comptime scales_smem_layout = row_major[ceildiv(BK, group_size), BN]()

    var scales_smem_iter = TileTensorIter[
        scales_type,
        type_of(scales_smem_layout),
        _,
        address_space=.SHARED,
        circular=True,
        linear_idx_type=scales.linear_idx_type,
    ](
        scales_smem.unsafe_offset(warp_k_part_id * scales_smem_size),
        scales_smem_layout,
        bound=scales_smem_size,
    )

    var scales_native_ring = TileTensor[
        linear_idx_type=scales.linear_idx_type, address_space=.SHARED
    ](
        scales_smem.unsafe_offset(warp_k_part_id * scales_smem_size),
        row_major[num_scales_stages * ceildiv(BK, group_size), BN](),
    )

    # Global memory iterators: each starts at this partition's first K tile
    # and steps one tile along K.
    var bk_start: Int = (K // BK // num_warp_k_partitions) * warp_k_part_id
    var a_gmem_tile = a_tt.tile[BM, BK]((block_idx[1], bk_start))
    var a_gmem_iter = TileTensorIter[
        a_type,
        type_of(a_gmem_tile).LayoutType,
        _,
        linear_idx_type=a_tt.linear_idx_type,
    ](a_gmem_tile, a_gmem_tile.layout, bound=K, stride=BK)

    var b_tile_coords = args_to_tuple[transpose_b](bk_start, block_idx[0])
    comptime b_tile_axis = 1 if transpose_b else 0
    comptime b_tile_extent = BD_1 if transpose_b else BD_0
    comptime b_axis_stride = type_of(b).static_stride[b_tile_axis]
    var b_gmem_tile = b.tile[BD_0, BD_1]((b_tile_coords[0], b_tile_coords[1]))
    var b_gmem_iter = TileTensorIter[
        b_type,
        type_of(b_gmem_tile).LayoutType,
        _,
        linear_idx_type=b.linear_idx_type,
    ](
        b_gmem_tile,
        b_gmem_tile.layout,
        bound=type_of(b).static_shape[b_tile_axis] * b_axis_stride,
        stride=b_tile_extent * b_axis_stride,
    )

    comptime groups_per_iter = ceildiv(BK, group_size)
    var bk_scales_start: Int = (
        K // (groups_per_iter * group_size) // num_warp_k_partitions
    ) * warp_k_part_id
    var scales_gmem_tile = scales.tile[groups_per_iter, BN](
        (bk_scales_start, Int(block_idx[0]))
    )
    var scales_gmem_iter = TileTensorIter[
        scales_type,
        type_of(scales_gmem_tile).LayoutType,
        _,
        linear_idx_type=scales.linear_idx_type,
    ](
        scales_gmem_tile,
        scales_gmem_tile.layout,
        bound=(K // group_size) * N,
        stride=groups_per_iter * N,
    )

    var scales_native_src = scales_gmem_tile.address_space_cast[.GENERIC]()
    # The native stream handles complete aligned scale groups. Keep the
    # original iterator path for tails and other group geometries.
    comptime native_scale_stream = (
        is_nvidia_gpu()
        and scales_type == DType.bfloat16
        and N % BN == 0
        and BN in (64, 128, 256)
        and WN == 64
        and group_size >= BK
        and group_size % BK == 0
        and K % (group_size * num_warp_k_partitions) == 0
        and K // group_size // num_warp_k_partitions
        >= ceildiv(
            num_pipeline_stages - 1,
            group_size // BK if group_size >= BK else 1,
        )
    )

    comptime mma_shape = get_mma_shape[a_type, get_accum_type[a_type]()]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N

    comptime accum_type = get_accum_type[a_type]()
    comptime frag_size = get_fragment_size[mma_shape]()
    comptime c_frag_size = frag_size[2]

    var c_reg_tile = stack_allocation[accum_type, address_space=.LOCAL](
        row_major[num_m_mmas * num_n_mmas, c_frag_size]()
    ).fill(0)

    multistage_mma_q[
        BM,
        BN,
        BK,
        WM,
        WN,
        num_threads_per_warp_k_part,
        num_pipeline_stages,
        transpose_b,
        group_size,
        pack_factor,
        native_scale_stream=native_scale_stream,
    ](
        c_reg_tile,
        a_gmem_iter,
        b_gmem_iter,
        a_smem_iter,
        b_smem_iter,
        scales_smem_iter,
        scales_gmem_iter,
        scales_native_src,
        scales_native_ring,
        ceildiv(K // num_warp_k_partitions, BK),
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

    # Map global memory tile down to thread. The tiles keep static shapes;
    # their linear offsets from `c_tt` recover each element's (m, n), and
    # `c_rows_left` bounds the stores of a partial M tile.
    var c_gmem_tile, _, c_tile_offset = c_tt.tile_with_offset[BM, BN](
        Coord(block_idx[1], block_idx[0])
    )
    var c_gmem_warp_tile, _, c_warp_offset = c_gmem_tile.tile_with_offset[
        WM, WN
    ](Coord(Int(warp_y), Int(warp_x)))
    var c_rows_left = M - block_idx[1] * BM - Int(warp_y) * WM

    @inline(.always)
    def apply_epilogue() {imm}:
        # This block is identical to the one used for f32 case
        # but putting this in a lambda function leads to test failures
        # TODO: Refactor to remove code duplication
        comptime assert has_epilogue_fn, "epilogue_fn is not valid"
        var c_gmem_frag, _, c_frag_offset = c_gmem_warp_tile.vectorize[
            1, 2
        ]().distribute_with_offset[row_major[8, 4]()](Int(ln_id))
        var c_reg_frag = c_reg_tile.vectorize[1, 2]().transpose()
        var thread_offset = c_tile_offset + c_warp_offset + c_frag_offset

        comptime for i in range(type_of(c_gmem_frag).LayoutType.static_product):
            var dst_idx = Int(c_gmem_frag.layout(Idx[i]))
            comptime alignment = align_of[SIMD[c_type, 2]]()
            var m, n = divmod(thread_offset + dst_idx, N)
            if m < M and n < N:
                var vec = c_reg_frag.load[width=2, alignment=alignment](
                    (Idx[i],)
                )
                epilogue_fn[alignment=alignment]((m, n), vec)

    # Store FP32 mma results to half precision buffer in global memory.
    # Each thread's fragment has 2x2 fp32 values. Casting to half float and
    # directly storing to global memory results in 2 4B writes. Following cutlass,
    # we stage the fragments in shared memory so that each thread can store 16B.

    comptime if c_type.is_half_float():
        comptime swizzle = make_swizzle[
            num_rows=MMA_M // 2, row_size=WN, access_size=MMA_N
        ]()

        var accum_smem_warp_tile = TileTensor(
            a_smem.unsafe_bitcast[Scalar[c_type]]().unsafe_offset(
                warp_id * WM * WN
            ),
            row_major[WM, WN](),
        )

        copy_local_to_shared[row_major[8, 4](), swizzle=swizzle](
            accum_smem_warp_tile.vectorize[1, 2](),
            c_reg_tile.vectorize[1, 2]().transpose(),
        )

        # Guard writing to shared memory.
        barrier()

        comptime warp_layout = row_major[
            WARP_SIZE * simd_size // WN, WN // simd_size
        ]()

        # The local-to-shared copy has already cast the accumulators to half
        # precision. Read the staged values in vectors for 16-byte global stores.
        comptime if has_epilogue_fn:
            var c_gmem_frag, _, c_frag_offset = c_gmem_warp_tile.vectorize[
                1, simd_size
            ]().distribute_with_offset[warp_layout](Int(thread_idx.x))
            var c_smem_frag, _, c_smem_frag_offset = (
                accum_smem_warp_tile.vectorize[
                    1, simd_size
                ]().distribute_with_offset[warp_layout](Int(thread_idx.x))
            )
            var thread_offset = c_tile_offset + c_warp_offset + c_frag_offset
            comptime num_stores_per_thread = type_of(
                c_gmem_frag
            ).LayoutType.static_product
            var accum_smem_flat = accum_smem_warp_tile.reshape(
                row_major[WM * WN]()
            )

            comptime for i in range(num_stores_per_thread):
                var src_idx = Int(c_smem_frag.layout(Idx[i]))
                var src_idx_base = src_idx % swizzle.size()
                var src_idx_diff = src_idx - src_idx_base
                var swizzled_idx = (
                    swizzle(c_smem_frag_offset + src_idx_base) + src_idx_diff
                )

                var dst_idx = Int(c_gmem_frag.layout(Idx[i]))
                var m, n = divmod(thread_offset + dst_idx, N)
                comptime alignment = align_of[SIMD[c_type, simd_size]]()
                if m < M and n < N:
                    epilogue_fn[alignment=alignment](
                        (m, n),
                        accum_smem_flat.load[
                            width=simd_size, alignment=alignment
                        ](Coord(swizzled_idx)).cast[c_type](),
                    )
        else:
            copy_sram_to_dram[warp_layout, swizzle=swizzle, masked=True](
                c_gmem_warp_tile.vectorize[1, simd_size](),
                accum_smem_warp_tile.vectorize[1, simd_size](),
                c_rows_left,
            )

    # Store FP32 results to FP32 buffer in global memory.
    else:
        comptime if has_epilogue_fn:
            apply_epilogue()
        else:
            copy_local_to_dram[row_major[8, 4]()](
                c_gmem_warp_tile.vectorize[1, 2](),
                c_reg_tile.vectorize[1, 2]().transpose(),
                c_rows_left,
            )


# For a 4-bit weight matrix of shape (64 * TN, 16 * TK), we first
# repack elements within each 64x16 tile, and in memory every 8
# 4-bit elements are packed in one uint32.
#              K
#       0       16      32
#      0+-------+-------+-----+
#       | 64x16 | 64x16 | ... |
#       | tile  | tile  |     |
# N   64+-------+-------+-----+   Matrix with 4-bit elements
#       | 64x16 | 64x16 | ... |
#       | tile  | tile  |     |
#    128+-------+-------+-----+
#
#
# Stored as uint32 tensor (64 * TN, 2 * TK):
#              K
#     0       2      4
#     0+------+------+-----+
#      | 64x2 | 64x2 | ... |
#      | tile | tile |     |
# N  64+------+------+-----+    uint32 Matrix
#      | 64x2 | 64x2 | ... |
#      | tile | tile |     |
#   128+------+------+-----+
#
# Elements within each tile are stored continuously
#                  K
#      0     1     2     3     4
#     0+-----+-----+-----+-----+
#      |  0  |  1  | 128 | 129 |
#     1+-----+-----+-----+-----+
#      |  2  |  3  | 130 | 131 |
#     2+-----+-----+-----+-----+
#      |  4  |  5  | 132 | 133 |
# N   3+-----+-----+-----+-----+
#      | ... | ... | ... | ... |
#    64+-----+-----+-----+-----+
#      |TK*  |TK*  | ... | ... |
#      |128  |128+1| ... | ... |
#    65+-----+-----+-----+-----+
#      |TK*  |TK*  | ... | ... |
#      |128+2|128+3| ... | ... |
#    65+-----+-----+-----+-----+
# The repacker writes a row-major uint32 TileTensor of shape (TN, TK * 128),
# where each 64x16 weight tile occupies 128 consecutive words.
@inline(.always)
@doc_hidden
def pack_Q_tile(input: SIMD[.uint8, 16]) -> SIMD[.uint32, 4]:
    """Packs sixteen bytes (thirty-two 4-bit weights, two per byte) into four `uint32` lanes for the repacked weight layout.

    Args:
        input: A SIMD vector of sixteen `uint8` values; both nibbles of each byte hold a 4-bit weight.

    Returns:
        A SIMD vector of four `uint32` values with the nibbles interleaved into the repacked layout.
    """
    # Q-tile is the smallest indivisible unit when performing gemm
    # operations with quantized matrices.

    var res: SIMD[.uint32, 4] = 0

    comptime for i in range(4):
        res[i] |= input[i * 4 + 0].cast[.uint32]() & 0x0F
        res[i] |= (input[i * 4 + 0].cast[.uint32]() & 0xF0) << 12
        res[i] |= (input[i * 4 + 1].cast[.uint32]() & 0x0F) << 4
        res[i] |= (input[i * 4 + 1].cast[.uint32]() & 0xF0) << 16

        res[i] |= (input[i * 4 + 2].cast[.uint32]() & 0x0F) << 8
        res[i] |= (input[i * 4 + 2].cast[.uint32]() & 0xF0) << 20
        res[i] |= (input[i * 4 + 3].cast[.uint32]() & 0x0F) << 12
        res[i] |= (input[i * 4 + 3].cast[.uint32]() & 0xF0) << 24

    return res


@inline(.always)
@doc_hidden
def unpack_4bit_int(val: SIMD[.uint32, _], idx: Int) -> UInt8:
    """Extracts a single 4-bit value from the packed `uint32` lane at the given nibble index.

    Args:
        val: The packed `uint32` SIMD value.
        idx: The nibble index of the 4-bit element to extract.

    Returns:
        The extracted 4-bit value as a `UInt8`.
    """
    var u32_val = rebind[UInt32](val)
    return (u32_val >> UInt32(idx * 4)).cast[.uint8]() & 0x0F


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](128))
@__name(t"repack_Q4_0_for_sm8x_{scales_type}")
@doc_hidden
def repack_Q4_0_for_sm8x[
    q_layout: TensorLayout,
    repack_layout: TensorLayout,
    scales_type: DType,
](
    q_weight: TileTensor[mut=False, .uint8, q_layout, ImmutAnyOrigin],
    q_packed_weight: TileTensor[mut=True, .uint8, repack_layout, MutAnyOrigin],
):
    """Repacks Q4_0 quantized weights into the tiled layout expected by the SM8x quantized GEMM kernel.

    Parameters:
        q_layout: The input Q4_0 weight buffer layout.
        repack_layout: The output repacked weight buffer layout.
        scales_type: The dtype to cast the per-group scales to.

    Args:
        q_weight: The input Q4_0 quantized weight tensor in global memory.
        q_packed_weight: The output repacked weight tensor in global memory.
    """
    comptime group_size = 32
    comptime group_bytes = size_of[DType.float16]() + (group_size // 2)
    comptime pack_factor = 8
    comptime repack_tile = (64, 16)
    comptime WARP_SIZE = 32
    comptime BN = 128
    comptime BK = 1024

    var tid = thread_idx.x
    var warp_id = ufloordiv(tid, WARP_SIZE)
    comptime num_warps_x = BN // repack_tile[0]
    var warp_x = umod(warp_id, num_warps_x)
    var warp_y = ufloordiv(warp_id, num_warps_x)
    var lane_id: Int = umod(tid, WARP_SIZE)
    var block_idx = Index(block_idx.x, block_idx.y)

    comptime N = q_weight.static_shape[0]
    comptime K = q_weight.static_shape[1] // group_bytes * group_size

    comptime K_groups = K // group_size
    comptime BK_groups = BK // group_size

    @inline(.always)
    def convert_bytes_to_bf16[
        scales_type: DType
    ](input_bytes: SIMD[.uint8, _]) -> Scalar[scales_type]:
        var f32_values = bitcast[.float16, 1](input_bytes).cast[.float32]()
        return bitcast[scales_type, 2](f32_values)[1]

    # Each 64x16 weight tile occupies 128 contiguous uint32 values.
    var repack_weights = q_packed_weight.bitcast[.uint32]().reshape(
        row_major[N // 64, K * 64 // pack_factor]()
    )

    comptime packed_bytes = repack_layout.static_product
    var repack_scales = (
        q_packed_weight.reshape(row_major[packed_bytes]())
        .slice[N * K // 2 : packed_bytes]()
        .bitcast[scales_type]()
        .reshape(row_major[K_groups, N]())
    )

    # We keep 128x2 Q4_0 GGUF blocks in smem
    var smem = external_memory[
        UInt8,
        address_space=.SHARED,
        alignment=align_of[UInt8](),
    ]()
    var qb_smem = TileTensor(smem, row_major[BN, 2 * group_bytes]())

    # We load 128x2 Q4_0 GGUF blocks to smem.
    # Each warp repacks 64x1 Q4_0 GGUF blocks, which are
    # 64x32 4-bit weights. We repack weights into 64x16
    # tiles for our quantized matmul kernel, so there are
    # two tiles for each warp.
    # frag_0 stores frags of the first 64x16 tile,
    # frag_1 stores frags of the second,
    for i in range(ceildiv(BK_groups, 2)):
        barrier()
        var q_gmem_tile = q_weight.tile[BN, 2 * group_bytes](
            block_idx[0], block_idx[1] * (BK_groups // 2) + i
        )
        copy_dram_to_sram[thread_layout=row_major[128, 1]()](
            qb_smem.vectorize[1, 4](), q_gmem_tile.vectorize[1, 4]()
        )
        barrier()
        var q_warp_tile = qb_smem.tile[repack_tile[0], group_bytes](
            warp_x, warp_y
        )

        if (BK_groups * block_idx[1] + i * 2 + warp_y) < K_groups:
            var frag_0: SIMD[.uint8, 16] = 0
            var frag_1: SIMD[.uint8, 16] = 0
            comptime thd_layout = row_major[8, 4]()
            # The first 2 Bytes is the scale for this Q4_0 block
            # GGUF pack elements 0-15 in the lower 4-bit of the 16 Bytes,
            # and elements 16-31 in the higher 4-bit of the 16 Bytes.
            #
            # This gets elements 0, 1, 8, 9, 16, 17, 24, 25 for
            # thread 0.
            var thread_tile = (
                q_warp_tile.slice[:, 2:]()
                .vectorize[1, 2]()
                .distribute[thd_layout](lane_id)
            )

            comptime for i_e in range(16):
                var val = thread_tile.load[2]((i_e // 2, i_e % 2))
                frag_0[i_e] = (val[0] & 0x0F) | ((val[1] & 0x0F) << 4)
                frag_1[i_e] = ((val[0] & 0xF0) >> 4) | (val[1] & 0xF0)

            var repack_warp_tile = repack_weights.tile[1, 256](
                block_idx[0] * (BN // 64) + warp_x,
                block_idx[1] * BK_groups + 2 * i + warp_y,
            )
            repack_warp_tile.store((0, 4 * lane_id), pack_Q_tile(frag_0))
            repack_warp_tile.store((0, 128 + 4 * lane_id), pack_Q_tile(frag_1))

            # Reorder scales to match the lanes consuming packed weights.
            var lane_group, lane_offset = divmod(lane_id, 4)
            var scale_row = lane_offset * 16 + lane_group
            var scales_warp_tile = repack_scales.tile[1, 64](
                block_idx[1] * BK_groups + 2 * i + warp_y,
                block_idx[0] * (BN // 64) + warp_x,
            )

            scales_warp_tile[0, 2 * lane_id] = convert_bytes_to_bf16[
                scales_type
            ](q_warp_tile.vectorize[1, 2]()[scale_row, 0])

            scales_warp_tile[0, 2 * lane_id + 1] = convert_bytes_to_bf16[
                scales_type
            ](q_warp_tile.vectorize[1, 2]()[scale_row + 8, 0])


# Tensors of GPTQ format are stored in a non-transposed way.
# Assume the original transposed matrix is of shape [N, K], the qweight shard
# will be a uint32 matrix of shape [K // 8, N], and scales will be of shape
# [K_groups, N]. The input is a uint8 tensor of shape
# [K_groups * group_bytes, N].
comptime _PermutationTensor = TileTensor[
    .int32, type_of(row_major(Int(0))), ImmutAnyOrigin
]


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](128))
@__name(t"repack_GPTQ_for_sm8x_{scales_type}_g{group_size}_{has_perm}")
@doc_hidden
def repack_GPTQ_for_sm8x[
    in_layout: TensorLayout,
    out_layout: TensorLayout,
    scales_type: DType,
    group_size: Int,
    has_perm: Bool,
](
    in_tensor: TileTensor[mut=False, .uint8, in_layout, ImmutAnyOrigin],
    out_tensor: TileTensor[mut=True, .uint8, out_layout, MutAnyOrigin],
    perm_idx: _PermutationTensor,
):
    """Repacks GPTQ quantized weights into the tiled layout expected by the SM8x quantized GEMM kernel.

    Parameters:
        in_layout: The layout of the input GPTQ weight buffer.
        out_layout: The layout of the output repacked weight buffer.
        scales_type: The dtype to cast the per-group scales to.
        group_size: The number of K elements sharing a single scale.
        has_perm: Whether a permutation index is applied to the K dimension.

    Args:
        in_tensor: The input GPTQ quantized weight tensor in global memory.
        out_tensor: The output repacked weight tensor in global memory.
        perm_idx: The permutation index tensor, or an empty view when `has_perm` is False.
    """
    comptime raw_scales_type = DType.float16
    comptime weights_bytes_per_group = group_size // 2
    comptime group_bytes = size_of[DType.float16]() + weights_bytes_per_group
    comptime pack_factor = 8
    comptime repack_tile = (64, 16)
    comptime BN = 128
    comptime BK = 1024

    var tid = thread_idx.x
    var warp_id = ufloordiv(tid, WARP_SIZE)
    comptime num_warps_x = BN // repack_tile[0]
    var warp_x = umod(warp_id, num_warps_x)
    var warp_y = ufloordiv(warp_id, num_warps_x)
    var lane_id: Int = umod(tid, WARP_SIZE)
    var block_idx = Index(block_idx.x, block_idx.y)

    comptime N = in_tensor.static_shape[1]
    comptime K = in_tensor.static_shape[0] // group_bytes * group_size

    comptime K_groups = K // group_size
    comptime BK_groups = BK // group_size

    comptime uint_K = K // pack_factor

    @inline(.always)
    def convert_bytes_to_bf16[
        scales_type: DType
    ](input_bytes: SIMD[raw_scales_type, _]) -> Scalar[scales_type]:
        var f32_values = bitcast[.float16, 1](input_bytes).cast[.float32]()
        return bitcast[scales_type, 2](f32_values)[1]

    var raw_weights = (
        in_tensor.bitcast[.uint32]().reshape(row_major[uint_K, N]()).transpose()
    )
    comptime in_bytes = in_layout.static_product
    var raw_scales = (
        in_tensor.reshape(row_major[in_bytes]())
        .slice[N * K // 2 : in_bytes]()
        .bitcast[raw_scales_type]()
        .reshape(row_major[K_groups, N]())
        .transpose()
    )

    var repack_weights = out_tensor.bitcast[.uint32]().reshape(
        row_major[N // 64, K * 64 // pack_factor]()
    )
    comptime out_bytes = out_layout.static_product
    var repack_scales = (
        out_tensor.reshape(row_major[out_bytes]())
        .slice[N * K // 2 : out_bytes]()
        .bitcast[scales_type]()
        .reshape(row_major[K_groups, N]())
    )

    # We keep 128x2 GPTQ blocks in smem
    var smem = external_memory[
        UInt8,
        address_space=.SHARED,
        alignment=align_of[UInt8](),
    ]()
    var weights_smem = TileTensor(
        smem, row_major[BN, 2 * weights_bytes_per_group]()
    )
    var weights_smem_uint4 = TileTensor(
        smem.unsafe_bitcast[UInt32](),
        row_major[BN, 2 * group_size // pack_factor](),
    )

    # We load 128x2 GPTQ blocks to smem.
    # Each warp repacks 64x1 GPTQ blocks, which are
    # 64xgroup_size 4-bit weights. We repack weights into 64x16
    # tiles for our quantized matmul kernel, so there are
    # (group_size // 16) tiles for each warp.
    for i in range(ceildiv(BK_groups, 2)):
        comptime if not has_perm:
            barrier()
            var raw_weights_gmem_tile = raw_weights.tile[
                BN, 2 * group_size // pack_factor
            ](block_idx[0], block_idx[1] * (BK_groups // 2) + i)
            copy_dram_to_sram[thread_layout=row_major[128, 1]()](
                weights_smem_uint4, raw_weights_gmem_tile
            )
            barrier()

        if (BK_groups * block_idx[1] + i * 2 + warp_y) < K_groups:
            var repacked_warp_tile = repack_weights.tile[
                1, group_size * 64 // pack_factor
            ](
                block_idx[0] * (BN // 64) + warp_x,
                block_idx[1] * BK_groups + i * 2 + warp_y,
            )

            comptime for i_Q_tile in range(group_size // repack_tile[1]):
                var tmp: SIMD[.uint8, 16] = 0
                comptime thd_layout = row_major[8, 4]()

                comptime if has_perm:
                    var p_block_idx = perm_idx.tile[BK](block_idx[1])
                    var p_group_idx = p_block_idx.tile[group_size](
                        2 * i + warp_y
                    )
                    var p_Qtile_idx = p_group_idx.tile[repack_tile[1]](i_Q_tile)
                    var thd_idx = p_Qtile_idx.vectorize[2]().distribute[
                        row_major[4]()
                    ](lane_id % 4)
                    var n_idx = lane_id // 4

                    var weights_K = raw_weights.tile[BN, uint_K](
                        block_idx[0], 0
                    )
                    var weights_K_wrap = weights_K.tile[repack_tile[0], uint_K](
                        warp_x, 0
                    )

                    comptime for i_e in range(16):
                        if i_e % 2 == 0 and i_e > 0:
                            n_idx += 8
                        var k_idx: Int = Int(thd_idx[i_e % 2][0])
                        var packed_int = weights_K_wrap[
                            n_idx, k_idx // pack_factor
                        ]
                        tmp[i_e] |= unpack_4bit_int(packed_int, k_idx % 8)

                        k_idx = Int(thd_idx[i_e % 2][1])
                        packed_int = weights_K_wrap[n_idx, k_idx // pack_factor]
                        tmp[i_e] |= unpack_4bit_int(packed_int, k_idx % 8) << 4

                else:
                    var raw_weights_warp_tile = weights_smem.tile[
                        repack_tile[0], weights_bytes_per_group
                    ](warp_x, warp_y)
                    var raw_Q_tile = raw_weights_warp_tile.tile[
                        repack_tile[0], repack_tile[1] // 2
                    ](0, i_Q_tile)
                    # This gets elements 0, 1, 8, 9 in each mma_tile for
                    # thread 0.
                    var thread_tile = raw_Q_tile.distribute[thd_layout](lane_id)

                    comptime for i_e in range(16):
                        tmp[i_e] = thread_tile[i_e // 2, i_e % 2]

                var repacked_Q_tile = repacked_warp_tile.tile[1, 128](
                    0, i_Q_tile
                )
                repacked_Q_tile.store((0, 4 * lane_id), pack_Q_tile(tmp))

            var group = block_idx[1] * BK_groups + i * 2 + warp_y
            var row_tile = block_idx[0] * (BN // 64) + warp_x
            var scales_warp_tile = repack_scales.tile[1, 64](group, row_tile)
            var raw_scales_warp_tile = raw_scales.tile[64, 1](row_tile, group)
            var lane_group, lane_offset = divmod(lane_id, 4)
            var scale_row = lane_offset * 16 + lane_group

            scales_warp_tile[0, 2 * lane_id] = convert_bytes_to_bf16[
                scales_type
            ](raw_scales_warp_tile[scale_row, 0])

            scales_warp_tile[0, 2 * lane_id + 1] = convert_bytes_to_bf16[
                scales_type
            ](raw_scales_warp_tile[scale_row + 8, 0])


@inline(.always)
@doc_hidden
def q_smem_usage[config: MatmulConfig, group_size: Int]() -> Int:
    """Computes the shared memory footprint in bytes for the quantized GEMM kernel under the given configuration.

    Parameters:
        config: The matmul configuration describing tile and warp shapes.
        group_size: The number of K elements sharing a single scale.

    Returns:
        The maximum shared memory in bytes needed across A, B, scales, accumulator, and split-K reduction buffers.
    """
    comptime num_warp_k_partitions = config.num_warp_k_partitions
    comptime block_mnk = config.block_tile_shape
    comptime num_pipeline_stages = config.num_pipeline_stages
    comptime pack_factor = 8

    # fmt: off
    var a_usage = block_mnk[0] * block_mnk[2] * num_pipeline_stages * size_of[config.a_type]()
    var b_usage = block_mnk[1] * block_mnk[2] * num_pipeline_stages * size_of[DType.uint32]() // pack_factor
    var c_usage = block_mnk[0] * block_mnk[1] * size_of[DType.float32]()
    var num_scales_stages = uceildiv((num_pipeline_stages - 1) * block_mnk[2], group_size) + 1
    var scales_usage = block_mnk[1] * ceildiv(
        block_mnk[2], group_size
    ) * num_scales_stages * size_of[config.a_type]()
    var slice_k_reduction = Int(block_mnk[0] * block_mnk[1] * ufloordiv(num_warp_k_partitions, 2) * size_of[DType.float32]())
    # fmt: on

    var smem_usage = num_warp_k_partitions * (a_usage + b_usage + scales_usage)
    return max(c_usage, smem_usage, slice_k_reduction)


@doc_hidden
@inline(.always)
def multistage_gemm_q[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    EpilogueFnType: ElementwiseEpilogueFn = type_of(no_epilogue_fn),
    //,
    *,
    group_size: Int,
    pack_factor: Int,
    config: MatmulConfig[a_type, b_type, c_type, True],
    has_epilogue_fn: Bool = False,
](
    c: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, b_type, address_space=.GENERIC, ...],
    runtime_config: MatmulConfig[a_type, b_type, c_type, True],
    ctx: DeviceContext,
    epilogue_fn: EpilogueFnType = no_epilogue_fn,
) raises:
    """Enqueues the multi-stage quantized GEMM kernel, reducing pipeline stages or warp partitions when the shared memory budget is exceeded.

    Parameters:
        c_type: The dtype of the output matrix.
        a_type: The dtype of the A matrix elements.
        b_type: The dtype of the packed B weight buffer.
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        group_size: The number of K elements sharing a single scale.
        pack_factor: The number of 4-bit elements packed into one `uint32`.
        config: The matmul configuration describing tile and warp shapes.
        has_epilogue_fn: Whether `epilogue_fn` stores the output instead of
            the kernel.

    Args:
        c: The output matrix in global memory.
        a: The left-hand (activation) matrix in global memory.
        b: The packed quantized weight buffer in global memory.
        runtime_config: The runtime matmul configuration used for grid and block dimensions.
        ctx: The device context used to enqueue the kernel.
        epilogue_fn: Epilogue that stores each output element when
            `has_epilogue_fn` is set.

    Raises:
        An error if the input tensors are not rank-2.
    """
    comptime assert c.rank == 2
    comptime assert a.rank == 2
    comptime assert b.rank == 2
    var M = Int(c.dim[0]())
    var N = Int(c.dim[1]())

    comptime smem_usage = q_smem_usage[config, group_size]()
    comptime max_smem = ctx.default_device_info.shared_memory_per_multiprocessor

    comptime if smem_usage > max_smem:
        # Strategy:
        # 1. First attempt: Reduce pipeline stages until minimum of 3
        # 2. If still insufficient: Halve the number of warp partitions
        # and retry pipeline stages reduction
        comptime for partition_reduction in range(
            log2_floor(config.num_warp_k_partitions) + 1
        ):
            comptime for num_stages in range(config.num_pipeline_stages, 2, -1):
                comptime adjusted_config = MatmulConfig[
                    a_type, b_type, c_type, True
                ](
                    block_tile_shape=config.block_tile_shape,
                    warp_tile_shape=config.warp_tile_shape,
                    num_pipeline_stages=num_stages,
                    num_k_partitions=config.num_k_partitions,
                    num_warp_k_partitions=(
                        config.num_warp_k_partitions
                        // (2**partition_reduction)
                    ),  # Reduce warp partitions by powers of 2
                )

                comptime adjusted_smem = q_smem_usage[
                    adjusted_config, group_size
                ]()

                comptime if adjusted_smem < max_smem:
                    comptime gemm_kernel_type = multistage_qgemm_kernel[
                        c_type,  # c_type
                        c.LayoutType,
                        a_type,  # a_type
                        a.LayoutType,
                        b_type,  # b_type
                        b.LayoutType,
                        group_size,
                        pack_factor,
                        True,
                        adjusted_config,
                        EpilogueFnType,
                        has_epilogue_fn,
                    ]

                    ctx.enqueue_function[gemm_kernel_type](
                        c,
                        a,
                        b,
                        host_args=(epilogue_fn,),
                        grid_dim=adjusted_config.grid_dim(M, N),
                        block_dim=adjusted_config.block_dim(),
                        shared_mem_bytes=adjusted_smem,
                        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                            UInt32(adjusted_smem),
                        ),
                    )

                    return

    comptime gemm_kernel_type = multistage_qgemm_kernel[
        c_type,  # c_type
        c.LayoutType,
        a_type,  # a_type
        a.LayoutType,
        b_type,  # b_type
        b.LayoutType,
        group_size,
        pack_factor,
        True,
        config,
        EpilogueFnType,
        has_epilogue_fn,
    ]

    ctx.enqueue_function[gemm_kernel_type](
        c,
        a,
        b,
        host_args=(epilogue_fn,),
        grid_dim=runtime_config.grid_dim(M, N),
        block_dim=runtime_config.block_dim(),
        shared_mem_bytes=smem_usage,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_usage),
        ),
    )


@inline(.always)
def matmul_gpu_qint4[
    c_type: DType,
    a_type: DType,
    //,
    group_size: Int,
    target: StaticString,
](
    c_tt: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a_tt: TileTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b_tt: TileTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Launches a GPU int4 quantized matrix multiplication for the given tile tensors.

    Parameters:
        c_type: The dtype of the output matrix.
        a_type: The dtype of the A matrix elements.
        group_size: The number of K elements sharing a single scale.
        target: The target platform string, which must identify a GPU.

    Args:
        c_tt: The output tile tensor in global memory.
        a_tt: The left-hand (activation) tile tensor in global memory.
        b_tt: The packed quantized weight tile tensor in global memory.
        ctx: The device context used to enqueue the kernel.

    Raises:
        An error if the input tensors are not rank-2 or the target is not a GPU.

    Constraints:
        Requires an NVIDIA GPU target. `a_type` and `c_type` must both be
        `bfloat16`.
    """
    comptime assert c_tt.rank == 2
    comptime assert a_tt.rank == 2
    comptime assert b_tt.rank == 2
    comptime assert is_gpu[target](), "unsupported target"
    var cuda_ctx = ctx.value()

    matmul_gpu_qint4_impl[group_size, target](c_tt, a_tt, b_tt, cuda_ctx)


@inline(.always)
def matmul_gpu_qint4[
    c_type: DType,
    a_type: DType,
    EpilogueFnType: ElementwiseEpilogueFn,
    //,
    group_size: Int,
    target: StaticString,
](
    c_tt: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a_tt: TileTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b_tt: TileTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    epilogue_fn: EpilogueFnType,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Same as the plain overload, storing the output through `epilogue_fn`.

    Parameters:
        c_type: The dtype of the output matrix.
        a_type: The dtype of the A matrix elements.
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        group_size: The number of K elements sharing a single scale.
        target: The target platform string, which must identify a GPU.

    Args:
        c_tt: The output tile tensor; the kernel does not write it.
        a_tt: The left-hand (activation) tile tensor in global memory.
        b_tt: The packed quantized weight tile tensor in global memory.
        epilogue_fn: Epilogue that stores each output element.
        ctx: The device context used to enqueue the kernel.

    Raises:
        An error if the input tensors are not rank-2 or the target is not a GPU.

    Constraints:
        Requires an NVIDIA GPU target. `a_type` and `c_type` must both be
        `bfloat16`.
    """
    comptime assert c_tt.rank == 2
    comptime assert a_tt.rank == 2
    comptime assert b_tt.rank == 2
    comptime assert is_gpu[target](), "unsupported target"
    var cuda_ctx = ctx.value()

    matmul_gpu_qint4_impl[group_size, target, has_epilogue_fn=True](
        c_tt, a_tt, b_tt, cuda_ctx, epilogue_fn
    )


@inline(.always)
def matmul_gpu_qint4_impl[
    c_type: DType,
    a_type: DType,
    EpilogueFnType: ElementwiseEpilogueFn = type_of(no_epilogue_fn),
    //,
    group_size: Int,
    target: StaticString,
    *,
    has_epilogue_fn: Bool = False,
](
    c: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    ctx: Optional[DeviceContext],
    epilogue_fn: EpilogueFnType = no_epilogue_fn,
) raises:
    """Dispatches a GPU int4 quantized matrix multiplication to the tuned kernel configuration for the runtime M dimension.

    Selects a `MatmulConfig` specialized for the static K and N dimensions and
    the runtime M value, then delegates to `multistage_gemm_q`.

    Parameters:
        c_type: The dtype of the output matrix.
        a_type: The dtype of the A matrix elements.
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        group_size: The number of K elements sharing a single scale.
        target: The target platform string, which must identify a GPU.
        has_epilogue_fn: Whether `epilogue_fn` stores the output instead of
            the kernel.

    Args:
        c: The output matrix in global memory.
        a: The left-hand (activation) matrix in global memory.
        b: The packed quantized weight buffer in global memory.
        ctx: The device context used to enqueue the kernel.
        epilogue_fn: Epilogue that stores each output element when
            `has_epilogue_fn` is set.

    Raises:
        An error if the input tensors are not rank-2.
    """
    comptime assert c.rank == 2
    comptime assert a.rank == 2
    comptime assert b.rank == 2
    # comptime assert is_gpu[target](), "unsupported target"
    var cuda_ctx = ctx.value()

    comptime pack_factor = 8

    var shape = GemmShape.get[transpose_b=True](c, a, b)
    var m = shape.M

    comptime static_K = a.static_shape[1]
    comptime static_N = c.static_shape[1]

    comptime if static_K == 4096 and static_N == 4096:
        if m <= 16:
            comptime M16_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=(16, 64, 128),
                warp_tile_shape=(16, 64, 32),
                num_pipeline_stages=4,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M16_config,
                has_epilogue_fn=has_epilogue_fn,
            ](
                c,
                a,
                b,
                M16_config,
                cuda_ctx,
                epilogue_fn,
            )
            return
        if 16 < m <= 32:
            comptime M32_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=(32, 64, 128),
                warp_tile_shape=(16, 64, 32),
                num_pipeline_stages=3,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M32_config,
                has_epilogue_fn=has_epilogue_fn,
            ](
                c,
                a,
                b,
                M32_config,
                cuda_ctx,
                epilogue_fn,
            )
            return
        if 32 < m <= 64:
            comptime M64_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=(64, 64, 32),
                warp_tile_shape=(64, 64, 32),
                num_pipeline_stages=5,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M64_config,
                has_epilogue_fn=has_epilogue_fn,
            ](
                c,
                a,
                b,
                M64_config,
                cuda_ctx,
                epilogue_fn,
            )
            return
        if 64 < m <= 128:
            comptime M128_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=(64, 128, 32),
                warp_tile_shape=(64, 64, 32),
                num_pipeline_stages=4,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M128_config,
                has_epilogue_fn=has_epilogue_fn,
            ](
                c,
                a,
                b,
                M128_config,
                cuda_ctx,
                epilogue_fn,
            )
            return
        if 128 < m <= 512:
            comptime M512_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=(64, 64, 32),
                warp_tile_shape=(64, 64, 32),
                num_pipeline_stages=3,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M512_config,
                has_epilogue_fn=has_epilogue_fn,
            ](
                c,
                a,
                b,
                M512_config,
                cuda_ctx,
                epilogue_fn,
            )
            return

    comptime if static_K == 4096 and static_N == 6144:
        if m <= 16:
            comptime M16_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=(16, 64, 128),
                warp_tile_shape=(16, 64, 32),
                num_pipeline_stages=4,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M16_config,
                has_epilogue_fn=has_epilogue_fn,
            ](
                c,
                a,
                b,
                M16_config,
                cuda_ctx,
                epilogue_fn,
            )
            return
        if 16 < m <= 32:
            comptime M32_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=(32, 64, 128),
                warp_tile_shape=(16, 64, 32),
                num_pipeline_stages=3,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M32_config,
                has_epilogue_fn=has_epilogue_fn,
            ](
                c,
                a,
                b,
                M32_config,
                cuda_ctx,
                epilogue_fn,
            )
            return
        if 32 < m <= 64:
            comptime M64_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=(64, 64, 32),
                warp_tile_shape=(64, 64, 32),
                num_pipeline_stages=5,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M64_config,
                has_epilogue_fn=has_epilogue_fn,
            ](
                c,
                a,
                b,
                M64_config,
                cuda_ctx,
                epilogue_fn,
            )
            return
        if 64 < m <= 128:
            comptime M128_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=(64, 128, 32),
                warp_tile_shape=(64, 64, 32),
                num_pipeline_stages=4,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M128_config,
                has_epilogue_fn=has_epilogue_fn,
            ](
                c,
                a,
                b,
                M128_config,
                cuda_ctx,
                epilogue_fn,
            )
            return
        if 128 < m <= 256:
            comptime M256_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=(128, 128, 32),
                warp_tile_shape=(64, 64, 32),
                num_pipeline_stages=3,
                num_k_partitions=1,
                num_warp_k_partitions=2,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M256_config,
                has_epilogue_fn=has_epilogue_fn,
            ](
                c,
                a,
                b,
                M256_config,
                cuda_ctx,
                epilogue_fn,
            )
            return

    comptime if static_K == 4096 and static_N == 14336:
        if m <= 16:
            comptime M16_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=(16, 64, 32),
                warp_tile_shape=(16, 64, 32),
                num_pipeline_stages=5,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M16_config,
                has_epilogue_fn=has_epilogue_fn,
            ](
                c,
                a,
                b,
                M16_config,
                cuda_ctx,
                epilogue_fn,
            )
            return
        if 16 < m <= 32:
            comptime M32_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=(16, 64, 32),
                warp_tile_shape=(16, 64, 32),
                num_pipeline_stages=5,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M32_config,
                has_epilogue_fn=has_epilogue_fn,
            ](
                c,
                a,
                b,
                M32_config,
                cuda_ctx,
                epilogue_fn,
            )
            return
        if 32 < m <= 64:
            comptime M64_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=(32, 64, 32),
                warp_tile_shape=(32, 64, 32),
                num_pipeline_stages=3,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M64_config,
                has_epilogue_fn=has_epilogue_fn,
            ](
                c,
                a,
                b,
                M64_config,
                cuda_ctx,
                epilogue_fn,
            )
            return
        if 64 < m <= 256:
            comptime M128_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=(64, 64, 32),
                warp_tile_shape=(64, 64, 32),
                num_pipeline_stages=4,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M128_config,
                has_epilogue_fn=has_epilogue_fn,
            ](
                c,
                a,
                b,
                M128_config,
                cuda_ctx,
                epilogue_fn,
            )
            return

    comptime if static_K == 14336 and static_N == 4096:
        if m <= 16:
            comptime M16_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=(16, 64, 32),
                warp_tile_shape=(16, 64, 32),
                num_pipeline_stages=4,
                num_k_partitions=1,
                num_warp_k_partitions=8,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M16_config,
                has_epilogue_fn=has_epilogue_fn,
            ](
                c,
                a,
                b,
                M16_config,
                cuda_ctx,
                epilogue_fn,
            )
            return
        if 16 < m <= 32:
            comptime M32_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=(16, 64, 32),
                warp_tile_shape=(16, 64, 32),
                num_pipeline_stages=4,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M32_config,
                has_epilogue_fn=has_epilogue_fn,
            ](
                c,
                a,
                b,
                M32_config,
                cuda_ctx,
                epilogue_fn,
            )
            return
        if 32 < m <= 64:
            comptime M64_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=(32, 64, 32),
                warp_tile_shape=(32, 64, 32),
                num_pipeline_stages=4,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M64_config,
                has_epilogue_fn=has_epilogue_fn,
            ](
                c,
                a,
                b,
                M64_config,
                cuda_ctx,
                epilogue_fn,
            )
            return
        if 64 < m <= 128:
            comptime M128_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=(32, 64, 32),
                warp_tile_shape=(32, 64, 32),
                num_pipeline_stages=3,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M128_config,
                has_epilogue_fn=has_epilogue_fn,
            ](
                c,
                a,
                b,
                M128_config,
                cuda_ctx,
                epilogue_fn,
            )
            return
        if 128 < m <= 512:
            comptime M512_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=(64, 64, 32),
                warp_tile_shape=(64, 64, 32),
                num_pipeline_stages=3,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M512_config,
                has_epilogue_fn=has_epilogue_fn,
            ](
                c,
                a,
                b,
                M512_config,
                cuda_ctx,
                epilogue_fn,
            )
            return

    comptime default_config = MatmulConfig[a_type, .uint8, c_type, True](
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=5,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    multistage_gemm_q[
        group_size=group_size,
        pack_factor=pack_factor,
        config=default_config,
        has_epilogue_fn=has_epilogue_fn,
    ](
        c,
        a,
        b,
        default_config,
        cuda_ctx,
        epilogue_fn,
    )


@inline(.always)
def gpu_qint4_repack_Q4_0[
    target: StaticString,
](
    b_tt: TileTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    b_packed_tt: TileTensor[mut=True, .uint8, address_space=.GENERIC, ...],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Launches the GPU kernel that repacks Q4_0 weights into the packed GEMM layout.

    Parameters:
        target: The target platform string, which must identify a GPU.

    Args:
        b_tt: The input Q4_0 quantized weight tile tensor in global memory.
        b_packed_tt: The output repacked weight tile tensor in global memory.
        ctx: The device context used to enqueue the kernel.

    Raises:
        An error if the input tensors are not rank-2 or the target is not a GPU.
    """
    comptime assert b_tt.rank == 2
    comptime assert b_packed_tt.rank == 2
    comptime assert is_gpu[target](), "unsupported target"
    var cuda_ctx = ctx.value()

    comptime pack_factor = 8
    comptime group_size = 32
    comptime group_bytes = 2 + (group_size // 2)
    comptime BN = 128
    comptime BK = 1024

    comptime N = b_tt.static_shape[0]
    comptime K = b_tt.static_shape[1] // group_bytes * group_size

    var smem_usage: Int = BN * 2 * group_bytes

    var b = TileTensor(b_tt.ptr, b_tt.layout)
    var b_packed = TileTensor(b_packed_tt.ptr, b_packed_tt.layout)
    comptime repack = repack_Q4_0_for_sm8x[
        b.LayoutType, b_packed.LayoutType, DType.bfloat16
    ]

    cuda_ctx.enqueue_function[repack](
        b,
        b_packed,
        grid_dim=(ceildiv(N, BN), ceildiv(K, BK), 1),
        block_dim=(128, 1, 1),
        shared_mem_bytes=smem_usage,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_usage)
        ),
    )


@inline(.always)
def gpu_qint4_repack_GPTQ[
    group_size: Int,
    target: StaticString,
](
    b_tt: TileTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    b_packed_tt: TileTensor[mut=True, .uint8, address_space=.GENERIC, ...],
    perm_idx: OptionalReg[_PermutationTensor] = None,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Launches the GPU kernel that repacks GPTQ weights into the packed GEMM layout.

    Parameters:
        group_size: The number of K elements sharing a single scale.
        target: The target platform string, which must identify a GPU.

    Args:
        b_tt: The input GPTQ quantized weight tile tensor in global memory.
        b_packed_tt: The output repacked weight tile tensor in global memory.
        perm_idx: An optional permutation index tensor for the K dimension.
        ctx: The device context used to enqueue the kernel.

    Raises:
        An error if the input tensors are not rank-2, the target is not a GPU, or the input and output dimensions are mismatched.
    """
    var b = TileTensor(b_tt.ptr, b_tt.layout)
    var b_packed = TileTensor(b_packed_tt.ptr, b_packed_tt.layout)
    comptime assert b.rank == 2
    comptime assert b_packed.rank == 2
    comptime assert is_gpu[target](), "unsupported target"
    var cuda_ctx = ctx.value()

    comptime pack_factor = 8
    comptime group_bytes = 2 + (group_size // 2)
    comptime BN = 128
    comptime BK = 1024

    comptime N = b.static_shape[1]
    comptime K = b.static_shape[0] // group_bytes * group_size

    comptime assert (
        N == b_packed.static_shape[0]
    ), "qmatmul: Mismatched input/output dimension."
    comptime assert K == (
        b_packed.static_shape[1] // group_bytes * group_size
    ), "qmatmul: Mismatched input/output dimension."

    var smem_usage: Int = BN * 2 * group_bytes

    if case .Some(perm_idx) = perm_idx:
        comptime repack = repack_GPTQ_for_sm8x[
            b.LayoutType,
            b_packed.LayoutType,
            DType.bfloat16,
            group_size,
            True,
        ]

        cuda_ctx.enqueue_function[repack](
            b,
            b_packed,
            perm_idx,
            grid_dim=(ceildiv(N, BN), ceildiv(K, BK), 1),
            block_dim=(128, 1, 1),
        )

    else:
        comptime repack = repack_GPTQ_for_sm8x[
            b.LayoutType,
            b_packed.LayoutType,
            DType.bfloat16,
            group_size,
            False,
        ]

        var empty_permutation = b.bitcast[.int32]().reshape(row_major(Int(0)))

        cuda_ctx.enqueue_function[repack](
            b,
            b_packed,
            empty_permutation,
            grid_dim=(ceildiv(N, BN), ceildiv(K, BK), 1),
            block_dim=(128, 1, 1),
            shared_mem_bytes=smem_usage,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(smem_usage)
            ),
        )
