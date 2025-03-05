# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import InlineArray, OptionalReg
from gpu import (
    barrier,
    lane_id,
    block_dim,
    block_idx,
    global_idx,
    thread_idx,
    WARP_SIZE,
    grid_dim,
    MAX_THREADS_PER_BLOCK_METADATA,
)
from gpu.sync import (
    schedule_barrier,
    schedule_group_barrier,
    AMDScheduleBarrierMask,
)
import gpu.warp as warp
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, IntTuple
from layout.layout_tensor import (
    copy_dram_to_sram,
    copy_local_to_dram,
    copy_dram_to_local,
    copy_local_to_sram,
    ThreadScope,
)
from layout.runtime_layout import RuntimeLayout
from layout.tensor_builder import LayoutTensorBuild as tb, static
from layout.tensor_core import TensorCore
from linalg.utils import GemmShape
from math import align_down, ceildiv, align_up
from memory import UnsafePointer
from sys import simdwidthof, alignof
from utils import Index, IndexList, StaticTuple
from utils.numerics import get_accum_type
from .utils_gpu import MatmulConfig
from .utils import apply_epilogue, elementwise_epilogue_type
from layout.swizzle import Swizzle


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](config.num_threads())
)
fn gemm_kernel[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout, address_space = AddressSpace.GLOBAL],
    a: LayoutTensor[a_type, a_layout, address_space = AddressSpace.GLOBAL],
    b: LayoutTensor[b_type, b_layout, address_space = AddressSpace.GLOBAL],
):
    constrained[transpose_b, "Transpose b must be true"]()
    constrained[
        a_type == b_type,
        "a and b must have same type",
    ]()

    constrained[b_layout.all_dims_known(), "b_layout must be known"]()

    alias accum_type = get_accum_type[a_type]()

    alias BM = config.block_tile_shape[0]
    alias BN = config.block_tile_shape[1]
    alias BK = config.block_tile_shape[2]
    alias WM = config.warp_tile_shape[0]
    alias WN = config.warp_tile_shape[1]
    alias MMA_M = config.mma_shape[0]
    alias MMA_N = config.mma_shape[1]
    alias MMA_K = config.mma_shape[2]

    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN
    alias num_warps = num_warps_m * num_warps_n

    # TODO: Remove this once KERN-1609 is fixed
    constrained[
        (BM // num_warps) % 16 == 0,
        "BM per warp (" + String(BM // num_warps) + ") must be divisible by 16",
    ]()
    constrained[
        (BN // num_warps) % 16 == 0,
        "BN per warp (" + String(BN // num_warps) + ") must be divisible by 16",
    ]()

    alias simd_width = simdwidthof[a_type]()
    alias k_group_size = 16 // simd_width
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N
    alias num_k_mmas2 = BK // (MMA_K * k_group_size)

    var M = a.dim(0)
    alias N = b.shape[0]() if transpose_b else b.shape[1]()
    alias K = b.shape[1]() if transpose_b else b.shape[0]()

    var c_reg_tile = tb[accum_type]().row_major[
        num_m_mmas * num_n_mmas, 4
    ]().local().alloc().fill(0)

    var flat_thread_idx = thread_idx.x
    var warp_id = flat_thread_idx // 64

    @parameter
    fn get_smem_layout(tile_size: Int, block_size: Int) -> Layout:
        return Layout(
            IntTuple(
                IntTuple(tile_size, block_size // tile_size),
                IntTuple(k_group_size * MMA_K, BK // (k_group_size * MMA_K)),
            ),
            IntTuple(
                IntTuple(k_group_size * MMA_K, BK * tile_size),
                IntTuple(1, k_group_size * MMA_K * tile_size),
            ),
        )

    alias smem_alignment = alignof[SIMD[a_type, simd_width]]()
    alias a_smem_layout = get_smem_layout(MMA_M, BM)
    var a_smem_tensor = LayoutTensor[
        a_type,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=smem_alignment,
    ].stack_allocation()
    var a_smem_warp_tile = a_smem_tensor.tile[BM // num_warps, BK](warp_id, 0)

    alias b_smem_layout = get_smem_layout(MMA_N, BN)

    var b_smem_tensor = LayoutTensor[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=smem_alignment,
    ].stack_allocation()
    var b_smem_warp_tile = b_smem_tensor.tile[BN // num_warps, BK](warp_id, 0)

    var mma_op = TensorCore[
        accum_type,
        a_type,
        config.mma_shape,
        transpose_b=True,
    ]()

    var a_tile = a.tile[BM, K](block_idx.y, 0)
    var b_tile = b.tile[BN, K](block_idx.x, 0)

    var a_gmem_iter = a_tile.tiled_iterator[BM // num_warps, BK, axis=1](
        warp_id, 0
    )
    var b_gmem_iter = b_tile.tiled_iterator[BN // num_warps, BK, axis=1](
        warp_id, 0
    )

    alias swizzle = Swizzle(2, 0, 2)
    alias thread_layout = Layout.row_major(16, 4)

    var a_load_tile = tb[a_type]().row_major[
        (BM // num_warps) // MMA_M * num_k_mmas2, simd_width
    ]().local().alloc()

    var b_load_tile = tb[b_type]().row_major[
        (BN // num_warps) // MMA_N * num_k_mmas2, simd_width
    ]().local().alloc()

    var a_reg_tile = tb[a_type]().row_major[
        num_m_mmas * num_k_mmas2, simd_width
    ]().local().alloc()

    var b_reg_tile = tb[b_type]().row_major[
        num_n_mmas * num_k_mmas2, simd_width
    ]().local().alloc()

    var a_mma_tile = a_smem_tensor.tile[BM // num_warps_m, BK](
        warp_id // num_warps_n, 0
    )
    var b_mma_tile = b_smem_tensor.tile[BN // num_warps_n, BK](
        warp_id % num_warps_n, 0
    )

    @always_inline
    @parameter
    fn _copy_sram_to_local():
        @parameter
        for k_mma in range(num_k_mmas2):
            mma_op.load_a[swizzle=True](
                a_mma_tile,
                a_reg_tile.tile[num_m_mmas, simd_width](k_mma, 0).vectorize[
                    1, simd_width
                ](),
                k_mma,
            )

            mma_op.load_b[swizzle=swizzle](
                b_mma_tile,
                b_reg_tile.tile[num_n_mmas, simd_width](k_mma, 0).vectorize[
                    1, simd_width
                ](),
                k_mma,
            )

    @always_inline
    @parameter
    fn _mma():
        @parameter
        for k_mma in range(num_k_mmas2):

            @parameter
            for k in range(k_group_size):
                alias elements_per_thread = simd_width // k_group_size
                var a_reg_k = a_reg_tile.tile[num_m_mmas, elements_per_thread](
                    k_mma, k
                ).vectorize[1, elements_per_thread]()
                var b_reg_k = b_reg_tile.tile[num_n_mmas, elements_per_thread](
                    k_mma, k
                ).vectorize[1, elements_per_thread]()
                mma_op.mma(a_reg_k, b_reg_k, c_reg_tile.vectorize[1, 4]())

    @always_inline
    @parameter
    fn _copy_dram_to_local():
        @always_inline
        @parameter
        fn _copy_dram_to_local_impl(
            reg_tile: LayoutTensor,
            gmem_warp_tile: LayoutTensor,
            gmem: LayoutTensor,
        ):
            copy_dram_to_local[
                src_thread_layout=thread_layout,
                thread_scope = ThreadScope.WARP,
            ](
                reg_tile.vectorize[1, simd_width](),
                gmem_warp_tile.vectorize[1, simd_width](),
                gmem,
            )

        _copy_dram_to_local_impl(a_load_tile, a_gmem_iter[], a)
        _copy_dram_to_local_impl(b_load_tile, b_gmem_iter[], b)
        a_gmem_iter._incr()
        b_gmem_iter._incr()

    @always_inline
    @parameter
    fn _copy_local_to_sram():
        @always_inline
        @parameter
        fn _copy_local_to_sram_impl(
            smem_warp_tile: LayoutTensor,
            reg_tile: LayoutTensor,
        ):
            copy_local_to_sram[
                thread_layout=thread_layout,
                swizzle=swizzle,
                thread_scope = ThreadScope.WARP,
                row_major=True,
            ](
                smem_warp_tile.vectorize[1, simd_width](),
                reg_tile.vectorize[1, simd_width](),
            )

        _copy_local_to_sram_impl(a_smem_warp_tile, a_load_tile)
        _copy_local_to_sram_impl(b_smem_warp_tile, b_load_tile)

    _copy_dram_to_local()
    _copy_local_to_sram()

    barrier()

    _copy_sram_to_local()
    _copy_dram_to_local()

    schedule_barrier()
    for _ in range(0, K - 2 * BK, BK):
        barrier()

        _copy_local_to_sram()
        _copy_dram_to_local()
        _mma()

        barrier()

        _copy_sram_to_local()

        # these schedules will change soon in the future
        # will add more info once they are stabilized
        @parameter
        for _ in range(
            ((BN // 4) * BK + (BM // 4) * BK) // (WARP_SIZE * simd_width)
        ):
            schedule_group_barrier(AMDScheduleBarrierMask.DS_WRITE, 1, 0)
            schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 1, 0)
            schedule_group_barrier(AMDScheduleBarrierMask.VMEM_READ, 1, 0)
            schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 5, 0)

        @parameter
        for _ in range(num_n_mmas * num_k_mmas2 + num_m_mmas * num_k_mmas2):
            schedule_group_barrier(AMDScheduleBarrierMask.DS_READ, 1, 0)
            schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 1, 0)

    schedule_barrier()
    barrier()

    _copy_local_to_sram()
    _mma()
    barrier()

    _copy_sram_to_local()
    _mma()

    # write to output tensor
    var c_block_tile = c.tile[BM, BN](block_idx.y, block_idx.x)
    var c_warp_tile = c_block_tile.tile[WM, WN](
        warp_id // num_warps_n, warp_id % num_warps_n
    )

    @parameter
    if elementwise_lambda_fn:
        constrained[
            elementwise_lambda_fn is not None,
            "elementwise_lambda_fn is not valid",
        ]()
        alias epilogue = elementwise_lambda_fn.value()
        alias thread_layout = Layout.row_major(4, 16)
        var c_gmem_frag = c_warp_tile.vectorize[4, 1]().distribute[
            thread_layout
        ](lane_id())
        var c_reg_frag = c_reg_tile.vectorize[1, 4]()
        var thread_offset = c_gmem_frag.distance(c.ptr)

        @parameter
        for i in range(__type_of(c_gmem_frag).layout.size()):
            alias src_idx = c_reg_frag.layout(i)
            alias dst_static_idx: UInt = __type_of(c_gmem_frag).layout(i)
            var dst_idx = 0

            @parameter
            if c_gmem_frag.layout.all_dims_known():
                dst_idx = dst_static_idx
            else:
                dst_idx = c_gmem_frag.runtime_layout(i)
            var m = Int((thread_offset + dst_idx) // N)
            var n = Int((thread_offset + dst_idx) % N)
            if m < M and n < N:
                var vec = c_reg_frag.ptr.offset(src_idx).load[
                    width=4,
                    alignment = alignof[SIMD[c_type, 4]](),
                ]()

                @parameter
                for j in range(4):
                    if m + j < M:
                        epilogue[alignment = alignof[SIMD[c_type, 1]]()](
                            (m + j, n), vec[j].cast[c_type]()
                        )
    else:
        copy_local_to_dram[
            Layout.row_major(4, 16), thread_scope = ThreadScope.WARP
        ](c_warp_tile.vectorize[4, 1](), c_reg_tile.vectorize[1, 4](), c)
