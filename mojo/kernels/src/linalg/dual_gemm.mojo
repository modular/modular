# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg
from math import ceildiv, exp
from os import abort
from sys import alignof, is_defined, simdwidthof
from collections.string import StaticString

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
import gpu.warp as warp
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.info import A100, is_gpu
from gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_group,
    external_memory,
)
from layout import Layout
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy,
    copy_sram_to_dram,
)
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.runtime_layout import RuntimeLayout
from layout.runtime_tuple import RuntimeTuple
from layout.swizzle import make_swizzle, Swizzle, make_ldmatrix_swizzle
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.tensor_core import (
    TensorCore,
    get_fragment_size,
    get_mma_shape,
)
from utils.numerics import get_accum_type
from memory import memset_zero, stack_allocation
from memory.pointer import _GPUAddressSpace as AddressSpace
from register import register_internal
from runtime.asyncrt import DeviceContextPtr
from runtime.tracing import Trace, TraceLevel, trace_arg

from utils import StaticTuple
from utils.index import Index, IndexList

from .utils import GemmShape, elementwise_epilogue_type
from .utils_gpu import (
    MatmulConfig,
    MatmulKernels,
    _bk_base,
    block_swizzle,
)


@always_inline
fn multistage_dual_mma[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    a_smem_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    b_smem_layout: Layout, //,
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
    static_num_iters: Dim = Dim(),
    k_group_size: UInt = 1,
](
    c0: LayoutTensor[
        c_type,
        c_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL, **_,
    ],
    c1: LayoutTensor[
        c_type,
        c_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL, **_,
    ],
    a_iter_arg: LayoutTensorIter[_, a_layout, MutableAnyOrigin, **_],
    b0_iter_arg: LayoutTensorIter[b_type, b_layout, MutableAnyOrigin, **_],
    b1_iter_arg: LayoutTensorIter[b_type, b_layout, MutableAnyOrigin, **_],
    a_smem_iter_arg: LayoutTensorIter[
        a_type,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED, **_,
    ],
    mut b0_smem_iter: LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED, **_,
    ],
    mut b1_smem_iter: LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED, **_,
    ],
    num_iters: Int,
    /,
    *,
    num_b_rows: OptionalReg[Int] = None,
):
    constrained[
        b0_iter_arg.address_space == b1_iter_arg.address_space,
        "b0 and b1 should have the same address space",
    ]()
    alias simd_size = simdwidthof[a_type]()

    var tid: UInt32 = thread_idx.x
    var warp_id = warp.broadcast(tid // WARP_SIZE)

    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN
    var warp_x = warp_id % num_warps_n
    var warp_y = warp_id // num_warps_n

    var a_iter = a_iter_arg
    var b0_iter = b0_iter_arg
    var b1_iter = b1_iter_arg
    var a_smem_iter = a_smem_iter_arg

    alias async_copy_a_layout = Layout.row_major(
        num_threads * simd_size // BK, BK // simd_size
    )

    alias async_copy_b_layout = Layout.row_major(
        num_threads * simd_size // b_smem_layout.stride[0].value(),
        b_smem_layout.stride[0].value() // simd_size,
    )
    alias swizzle_b = transpose_b or b_type.is_half_float()

    @always_inline
    @parameter
    fn _mask_tensor_row(
        tensor: LayoutTensor, num_rows: Int, out result: __type_of(tensor)
    ):
        return __type_of(tensor)(
            tensor.ptr,
            RuntimeLayout[
                element_type = tensor.layout_int_type,
                linear_idx_type = tensor.linear_idx_type,
            ](
                RuntimeTuple[
                    tensor.layout.shape, element_type = tensor.layout_int_type
                ](num_rows, tensor.dim(1)),
                tensor.runtime_layout.stride,
            ),
        )

    @always_inline
    @parameter
    fn _copy_single_tensor_to_sram(dst: LayoutTensor, src: LayoutTensor):
        copy_dram_to_sram_async[
            thread_layout=async_copy_a_layout,
            swizzle=swizzle_a,
        ](
            dst.vectorize[1, simd_size](),
            src.vectorize[1, simd_size](),
        )

    @always_inline
    @parameter
    fn _copy_dual_tensor_to_sram(
        b0_dst: LayoutTensor,
        b1_dst: LayoutTensor,
        b0_src: LayoutTensor,
        b1_src: LayoutTensor,
    ):
        copy_dram_to_sram_async[
            thread_layout=async_copy_b_layout,
            swizzle=swizzle_b,
        ](
            b0_dst.vectorize[1, simd_size](),
            b0_src.vectorize[1, simd_size](),
        )
        copy_dram_to_sram_async[
            thread_layout=async_copy_b_layout,
            swizzle=swizzle_b,
        ](
            b1_dst.vectorize[1, simd_size](),
            b1_src.vectorize[1, simd_size](),
        )

    # Prefetch (num_pipeline_stages - 1) stages.

    @parameter
    for stage in range(num_pipeline_stages - 1):
        var a_smem_tile = a_smem_iter.next_unsafe(stage)[]

        _copy_single_tensor_to_sram(a_smem_tile, a_iter[])

        a_iter._incr()

        var b0_smem_tile = b0_smem_iter.next_unsafe(stage)[]
        var b1_smem_tile = b1_smem_iter.next_unsafe(stage)[]
        if num_b_rows:
            var num_rows_bound = num_b_rows.value() if transpose_b else max(
                0, num_b_rows.value() - stage * BK
            )
            _copy_dual_tensor_to_sram(
                b0_smem_tile,
                b1_smem_tile,
                _mask_tensor_row(b0_iter[], num_rows_bound),
                _mask_tensor_row(b1_iter[], num_rows_bound),
            )
        else:
            _copy_dual_tensor_to_sram(
                b0_smem_tile, b1_smem_tile, b0_iter[], b1_iter[]
            )
        b0_iter._incr()
        b1_iter._incr()

        async_copy_commit_group()

    # Guard stage 0.
    async_copy_wait_group(num_pipeline_stages - 2)
    barrier()

    alias accum_type = get_accum_type[a_type, preferred_accum_type=c_type]()
    alias mma_shape = get_mma_shape[a_type, accum_type]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_k_mmas: UInt = BK // MMA_K
    alias num_k_mma_iters: UInt = num_k_mmas // k_group_size
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // (2 * MMA_N)
    constrained[
        num_k_mmas % (2 * k_group_size) == 0,
        "num_k_mmas must be an integer multiple of 2*k_group_size",
    ]()
    constrained[num_n_mmas % 2 == 0]()

    alias frag_size = get_fragment_size[mma_shape]()
    alias a_frag_size = frag_size[0]
    alias b_frag_size = frag_size[1]
    alias c_frag_size = frag_size[2]

    alias num_reg_tiles = 2 * k_group_size
    # Register tiles.
    var a_reg_tiles = tb[a_type]().row_major[
        2 * k_group_size * num_m_mmas, a_frag_size
    ]().local().alloc().split[2 * k_group_size]()

    var b0_reg_tiles = tb[b_type]().row_major[
        2 * k_group_size * num_n_mmas, b_frag_size
    ]().local().alloc().vectorize[1, b_frag_size]().split[2 * k_group_size]()
    var b1_reg_tiles = tb[b_type]().row_major[
        2 * k_group_size * num_n_mmas, b_frag_size
    ]().local().alloc().vectorize[1, b_frag_size]().split[2 * k_group_size]()

    var a_warp_tile = a_smem_iter[].tile[WM, BK](Int(warp_y), 0)

    alias b_wtile_dim0 = WN // 2 if transpose_b else BK
    alias b_wtile_dim1 = BK if transpose_b else WN // 2
    var b_wtile_coord0 = Int(warp_x) if transpose_b else 0
    var b_wtile_coord1 = 0 if transpose_b else Int(warp_x)
    var b0_warp_tile = b0_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
        b_wtile_coord0, b_wtile_coord1
    )
    var b1_warp_tile = b1_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
        b_wtile_coord0, b_wtile_coord1
    )

    var mma_op = TensorCore[accum_type, a_type, mma_shape, transpose_b]()

    alias swizzle_a_pattern = make_ldmatrix_swizzle[
        a_type, a_warp_tile.stride[0]()
    ]() if swizzle_a else OptionalReg[Swizzle](None)

    @parameter
    for i in range(Int(k_group_size)):
        mma_op.load_a[swizzle_a_pattern](
            a_warp_tile, a_reg_tiles[i].vectorize[1, a_frag_size](), i
        )
        mma_op.load_b(b0_warp_tile, b0_reg_tiles[i], i, Int(warp_x))
        mma_op.load_b(b1_warp_tile, b1_reg_tiles[i], i, Int(warp_x))

    for k_tile_id in range(num_iters):
        var a_warp_tile = a_smem_iter[].tile[WM, BK](Int(warp_y), 0)
        var b0_warp_tile = b0_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
            b_wtile_coord0,
            b_wtile_coord1,
        )
        var b1_warp_tile = b1_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
            b_wtile_coord0,
            b_wtile_coord1,
        )

        # Perform prefetch registers and mma until current shared memory tile's
        # data has all been loaded to registers.
        @parameter
        for k_mma0 in range(Int(num_k_mma_iters)):

            @parameter
            for k_mma1 in range(Int(k_group_size)):
                alias k_mma = UInt32(k_mma0 * k_group_size + k_mma1)
                alias current = k_mma % num_reg_tiles
                alias k_mma_next = k_mma + k_group_size
                alias next = Int(k_mma_next % num_reg_tiles)

                @parameter
                if k_mma_next == num_k_mmas:
                    var prefetch_tile_id = k_tile_id + num_pipeline_stages - 1

                    # Prefetch one k tile (if valid) from global memory to current
                    # shared memory buffer.
                    if prefetch_tile_id < num_iters:
                        var a_smem_prefetch_tile = a_smem_iter.next_unsafe(
                            num_pipeline_stages - 1
                        )[]
                        _copy_single_tensor_to_sram(
                            a_smem_prefetch_tile, a_iter[]
                        )
                        a_iter._incr()

                        var b0_smem_prefetch_tile = b0_smem_iter.next_unsafe(
                            num_pipeline_stages - 1
                        )[]
                        var b1_smem_prefetch_tile = b1_smem_iter.next_unsafe(
                            num_pipeline_stages - 1
                        )[]
                        if num_b_rows:
                            var num_rows_bound = num_b_rows.value() if transpose_b else max(
                                0,
                                num_b_rows.value() - prefetch_tile_id * BK,
                            )
                            _copy_dual_tensor_to_sram(
                                b0_smem_prefetch_tile,
                                b1_smem_prefetch_tile,
                                _mask_tensor_row(b0_iter[], num_rows_bound),
                                _mask_tensor_row(b1_iter[], num_rows_bound),
                            )
                        else:
                            _copy_dual_tensor_to_sram(
                                b0_smem_prefetch_tile,
                                b1_smem_prefetch_tile,
                                b0_iter[],
                                b1_iter[],
                            )
                        b0_iter._incr()
                        b1_iter._incr()

                    async_copy_commit_group()

                    # Guard the next k tile's shared memory buffer.
                    async_copy_wait_group(num_pipeline_stages - 2)
                    barrier()

                    a_smem_iter._incr()
                    b0_smem_iter._incr()
                    b1_smem_iter._incr()

                    a_warp_tile = a_smem_iter[].tile[WM, BK](Int(warp_y), 0)
                    b0_warp_tile = b0_smem_iter[].tile[
                        b_wtile_dim0, b_wtile_dim1
                    ](b_wtile_coord0, b_wtile_coord1)
                    b1_warp_tile = b1_smem_iter[].tile[
                        b_wtile_dim0, b_wtile_dim1
                    ](b_wtile_coord0, b_wtile_coord1)

                alias kidx = Int(k_mma_next % num_k_mmas)
                mma_op.load_a[swizzle_a_pattern](
                    a_warp_tile,
                    a_reg_tiles[next].vectorize[1, a_frag_size](),
                    kidx,
                )

                mma_op.load_b(
                    b0_warp_tile,
                    b0_reg_tiles[next],
                    kidx,
                    Int(warp_x),
                )
                mma_op.load_b(
                    b1_warp_tile,
                    b1_reg_tiles[next],
                    kidx,
                    Int(warp_x),
                )

            @parameter
            for k_mma1 in range(Int(k_group_size)):
                alias k_mma = UInt32(k_mma0 * k_group_size + k_mma1)
                alias current = k_mma % num_reg_tiles
                mma_op.mma(
                    a_reg_tiles[Int(current)].vectorize[1, a_frag_size](),
                    b0_reg_tiles[Int(current)],
                    c0.vectorize[1, c_frag_size](),
                )
                mma_op.mma(
                    a_reg_tiles[Int(current)].vectorize[1, a_frag_size](),
                    b1_reg_tiles[Int(current)],
                    c1.vectorize[1, c_frag_size](),
                )


alias binary_fn_type = fn[type: DType, width: Int] (
    SIMD[type, width], SIMD[type, width]
) -> SIMD[type, width]


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](config.num_threads())
)
fn multistage_dual_gemm_kernel[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    binary_lambda_fn: binary_fn_type,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    a: LayoutTensor[a_type, a_layout, MutableAnyOrigin],
    b0: LayoutTensor[b_type, b_layout, MutableAnyOrigin],
    b1: LayoutTensor[b_type, b_layout, MutableAnyOrigin],
):
    # Hold on adding fp16 because it counld have differnet precisions than bf16.
    constrained[
        a_type in (DType.float32, DType.bfloat16, DType.float16)
        and a_type == b_type,
        "Pipeline gemm only supports tf32, BF16 mma, or fp16",
    ]()

    alias simd_size = simdwidthof[c_type]()

    var M: UInt = c.dim(0)
    var N: UInt = b0.dim(0) if transpose_b else b0.dim(1)
    var K: UInt = b0.dim(1) if transpose_b else b0.dim(0)
    # we require b0 and b1 to be of the same size

    alias BM = config.block_tile_shape[0]
    alias BN = config.block_tile_shape[1]
    alias BK = config.block_tile_shape[2]
    alias WM = config.warp_tile_shape[0]
    alias WN = config.warp_tile_shape[1]
    alias num_pipeline_stages = config.num_pipeline_stages

    alias num_warps_m = config.num_warps_m()
    alias num_warps_n = config.num_warps_n()
    alias num_threads = config.num_threads()

    var tid = thread_idx.x
    var ln_id = lane_id()
    var warp_id = warp.broadcast(tid // WARP_SIZE)

    # Only apply block swizzling for half precision types.
    alias swizzle_block = a_type.is_half_float() and b_type.is_half_float()

    # NOTE: the condition ( not (N // BN & 1)) is for a temporary solution
    # for solving mismatches in some shapes
    var block_idx = block_swizzle(
        Index[type = DType.uint32](block_idx.x, block_idx.y),
        Index[type = DType.uint32](grid_dim.x, grid_dim.y),
    ) if swizzle_block else Index[type = DType.uint32](block_idx.x, block_idx.y)

    # Coordinates of the current warp.
    warp_y, warp_x = divmod(warp_id, num_warps_n)

    # Prepare circular shared memory buffer for A and B.
    # Each pipeline stage has its own buffer.
    var a_smem = external_memory[
        Scalar[a_type],
        address_space = AddressSpace.SHARED,
        alignment = alignof[SIMD[a_type, simd_size]](),
    ]()
    alias a_smem_size = num_pipeline_stages * BM * BK
    var a_smem_iter = LayoutTensorIter[
        a_type,
        Layout.row_major(BM, BK),
        address_space = a_smem.address_space,
        alignment = a_smem.alignment,
        circular=True,
    ](
        rebind[
            __type_of(
                LayoutTensorIter[
                    a_type,
                    Layout.row_major(BM, BK),
                    MutableAnyOrigin,
                    address_space = a_smem.address_space,
                    alignment = a_smem.alignment,
                    circular=True,
                ]().ptr
            )
        ](a_smem),
        a_smem_size,
    )

    # There is one pre-allocated shared buffer. Explicitly offset B after at A's end.
    var b_smem = (a_smem + a_smem_size).bitcast[Scalar[b_type]]()
    alias b_smem_size = num_pipeline_stages * BK * BN // 2
    alias BD_0 = BN // 2 if transpose_b else BK
    alias BD_1 = BK if transpose_b else BN // 2
    alias b_smem_layout = Layout.row_major(BD_0, BD_1)
    var b0_smem_iter = LayoutTensorIter[
        b_type,
        b_smem_layout,
        address_space = AddressSpace.SHARED,
        circular=True,
    ](b_smem, b_smem_size)
    var b1_smem_iter = LayoutTensorIter[
        b_type,
        b_smem_layout,
        address_space = AddressSpace.SHARED,
        circular=True,
    ](b_smem + b_smem_size, b_smem_size)

    # create input layout tensors A and Bv
    # global memory iterator
    var a_gmem_iter = a.tiled_iterator[BM, BK, axis=1](block_idx[1], 0)
    var b_tile_coords = (block_idx[0], 0) if transpose_b else (0, block_idx[0])
    alias b_tile_axis = 1 if transpose_b else 0
    var b0_gmem_iter = b0.tiled_iterator[BD_0, BD_1, axis=b_tile_axis](
        b_tile_coords[0], b_tile_coords[1]
    )
    var b1_gmem_iter = rebind[__type_of(b0_gmem_iter)](
        b1.tiled_iterator[BD_0, BD_1, axis=b_tile_axis](
            b_tile_coords[0], b_tile_coords[1]
        )
    )

    # Compute MMA config
    alias accum_type = get_accum_type[a_type, preferred_accum_type=c_type]()
    alias mma_shape = get_mma_shape[a_type, accum_type]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_k_mmas = BK // MMA_K
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // (2 * MMA_N)

    alias frag_size = get_fragment_size[mma_shape]()
    alias c_frag_size = frag_size[2]
    var c0_reg_tile = tb[accum_type]().row_major[
        num_m_mmas * num_n_mmas, c_frag_size
    ]().local().alloc().fill(0)
    var c1_reg_tile = tb[accum_type]().row_major[
        num_m_mmas * num_n_mmas, c_frag_size
    ]().local().alloc().fill(0)

    multistage_dual_mma[
        BM,
        BN,
        BK,
        WM,
        WN,
        num_threads,
        num_pipeline_stages,
        transpose_b,
        k_group_size = config.k_group_size,
    ](
        c0_reg_tile,
        c1_reg_tile,
        a_gmem_iter,
        b0_gmem_iter,
        b1_gmem_iter,
        a_smem_iter,
        b0_smem_iter,
        b1_smem_iter,
        ceildiv(K, BK),
    )

    alias HWN = WN // 2
    # c_reg_tile is indeced in `mma` as follows:
    # c_frag[n_mma * num_m_mmas + m_mma, 0]
    # so, the first dim can be viewed as `row_major(num_n_mmas, num_m_mmas)`
    # we combine the second half to the first half
    var c0_vec = c0_reg_tile.vectorize[1, c_frag_size]()
    var c1_vec = c1_reg_tile.vectorize[1, c_frag_size]()

    @parameter
    for n_mma in range(c0_vec.layout.size()):
        c0_vec[n_mma, 0] = binary_lambda_fn(c0_vec[n_mma, 0], c1_vec[n_mma, 0])

    # Map global memory tile down to thread.
    var c_gmem_tile = c.tile[BM, BN // 2](block_idx[1], block_idx[0])
    var c_gmem_warp_tile = c_gmem_tile.tile[WM, HWN](Int(warp_y), Int(warp_x))

    # Store FP32 mma results to half precision buffer in global memory.
    # Each thread's fragment has 2x2 fp32 values. Casting to half float and
    # directly storing to global memory results in 2 4B writes. Following cutlass,
    # we stage the fragments in shared memory so that each thread can store 16B.
    @parameter
    if c_type.is_half_float():
        alias swizzle = make_swizzle[
            num_rows = MMA_M // 2, row_size=HWN, access_size=MMA_N
        ]()

        var accum_smem_warp_tile = tb[c_type]().row_major[
            WM, HWN
        ]().shared().view(a_smem.bitcast[Scalar[c_type]]() + warp_id * WM * HWN)

        copy[thread_layout = Layout.row_major(8, 4), swizzle=swizzle,](
            accum_smem_warp_tile.vectorize[1, 2](),
            c0_reg_tile.vectorize[1, 2]().transpose(),
        )

        # Guard writing to shared memory.
        barrier()

        # Vectorized copy from shared to global memory, during which every 2 FP32
        # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
        # vector and stored using 16B store instruction.
        @parameter
        if elementwise_lambda_fn:
            alias epilogue = elementwise_lambda_fn.value()
            alias warp_layout = Layout.row_major(
                WARP_SIZE * simd_size // HWN, HWN // simd_size
            )
            var c_gmem_frag = c_gmem_warp_tile.vectorize[
                1, simd_size
            ]().distribute[warp_layout](thread_idx.x)
            var c_smem_frag = accum_smem_warp_tile.vectorize[
                1, simd_size
            ]().distribute[warp_layout](thread_idx.x)
            var thread_offset = c_gmem_frag.distance(c.ptr)
            alias num_stores_per_thread = __type_of(c_gmem_frag).layout.size()

            var c_smem_frag_offset = c_smem_frag.distance(
                accum_smem_warp_tile.ptr
            )

            @parameter
            for i in range(num_stores_per_thread):
                alias src_idx = __type_of(c_smem_frag).layout(i)
                alias src_idx_base = src_idx % swizzle.size()
                alias src_idx_diff = src_idx - src_idx_base
                var swizzled_idx = swizzle(
                    c_smem_frag_offset + src_idx_base
                ) + src_idx_diff

                alias dst_static_idx = __type_of(c_gmem_frag).layout(i)
                var dst_idx = 0

                @parameter
                if c_layout.all_dims_known():
                    dst_idx = dst_static_idx
                else:
                    dst_idx = Int(c_gmem_frag.runtime_layout(i))

                var m = (Int(thread_offset) + dst_idx) // N
                var n = (Int(thread_offset) + dst_idx) % N
                alias alignment = alignof[SIMD[c_type, simd_size]]()
                if m < M and n < N:
                    epilogue[alignment=alignment](
                        (m, n),
                        accum_smem_warp_tile.ptr.load[
                            width=simd_size, alignment=alignment
                        ](swizzled_idx).cast[c_type](),
                    )
        else:
            copy_sram_to_dram[
                thread_layout = Layout.row_major(
                    WARP_SIZE * simd_size // HWN, HWN // simd_size
                ),
                swizzle=swizzle,
            ](
                c_gmem_warp_tile.vectorize[1, simd_size](),
                accum_smem_warp_tile.vectorize[1, simd_size](),
            )

    # Store FP32 results to FP32 buffer in global memory.
    else:

        @parameter
        if elementwise_lambda_fn:
            alias epilogue = elementwise_lambda_fn.value()
            var c_gmem_frag = c_gmem_warp_tile.vectorize[1, 2]().distribute[
                Layout.row_major(8, 4)
            ](ln_id)
            var c_reg_frag = c0_reg_tile.vectorize[1, 2]().transpose()
            var thread_offset = c_gmem_frag.distance(c.ptr)

            @parameter
            for i in range(__type_of(c_gmem_frag).layout.size()):
                alias src_idx = c_reg_frag.layout(i)
                alias dst_static_idx: UInt = __type_of(c_gmem_frag).layout(i)
                var dst_idx = 0

                @parameter
                if c_layout.all_dims_known():
                    dst_idx = dst_static_idx
                else:
                    dst_idx = Int(c_gmem_frag.runtime_layout(i))

                alias alignment = alignof[SIMD[c_type, 2]]()
                var m = (Int(thread_offset) + dst_idx) // N
                var n = (Int(thread_offset) + dst_idx) % N
                if m < M and n < N:
                    var vec = c_reg_frag.ptr.offset(src_idx).load[
                        width=2, alignment = alignof[SIMD[c_type, 2]]()
                    ]()
                    epilogue[alignment=alignment]((m, n), vec)

        else:
            copy_local_to_dram[dst_thread_layout = Layout.row_major(8, 4)](
                c_gmem_warp_tile.vectorize[1, 2](),
                c0_reg_tile.vectorize[1, 2]().transpose(),
            )


fn swilu[
    type: DType, width: Int
](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
    return (x * y) / (1 + exp(-x))


@always_inline
fn multistage_dual_gemm[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout, //,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    binary_lambda_fn: binary_fn_type = swilu,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout],
    a: LayoutTensor[a_type, a_layout],
    b0: LayoutTensor[b_type, b_layout],
    b1: LayoutTensor[b_type, b_layout],
    ctx: DeviceContext,
) raises:
    var M = c.dim(0)
    var N = c.dim(1)

    alias smem_usage = config.shared_mem_usage()
    constrained[
        smem_usage <= ctx.device_info.shared_memory_per_multiprocessor,
        "using too much shared memory",
    ]()
    alias gemm_kernel_type = multistage_dual_gemm_kernel[
        c_type,
        c_layout,
        a_type,
        a_layout,
        b_type,
        b_layout,
        transpose_b,
        config,
        binary_lambda_fn=binary_lambda_fn,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]

    ctx.enqueue_function[gemm_kernel_type](
        c,
        a,
        b0,
        b1,
        grid_dim=config.grid_dim(M, 2 * N),
        block_dim=config.block_dim(),
        shared_mem_bytes=smem_usage,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_usage),
    )
    ctx.synchronize()


fn multistage_dual_gemm[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList, //,
    *,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    binary_lambda_fn: binary_fn_type = swilu,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    num_k_partitions: Int = 1,
](
    c: NDBuffer[c_type, 2, _, c_shape],
    a: NDBuffer[a_type, 2, _, a_shape],
    b0: NDBuffer[b_type, 2, _, b_shape],
    b1: NDBuffer[b_type, 2, _, b_shape],
    ctx: DeviceContext,
) raises:
    var tensor_c = from_ndbuffer_row_major(c)
    var tensor_a = from_ndbuffer_row_major(a)
    var tensor_b0 = from_ndbuffer_row_major(b0)
    var tensor_b1 = from_ndbuffer_row_major(b1)
    multistage_dual_gemm[
        transpose_b=transpose_b,
        config=config,
        binary_lambda_fn=binary_lambda_fn,
    ](tensor_c, tensor_a, tensor_b0, tensor_b1, ctx)


fn config_in_smem[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool, //,
    max_smem: Int,
](
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    out res: MatmulConfig[a_type, b_type, c_type, transpose_b],
):
    var c: __type_of(res) = config
    var i = 0
    while c.shared_mem_usage() > max_smem:
        if c.block_tile_shape[1] >= 256:
            c = __type_of(res)(
                block_tile_shape=Index(
                    c.block_tile_shape[0],
                    c.block_tile_shape[1] // 2,
                    c.block_tile_shape[2],
                ),
                warp_tile_shape=Index(
                    c.warp_tile_shape[0],
                    c.warp_tile_shape[1] if c.warp_tile_shape[1]
                    >= c.block_tile_shape[1]
                    // 2 else c.block_tile_shape[1]
                    // 2,
                    c.warp_tile_shape[2],
                ),
                num_pipeline_stages=c.num_pipeline_stages,
                num_k_partitions=c.num_k_partitions,
            )
        else:
            c = __type_of(res)(
                block_tile_shape=Index(
                    c.block_tile_shape[0] // 2,
                    c.block_tile_shape[1],
                    c.block_tile_shape[2],
                ),
                warp_tile_shape=Index(
                    c.warp_tile_shape[0] if c.warp_tile_shape[0]
                    >= c.block_tile_shape[0]
                    // 2 else c.block_tile_shape[0]
                    // 2,
                    c.warp_tile_shape[1],
                    c.warp_tile_shape[2],
                ),
                num_pipeline_stages=c.num_pipeline_stages,
                num_k_partitions=c.num_k_partitions,
            )
        i += 1
        if i > 8:
            abort("too many iterations")
    return c


fn dual_gemm[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList, //,
    *,
    transpose_b: Bool,
    binary_lambda_fn: binary_fn_type = swilu,
    config: OptionalReg[
        MatmulConfig[a_type, b_type, c_type, transpose_b]
    ] = None,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: NDBuffer[c_type, 2, MutableAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    b0: NDBuffer[b_type, 2, MutableAnyOrigin, b_shape],
    b1: NDBuffer[b_type, 2, MutableAnyOrigin, b_shape],
    ctx: DeviceContext,
) raises:
    # TODO: Autotune. Currently, these are a copy+paste of `_matmul_gpu`.
    #       These should be roughly optimal, as the dual gemm is sort of
    #       like one gemm where `B` has twice as many columns, but it
    #       isn't precisely that. Thus, autotuning is reasonable.
    #       In particular, we will still want to coalesce loads from
    #       `B0` and `B1`, which means we want
    #       BN*sizeof[b_type]() >= 128
    #       although all existing configs from `_matmul_gpu` currently
    #       satisfy that.
    var shape = GemmShape.get[transpose_b=transpose_b](c, a, b0)
    var M = shape.M
    var N = 2 * shape.N
    var K = shape.K
    var multi_gemm_cond = (M > 1 and N % 128 == 0 and K % 32 == 0 and K >= 128)
    alias multistage_gemm_supported_shape = b_shape.all_known[
        2
    ]() and a_shape.has_value[1]() and c_shape.has_value[1]()
    alias matmul_supported_format = (
        a_type in (DType.float32, DType.bfloat16, DType.float16)
        and b_type in (DType.float32, DType.bfloat16, DType.float16)
        and c_type in (DType.float32, DType.bfloat16, DType.float16)
    )
    alias max_smem = ctx.device_info.shared_memory_per_multiprocessor

    constrained[
        matmul_supported_format,
        String("unsupported dual_gemm dtypes", a_type, b_type, c_type),
    ]()
    constrained[
        multistage_gemm_supported_shape,
        String("unsupported dual_gemm shapes", a_shape, b_shape, c_shape),
    ]()
    if multi_gemm_cond:
        alias kernels = MatmulKernels[a_type, b_type, c_type, transpose_b]()

        @parameter
        if is_defined["AUTOTUNING_MODE"]():
            multistage_dual_gemm[
                transpose_b=transpose_b,
                config = kernels.tuning_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                rebind[NDBuffer[b_type, 2, b0.origin, b_shape]](b0),
                rebind[NDBuffer[b_type, 2, b1.origin, b_shape]](b1),
                ctx,
            )
            return

        # Allow caller to overwrite dispatch heuristic with their own config.
        @parameter
        if config:
            multistage_dual_gemm[
                transpose_b=transpose_b,
                config = config.value(),
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                rebind[NDBuffer[b_type, 2, b0.origin, b_shape]](b0),
                rebind[NDBuffer[b_type, 2, b1.origin, b_shape]](b1),
                ctx,
            )
            return

        @parameter
        if (
            a_type == b_type
            and a_type.is_half_float()
            and ctx.device_info is A100
            and transpose_b
        ):
            alias static_K = a_shape.get[1]()
            alias static_N = c_shape.get[1]()
            alias warp_shape = Index(64, 64, _bk_base[a_type]())

            @parameter
            if static_N == 28672 and static_K == 4096:
                if M <= 128:
                    alias M128_N28672_K4096_config = config_in_smem[max_smem](
                        MatmulConfig[a_type, b_type, c_type, transpose_b](
                            block_tile_shape=Index(128, 128, 32),
                            warp_tile_shape=warp_shape,
                            num_pipeline_stages=4,
                            num_k_partitions=1,
                        )
                    )
                    multistage_dual_gemm[
                        transpose_b=transpose_b,
                        config=M128_N28672_K4096_config,
                        elementwise_lambda_fn=elementwise_lambda_fn,
                    ](
                        rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                        rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                        rebind[NDBuffer[b_type, 2, b0.origin, b_shape]](b0),
                        rebind[NDBuffer[b_type, 2, b1.origin, b_shape]](b1),
                        ctx,
                    )
                    return
                if M <= 256:
                    alias M256_N28672_K4096_config = config_in_smem[max_smem](
                        MatmulConfig[a_type, b_type, c_type, transpose_b](
                            block_tile_shape=Index(64, 256, 64),
                            warp_tile_shape=warp_shape,
                            num_pipeline_stages=4,
                            num_k_partitions=1,
                        )
                    )
                    multistage_dual_gemm[
                        transpose_b=transpose_b,
                        config=M256_N28672_K4096_config,
                        elementwise_lambda_fn=elementwise_lambda_fn,
                    ](
                        rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                        rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                        rebind[NDBuffer[b_type, 2, b0.origin, b_shape]](b0),
                        rebind[NDBuffer[b_type, 2, b1.origin, b_shape]](b1),
                        ctx,
                    )
                    return
                if M <= 512:
                    alias M512_N28672_K4096_config = config_in_smem[max_smem](
                        MatmulConfig[a_type, b_type, c_type, transpose_b](
                            block_tile_shape=Index(128, 128, 32),
                            warp_tile_shape=warp_shape,
                            num_pipeline_stages=4,
                            num_k_partitions=1,
                        )
                    )
                    multistage_dual_gemm[
                        transpose_b=transpose_b,
                        config=M512_N28672_K4096_config,
                        elementwise_lambda_fn=elementwise_lambda_fn,
                    ](
                        rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                        rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                        rebind[NDBuffer[b_type, 2, b0.origin, b_shape]](b0),
                        rebind[NDBuffer[b_type, 2, b1.origin, b_shape]](b1),
                        ctx,
                    )
                    return
                if M <= 1606:
                    alias M1606_N28672_K4096_config = config_in_smem[max_smem](
                        MatmulConfig[a_type, b_type, c_type, transpose_b](
                            block_tile_shape=Index(128, 256, 64),
                            warp_tile_shape=warp_shape,
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                    )
                    multistage_dual_gemm[
                        transpose_b=transpose_b,
                        config=M1606_N28672_K4096_config,
                        elementwise_lambda_fn=elementwise_lambda_fn,
                    ](
                        rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                        rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                        rebind[NDBuffer[b_type, 2, b0.origin, b_shape]](b0),
                        rebind[NDBuffer[b_type, 2, b1.origin, b_shape]](b1),
                        ctx,
                    )
                    return
                if M <= 2048:
                    alias M2048_N28672_K4096_config = config_in_smem[max_smem](
                        MatmulConfig[a_type, b_type, c_type, transpose_b](
                            block_tile_shape=Index(256, 128, 64),
                            warp_tile_shape=warp_shape,
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                    )
                    multistage_dual_gemm[
                        transpose_b=transpose_b,
                        config=M2048_N28672_K4096_config,
                        elementwise_lambda_fn=elementwise_lambda_fn,
                    ](
                        rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                        rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                        rebind[NDBuffer[b_type, 2, b0.origin, b_shape]](b0),
                        rebind[NDBuffer[b_type, 2, b1.origin, b_shape]](b1),
                        ctx,
                    )
                    return

        multistage_dual_gemm[
            transpose_b=transpose_b,
            config = config_in_smem[max_smem](
                MatmulConfig[a_type, b_type, c_type, transpose_b]()
            ),
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](
            rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
            rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
            rebind[NDBuffer[b_type, 2, b0.origin, b_shape]](b0),
            rebind[NDBuffer[b_type, 2, b1.origin, b_shape]](b1),
            ctx,
        )
        return

    # Gemv
    elif M == 1:
        dual_gemv[elementwise_lambda_fn=elementwise_lambda_fn](
            c, a, b0, b1, ctx
        )
        return

    raise "dual gemm size unsupported."


# ---------------------------------------------------------------------------- #
# Dual-Gemv
# ---------------------------------------------------------------------------- #


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads)
)
fn dual_gemv_kernel[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    simd_width: UInt,
    tile_m: UInt,
    tile_n: UInt,
    num_threads: UInt,
    binary_lambda_fn: binary_fn_type,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
](
    c: NDBuffer[c_type, 2, MutableAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    b0: NDBuffer[b_type, 2, MutableAnyOrigin, b_shape],
    b1: NDBuffer[b_type, 2, MutableAnyOrigin, b_shape],
):
    var m: UInt = c.dim(0)
    var n: UInt = b0.dim(0)
    var k: UInt = b0.dim(1)

    var tid = thread_idx.x

    # tile_m is number of rows in A, defaults to 1.
    # tile_n is number of rows in B0 and B1, each gets tile_n // 2.
    alias tile_n_per_B = tile_n // 2
    var tile_id_m = block_idx.x * tile_m
    var tile_id_n = block_idx.y * tile_n_per_B

    alias tile_k = simd_width * num_threads
    var tile_a = stack_allocation[
        simd_width, a_type, address_space = AddressSpace.LOCAL
    ]()
    var tile_w = stack_allocation[
        tile_n * simd_width, b_type, address_space = AddressSpace.LOCAL
    ]()
    var acc = stack_allocation[
        tile_m * tile_n, s_type, address_space = AddressSpace.LOCAL
    ]()

    var tile_b0 = tile_w
    var tile_b1 = tile_w.offset(tile_n_per_B * simd_width)

    alias align_act = alignof[SIMD[a_type, simd_width]]()
    alias align_weight = alignof[SIMD[b_type, simd_width]]()

    memset_zero[count = tile_m * tile_n](acc)

    var act_idx = tile_id_m * k
    var weight_idx = tile_id_n * k
    var output_idx = tile_id_m * n + tile_id_n

    # Each thread sums local data in K.
    for idxK in range(tid * simd_width, k, tile_k):

        @parameter
        for i in range(Int(tile_n_per_B)):
            var b0_vec = b0.data.load[width=simd_width, alignment=align_weight](
                weight_idx + i * k + idxK
            )

            tile_b0.store[alignment=align_weight](i * simd_width, b0_vec)

        @parameter
        for i in range(Int(tile_n_per_B)):
            var b1_vec = b1.data.load[width=simd_width, alignment=align_weight](
                weight_idx + i * k + idxK
            )

            tile_b1.store[alignment=align_weight](i * simd_width, b1_vec)

        @parameter
        for i in range(Int(tile_m)):
            var a_vec = a.data.load[width=simd_width, alignment=align_act](
                act_idx + i * k + idxK
            )

            tile_a.store[alignment=align_act](i * simd_width, a_vec)

            @parameter
            for j in range(Int(tile_n)):

                @parameter
                for l in range(Int(simd_width)):
                    acc[i * tile_n + j] += (
                        tile_a[l].cast[s_type]()
                        * tile_w[j * simd_width + l].cast[s_type]()
                    )

    # Warps are arranged along K.
    alias k_warp_num = num_threads // WARP_SIZE
    var warp_id = warp.broadcast(tid // WARP_SIZE)
    var lane_id = tid % WARP_SIZE
    var shmem = stack_allocation[
        k_warp_num * tile_m * tile_n,
        s_type,
        address_space = AddressSpace.SHARED,
    ]()

    # Each warp sums across its threads and stages results in shared memory.
    # Shared memory data is row mojor (num_warps, tile_m, tile_n) stored in 1D.
    @parameter
    for mi in range(Int(tile_m)):

        @parameter
        for ni in range(Int(tile_n)):
            var val = warp.sum(acc[mi * tile_n + ni])
            if lane_id == 0:
                shmem[mi * tile_n + ni + warp_id * tile_m * tile_n] = val

    barrier()

    # Sum across warps' results in shared memory and apply the binary op.
    for ii in range(tid, tile_m * tile_n_per_B, num_threads):
        var mid = ii // tile_n_per_B
        var nid = ii % tile_n_per_B
        var val0 = Scalar[s_type]()
        var val1 = Scalar[s_type]()

        @parameter
        for jj in range(Int(k_warp_num)):
            val0 += shmem[jj * tile_m * tile_n + mid * tile_n + nid]
            val1 += shmem[
                jj * tile_m * tile_n + mid * tile_n + nid + tile_n_per_B
            ]

        val0 = binary_lambda_fn(val0, val1)

        @parameter
        if elementwise_lambda_fn:
            alias elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](
                Index(0, output_idx + mid * n + nid), val0.cast[c_type]()
            )
        else:
            c.data.store(output_idx + mid * n + nid, val0.cast[c_type]())


fn dual_gemv[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList, //,
    *,
    binary_lambda_fn: binary_fn_type = swilu,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: NDBuffer[c_type, 2, MutableAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    b0: NDBuffer[b_type, 2, MutableAnyOrigin, b_shape],
    b1: NDBuffer[b_type, 2, MutableAnyOrigin, b_shape],
    ctx: DeviceContext,
) raises:
    var M = c.dim(0)
    var N = c.dim(1)

    alias simd_width = simdwidthof[a_type]()
    alias tile_m = 1
    alias tile_n = 2
    alias num_threads = 128

    alias kernel_type = dual_gemv_kernel[
        c_type,
        c_shape,
        a_type,
        a_shape,
        b_type,
        b_shape,
        simd_width,
        tile_m,
        tile_n,
        num_threads,
        binary_lambda_fn,
        elementwise_lambda_fn,
    ]

    ctx.enqueue_function[kernel_type](
        c,
        a,
        b0,
        b1,
        grid_dim=(ceildiv(M, tile_m), ceildiv(N, tile_n // 2)),
        block_dim=num_threads,
    )


# ---------------------------------------------------------------------------- #
# SwiGLU Layer
# ---------------------------------------------------------------------------- #


@register_internal("swishGLU")
@always_inline
fn swishGLU[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList, //,
    target: StaticString = "cpu",
](
    a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    b0: NDBuffer[b_type, 2, MutableAnyOrigin, b_shape],
    b1: NDBuffer[b_type, 2, MutableAnyOrigin, b_shape],
    c: NDBuffer[c_type, 2, MutableAnyOrigin, c_shape],
    ctx: DeviceContextPtr,
) raises:
    """
    Reference:
        GLU Variants Improve Transformer
        by Noam Shazeer
        https://arxiv.org/pdf/2002.05202v1
    The implementation follows cutlass, using one kernel invocation
    and writing to the destination once.
    """

    constrained[is_gpu[target](), "only valid on GPUs"]()

    @always_inline
    @parameter
    fn description_fn() -> String:
        var shape = GemmShape.get[True](c, a, b0)
        return String(";").join(
            String(target),
            trace_arg("A", IndexList[2](shape.M, shape.K), a.type),
            trace_arg("B0", IndexList[2](shape.K, shape.N), b0.type),
            trace_arg("B1", IndexList[2](shape.K, shape.N), b1.type),
            trace_arg("C", IndexList[2](shape.M, shape.N), c.type),
        )

    with Trace[TraceLevel.OP, target=target](
        "swish_glu",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ):
        dual_gemm[transpose_b=True](c, a, b0, b1, ctx=ctx.get_device_context())
