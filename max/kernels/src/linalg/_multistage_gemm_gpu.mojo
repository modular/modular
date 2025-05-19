# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections import OptionalReg
from math import ceildiv
from sys import (
    alignof,
    has_amd_gpu_accelerator,
    is_nvidia_gpu,
    simdwidthof,
    sizeof,
)

import gpu.warp as warp
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from gpu.host import FuncAttribute
from gpu.memory import (
    CacheEviction,
    Fill,
    async_copy_commit_group,
    async_copy_wait_group,
    external_memory,
)
from gpu.mma import ld_matrix, mma
from gpu.semaphore import Semaphore
from layout.int_tuple import UNKNOWN_VALUE, IntTuple
from layout.layout import *
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    _swizzle_signature,
    copy,
    copy_dram_to_sram,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_local_to_local,
    copy_sram_to_dram,
)
from layout.runtime_layout import RuntimeLayout
from layout.runtime_tuple import RuntimeTuple
from layout.swizzle import Swizzle, make_ldmatrix_swizzle, make_swizzle
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.tensor_core import TensorCore, get_fragment_size, get_mma_shape
from memory import UnsafePointer
from memory.pointer import _GPUAddressSpace as AddressSpace

from utils import StaticTuple
from utils.index import Index, IndexList
from utils.numerics import get_accum_type

from ._amd_gemm_gpu import gemm_kernel as amd_gemm_kernel
from .matmul_gpu import matmul_kernel_naive
from .utils import apply_epilogue, elementwise_epilogue_type
from .utils_gpu import MatmulConfig, MatmulKernels, block_swizzle


@always_inline
fn distance[
    type: DType, //
](arg0: UnsafePointer[Scalar[type]], arg1: UnsafePointer[Scalar[type]]) -> Int:
    return (Int(arg0) - Int(arg1)) // sizeof[arg1.type]()


@always_inline
fn warp_split_k_reduction[
    c_type: DType,
    c_layout: Layout, //,
    BM: Int,
    BN: Int,
    num_threads_per_warp_k_part: Int,
    num_warp_k_partitions: Int,
](
    warp_k_part_id: Int,
    c_reg_tile: LayoutTensor[
        c_type, c_layout, address_space = AddressSpace.LOCAL, **_
    ],
):
    alias red_layout = Layout.row_major(1, num_threads_per_warp_k_part)

    alias num_mmas = c_layout.shape[0].value()
    alias c_frag_size = c_layout.shape[1].value()

    var i_red = num_warp_k_partitions // 2
    var tid = thread_idx.x

    var smem = external_memory[
        Scalar[c_type],
        address_space = AddressSpace.SHARED,
        alignment = alignof[SIMD[c_type, c_frag_size]](),
    ]()

    while i_red > 0:
        barrier()
        var red_tb_smem = tb[c_type]().row_major[1, BM * BN]().shared().view(
            smem.bitcast[Scalar[c_type]]()
            + ((warp_k_part_id % i_red) * BM * BN)
        ).vectorize[1, c_frag_size]()
        if i_red <= warp_k_part_id < 2 * i_red:
            copy[thread_layout=red_layout](
                red_tb_smem,
                c_reg_tile.vectorize[1, c_frag_size](),
            )
        barrier()
        if warp_k_part_id < i_red:
            var red_tb_thread_tile = red_tb_smem.distribute[red_layout](tid)
            var c_reg_tile_vectorized = c_reg_tile.vectorize[
                1, c_frag_size
            ]().transpose()

            @parameter
            for i in range(num_mmas):
                c_reg_tile_vectorized[0, i] += rebind[
                    __type_of(c_reg_tile_vectorized[0, i])
                ](red_tb_thread_tile[0, i])
        i_red //= 2


@always_inline
fn multistage_mma[
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
    prefetch_init: Bool = True,
    continue_prefetch_b: Bool = False,
    transpose_b_next: Bool = False,
    b_next_gmem_layout: Layout = Layout(),
    b_next_smem_layout: Layout = Layout(),
    next_op_b_iter_masked: Bool = False,
    next_op_b_iter_alignment: Int = alignof[b_type](),
    next_op_b_layout_int_type: DType = DType.int64,
    next_op_b_linear_idx_type: DType = DType.int64,
    k_group_size: UInt = 1,
](
    c: LayoutTensor[c_type, c_layout, address_space = AddressSpace.LOCAL, **_],
    a_iter_arg: LayoutTensorIter[_, a_layout, **_],
    b_iter_arg: LayoutTensorIter[b_type, b_layout, **_],
    a_smem_iter_arg: LayoutTensorIter[
        a_type, a_smem_layout, address_space = AddressSpace.SHARED, **_
    ],
    mut b_smem_iter: LayoutTensorIter[
        b_type, b_smem_layout, address_space = AddressSpace.SHARED, **_
    ],
    num_iters: Int,
    /,
    *,
    num_b_rows: OptionalReg[Int] = None,
    next_op_b_iter: LayoutTensorIter[
        b_type,
        b_next_gmem_layout,
        MutableAnyOrigin,
        alignment=next_op_b_iter_alignment,
        layout_int_type=next_op_b_layout_int_type,
        linear_idx_type=next_op_b_linear_idx_type,
        masked=next_op_b_iter_masked,
    ] = LayoutTensorIter[
        b_type,
        b_next_gmem_layout,
        MutableAnyOrigin,
        alignment=next_op_b_iter_alignment,
        layout_int_type=next_op_b_layout_int_type,
        linear_idx_type=next_op_b_linear_idx_type,
        masked=next_op_b_iter_masked,
    ](),
):
    alias simd_size = simdwidthof[a_type]()

    # In the slice-K method, we pass `num_threads_per_warp_k_part` as `num_threads`
    # in the parameters. This ensures that `tid` represents the relative thread position
    # within each warp_k_part_id groups.
    var tid: UInt32 = thread_idx.x % num_threads
    var warp_id = warp.broadcast(tid // WARP_SIZE)

    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN
    var warp_x = warp_id % num_warps_n
    var warp_y = warp_id // num_warps_n

    var a_iter = a_iter_arg
    var b_iter = b_iter_arg
    var a_smem_iter = a_smem_iter_arg
    # work around mut argument can't have default value.
    var next_b_iter = next_op_b_iter

    # If there are more threads than vectors, thread layout should be based on
    # the latter so that a vector is only mapped to one thread.
    alias a_num_vecs = BM * BK // simd_size
    alias async_copy_a_layout = Layout.row_major(
        min(num_threads, a_num_vecs) * simd_size // BK, BK // simd_size
    )

    alias b_num_ves = BN * BK // simd_size
    alias async_copy_b_layout = Layout.row_major(
        min(num_threads, b_num_ves)
        * simd_size
        // b_smem_layout.shape[1].value(),
        b_smem_layout.shape[1].value() // simd_size,
    )

    # TODO (KERN-1337): Enable swizzle for matrix B for FP8 data type and tranpose_b==False
    alias swizzle_b = (
        transpose_b or b_type.is_half_float()
    ) and is_nvidia_gpu()

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
                ](num_rows, tensor.dim[1]()),
                tensor.runtime_layout.stride,
            ),
        )

    @always_inline
    @parameter
    fn _copy_tensor_to_sram[
        thread_layout: Layout, swizzle: Bool
    ](dst: LayoutTensor, src: LayoutTensor):
        @parameter
        if is_nvidia_gpu():
            copy_dram_to_sram_async[
                thread_layout=thread_layout,
                swizzle=swizzle,
                num_threads=num_threads,
            ](
                dst.vectorize[1, simd_size](),
                src.vectorize[1, simd_size](),
            )
        else:
            copy_dram_to_sram[thread_layout=thread_layout](
                dst.vectorize[1, simd_size](),
                src.vectorize[1, simd_size](),
            )

    # Prefetch (num_pipeline_stages - 1) stages.
    @parameter
    if prefetch_init:

        @parameter
        for stage in range(num_pipeline_stages - 1):

            @parameter
            if a_iter.address_space == AddressSpace.GENERIC:
                var a_smem_tile = a_smem_iter.next_unsafe(stage)[]
                _copy_tensor_to_sram[async_copy_a_layout, swizzle_a](
                    a_smem_tile, a_iter[]
                )

                a_iter._incr()

            @parameter
            if b_iter.address_space == AddressSpace.GENERIC:
                var b_smem_tile = b_smem_iter.next_unsafe(stage)[]

                if num_b_rows:
                    var num_rows_bound = num_b_rows.value() if transpose_b else max(
                        0, num_b_rows.value() - stage * BK
                    )
                    var b_tensor = _mask_tensor_row(b_iter[], num_rows_bound)
                    _copy_tensor_to_sram[async_copy_b_layout, swizzle_b](
                        b_smem_tile, b_tensor
                    )
                else:
                    _copy_tensor_to_sram[async_copy_b_layout, swizzle_b](
                        b_smem_tile, b_iter[]
                    )

                b_iter._incr()

            async_copy_commit_group()

        # Guard stage 0.
        async_copy_wait_group(num_pipeline_stages - 2)
        barrier()

    alias mma_shape = get_mma_shape[a_type, get_accum_type[a_type]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_k_mmas: UInt = BK // MMA_K
    alias num_k_mma_iters: UInt = num_k_mmas // k_group_size
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N
    constrained[
        num_k_mmas % (2 * k_group_size) == 0,
        "num_k_mmas must be an integer multiple of 2*k_group_size",
    ]()

    alias accum_type = get_accum_type[a_type]()
    alias frag_size = get_fragment_size[mma_shape]()
    alias a_frag_size = frag_size[0]
    alias b_frag_size = frag_size[1]
    alias c_frag_size = frag_size[2]

    alias num_reg_tiles = 2 * k_group_size
    # Register tiles.
    var a_reg_tiles = tb[a_type]().row_major[
        2 * k_group_size * num_m_mmas, a_frag_size
    ]().local().alloc().split[2 * k_group_size]()

    var b_reg_tiles = tb[b_type]().row_major[
        2 * k_group_size * num_n_mmas, b_frag_size
    ]().local().alloc().vectorize[1, b_frag_size]().split[2 * k_group_size]()

    var a_warp_tile = a_smem_iter[].tile[WM, BK](Int(warp_y), 0)

    alias b_wtile_dim0 = WN if transpose_b else BK
    alias b_wtile_dim1 = BK if transpose_b else WN
    var b_wtile_coord0 = Int(warp_x) if transpose_b else 0
    var b_wtile_coord1 = 0 if transpose_b else Int(warp_x)
    var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
        b_wtile_coord0, b_wtile_coord1
    )

    var mma_op = TensorCore[accum_type, a_type, mma_shape, transpose_b]()

    alias swizzle_a_pattern = make_ldmatrix_swizzle[
        a_type, a_warp_tile.stride[0]()
    ]() if swizzle_a else OptionalReg[Swizzle](None)

    @parameter
    for i in range(k_group_size):

        @parameter
        if a_iter.address_space == AddressSpace.LOCAL:
            # Assume input is the 16x8 output of 16x8x16 or 16x8x8 mma.
            # Need to cast address space because it's not known at parse time to be LOCAL.
            copy_local_to_local(a_reg_tiles[i], a_iter[])
            a_iter._incr()
        else:
            mma_op.load_a[swizzle_a_pattern](
                a_warp_tile, a_reg_tiles[i].vectorize[1, a_frag_size](), i
            )

        mma_op.load_b(b_warp_tile, b_reg_tiles[i], i, Int(warp_x))

    @parameter
    if static_num_iters.has_value():
        constrained[
            a_iter.address_space == AddressSpace.SHARED
            or a_iter.address_space == AddressSpace.LOCAL,
            (
                "Using input in registers or shared memory requires static"
                " iteration bound.\n"
            ),
        ]()

        @parameter
        for k_tile_id in range(static_num_iters.get()):
            var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
                b_wtile_coord0,
                b_wtile_coord1,
            )

            # Perform prefetch registers and mma until current shared memory tile's
            # data has all been loaded to registers.
            @parameter
            for k_mma0 in range(num_k_mma_iters):

                @parameter
                for k_mma1 in range(k_group_size):
                    alias k_mma = UInt32(k_mma0 * k_group_size + k_mma1)
                    alias current = k_mma % num_reg_tiles
                    alias k_mma_next = k_mma + k_group_size
                    alias next = Int(k_mma_next % num_reg_tiles)

                    @parameter
                    if k_mma_next == num_k_mmas:
                        alias prefetch_tile_id = k_tile_id + num_pipeline_stages - 1

                        # Prefetch one k tile (if valid) from global memory to current
                        # shared memory buffer.
                        @parameter
                        if b_iter.address_space == AddressSpace.GENERIC:

                            @parameter
                            if prefetch_tile_id < static_num_iters.get():
                                var b_smem_prefetch_tile = b_smem_iter.next_unsafe(
                                    num_pipeline_stages - 1
                                )[]

                                if num_b_rows:
                                    var num_rows_bound = num_b_rows.value() if transpose_b else max(
                                        0,
                                        num_b_rows.value()
                                        - prefetch_tile_id * BK,
                                    )
                                    var b_tensor = _mask_tensor_row(
                                        b_iter[], num_rows_bound
                                    )
                                    _copy_tensor_to_sram[
                                        async_copy_b_layout, swizzle_b
                                    ](b_smem_prefetch_tile, b_tensor)
                                else:
                                    _copy_tensor_to_sram[
                                        async_copy_b_layout, swizzle_b
                                    ](b_smem_prefetch_tile, b_iter[])

                                b_iter._incr()

                            async_copy_commit_group()

                            # Guard the next k tile's shared memory buffer.
                            async_copy_wait_group(num_pipeline_stages - 2)
                            barrier()

                        @parameter
                        if a_iter.address_space == AddressSpace.SHARED:
                            a_smem_iter._incr()
                        b_smem_iter._incr()

                        a_warp_tile = a_smem_iter[].tile[WM, BK](Int(warp_y), 0)
                        b_warp_tile = b_smem_iter[].tile[
                            b_wtile_dim0, b_wtile_dim1
                        ](b_wtile_coord0, b_wtile_coord1)

                    alias kidx = k_mma_next % num_k_mmas

                    @parameter
                    if a_iter.address_space == AddressSpace.SHARED:
                        mma_op.load_a[swizzle_a_pattern](
                            a_warp_tile,
                            a_reg_tiles[next].vectorize[1, a_frag_size](),
                            Int(kidx),
                        )
                    else:
                        # Assume input is the 16x8 output of 16x8x16 or 16x8x8 mma.
                        copy_local_to_local(a_reg_tiles[Int(next)], a_iter[])
                        a_iter._incr()

                    mma_op.load_b(
                        b_warp_tile,
                        b_reg_tiles[Int(next)],
                        Int(kidx),
                        Int(warp_x),
                    )

                @parameter
                for k_mma1 in range(k_group_size):
                    alias k_mma = UInt32(k_mma0 * k_group_size + k_mma1)
                    alias current = k_mma % num_reg_tiles
                    mma_op.mma(
                        a_reg_tiles[Int(current)].vectorize[1, a_frag_size](),
                        b_reg_tiles[Int(current)],
                        c.vectorize[1, c_frag_size](),
                    )

        return

    for k_tile_id in range(num_iters):
        var a_warp_tile = a_smem_iter[].tile[WM, BK](Int(warp_y), 0)
        var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
            b_wtile_coord0,
            b_wtile_coord1,
        )

        # Perform prefetch registers and mma until current shared memory tile's
        # data has all been loaded to registers.
        @parameter
        for k_mma0 in range(num_k_mma_iters):

            @parameter
            for k_mma1 in range(k_group_size):
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

                        @parameter
                        if a_iter.address_space == AddressSpace.GENERIC:
                            var a_smem_prefetch_tile = a_smem_iter.next_unsafe(
                                num_pipeline_stages - 1
                            )[]
                            _copy_tensor_to_sram[
                                async_copy_a_layout, swizzle_a
                            ](a_smem_prefetch_tile, a_iter[])

                            a_iter._incr()

                        @parameter
                        if b_iter.address_space == AddressSpace.GENERIC:
                            var b_smem_prefetch_tile = b_smem_iter.next_unsafe(
                                num_pipeline_stages - 1
                            )[]

                            if num_b_rows:
                                var num_rows_bound = num_b_rows.value() if transpose_b else max(
                                    0,
                                    num_b_rows.value() - prefetch_tile_id * BK,
                                )
                                var b_tensor = _mask_tensor_row(
                                    b_iter[], num_rows_bound
                                )
                                _copy_tensor_to_sram[
                                    async_copy_b_layout, swizzle_b
                                ](b_smem_prefetch_tile, b_tensor)
                            else:
                                _copy_tensor_to_sram[
                                    async_copy_b_layout, swizzle_b
                                ](b_smem_prefetch_tile, b_iter[])

                            b_iter._incr()
                    else:

                        @parameter
                        if continue_prefetch_b:
                            var b_smem_prefetch_tile = b_smem_iter.next_unsafe(
                                num_pipeline_stages - 1
                            )[].reshape[b_next_smem_layout]()

                            alias row_size = b_next_smem_layout.stride[
                                0
                            ].value()

                            alias b_prefetch_thread_layout = Layout.row_major(
                                num_threads * simd_size // row_size,
                                row_size // simd_size,
                            )
                            alias swizzle_prefetch_b = (
                                transpose_b_next or b_type.is_half_float()
                            ) and is_nvidia_gpu()

                            if num_b_rows:
                                # TODO: can we guard at compile time num_b_rows is set here?
                                var num_rows_bound = num_b_rows.value() if transpose_b_next else max(
                                    0,
                                    num_b_rows.value()
                                    - (prefetch_tile_id - num_iters) * BK,
                                )

                                var b_tensor = _mask_tensor_row(
                                    next_b_iter[], num_rows_bound
                                )
                                _copy_tensor_to_sram[
                                    b_prefetch_thread_layout, swizzle_prefetch_b
                                ](b_smem_prefetch_tile, b_tensor)

                            else:
                                _copy_tensor_to_sram[
                                    b_prefetch_thread_layout, swizzle_prefetch_b
                                ](b_smem_prefetch_tile, next_b_iter[])

                            next_b_iter._incr()

                    async_copy_commit_group()

                    # Guard the next k tile's shared memory buffer.
                    async_copy_wait_group(num_pipeline_stages - 2)
                    barrier()

                    a_smem_iter._incr()
                    b_smem_iter._incr()

                    a_warp_tile = a_smem_iter[].tile[WM, BK](Int(warp_y), 0)
                    b_warp_tile = b_smem_iter[].tile[
                        b_wtile_dim0, b_wtile_dim1
                    ](b_wtile_coord0, b_wtile_coord1)

                alias kidx = Int(k_mma_next % num_k_mmas)
                mma_op.load_a[swizzle_a_pattern](
                    a_warp_tile,
                    a_reg_tiles[next].vectorize[1, a_frag_size](),
                    kidx,
                )
                mma_op.load_b(
                    b_warp_tile,
                    b_reg_tiles[next],
                    kidx,
                    Int(warp_x),
                )

            @parameter
            for k_mma1 in range(k_group_size):
                alias k_mma = UInt32(k_mma0 * k_group_size + k_mma1)
                alias current = k_mma % num_reg_tiles
                mma_op.mma(
                    a_reg_tiles[Int(current)].vectorize[1, a_frag_size](),
                    b_reg_tiles[Int(current)],
                    c.vectorize[1, c_frag_size](),
                )


fn multistage_gemm_kernel[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    transpose_b: Bool,
    c_layout_int_type: DType,
    a_layout_int_type: DType,
    b_layout_int_type: DType,
    c_linear_idx_type: DType,
    a_linear_idx_type: DType,
    b_linear_idx_type: DType,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    serial_reduction: Bool = False,
](
    c: LayoutTensor[
        c_type,
        c_layout,
        MutableAnyOrigin,
        layout_int_type=c_layout_int_type,
        linear_idx_type=c_linear_idx_type,
    ],
    a: LayoutTensor[
        a_type,
        a_layout,
        MutableAnyOrigin,
        layout_int_type=a_layout_int_type,
        linear_idx_type=a_linear_idx_type,
    ],
    b: LayoutTensor[
        b_type,
        b_layout,
        MutableAnyOrigin,
        layout_int_type=b_layout_int_type,
        linear_idx_type=b_linear_idx_type,
    ],
    locks: UnsafePointer[Int32],
):
    # Hold on adding fp16 because it counld have differnet precisions than bf16.
    constrained[
        (a_type in (DType.float32, DType.bfloat16) and a_type == b_type)
        or (
            a_type in (DType.float8_e4m3fn, DType.float8_e5m2)
            and a_type == b_type
            and c_type is DType.float32
        ),
        "Pipeline gemm only supports tf32, BF16, E4M3, and E5M2 mma",
    ]()
    alias simd_size = simdwidthof[c_type]()

    var M: UInt = c.dim[0]()
    var N: UInt = b.dim[0 if transpose_b else 1]()
    var K: UInt = b.dim[1 if transpose_b else 0]()

    alias BM = config.block_tile_shape[0]
    alias BN = config.block_tile_shape[1]
    alias BK = config.block_tile_shape[2]
    alias WM = config.warp_tile_shape[0]
    alias WN = config.warp_tile_shape[1]
    alias num_pipeline_stages = config.num_pipeline_stages

    alias num_warps_m = config.num_warps_m()
    alias num_warps_n = config.num_warps_n()
    alias num_threads = config.num_threads()

    alias num_warp_k_partitions = config.num_warp_k_partitions
    alias num_threads_per_warp_k_part = num_threads // num_warp_k_partitions

    var tid = thread_idx.x
    var ln_id = lane_id()
    var warp_k_part_id = tid // num_threads_per_warp_k_part if num_warp_k_partitions > 1 else 0
    var warp_id = warp.broadcast(
        (tid % num_threads_per_warp_k_part) // WARP_SIZE
    )

    # Only apply block swizzling for half precision types.
    alias swizzle_block = a_type.is_half_float() and b_type.is_half_float() and is_nvidia_gpu()

    # NOTE: the condition ( not (N // BN & 1)) is for a temporary solution
    # for solving mismatches in some shapes
    var block_idx_swizzle = block_swizzle(
        Index[dtype = DType.uint32](block_idx.x, block_idx.y),
        Index[dtype = DType.uint32](grid_dim.x, grid_dim.y),
    ) if swizzle_block else Index[dtype = DType.uint32](
        block_idx.x, block_idx.y
    )

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
        ](a_smem)
        + warp_k_part_id * a_smem_size,
        a_smem_size,
    )

    # There is one pre-allocated shared buffer. Explicitly offset B after at A's end.
    var b_smem = (a_smem + num_warp_k_partitions * a_smem_size).bitcast[
        Scalar[b_type]
    ]()
    alias b_smem_size = num_pipeline_stages * BK * BN
    alias BD_0 = BN if transpose_b else BK
    alias BD_1 = BK if transpose_b else BN
    alias b_smem_layout = Layout.row_major(BD_0, BD_1)
    var b_smem_iter = LayoutTensorIter[
        b_type,
        b_smem_layout,
        address_space = AddressSpace.SHARED,
        circular=True,
    ](b_smem + warp_k_part_id * b_smem_size, b_smem_size)

    # create input layout tensors A and Bv
    # global memory iterator
    var bk_start: Int = (K // BK // num_warp_k_partitions) * warp_k_part_id
    var a_gmem_iter = a.tiled_iterator[BM, BK, axis=1](
        block_idx_swizzle[1], bk_start
    )
    var b_tile_coords = (block_idx_swizzle[0], bk_start) if transpose_b else (
        bk_start,
        block_idx_swizzle[0],
    )
    alias b_tile_axis = 1 if transpose_b else 0
    var b_gmem_iter = b.tiled_iterator[BD_0, BD_1, axis=b_tile_axis](
        b_tile_coords[0], b_tile_coords[1]
    )

    # Compute MMA config
    alias mma_shape = get_mma_shape[a_type, get_accum_type[a_type]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_k_mmas = BK // MMA_K
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N

    alias accum_type = get_accum_type[a_type]()
    alias frag_size = get_fragment_size[mma_shape]()
    alias c_frag_size = frag_size[2]
    var c_reg_tile = tb[accum_type]().row_major[
        num_m_mmas * num_n_mmas, c_frag_size
    ]().local().alloc().fill(0)

    multistage_mma[
        BM,
        BN,
        BK,
        WM,
        WN,
        num_threads_per_warp_k_part,
        num_pipeline_stages,
        transpose_b,
        k_group_size = config.k_group_size,
        swizzle_a = is_nvidia_gpu(),
    ](
        c_reg_tile,
        a_gmem_iter,
        b_gmem_iter,
        a_smem_iter,
        b_smem_iter,
        ceildiv(K // num_warp_k_partitions, BK),
    )

    # reduce within the threadblock
    @parameter
    if num_warp_k_partitions > 1:
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
    var c_gmem_tile = c.tile[BM, BN](block_idx_swizzle[1], block_idx_swizzle[0])
    var c_gmem_warp_tile = c_gmem_tile.tile[WM, WN](Int(warp_y), Int(warp_x))

    @always_inline
    @parameter
    fn apply_epilogue():
        # This block is identical to the one used for f32 case
        # but putting this in a lambda function leads to test failures
        # TODO: Refactor to remove code duplication
        constrained[
            elementwise_lambda_fn is not None,
            "elementwise_lambda_fn is not valid",
        ]()
        alias thread_layout = Layout.row_major(
            8, 4
        ) if is_nvidia_gpu() else Layout.row_major(4, 16)
        alias dst_simd_width_x = 1 if is_nvidia_gpu() else 4
        alias dst_simd_width_y = 2 if is_nvidia_gpu() else 1
        alias src_simd_width_x = 1 if is_nvidia_gpu() else 1
        alias src_simd_width_y = 2 if is_nvidia_gpu() else 4
        alias epilogue = elementwise_lambda_fn.value()
        var c_gmem_frag = c_gmem_warp_tile.vectorize[
            dst_simd_width_x, dst_simd_width_y
        ]().distribute[thread_layout](ln_id)
        var c_reg_frag = c_reg_tile.vectorize[
            src_simd_width_x, src_simd_width_y
        ]().transpose()
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
                dst_idx = Int(c_gmem_frag.runtime_layout(i))
            alias alignment = alignof[SIMD[c_type, src_simd_width_y]]()
            var m = (Int(thread_offset) + dst_idx) // N
            var n = (Int(thread_offset) + dst_idx) % N
            if m < M and n < N:
                var vec = c_reg_frag.ptr.offset(src_idx).load[
                    width=src_simd_width_y,
                    alignment = alignof[SIMD[c_type, src_simd_width_y]](),
                ]()

                @parameter
                if dst_simd_width_x == 1:
                    epilogue[alignment=alignment]((m, n), vec)
                else:

                    @parameter
                    for j in range(dst_simd_width_x):
                        if m + j < M:
                            epilogue[alignment=alignment](
                                (m + j, n), vec[j].cast[c_type]()
                            )

    # Store FP32 mma results to half precision buffer in global memory.
    # Each thread's fragment has 2x2 fp32 values. Casting to half float and
    # directly storing to global memory results in 2 4B writes. Following cutlass,
    # we stage the fragments in shared memory so that each thread can store 16B.
    @parameter
    if c_type.is_half_float() and is_nvidia_gpu():
        alias swizzle = make_swizzle[
            num_rows = MMA_M // 2, row_size=WN, access_size=MMA_N
        ]()

        var accum_smem_warp_tile = tb[c_type]().row_major[
            WM, WN
        ]().shared().view(a_smem.bitcast[Scalar[c_type]]() + warp_id * WM * WN)

        copy[thread_layout = Layout.row_major(8, 4), swizzle=swizzle,](
            accum_smem_warp_tile.vectorize[1, 2](),
            c_reg_tile.vectorize[1, 2]().transpose(),
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
                WARP_SIZE * simd_size // WN, WN // simd_size
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
                if c_gmem_frag.layout.all_dims_known():
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
            var num_parts = grid_dim.z
            if serial_reduction and num_parts > 1:
                var bid = block_idx_swizzle[
                    1
                ] + block_dim.x * block_idx_swizzle[0]
                var semaphore = Semaphore(locks.offset(bid), thread_idx.x)
                semaphore.fetch()
                semaphore.wait(block_idx.z)

                # For the very first block the comes in, it needs to just copy and not reduce_copy
                if block_idx.z == 0:
                    copy_sram_to_dram[
                        thread_layout = Layout.row_major(
                            WARP_SIZE * simd_size // WN, WN // simd_size
                        ),
                        swizzle=swizzle,
                    ](
                        c_gmem_warp_tile.vectorize[1, simd_size](),
                        accum_smem_warp_tile.vectorize[1, simd_size](),
                    )
                else:

                    @always_inline
                    fn add_op[
                        type: DType, width: Int
                    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[
                        type, width
                    ]:
                        return lhs + rhs

                    copy_sram_to_dram[
                        thread_layout = Layout.row_major(
                            WARP_SIZE * simd_size // WN, WN // simd_size
                        ),
                        swizzle=swizzle,
                        binary_op=add_op,
                    ](
                        c_gmem_warp_tile.vectorize[1, simd_size](),
                        accum_smem_warp_tile.vectorize[1, simd_size](),
                    )

                var lock_flag: Int
                if num_parts == (block_idx.z + 1):
                    lock_flag = 0
                else:
                    lock_flag = block_idx.z + 1
                semaphore.release(lock_flag)

            else:
                copy_sram_to_dram[
                    thread_layout = Layout.row_major(
                        WARP_SIZE * simd_size // WN, WN // simd_size
                    ),
                    swizzle=swizzle,
                ](
                    c_gmem_warp_tile.vectorize[1, simd_size](),
                    accum_smem_warp_tile.vectorize[1, simd_size](),
                )

    elif c_type.is_half_float() and not is_nvidia_gpu():

        @parameter
        if elementwise_lambda_fn:
            apply_epilogue()

        else:
            var c_reg_tile_out = LayoutTensor[
                c_type,
                c_reg_tile.layout,
                MutableAnyOrigin,
                address_space = AddressSpace.LOCAL,
            ].stack_allocation()

            @parameter
            for i in range(c_reg_tile.shape[0]()):

                @parameter
                for j in range(c_reg_tile.shape[1]()):
                    c_reg_tile_out[i, j] = c_reg_tile[i, j].cast[c_type]()
            copy_local_to_dram[dst_thread_layout = Layout.row_major(4, 16)](
                c_gmem_warp_tile.vectorize[4, 1](),
                c_reg_tile_out.vectorize[1, 4](),
            )
    # Store FP32 results to FP32 buffer in global memory.
    else:

        @parameter
        if elementwise_lambda_fn:
            apply_epilogue()
        else:

            @parameter
            if is_nvidia_gpu():
                copy_local_to_dram[dst_thread_layout = Layout.row_major(8, 4)](
                    c_gmem_warp_tile.vectorize[1, 2](),
                    c_reg_tile.vectorize[1, 2]().transpose(),
                )
            else:
                copy_local_to_dram[dst_thread_layout = Layout.row_major(4, 16)](
                    c_gmem_warp_tile.vectorize[4, 1](),
                    c_reg_tile.vectorize[1, 4](),
                )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](config.num_threads())
)
fn multistage_gemm_split_k_kernel[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    work_space_type: DType,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    serial_reduction: Bool = False,
](
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    a: LayoutTensor[a_type, a_layout, MutableAnyOrigin],
    b: LayoutTensor[b_type, b_layout, MutableAnyOrigin],
    work_space: NDBuffer[work_space_type, 3, MutableAnyOrigin],
    num_partitions: UInt,
    locks: UnsafePointer[Int32],
):
    var M = c.dim[0]()
    alias N = b.shape[0]() if transpose_b else b.shape[1]()
    alias K = b.shape[1]() if transpose_b else b.shape[0]()
    alias BK = config.block_tile_shape[2]

    # If K is not divisible by num_partitions, the first num_partitions-1 parts
    # will be rounded up to multiple of BK.
    var a_part = a.split[axis=1, alignment=BK](num_partitions, block_idx.z)
    var b_part = b.split[axis= 1 if transpose_b else 0, alignment=BK](
        num_partitions, block_idx.z
    )

    @parameter
    if serial_reduction:
        alias k_partition_config = MatmulConfig[
            a_type, b_type, c_type, transpose_b
        ](
            block_tile_shape=config.block_tile_shape,
            warp_tile_shape=config.warp_tile_shape,
            num_pipeline_stages=config.num_pipeline_stages,
        )

        multistage_gemm_kernel[
            c_type,
            c.layout,
            a_type,
            a_part.layout,
            b_type,
            b_part.layout,
            transpose_b,
            config=k_partition_config,
            serial_reduction=serial_reduction,
        ](c, a_part, b_part, locks)
    else:
        alias work_space_tensor_type = LayoutTensor[
            work_space_type, c_layout, MutableAnyOrigin
        ]

        var work_space_part = work_space_tensor_type(
            work_space.data + block_idx.z * M * N,
            RuntimeLayout[
                c_layout,
                element_type = work_space_tensor_type.layout_int_type,
                linear_idx_type = work_space_tensor_type.linear_idx_type,
            ].row_major(
                IndexList[
                    2, element_type = work_space_tensor_type.layout_int_type
                ](M, N)
            ),
        )
        alias k_partition_config = MatmulConfig[
            a_type,
            b_type,
            work_space_type,
            transpose_b,
        ](
            block_tile_shape=config.block_tile_shape,
            warp_tile_shape=config.warp_tile_shape,
            num_pipeline_stages=config.num_pipeline_stages,
        )

        @parameter
        if has_amd_gpu_accelerator() and transpose_b:
            amd_gemm_kernel[
                work_space_type,
                work_space_part.layout,
                a_type,
                a_part.layout,
                b_type,
                b_part.layout,
                transpose_b,
                config=k_partition_config,
            ](work_space_part, a_part, b_part)

        else:
            multistage_gemm_kernel[
                work_space_type,
                work_space_part.layout,
                a_type,
                a_part.layout,
                b_type,
                b_part.layout,
                transpose_b,
                config=k_partition_config,
                serial_reduction=serial_reduction,
            ](work_space_part, a_part, b_part, locks)
