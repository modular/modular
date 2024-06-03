# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s

from math import ceildiv, isclose
from buffer import NDBuffer
from buffer.list import DimList
from collections.optional import OptionalReg
from memory.unsafe import DTypePointer
from memory.reference import _GPUAddressSpace as AddressSpace
from pathlib import Path

from LinAlg.MatmulGPU import matmul_kernel_naive
from gpu import (
    WARP_SIZE,
    BlockIdx,
    ThreadIdx,
    barrier,
    lane_id,
)
from gpu.host import Context, Function, synchronize, Stream, FuncAttribute
from gpu.host.event import time_function
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from gpu.mma import mma, ld_matrix
from gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_group,
    dynamic_shared_memory,
)
from testing import assert_almost_equal
from sys import argv
from layout.int_tuple import IntTuple
from layout.layout import *
from layout.layout_tensor import LayoutTensor, _swizzle_signature
from layout.nd_buffer_stub import copy_from_nd_buffer, distribute, vectorize
from layout.swizzle import Swizzle
from layout.tensor_core import get_accum_type, get_mma_shape, get_fragment_size


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


# Mask ^ tid's 2 least significant and every 8 threads share one mask.
# This reproduces the thread map in Cutlass when BK=16.
@always_inline
fn xor_2bits_per8T[type: DType](tid: Scalar[type]) -> Scalar[type]:
    return Swizzle[2, 0, 3]()(tid)


@always_inline
fn ld_mma[
    num_matrices: Int,
    # Refactor the three parameters with ComposedLayout
    thread_layout: Layout,
    swizzle: OptionalReg[_swizzle_signature] = None,
    *,
    # work around parameter deduction
    __layout: Layout,
    __element_layout: Layout,
    __index_type: DType,
    __masked: Bool,
](
    mat: LayoutTensor[
        _,
        __layout,
        address_space = AddressSpace.SHARED,
        element_layout=__element_layout,
        index_type=__index_type,
        masked=__masked,
    ],
    offset: Int,
) -> SIMD[mat.dtype, num_matrices]:
    constrained[
        num_matrices == 2 or num_matrices == 4,
        "Only support loading 2 or 4 matrices.",
    ]()

    # TODO: Either optimize signed int division or restrict this to uint32.
    var lane_id = UInt32(ThreadIdx.x()) % WARP_SIZE

    alias stride = thread_layout.stride[0].value()
    alias simd_size = simdwidthof[mat.dtype]()

    # TODO: the index calculation can be refactored when layout(i) works on GPU.
    # var row_offset = thread_layout(lane_id)
    fn get_row_offset() -> Int:
        var quo: Int
        var rem: Int
        quo, rem = divmod(int(lane_id), 16 if num_matrices == 4 else 8)
        return rem * stride + quo + offset

    var row_offset = UInt32(get_row_offset())

    @parameter
    if swizzle:
        alias swizzle_fn = swizzle.value()
        row_offset = swizzle_fn(row_offset)

    row_offset = row_offset * simd_size

    return ld_matrix[mat.dtype, num_matrices](mat.ptr + row_offset)


fn multistage_gemm[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    transpose_b: Bool,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
](
    c: NDBuffer[c_type, 2, c_shape],
    a: NDBuffer[a_type, 2, a_shape],
    b: NDBuffer[b_type, 2, b_shape],
):
    constrained[
        c_type == DType.float32
        and a_type == DType.float32
        and b_type == DType.float32,
        "Only support tf32 mma",
    ]()

    constrained[BK == 16, "Only support BK = 16."]()

    alias simd_size = simdwidthof[c_type]()

    var M = c.dim[0]()
    var N = c.dim[1]()
    var K = a.dim[1]()

    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN

    constrained[
        num_warps_m * num_warps_n == num_threads // WARP_SIZE,
        "Number of warps doesn't match warp tile sizes.",
    ]()

    var tid: UInt32 = ThreadIdx.x()
    var warp_id = tid // WARP_SIZE
    var lane_id = lane_id()

    # Coordinates of the current warp.
    var warp_x = warp_id % num_warps_n
    var warp_y = warp_id // num_warps_n

    var a_smem = dynamic_shared_memory[Scalar[a_type], alignment=4]()
    var b_smem = (a_smem + num_pipeline_stages * BM * BK).bitcast[
        Scalar[b_type]
    ]()

    alias thread_async_copy_a_layout = Layout.row_major(
        num_threads * simd_size // BK, BK // simd_size
    )

    alias thread_async_copy_b_layout = Layout.row_major(
        num_threads * simd_size // BK, BK // simd_size
    ) if transpose_b else Layout.row_major(
        num_threads * simd_size // BN, BN // simd_size
    )

    alias b_smem_layout = Layout.row_major(
        BN, BK
    ) if transpose_b else Layout.row_major(BK, BN)

    # Prefetch (num_pipeline_stages - 1) stages.
    @parameter
    for stage in range(num_pipeline_stages - 1):
        var a_smem_tile = LayoutTensor[
            a_type,
            Layout.row_major(BM, BK),
            address_space = AddressSpace.SHARED,
        ](a_smem + stage * BM * BK)

        var b_smem_tile = LayoutTensor[
            b_type,
            b_smem_layout,
            address_space = AddressSpace.SHARED,
        ](b_smem + stage * BN * BK)

        copy_from_nd_buffer[
            thread_layout=thread_async_copy_a_layout,
            is_async=True,
            swizzle=xor_2bits_per8T,
        ](
            a_smem_tile.vectorize[1, simd_size]().distribute[
                thread_async_copy_a_layout
            ](ThreadIdx.x()),
            a.tile[BM, BK]((BlockIdx.y(), stage)),
            ThreadIdx.x(),
        )

        # fmt: off
        @parameter
        if transpose_b:
            copy_from_nd_buffer[
                thread_layout=thread_async_copy_b_layout,
                is_async=True,
                swizzle=xor_2bits_per8T,
            ](
                b_smem_tile.vectorize[1, simd_size]().distribute[
                    thread_async_copy_b_layout
                ](ThreadIdx.x()),
                b.tile[BN, BK]((BlockIdx.x(), stage)),
                ThreadIdx.x(),
            )
        else:
            copy_from_nd_buffer[
                thread_layout=thread_async_copy_b_layout,
                is_async=True,
                swizzle=xor_2bits_per8T,
            ](
                b_smem_tile.vectorize[1, simd_size]().distribute[
                    thread_async_copy_b_layout
                ](ThreadIdx.x()),
                b.tile[BK, BN]((stage, BlockIdx.x())),
                ThreadIdx.x(),
            )
        # fmt: on

        async_copy_commit_group()

    # Guard stage 0.
    async_copy_wait_group(num_pipeline_stages - 2)
    barrier()

    alias mma_shape = get_mma_shape[a_type, get_accum_type[a_type]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_k_mmas = BK // MMA_K
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N

    alias frag_size = get_fragment_size[mma_shape]()
    alias a_frag_size = frag_size[0]
    alias b_frag_size = frag_size[1]
    alias c_frag_size = frag_size[2]

    # Register tiles.
    # TODO: parameterize fragment size based on data type.
    var a_reg_tiles = LayoutTensor[
        a_type, Layout.row_major(2 * num_m_mmas, a_frag_size)
    ].stack_allocation().vectorize[1, a_frag_size]().split[2]()
    var b_reg_tiles = LayoutTensor[
        b_type, Layout.row_major(2 * num_n_mmas, b_frag_size)
    ].stack_allocation().vectorize[1, b_frag_size]().split[2]()
    var c_reg_tile = LayoutTensor[
        c_type, Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size)
    ].stack_allocation().vectorize[1, c_frag_size]()

    c_reg_tile.fill(0)

    alias a_frag_type = __type_of(a_reg_tiles).ElementType.element_type
    alias b_frag_type = __type_of(b_reg_tiles).ElementType.element_type

    # Load shared -> registers for stage 0's mma.
    # TODO: remove the cast.
    var a_smem_tile0 = LayoutTensor[
        a_type,
        Layout.row_major(BM, BK),
        address_space = AddressSpace.SHARED,
    ](a_smem)

    var b_smem_tile0 = LayoutTensor[
        b_type,
        b_smem_layout,
        address_space = AddressSpace.SHARED,
    ](b_smem)

    var a_warp_tile = a_smem_tile0.tile[WM, BK](int(warp_y), 0)

    # TODO: warp the following in the tile method, maybe tile[shape: IntTuple].
    # I can't use b_warp_tile = ... if transpose_b else ... because the operands
    # are deduced as different types since their layout are different.
    alias b_wtile_dim0 = WN if transpose_b else BK
    alias b_wtile_dim1 = BK if transpose_b else WN
    var b_wtile_coord0 = int(warp_x) if transpose_b else 0
    var b_wtile_coord1 = 0 if transpose_b else int(warp_x)
    var b_warp_tile = b_smem_tile0.tile[b_wtile_dim0, b_wtile_dim1](
        b_wtile_coord0, b_wtile_coord1
    )

    # TODO: possbile to not rebind?
    @parameter
    for m_mma in range(num_m_mmas):
        var a_mma_tile = a_warp_tile.tile[MMA_M, BK](m_mma, 0)
        a_reg_tiles[0][m_mma, 0] = rebind[a_frag_type](
            ld_mma[
                4,
                Layout(IntTuple(16, 2), IntTuple(BK // simd_size, 1)),
                swizzle=xor_2bits_per8T,
            ](a_mma_tile, 0)
        )

    @parameter
    if transpose_b:
        # Use ld_matrix because with transposed B the thread layout in mma is
        # row-major(8, 4). TF32 mma needs 2 8x4 matrices but we can combine 2
        # mmas and load 4 matrics per iteration, reducing instruction count.
        @parameter
        for n_mma2 in range(num_n_mmas // 2):
            var b_mma_tile = b_warp_tile.tile[2 * MMA_N, BK](n_mma2, 0)
            var vec = ld_mma[
                4,
                Layout(IntTuple(16, 2), IntTuple(BK // simd_size, 1)),
                swizzle=xor_2bits_per8T,
            ](b_mma_tile, 0)
            b_reg_tiles[0][2 * n_mma2, 0] = rebind[b_frag_type](
                SIMD[b_type, 2](vec[0], vec[2])
            )
            b_reg_tiles[0][2 * n_mma2 + 1, 0] = rebind[b_frag_type](
                SIMD[b_type, 2](vec[1], vec[3])
            )

    else:
        # Use normal scalar load because the thread layout in mma is column-majored.
        @parameter
        for n_mma in range(num_n_mmas):
            var b_mma_tile = b_warp_tile.tile[MMA_K, MMA_N](0, n_mma)
            var b_mma_frag = b_mma_tile.distribute[Layout.col_major(4, 8)](
                int(lane_id)
            )
            b_reg_tiles[0][n_mma, 0] = rebind[b_frag_type](
                SIMD[b_type, 2](
                    rebind[Scalar[b_type]](b_mma_frag[0]),
                    rebind[Scalar[b_type]](b_mma_frag[1]),
                )
            )

    var num_k_tiles = ceildiv(K, BK)

    for k_tile_id in range(num_k_tiles):
        var stage = k_tile_id % num_pipeline_stages

        var a_smem_tile = LayoutTensor[
            a_type,
            Layout.row_major(BM, BK),
            address_space = AddressSpace.SHARED,
        ](a_smem + stage * BM * BK)

        var b_smem_tile = LayoutTensor[
            b_type,
            b_smem_layout,
            address_space = AddressSpace.SHARED,
        ](b_smem + stage * BN * BK)

        var a_warp_tile = a_smem_tile.tile[WM, BK](int(warp_y), 0)
        var b_warp_tile = b_smem_tile.tile[b_wtile_dim0, b_wtile_dim1](
            b_wtile_coord0, b_wtile_coord1
        )

        # Perform prefetch registers and mma until current shared memory tile's
        # data has all been loaded to registers.
        @parameter
        for k_mma in range(num_k_mmas):
            var current = k_mma % 2
            var next = (k_mma + 1) % 2
            var next_stage = (k_tile_id + 1) % num_pipeline_stages

            if k_mma == num_k_mmas - 1:
                var a_smem_next_tile = LayoutTensor[
                    a_type,
                    Layout.row_major(BM, BK),
                    address_space = AddressSpace.SHARED,
                ](a_smem + next_stage * BM * BK)

                var b_smem_next_tile = LayoutTensor[
                    b_type,
                    b_smem_layout,
                    address_space = AddressSpace.SHARED,
                ](b_smem + next_stage * BN * BK)

                a_warp_tile = a_smem_next_tile.tile[WM, BK](int(warp_y), 0)
                b_warp_tile = b_smem_next_tile.tile[b_wtile_dim0, b_wtile_dim1](
                    b_wtile_coord0, b_wtile_coord1
                )

            @parameter
            for m_mma in range(num_m_mmas):
                var a_mma_tile = a_warp_tile.tile[MMA_M, BK](m_mma, 0)
                a_reg_tiles[next][m_mma, 0] = rebind[a_frag_type](
                    ld_mma[
                        4,
                        Layout(IntTuple(16, 2), IntTuple(BK // simd_size, 1)),
                        swizzle=xor_2bits_per8T,
                    ](
                        a_mma_tile,
                        (k_mma + 1) % num_k_mmas * MMA_K // simd_size,
                    )
                )

            @parameter
            if transpose_b:

                @parameter
                for n_mma2 in range(num_n_mmas // 2):
                    var b_mma_tile = b_warp_tile.tile[2 * MMA_N, BK](n_mma2, 0)
                    var vec = ld_mma[
                        4,
                        Layout(IntTuple(16, 2), IntTuple(BK // simd_size, 1)),
                        swizzle=xor_2bits_per8T,
                    ](
                        b_mma_tile,
                        (k_mma + 1) % num_k_mmas * MMA_K // simd_size,
                    )
                    b_reg_tiles[next][2 * n_mma2, 0] = rebind[b_frag_type](
                        SIMD[b_type, 2](vec[0], vec[2])
                    )
                    b_reg_tiles[next][2 * n_mma2 + 1, 0] = rebind[b_frag_type](
                        SIMD[b_type, 2](vec[1], vec[3])
                    )
            else:

                @parameter
                for n_mma in range(num_n_mmas):
                    var b_mma_tile = b_warp_tile.tile[MMA_K, MMA_N](
                        (k_mma + 1) % num_k_mmas, n_mma
                    )
                    var b_mma_frag = b_mma_tile.distribute[
                        Layout.col_major(4, 8)
                    ](int(lane_id))
                    b_reg_tiles[next][n_mma, 0] = rebind[b_frag_type](
                        SIMD[b_type, 2](
                            rebind[Scalar[b_type]](b_mma_frag[0]),
                            rebind[Scalar[b_type]](b_mma_frag[1]),
                        )
                    )

            @parameter
            for m_mma in range(num_m_mmas):

                @parameter
                for n_mma in range(num_n_mmas):
                    mma(
                        c_reg_tile[m_mma * num_n_mmas + n_mma, 0],
                        a_reg_tiles[current][m_mma, 0],
                        b_reg_tiles[current][n_mma, 0],
                        c_reg_tile[m_mma * num_n_mmas + n_mma, 0],
                    )

            if k_mma + 2 == num_k_mmas:
                var prefetch_tile_id = k_tile_id + num_pipeline_stages - 1

                # Prefetch one k tile (if valid) from global memory to current
                # shared memory buffer.
                if prefetch_tile_id < num_k_tiles:
                    var prefetch_stage = prefetch_tile_id % num_pipeline_stages

                    var a_smem_prefetch_tile = LayoutTensor[
                        a_type,
                        Layout.row_major(BM, BK),
                        address_space = AddressSpace.SHARED,
                    ](a_smem + prefetch_stage * BM * BK)

                    var b_smem_prefetch_tile = LayoutTensor[
                        b_type,
                        b_smem_layout,
                        address_space = AddressSpace.SHARED,
                    ](b_smem + prefetch_stage * BN * BK)

                    # TODO: Extend the async copy instrinsic to creat dummy copies. The
                    # prefetch for the three two iterations should be dummy.
                    copy_from_nd_buffer[
                        thread_layout=thread_async_copy_a_layout,
                        is_async=True,
                        swizzle=xor_2bits_per8T,
                    ](
                        a_smem_prefetch_tile.vectorize[
                            1, simd_size
                        ]().distribute[thread_async_copy_a_layout](
                            ThreadIdx.x()
                        ),
                        a.tile[BM, BK](
                            (BlockIdx.y(), prefetch_tile_id % num_k_tiles)
                        ),
                        ThreadIdx.x(),
                    )

                    @parameter
                    if transpose_b:
                        copy_from_nd_buffer[
                            thread_layout=thread_async_copy_b_layout,
                            is_async=True,
                            swizzle=xor_2bits_per8T,
                        ](
                            b_smem_prefetch_tile.vectorize[
                                1, simd_size
                            ]().distribute[thread_async_copy_b_layout](
                                ThreadIdx.x()
                            ),
                            b.tile[BN, BK](
                                (BlockIdx.x(), prefetch_tile_id % num_k_tiles)
                            ),
                            ThreadIdx.x(),
                        )
                    else:
                        copy_from_nd_buffer[
                            thread_layout=thread_async_copy_b_layout,
                            is_async=True,
                            swizzle=xor_2bits_per8T,
                        ](
                            b_smem_prefetch_tile.vectorize[
                                1, simd_size
                            ]().distribute[thread_async_copy_b_layout](
                                ThreadIdx.x()
                            ),
                            b.tile[BK, BN](
                                (prefetch_tile_id % num_k_tiles, BlockIdx.x())
                            ),
                            ThreadIdx.x(),
                        )

                async_copy_commit_group()

                # Guard the next k tile's shared memory buffer.
                async_copy_wait_group(num_pipeline_stages - 2)
                barrier()

    # Map global memory tile down to thread.
    var c_gmem_tile = c.tile[BM, BN]((BlockIdx.y(), BlockIdx.x()))
    var c_gmem_warp_tile = c_gmem_tile.tile[WM, WN]((int(warp_y), int(warp_x)))

    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):
            var c_gmem_mma_tile = c_gmem_warp_tile.tile[MMA_M, MMA_N](
                (m_mma, n_mma)
            )
            var c_frag = vectorize[1, 2](c_gmem_mma_tile)
            var c_frag_local = distribute[
                thread_layout = Layout.row_major(8, 4)
            ](c_frag[0], int(lane_id))
            var c_reg = c_reg_tile[m_mma * num_n_mmas + n_mma, 0]
            SIMD[size=2].store[alignment = alignof[SIMD[c_type, 2]]()](
                c_frag_local._offset((0, 0)),
                SIMD[c_type, 2](c_reg[0], c_reg[1]),
            )
            SIMD[size=2].store[alignment = alignof[SIMD[c_type, 2]]()](
                c_frag_local._offset((0, 0)),
                SIMD[c_type, 2](c_reg[2], c_reg[3]),
            )


fn test[transpose_b: Bool]() raises:
    alias num_threads = 128
    alias num_pipeline_stages = 4
    alias M = 8192
    alias N = 8192
    alias K = 128
    alias BM = 128
    alias BN = 128
    alias BK = 16
    alias WM = 64
    alias WN = 64
    alias shared_mem_bytes = 80 * 1024

    var stream = Stream()

    alias a_layout = Layout.row_major(M, K)
    alias b_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    alias c_layout = Layout.row_major(M, N)

    var a_host = DTypePointer[DType.float32].alloc(M * K)
    var b_host = DTypePointer[DType.float32].alloc(K * N)
    var b_trans_host = DTypePointer[DType.float32].alloc(K * N)
    var c_host = DTypePointer[DType.float32].alloc(M * N)
    var c_host_ref = DTypePointer[DType.float32].alloc(M * N)

    for m in range(M):
        for k in range(K):
            a_host[m * K + k] = m * K + k

    for k in range(K):
        for n in range(N):
            b_host[k * N + n] = k * N + n

            @parameter
            if transpose_b:
                b_trans_host[n * K + k] = k * N + n
            else:
                b_trans_host[k * N + n] = k * N + n

    var a_device = _malloc[Float32](M * K)
    var b_device = _malloc[Float32](K * N)
    var c_device = _malloc[Float32](M * N)
    var c_device_ref = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_trans_host, K * N)

    alias b_shape = DimList(K, N) if not transpose_b else DimList(N, K)
    var c_buffer = NDBuffer[DType.float32, 2, DimList(M, N)](c_device)
    var a_buffer = NDBuffer[DType.float32, 2, DimList(M, K)](a_device)
    var b_buffer = NDBuffer[DType.float32, 2, b_shape](b_device)
    alias c_shape = DimList(M, N)
    alias a_shape = DimList(M, K)

    var c_tensor = LayoutTensor[DType.float32, c_layout](c_device)
    var a_tensor = LayoutTensor[DType.float32, a_layout](a_device)
    var b_tensor = LayoutTensor[DType.float32, b_layout](b_device)

    alias gemm = multistage_gemm[
        DType.float32,
        c_shape,
        DType.float32,
        a_shape,
        DType.float32,
        b_shape,
        transpose_b,
        BM,
        BN,
        BK,
        WM,
        WN,
        num_threads,
        num_pipeline_stages,
    ]
    # TODO: The cache config doesn't really help here, see #38391.
    var func = Function[gemm](
        threads_per_block=num_threads,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            shared_mem_bytes
        ),
        # dump_llvm=Path("./pipeline-gemm.ir"),
        # dump_ptx=Path("./pipeline-gemm.ptx"),
    )

    if is_benchmark():
        alias nrun = 200
        alias nwarmup = 2

        @always_inline
        @parameter
        fn run_func(stream: Stream) raises:
            for i in range(nrun):
                func(
                    c_buffer,
                    a_buffer,
                    b_buffer,
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
                    block_dim=(num_threads, 1, 1),
                    shared_mem_bytes=shared_mem_bytes,
                    stream=stream,
                )

        # Warmup
        for i in range(nwarmup):
            func(
                c_buffer,
                a_buffer,
                b_buffer,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
                block_dim=(num_threads, 1, 1),
                shared_mem_bytes=shared_mem_bytes,
                stream=stream,
            )

        var nstime = time_function[run_func](stream) / nrun
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * M * N * K * 1e-12
        print(
            "Tranpose B ",
            transpose_b,
            nrun,
            " runs avg(s)",
            sectime,
            "TFlops/s",
            TFlop / sectime,
        )

    func(
        c_buffer,
        a_buffer,
        b_buffer,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
        block_dim=(num_threads, 1, 1),
        shared_mem_bytes=shared_mem_bytes,
        stream=stream,
    )

    synchronize()

    _copy_device_to_host(c_host, c_device, M * N)
    _copy_host_to_device(b_device, b_host, K * N)

    # Naive gemm.
    alias BLOCK_DIM = 16
    alias gemm_naive = matmul_kernel_naive[
        DType.float32, DType.float32, DType.float32, BLOCK_DIM
    ]
    var func_naive = Function[gemm_naive](threads_per_block=256)
    var c_buffer_ref = NDBuffer[DType.float32, 2, DimList(M, N)](c_device_ref)
    func_naive(
        c_buffer_ref,
        a_buffer,
        b_buffer,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    synchronize()
    _copy_device_to_host(c_host_ref, c_device_ref, M * N)

    for i in range(M * N):
        if not isclose(c_host[i], c_host_ref[i], rtol=0.01):
            print(i, c_host[i], c_host_ref[i])
        assert_almost_equal(c_host[i], c_host_ref[i], rtol=0.01)

    _free(c_device)
    _free(c_device_ref)
    _free(a_device)
    _free(b_device)

    c_host.free()
    c_host_ref.free()
    a_host.free()
    b_host.free()
    b_trans_host.free()

    _ = func^
    _ = func_naive^
    _ = stream^


def main():
    try:
        with Context() as ctx:
            test[False]()
            test[True]()
    except e:
        print("ERROR:", e)
