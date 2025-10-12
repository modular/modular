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

from linalg.matmul.gpu.amd.structured import (
    AmdTileOperator,
    AmdWarpBlockScatterGather,
    ThreadRole,
    RingBuffer,
    SharedMemoryBuffer,
    batched_copy_dram_to_local,
)

from gpu import (
    WARP_SIZE,
    block_idx,
    thread_idx,
    barrier,
    warp_id as get_warp_id,
)

from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from layout.layout_tensor import (
    ThreadScope,
    copy_local_to_dram,
)
from gpu.host import DeviceContext
from utils import IndexList
from layout._fillers import random
from memory import stack_allocation
import linalg.matmul.vendor.blas as vendor_blas
from testing import assert_equal
from random import random_si64
from gpu import MAX_THREADS_PER_BLOCK_METADATA
from utils import StaticTuple


# NOTE: This is a hardcoded pipeline but in reality this should be struct
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        (producer_warps_a + producer_warps_b + consumer_warps) * WARP_SIZE
    )
)
fn test_producer_consumer[
    in_type: DType,
    out_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    WK: Int,
    producer_warps_a: Int = 1,
    producer_warps_b: Int = 1,
    consumer_warps: Int = 1,
    pipeline_stages: Int = 1,
](
    a: LayoutTensor[
        in_type, a_layout, MutableAnyOrigin, address_space = AddressSpace.GLOBAL
    ],
    b: LayoutTensor[
        in_type, b_layout, MutableAnyOrigin, address_space = AddressSpace.GLOBAL
    ],
    c: LayoutTensor[
        out_type,
        c_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.GLOBAL,
    ],
):
    var K = a.shape[1]()

    # NOTE: hardcoded MMA for now, but in theory this pipeline will work with
    # any MMA
    alias MMA_M = 16
    alias MMA_N = 16
    alias MMA_K = 16

    alias m_warps_per_block = BM // WM
    alias n_warps_per_block = BN // WN

    constrained[m_warps_per_block % producer_warps_a == 0]()
    constrained[n_warps_per_block % producer_warps_b == 0]()
    constrained[m_warps_per_block * n_warps_per_block % consumer_warps == 0]()

    var role: ThreadRole
    var warp_id = get_warp_id()
    alias producer_thread_count = (
        producer_warps_a + producer_warps_b
    ) * WARP_SIZE

    if thread_idx.x < UInt(producer_thread_count):
        role = ThreadRole.PRODUCER
    else:
        role = ThreadRole.CONSUMER

    alias thread_layout = Layout.row_major(16, 4)
    alias smem_layout_a = Layout.row_major(pipeline_stages, BM, BK)
    alias smem_layout_b = Layout.row_major(pipeline_stages, BN, BK)

    var smem_buffer_a = SharedMemoryBuffer[
        in_type,
        smem_layout_a,
        pipeline_stages,
        WM,
        WK,
    ]()

    var smem_buffer_b = SharedMemoryBuffer[
        in_type,
        smem_layout_b,
        pipeline_stages,
        WN,
        WK,
    ]()

    # NOTE: Every thread needs access to the ring buffer
    var ring_buffer = RingBuffer[
        pipeline_stages,
        smem_buffer_a.layout,
        smem_buffer_b.layout,
        in_type,
        in_type,
        WM,
        WN,
        WK,
        m_warps_per_block,
        n_warps_per_block,
    ](smem_buffer_a, smem_buffer_b)

    barrier()  # NOTE: probably not nessecary but I saw it in the HF code around the same point

    # if thread_idx.x == 0:
    #     for i in range(64):
    #         for j in range(64):
    #             print(b[i, j], end=" ")
    #         print()

    var tile_count = K // BK

    # NOTE: the 2 producer blocks are almost identical, you can proabbly make this a
    # a function
    if role is ThreadRole.PRODUCER:
        if warp_id < UInt(producer_warps_a):
            # NOTE: If there is a way to hide the phases that would be great, maybe the ringbuffer
            # handles the thread specific phase, or its encapsulated in the ThreadRole Struct.

            var phases = InlineArray[
                Int, pipeline_stages * (m_warps_per_block // producer_warps_a)
            ](
                fill=0,
            )

            var scatter_gather_a = AmdWarpBlockScatterGather[
                in_type,
                thread_layout,
                smem_buffer_a.WarpTileType.layout,
                8,  # NOTE: hardcoded simd width for now, but in theory this should be derived
                True,
                WM,
                WK,
            ]()

            for tile_num in range(tile_count):
                var a_tile = a.tile[BM, BK](block_idx.x, tile_num)
                var local_tile_count = 0

                # NOTE: producers and consumers can process more than one tile this loop
                # makes sure this is possible
                for warp_tile_idx in range(
                    warp_id, m_warps_per_block, producer_warps_a
                ):
                    # NOTE: my code already has support for double buffering, no extra
                    # work should be needed
                    var stage = tile_num % pipeline_stages

                    ref phase = phases[
                        local_tile_count * pipeline_stages + stage
                    ]

                    scatter_gather_a.load_compute_tile[
                        a_tile.dtype,
                        a_tile.layout,
                    ](
                        ring_buffer,
                        phase,
                        rebind[
                            LayoutTensor[
                                a_tile.dtype,
                                a_tile.layout,
                                MutableAnyOrigin,
                                address_space = AddressSpace.GLOBAL,
                            ]
                        ](a_tile),
                        stage,
                        warp_tile_idx,
                    )

                    local_tile_count += 1
        else:
            var relative_warp_id = warp_id - UInt(producer_warps_a)

            var phases = InlineArray[
                Int, pipeline_stages * (n_warps_per_block // producer_warps_b)
            ](
                fill=0,
            )

            var scatter_gather_b = AmdWarpBlockScatterGather[
                in_type,
                thread_layout,
                smem_buffer_b.WarpTileType.layout,
                8,
                False,
                WN,
                WK,
            ]()

            for tile_num in range(tile_count):
                var b_tile = b.tile[BN, BK](block_idx.y, tile_num)
                var local_tile_count = 0

                for warp_tile_idx in range(
                    relative_warp_id, n_warps_per_block, producer_warps_b
                ):
                    var stage = tile_num % pipeline_stages

                    ref phase = phases[
                        local_tile_count * pipeline_stages + stage
                    ]

                    scatter_gather_b.load_compute_tile[
                        b_tile.dtype,
                        b_tile.layout,
                    ](
                        ring_buffer,
                        phase,
                        rebind[
                            LayoutTensor[
                                b_tile.dtype,
                                b_tile.layout,
                                MutableAnyOrigin,
                                address_space = AddressSpace.GLOBAL,
                            ]
                        ](b_tile),
                        stage,
                        warp_tile_idx,
                    )

                    local_tile_count += 1

    else:
        alias total_consumer_operations = m_warps_per_block * n_warps_per_block

        var phases_a = InlineArray[
            Int, pipeline_stages * (total_consumer_operations // consumer_warps)
        ](
            fill=1,
        )

        var phases_b = InlineArray[
            Int, pipeline_stages * (total_consumer_operations // consumer_warps)
        ](
            fill=1,
        )

        var tile_operator = AmdTileOperator[
            in_type,
            out_type,
            smem_buffer_a.WarpTileType.layout,
            smem_buffer_b.WarpTileType.layout,
            IndexList[3](MMA_M, MMA_N, MMA_K),
            True,
            simd_width=8,
            warps_being_processed = total_consumer_operations // consumer_warps,
        ]()

        var relative_warp_id = warp_id - UInt(
            producer_warps_a + producer_warps_b
        )

        for i in range(tile_count):
            var local_tile_count = 0
            for warp_tile_idx in range(
                relative_warp_id, total_consumer_operations, consumer_warps
            ):
                var m_warp_idx = warp_tile_idx // n_warps_per_block
                var n_warp_idx = warp_tile_idx % n_warps_per_block

                var stage = i % pipeline_stages

                ref phase_a = phases_a[
                    local_tile_count * pipeline_stages + stage
                ]
                ref phase_b = phases_b[
                    local_tile_count * pipeline_stages + stage
                ]

                tile_operator.mma(
                    ring_buffer,
                    phase_a,
                    phase_b,
                    stage,
                    m_warp_idx,
                    n_warp_idx,
                    local_tile_count,
                )

                local_tile_count += 1

        var local_tile_count = 0
        for warp_tile_idx in range(
            relative_warp_id, total_consumer_operations, consumer_warps
        ):
            var m_warp_idx = warp_tile_idx // n_warps_per_block
            var n_warp_idx = warp_tile_idx % n_warps_per_block

            var c_reg_tile = tile_operator.get_c_reg_tile_slice(
                local_tile_count
            )
            store_c[
                c.dtype,
                c.layout,
                c_reg_tile.layout,
                BM,
                BN,
                WM,
                WN,
                c.shape[1](),
            ](c, c_reg_tile, m_warp_idx, n_warp_idx)

            local_tile_count += 1


fn store_c[
    c_type: DType,
    c_layout: Layout,
    c_reg_layout: Layout,
    BM: Int,
    BN: Int,
    WM: Int,
    WN: Int,
    static_N: Int,
](
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin, *_, **_],
    c_reg_tile: LayoutTensor[c_type, c_reg_layout, MutableAnyOrigin, *_, **_],
    warp_m: Int,
    warp_n: Int,
):
    var c_block_tile = c.tile[BM, BN](Int(block_idx.y), Int(block_idx.x))
    var c_warp_tile = c_block_tile.tile[WM, WN](Int(warp_m), Int(warp_n))

    # NOTE: these numbers are hardcoded based on register fragments shapes
    # these should be derived

    alias output_thread_layout = Layout.col_major(16, 4)

    copy_local_to_dram[output_thread_layout, thread_scope = ThreadScope.WARP](
        c_warp_tile.vectorize[1, 4](), c_reg_tile.vectorize[1, 4](), c
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](warps * WARP_SIZE)
)
fn async_gmem_to_local_kernel[
    dtype: DType,
    layout: Layout,
    tile_layout: Layout,
    simd_width: Int,
    thread_layout: Layout,
    warps: Int,
](
    input_tensor: LayoutTensor[dtype, layout, MutableAnyOrigin],
    output_tensor: LayoutTensor[dtype, layout, MutableAnyOrigin],
):
    alias thread_scope = ThreadScope.WARP if warps > 1 else ThreadScope.BLOCK
    var warp_id = get_warp_id()

    var input_tile = input_tensor.tile[
        tile_layout.shape[0].value(), tile_layout.shape[1].value()
    ](warp_id, 0)

    var output_tile = output_tensor.tile[
        tile_layout.shape[0].value(), tile_layout.shape[1].value()
    ](warp_id, 0)

    alias row_repeats = input_tile.shape[0]() // thread_layout.shape[0].value()
    alias col_repeats = (
        input_tile.shape[1]() // (thread_layout.shape[1].value() * simd_width)
    ) * simd_width

    alias local_tensor_layout = layout.row_major(row_repeats, col_repeats)

    var local_tensor = LayoutTensor[
        dtype,
        local_tensor_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()

    batched_copy_dram_to_local[thread_layout, thread_scope=thread_scope,](
        local_tensor.vectorize[1, simd_width](),
        input_tile.vectorize[1, simd_width](),
    )

    var out_frag = output_tile.vectorize[1, simd_width]().distribute[
        thread_layout
    ](thread_idx.x)
    var local_frag = local_tensor.vectorize[1, simd_width]()

    alias M = local_frag.shape[0]()
    alias N = local_frag.shape[1]()

    @parameter
    for i in range(M):

        @parameter
        for j in range(N):
            alias frag_type = __type_of(out_frag[i, j])
            var f = rebind[frag_type](local_frag[i, j])
            out_frag[i, j] = f


fn test_async_gmem_to_local[
    BM: Int, BK: Int, simd_width: Int, warps: Int
](ctx: DeviceContext) raises:
    var device_buffer = ctx.enqueue_create_buffer[DType.bfloat16](BM * BK)
    var device_buffer_2 = ctx.enqueue_create_buffer[DType.bfloat16](BM * BK)

    with device_buffer.map_to_host() as host_buffer:
        for i in range(BM * BK):
            var val = random_si64(0, 20)
            host_buffer[i] = val.cast[DType.bfloat16]()

        ctx.enqueue_copy(device_buffer, host_buffer)

        var device_tensor = LayoutTensor[
            DType.bfloat16,
            Layout.row_major(BM, BK),
            MutableAnyOrigin,
        ](device_buffer)

        var device_tensor_2 = LayoutTensor[
            DType.bfloat16,
            Layout.row_major(BM, BK),
            MutableAnyOrigin,
        ](device_buffer_2)

        alias kernel = async_gmem_to_local_kernel[
            DType.bfloat16,
            Layout.row_major(BM, BK),
            Layout.row_major(BM // warps, BK),
            simd_width,
            Layout.row_major(16, 4),
            warps,
        ]

        ctx.enqueue_function_checked[kernel, kernel](
            device_tensor,
            device_tensor_2,
            grid_dim=(1,),
            block_dim=(WARP_SIZE * warps,),
        )

        ctx.synchronize()

        with device_buffer_2.map_to_host() as host_buffer2:
            for i in range(BM):
                for j in range(BK):
                    assert_equal(
                        host_buffer2[i * BK + j], host_buffer[i * BK + j]
                    )


def test_warp_specialization_amd[
    M: Int,
    N: Int,
    K: Int,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    WK: Int,
    producer_warps_a: Int,
    producer_warps_b: Int,
    consumer_warps: Int,
    pipeline_stages: Int = 1,
](ctx: DeviceContext):
    var device_a = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var device_b = ctx.enqueue_create_buffer[DType.bfloat16](N * K)
    var device_c = ctx.enqueue_create_buffer[DType.float32](M * N)
    var device_c_ref = ctx.enqueue_create_buffer[DType.float32](M * N)

    with device_a.map_to_host() as host_a, device_b.map_to_host() as host_b, device_c.map_to_host() as host_c, device_c_ref.map_to_host() as host_c_ref:
        host_c = host_c.enqueue_fill(0)
        host_c_ref = host_c_ref.enqueue_fill(0)

        ctx.synchronize()

        for i in range(M * K):
            var val = random_si64(0, 20)
            host_a[i] = val.cast[DType.bfloat16]()

        for i in range(K * N):
            var val = random_si64(0, 20)
            host_b[i] = val.cast[DType.bfloat16]()

        ctx.enqueue_copy(device_a, host_a)
        ctx.enqueue_copy(device_b, host_b)
        ctx.enqueue_copy(device_c, host_c)
        ctx.enqueue_copy(device_c_ref, host_c_ref)
        ctx.synchronize()

        var a_device_tensor = LayoutTensor[
            DType.bfloat16,
            Layout.row_major(M, K),
        ](device_a)

        var b_device_tensor = LayoutTensor[
            DType.bfloat16, Layout.row_major(N, K)
        ](device_b)

        var c_device_tensor = LayoutTensor[
            DType.float32, Layout.row_major(M, N)
        ](device_c)

        var c_device_ref_tensor = LayoutTensor[
            DType.float32, Layout.row_major(M, N)
        ](device_c_ref)

        var global_c_device_tensor = c_device_tensor.address_space_cast[
            AddressSpace.GLOBAL
        ]()
        var global_a_device_tensor = a_device_tensor.address_space_cast[
            AddressSpace.GLOBAL
        ]()
        var global_b_device_tensor = b_device_tensor.address_space_cast[
            AddressSpace.GLOBAL
        ]()

        alias kernel = test_producer_consumer[
            a_device_tensor.dtype,
            c_device_tensor.dtype,
            a_device_tensor.layout,
            b_device_tensor.layout,
            c_device_tensor.layout,
            BM,
            BN,
            BK,
            WM,
            WN,
            WK,
            producer_warps_a=producer_warps_a,
            producer_warps_b=producer_warps_b,
            consumer_warps=consumer_warps,
        ]

        ctx.enqueue_function_checked[kernel, kernel](
            global_a_device_tensor,
            global_b_device_tensor,
            global_c_device_tensor,
            grid_dim=(1, 1),
            block_dim=(
                WARP_SIZE
                * (producer_warps_a + producer_warps_b + consumer_warps)
            ),
        )

        vendor_blas.matmul(
            ctx,
            c_device_ref_tensor,
            a_device_tensor,
            b_device_tensor,
            c_row_major=True,
            transpose_b=True,
        )

        ctx.synchronize()

        ctx.enqueue_copy(host_c, device_c)
        ctx.enqueue_copy(host_c_ref, device_c_ref)
        ctx.synchronize()

        var errors = 0
        for i in range(M * N):
            # print(i // N, i % N, c_host.tensor.data[i], c_host_ref.tensor.data[i])
            if host_c[i] != host_c_ref[i]:  # and errors < 100:
                print(i // N, i % N, host_c[i], host_c_ref[i])
                errors += 1

            # if errors < 100:
            #     print(i//N, i%N, host_c[i], host_c_ref[i])

        assert_equal(errors, 0)


def main():
    with DeviceContext() as ctx:
        print("Running AMD GPU batched copy tests")
        test_async_gmem_to_local[BM=64, BK=64, simd_width=8, warps=1](ctx)
        test_async_gmem_to_local[BM=32, BK=64, simd_width=8, warps=1](ctx)
        test_async_gmem_to_local[BM=64, BK=64, simd_width=4, warps=1](ctx)
        test_async_gmem_to_local[BM=128, BK=64, simd_width=8, warps=2](ctx)
        print("==== AMD GPU batched copy tests passed ====")

        print("Running AMD Warp Specialization Tests")
        test_warp_specialization_amd[
            64, 64, 64, 64, 64, 64, 32, 32, 64, 2, 2, 4
        ](ctx)

        test_warp_specialization_amd[
            32, 32, 32, 32, 32, 32, 16, 16, 32, 2, 2, 2
        ](ctx)
        print("==== AMD Warp Specialization Tests passed ====")
