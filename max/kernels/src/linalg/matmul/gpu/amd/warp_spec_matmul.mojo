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
from utils import IndexList
from gpu import MAX_THREADS_PER_BLOCK_METADATA
from utils import StaticTuple
from layout.layout import blocked_product


@parameter
fn pipeline_layout[layout: Layout, pipeline_stages: Int]() -> Layout:
    constrained[layout.rank() == 2]()
    return blocked_product(
        layout, Layout.row_major(1, pipeline_stages), coalesce_output=True
    )


# NOTE: This is a hardcoded pipeline but in reality this should be struct
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        (producer_warps_a + producer_warps_b + consumer_warps) * WARP_SIZE
    )
)
fn warp_specialized_matmul[
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
    alias K = a.shape[1]()

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
    constrained[
        consumer_warps >= producer_warps_a
        and consumer_warps >= producer_warps_b
    ]()
    constrained[
        consumer_warps.is_power_of_two(), "consumer_warps must be a power of 2"
    ]()

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
    alias smem_layout_a = Layout.row_major(BM, BK)
    alias smem_layout_b = Layout.row_major(BN, BK)

    alias pipelined_layout_a = pipeline_layout[smem_layout_a, pipeline_stages]()
    alias pipelined_layout_b = pipeline_layout[smem_layout_b, pipeline_stages]()

    alias SmemBufferTypeA = SharedMemoryBuffer[
        in_type, pipelined_layout_a, pipeline_stages, BM, BK, WM, WK
    ]
    alias SmemBufferTypeB = SharedMemoryBuffer[
        in_type, pipelined_layout_b, pipeline_stages, BN, BK, WN, WK
    ]

    var smem_buffer_a = SmemBufferTypeA()
    var smem_buffer_b = SmemBufferTypeB()

    # NOTE: Every thread needs access to the ring buffer
    var ring_buffer = RingBuffer[
        SmemBufferTypeA,
        SmemBufferTypeB,
        consumer_warps,
    ](smem_buffer_a, smem_buffer_b)

    barrier()  # NOTE: probably not nessecary but I saw it in the HF code around the same point

    alias tile_count = K // BK
    alias warps_processed_per_producer_a = Int(
        m_warps_per_block // producer_warps_a
    )
    alias warps_processed_per_producer_b = Int(
        n_warps_per_block // producer_warps_b
    )

    # NOTE: the 2 producer blocks are almost identical, you can proabbly make this a
    # a function
    if role is ThreadRole.PRODUCER:
        if warp_id < UInt(producer_warps_a):
            # NOTE: If there is a way to hide the phases that would be great, maybe the ringbuffer
            # handles the thread specific phase, or its encapsulated in the ThreadRole Struct.

            var phases = InlineArray[
                Int, pipeline_stages * warps_processed_per_producer_a
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

            @parameter
            for tile_num in range(tile_count):
                alias stage = tile_num % pipeline_stages
                var a_tile = a.tile[BM, BK](block_idx.x, tile_num)

                # NOTE: producers and consumers can process more than one tile this loop
                # makes sure this is possible
                @parameter
                for local_tile_count in range(warps_processed_per_producer_a):
                    var warp_tile_idx = (
                        Int(warp_id) + local_tile_count * producer_warps_a
                    )

                    # NOTE phase_idx needs to be known at compile time otherwise
                    # the compiler will not place phases
                    alias phase_idx = warps_processed_per_producer_a * stage + local_tile_count

                    ref phase = phases[phase_idx]

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

        else:
            var relative_warp_id = warp_id - UInt(producer_warps_a)

            var phases = InlineArray[
                Int, pipeline_stages * warps_processed_per_producer_b
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

            @parameter
            for tile_num in range(tile_count):
                var b_tile = b.tile[BN, BK](block_idx.y, tile_num)
                alias stage = tile_num % pipeline_stages

                @parameter
                for local_tile_count in range(warps_processed_per_producer_b):
                    var warp_tile_idx = (
                        Int(relative_warp_id)
                        + local_tile_count * producer_warps_b
                    )
                    alias phase_idx = warps_processed_per_producer_b * stage + local_tile_count

                    ref phase = phases[phase_idx]

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

        alias warps_computed_per_consumer = total_consumer_operations // consumer_warps

        @parameter
        for i in range(tile_count):
            alias stage = i % pipeline_stages

            @parameter
            for local_tile_count in range(warps_computed_per_consumer):
                var warp_tile_idx = (
                    Int(relative_warp_id) + local_tile_count * consumer_warps
                )
                var m_warp_idx = warp_tile_idx // n_warps_per_block
                var n_warp_idx = warp_tile_idx % n_warps_per_block

                alias phase_idx = warps_computed_per_consumer * stage + local_tile_count

                ref phase_a = phases_a[phase_idx]
                ref phase_b = phases_b[phase_idx]

                tile_operator.mma(
                    ring_buffer,
                    phase_a,
                    phase_b,
                    stage,
                    m_warp_idx,
                    n_warp_idx,
                    local_tile_count,
                    i,
                )

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
    var c_block_tile = c.tile[BM, BN](Int(block_idx.x), Int(block_idx.y))
    var c_warp_tile = c_block_tile.tile[WM, WN](Int(warp_m), Int(warp_n))

    # NOTE: these numbers are hardcoded based on register fragments shapes
    # these should be derived

    alias output_thread_layout = Layout.col_major(16, 4)

    copy_local_to_dram[output_thread_layout, thread_scope = ThreadScope.WARP](
        c_warp_tile.vectorize[1, 4](), c_reg_tile.vectorize[1, 4](), c
    )
