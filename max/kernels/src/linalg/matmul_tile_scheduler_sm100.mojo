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

from sys import sizeof

from utils.index import Index, IndexList
from gpu.memory import AddressSpace, fence_async_view_proxy
from gpu.id import block_idx, lane_id
from layout.tma_async import PipelineState, SharedMemBarrier
from gpu.cluster import (
    elect_one_sync,
    clusterlaunchcontrol_try_cancel,
    clusterlaunchcontrol_query_cancel_get_first_ctaid_v4,
    clusterlaunchcontrol_query_cancel_is_canceled,
)
from sys._assembly import inlined_assembly
from sys import _RegisterPackType


@fieldwise_init
@register_passable("trivial")
struct WorkInfo(Copyable, Movable, Stringable, Writable):
    # Coordinates in output matrix
    var m: UInt32
    var n: UInt32
    # Starting k index in A and B for the output tile's mma.
    var k_start: UInt32
    # Whether work tile is completely OOB.
    var is_valid_tile: Bool

    @always_inline
    fn is_valid(self) -> Bool:
        return self.is_valid_tile

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "(",
            self.m,
            ", ",
            self.n,
            ", ",
            self.k_start,
            ", ",
            self.is_valid_tile,
            ")",
        )


@register_passable("trivial")
struct TileScheduler[
    num_stages: Int,
    cluster_shape: IndexList[3] = Index(1, 1, 1),
]:
    alias cluster_size = cluster_shape[0] * cluster_shape[1] * cluster_shape[2]
    var clc_response: UnsafePointer[
        UInt128, address_space = AddressSpace.SHARED
    ]
    var full_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ]
    var empty_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ]

    @always_inline
    fn __init__(
        out self,
        clc_response_ptr: UnsafePointer[
            UInt128, address_space = AddressSpace.SHARED
        ],
        full_mbar_ptr: UnsafePointer[
            SharedMemBarrier, address_space = AddressSpace.SHARED
        ],
        empty_mbar_ptr: UnsafePointer[
            SharedMemBarrier, address_space = AddressSpace.SHARED
        ],
    ):
        self.clc_response = clc_response_ptr
        self.full_mbar = full_mbar_ptr
        self.empty_mbar = empty_mbar_ptr

    @always_inline
    @staticmethod
    fn work_info_from_clc_response(
        result: UnsafePointer[UInt128, address_space = AddressSpace.SHARED]
    ) -> WorkInfo:
        alias asm = """{
            .reg .pred p1;
            .reg .b128 clc_result;
            ld.shared.b128 clc_result, [$4];
            clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;
            selp.u32 $3, 1, 0, p1;
            @p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {$0, $1, $2, _}, clc_result;
        }"""
        var ret_val = inlined_assembly[
            asm,
            _RegisterPackType[UInt32, UInt32, UInt32, UInt32],
            has_side_effect=True,
            constraints="=r,=r,=r,=r,l",
        ](UInt32(Int(result)))

        fence_async_view_proxy()

        return WorkInfo(
            m=ret_val[0],
            n=ret_val[1],
            k_start=ret_val[2],
            is_valid_tile=Bool(ret_val[3] == 1),
        )

    @always_inline
    fn initial_work_info(self) -> WorkInfo:
        return WorkInfo(
            block_idx.x,
            block_idx.y,
            block_idx.z,
            is_valid_tile=False,
        )

    @always_inline
    fn fetch_next_work(
        self,
        work_info: WorkInfo,
        consumer_state: PipelineState[num_stages],
    ) -> WorkInfo:
        self.full_mbar[consumer_state.index()].wait(consumer_state.phase())
        var work_tile = self.work_info_from_clc_response(
            self.clc_response + consumer_state.index()
        )
        # Only cta 0 in a cluster is used for scheduling.
        self.empty_mbar[consumer_state.index()].arrive_cluster(0)
        return work_tile

    @always_inline
    fn advance_to_next_work(
        self,
        mut clc_state: PipelineState[num_stages],
    ) -> PipelineState[num_stages]:
        alias multicast = True if Self.cluster_size > 1 else False
        var lane_id = lane_id()
        var pred: UInt32 = 1 if lane_id < Self.cluster_size else 0
        self.empty_mbar[clc_state.index()].wait(clc_state.phase())
        self.full_mbar[clc_state.index()].arrive_and_expect_bytes(
            sizeof[UInt128](),
            UInt32(lane_id),
            pred,
        )

        if elect_one_sync():
            clusterlaunchcontrol_try_cancel[multicast=multicast](
                self.clc_response + clc_state.index(),
                (self.full_mbar + clc_state.index()).bitcast[Int64](),
            )

        return clc_state.next()
