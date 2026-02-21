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


from time.time import global_perf_counter_ns
from gpu import block_idx, thread_idx, WARP_SIZE
from gpu.host import DeviceContext
from gpu import sm_id


comptime MatmulWarpSpecializationWorkSpaceManager[
    max_entries_per_warp: UInt32
] = BlackwellWarpProfilingWorkspaceManager[1, 1, 1, 4, max_entries_per_warp]

comptime MatmulProfileWarp[
    warp_role: UInt32, max_entries_per_warp: UInt32
] = BlackwellProfileWarp[
    MatmulWarpSpecializationWorkSpaceManager[max_entries_per_warp](),
    warp_role=warp_role,
]


@fieldwise_init
struct BlackwellWarpProfilingWorkspaceManager[
    load_warps: UInt32,
    mma_warps: UInt32,
    scheduler_warps: UInt32,
    epilogue_warps: UInt32,
    max_entries_per_warp: UInt32,
](TrivialRegisterPassable):
    """
    This struct manages the profiling workspace. The workspaces consists of equal sized chunks, the total number of
    which is equal to the total number of active SMs. Each SM chunk consists of sequences of entries, with a maximum
    number of entries per warp role.

    Template Parameters:
        load_warps: Number of warps specialized for load operations
        mma_warps: Number of warps specialized for matrix multiply-accumulate operations
        scheduler_warps: Number of warps specialized for scheduling operations
        epilogue_warps: Number of warps specialized for epilogue operations
        max_entries_per_warp: Maximum number of entries per warp (common across all warp roles)
    """

    # load, scheduler, mma, epilogue
    comptime total_warp_roles = 4

    # how many values will be recorded per entry
    comptime total_data_points = 7

    # this header shows what each value in an entry symbolizes in a csv friendly format
    comptime header = "time_start,time_end,sm_id,block_idx_x,block_idx_y,role,entry_idx\n"

    comptime sm_count = B200.sm_count
    comptime entries_per_sm = Self.total_warp_roles * Self.max_entries_per_warp

    @staticmethod
    @parameter
    fn _get_warp_count[warp_role: UInt32]() -> UInt32:
        comptime if warp_role == 0:
            return Self.load_warps
        elif warp_role == 1:
            return Self.scheduler_warps
        elif warp_role == 2:
            return Self.mma_warps
        else:
            return Self.epilogue_warps

    @staticmethod
    @parameter
    fn _calculate_entries_before_role[warp_role: UInt32]() -> UInt32:
        return warp_role * Self.max_entries_per_warp

    @staticmethod
    @always_inline
    fn _get_workspace_offset[
        warp_role: UInt32
    ](sm_idx: UInt32, entry_idx: UInt32) -> UInt32:
        var sm_length = Self.total_data_points * Self.entries_per_sm

        return (
            (sm_idx * sm_length)
            + (
                Self._calculate_entries_before_role[warp_role]()
                * Self.total_data_points
            )
            + (entry_idx * Self.total_data_points)
        )

    @staticmethod
    @parameter
    fn _calculate_buffer_length() -> UInt32:
        return (
            UInt32(Self.sm_count) * Self.entries_per_sm * Self.total_data_points
        )

    @staticmethod
    @always_inline
    fn get_workspace(
        ctx: DeviceContext,
    ) raises -> Span[UInt64, MutAnyOrigin]:
        var length = Int(Self._calculate_buffer_length())
        var device_buffer = ctx.enqueue_create_buffer[DType.uint64](length)
        device_buffer.enqueue_fill(0)
        return Span[UInt64, MutAnyOrigin](
            ptr=device_buffer.unsafe_ptr(),
            length=length,
        )

    @staticmethod
    @always_inline
    fn write_to_workspace[
        warp_role: UInt32
    ](
        sm_idx: UInt32,
        entry_idx: UInt32,
        workspace: Span[UInt64, MutAnyOrigin],
        timeline: Tuple[UInt64, UInt64],
    ):
        comptime total_threads = UInt32(WARP_SIZE) * Self._get_warp_count[
            warp_role
        ]()

        var start_idx = Self._get_workspace_offset[warp_role](sm_idx, entry_idx)

        if UInt32(thread_idx.x) % total_threads == 0:
            workspace[start_idx] = timeline[0]
            workspace[start_idx + 1] = timeline[1]
            workspace[start_idx + 2] = UInt64(sm_idx)
            workspace[start_idx + 3] = UInt64(block_idx.x)
            workspace[start_idx + 4] = UInt64(block_idx.y)
            workspace[start_idx + 5] = UInt64(warp_role)
            workspace[start_idx + 6] = UInt64(entry_idx)

    @staticmethod
    @always_inline
    fn dump_workspace_as_csv(
        ctx: DeviceContext,
        workspace: Span[UInt64, MutAnyOrigin],
        filename: StaticString,
    ) raises:
        var length = Int(Self._calculate_buffer_length())
        var host_buffer = ctx.enqueue_create_host_buffer[DType.uint64](length)
        ctx.enqueue_copy(host_buffer, workspace.unsafe_ptr())
        ctx.synchronize()

        var host_span = host_buffer.as_span()

        var entries = len(host_buffer) // (Self.total_data_points)
        with open(filename + ".csv", "w") as f:
            f.write(Self.header)
            for entry in range(entries):
                var start = entry * Self.total_data_points
                f.write(
                    ", ".join(
                        [
                            String(x)
                            for x in host_span[
                                start : start + Self.total_data_points
                            ]
                        ]
                    )
                )
                f.write("\n")


struct BlackwellProfileWarp[
    load_warps: UInt32,
    mma_warps: UInt32,
    scheduler_warps: UInt32,
    epilogue_warps: UInt32,
    max_entries_per_warp: UInt32,
    //,
    WorkspaceManager: BlackwellWarpProfilingWorkspaceManager[
        load_warps,
        mma_warps,
        scheduler_warps,
        epilogue_warps,
        max_entries_per_warp,
    ],
    warp_role: UInt32 = 0,
](ImplicitlyCopyable):
    """
    This struct calculates execution time for a warp/s,
    and writes a single entry to the workspace.
    """

    comptime enable_profiling = Self.max_entries_per_warp > 0

    var timeline: Tuple[UInt64, UInt64]
    var workspace: Span[UInt64, MutAnyOrigin]

    # which entry is going to be written to the workspace for this warp
    var entry_idx: UInt32

    @always_inline
    fn __init__(
        out self,
        workspace: Span[UInt64, MutAnyOrigin],
        entry_idx: UInt32,
    ):
        self.timeline = (0, 0)
        self.workspace = workspace
        self.entry_idx = entry_idx

    @always_inline
    fn __enter__(mut self):
        comptime if Self.enable_profiling:
            self.timeline[0] = global_perf_counter_ns()

    @always_inline
    fn __exit__(mut self):
        comptime if Self.enable_profiling:
            self.timeline[1] = global_perf_counter_ns()
            Self.WorkspaceManager.write_to_workspace[Self.warp_role](
                UInt32(sm_id()),
                self.entry_idx,
                self.workspace,
                self.timeline,
            )
