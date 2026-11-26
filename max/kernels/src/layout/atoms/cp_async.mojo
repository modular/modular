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
"""Defines the transfer atom for NVIDIA's `cp.async` global-to-shared copy,
and the async engine that runs it.

`CpAsync` wraps `cp.async`: each thread copies 4, 8 or 16 bytes from global
to shared memory. `CpAsyncEngine` completes the copies two ways. Its
`Groups` route is `cp.async.commit_group`, which closes a group of the
thread's copies, and `cp.async.wait_group`, which waits until at most a
given number of groups are outstanding. Its `Barrier` route is
`cp.async.mbarrier.arrive`, which seals the thread's copies onto an
mbarrier that any thread can wait on.

The bulk copy, `cp.async.bulk`, is B200-specific and lives in
`layout.atoms.b200.cp_async`.

You can import these APIs from the `layout.atoms.cp_async` module:

```mojo
from layout.atoms.cp_async import CpAsync, CpAsyncEngine
```
"""

from layout import ImmTileTensor, MutTileTensor
from layout.atoms import (
    CommitAtom,
    CounterWaitAtom,
    Grouped,
    HasBarrier,
    HasGroups,
    SealsToBarrier,
    TransferAtom,
    vector_map,
)
from layout.atoms.mbarrier import MBarrierRef, MBarrierTryWait
from layout.tensor_engine import DefaultEngine, TensorEngine
from layout.layout import TensorLayout

from max.gpu.memory import (
    async_copy,
    async_copy_commit_group,
    async_copy_wait_all,
    async_copy_wait_group,
)
from max.gpu.sync import async_copy_arrive

from std.sys import size_of


@fieldwise_init
struct CpAsyncCommitGroup(CommitAtom, Defaultable, not Deinitable):
    """`cp.async.commit_group`: closes the thread's open `cp.async` group."""

    @always_inline
    def __call__(deinit self):
        """Closes the open group."""
        async_copy_commit_group()


@fieldwise_init
struct CpAsyncWaitGroup(CounterWaitAtom):
    """`cp.async.wait_group`: waits on the thread's committed `cp.async`
    groups."""

    comptime max_pending = Int.MAX
    """Any number of committed groups can stay outstanding."""

    @always_inline
    def __call__[pending: Int = 0](self):
        """Blocks until at most `pending` committed groups are outstanding,
        with `cp.async.wait_all` when `pending` is 0.

        Parameters:
            pending: The groups that may stay outstanding.
        """
        comptime assert (
            0 <= pending <= Self.max_pending
        ), "cp.async.wait_group: pending is out of range"
        comptime if pending == 0:
            async_copy_wait_all()
        else:
            async_copy_wait_group(Int32(pending))


struct CpAsyncGroups(Grouped):
    """`cp.async` groups: the issuing thread commits a group and waits on
    committed groups."""

    comptime Commit = CpAsyncCommitGroup
    """`cp.async.commit_group`."""

    comptime Wait = CpAsyncWaitGroup
    """`cp.async.wait_group`."""


@fieldwise_init
struct CpAsyncMBarrierArrive[origin: MutOrigin](CommitAtom, not Deinitable):
    """`cp.async.mbarrier.arrive`: arrives on an mbarrier once every
    `cp.async` the thread issued before it has completed.

    Parameters:
        origin: The origin of the barrier.
    """

    var barrier: MBarrierRef[Self.origin]
    """The mbarrier the seal arrives on."""

    @always_inline
    def __call__(deinit self):
        """Seals the thread's earlier copies onto `barrier`."""
        async_copy_arrive(self.barrier)


struct CpAsyncArrivals(SealsToBarrier):
    """`cp.async` copies sealed onto an mbarrier: any thread waits on it."""

    comptime Seal[origin: MutOrigin] = CpAsyncMBarrierArrive[origin]
    """`cp.async.mbarrier.arrive`, built with its barrier.

    Parameters:
        origin: The barrier's origin.
    """

    comptime Wait = MBarrierTryWait
    """`mbarrier.try_wait.parity`."""


struct CpAsyncEngine(HasBarrier, HasGroups):
    """The unit that runs `cp.async`. Its copies complete out of order, as
    groups or sealed onto an mbarrier."""

    comptime in_order = False
    """Copies may complete in any order."""

    comptime Groups = CpAsyncGroups
    """Commit and wait on groups of the thread's copies."""

    comptime Barrier = CpAsyncArrivals
    """Seal the thread's copies onto an mbarrier."""


@fieldwise_init
struct CpAsync[dt: DType, bytes: Int](TransferAtom):
    """`cp.async`: this thread copies `bytes` contiguous bytes from global
    to shared memory.

    Parameters:
        dt: The element type.
        bytes: The bytes one issue copies: 4, 8 or 16.
    """

    comptime shape_values = ParameterList.of[
        Self.bytes // size_of[Self.dt]()
    ]().values
    """One issue moves `bytes` bytes of elements."""
    comptime AsyncEngine = CpAsyncEngine
    """The copy completes through a `cp.async` group or an mbarrier."""
    comptime dtype = Self.dt
    """The element type the instruction moves."""
    comptime SrcEngine = DefaultEngine[element_width=1]
    """The engine of the memory the instruction reads."""
    comptime DstEngine = DefaultEngine[element_width=1]
    """The engine of the memory the instruction writes."""
    comptime src_map = vector_map(
        Self.bytes // size_of[Self.dt](), size_of[Self.dt]()
    )
    """Where each index reads the source."""
    comptime dst_map = vector_map(
        Self.bytes // size_of[Self.dt](), size_of[Self.dt]()
    )
    """Where each index writes the destination."""

    @always_inline
    def __call__[
        T: DType,
        LD: TensorLayout,
        ED: TensorEngine,
        LS: TensorLayout,
        ES: TensorEngine,
    ](
        self,
        dst: MutTileTensor[T, LD, Engine=ED, ...],
        src: ImmTileTensor[T, LS, Engine=ES, ...],
    ):
        """Issues the copy without waiting for it.

        Parameters:
            T: The operands' element type.
            LD: The shared view's layout.
            ED: The shared view's engine.
            LS: The global view's layout.
            ES: The global view's engine.

        Args:
            dst: Shared memory, from the copy's first element.
            src: Global memory, from the copy's first element.
        """
        comptime assert T == Self.dt, "CpAsync: operand dtype"
        comptime assert (
            Self.bytes == 4 or Self.bytes == 8 or Self.bytes == 16
        ), "CpAsync: copies 4, 8 or 16 bytes"
        comptime assert (
            type_of(src).address_space == .GLOBAL
        ), "CpAsync: src is global"
        comptime assert (
            type_of(dst).address_space == .SHARED
        ), "CpAsync: dst is shared"
        async_copy[Self.bytes](
            src.unsafe_ptr().unsafe_address_space_cast[.GLOBAL](),
            dst.unsafe_ptr().unsafe_address_space_cast[.SHARED](),
        )
