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
"""Defines the B200 bulk transfer atom: `cp.async.bulk` from global to
shared memory, completing on an mbarrier, and the TMA engine that runs it.

`CpAsyncBulk` wraps
`cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes`: one thread
copies a contiguous run of bytes, and the copy arrives on the mbarrier the
atom is built with, as transaction bytes. `TmaEngine`'s `Barrier` route
arms a phase with `mbarrier.arrive.expect_tx` and waits on it with
`mbarrier.try_wait.parity`.

You can import these APIs from the `layout.atoms.b200.cp_async` module:

```mojo
from layout.atoms.b200.cp_async import CpAsyncBulk, TmaEngine
```
"""

from layout import ImmTileTensor, MutTileTensor
from layout.atoms import Arrives, HasBarrier, TransferAtom, vector_map
from layout.atoms.mbarrier import (
    MBarrierExpectTx,
    MBarrierRef,
    MBarrierTryWait,
)
from layout.tensor_engine import DefaultEngine, TensorEngine
from layout.layout import TensorLayout

from max.gpu.memory import cp_async_bulk_shared_cluster_global

from std.sys import size_of


struct TmaArrivals(Arrives):
    """Each bulk copy arrives on its atom's mbarrier as transaction bytes;
    an arm sets the bytes a phase expects."""

    comptime Arm = MBarrierExpectTx
    """`mbarrier.arrive.expect_tx`."""

    comptime Wait = MBarrierTryWait
    """`mbarrier.try_wait.parity`."""


struct TmaEngine(HasBarrier):
    """The Tensor Memory Accelerator, which runs bulk copies. Its copies
    complete out of order, as transaction bytes on an mbarrier."""

    comptime in_order = False
    """Copies may complete in any order."""

    comptime Barrier = TmaArrivals
    """Every copy arrives on its atom's mbarrier."""


@fieldwise_init
struct CpAsyncBulk[dt: DType, bytes: Int, barrier_origin: MutOrigin](
    TransferAtom
):
    """`cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes`: one
    thread copies `bytes` contiguous bytes from global to shared memory,
    completing on `barrier`.

    Parameters:
        dt: The element type.
        bytes: The bytes one issue copies, a multiple of 16 and a power of
            two.
        barrier_origin: The origin of the barrier.
    """

    comptime shape_values = ParameterList.of[
        Self.bytes // size_of[Self.dt]()
    ]().values
    """One issue moves a run of `bytes` bytes of elements."""
    comptime AsyncEngine = TmaEngine
    """The copy's bytes arrive on the atom's mbarrier."""
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

    var barrier: MBarrierRef[Self.barrier_origin]
    """The mbarrier the copy's bytes arrive on."""

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
        """Issues the copy; its bytes arrive on `barrier`.

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
        comptime assert T == Self.dt, "CpAsyncBulk: operand dtype"
        comptime assert (
            Self.bytes % 16 == 0 and Self.bytes & (Self.bytes - 1) == 0
        ), "CpAsyncBulk: bytes is a power of two and a multiple of 16"
        comptime assert (
            type_of(dst).address_space == .SHARED
        ), "CpAsyncBulk: dst is shared"
        cp_async_bulk_shared_cluster_global(
            dst.unsafe_ptr().unsafe_address_space_cast[.SHARED](),
            src.unsafe_ptr(),
            Int32(Self.bytes),
            self.barrier,
        )
