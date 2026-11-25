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
"""Defines the NVIDIA mbarrier sync atoms that barrier routes share.

`MBarrierTryWait` waits for a phase of an mbarrier with
`mbarrier.try_wait.parity`, the wait of every route that arrives on an
mbarrier, whether its producer armed the phase or sealed it.
`MBarrierExpectTx` arms a phase with `mbarrier.arrive.expect_tx`, for
engines whose issues arrive on the barrier as transaction bytes. Both bind
the barrier traits' operands to the mbarrier's: `MBarrierRef`, a
shared-memory pointer to a `SharedMemBarrier`, an `MBarrierParity` and a
transaction byte count.

You can import these APIs from the `layout.atoms.mbarrier` module:

```mojo
from layout.atoms.mbarrier import MBarrierExpectTx, MBarrierParity
from layout.atoms.mbarrier import MBarrierRef, MBarrierTryWait
```
"""

from layout.atoms import ArmAtom, BarrierPhase, BarrierWaitAtom
from layout.tma_async import SharedMemBarrier

from max.gpu.sync import (
    mbarrier_arrive_expect_tx_shared,
    mbarrier_try_wait_parity_shared,
)

# CUTLASS's suspend-time hint for `mbarrier.try_wait`, in nanoseconds. The
# wait still returns only once the phase completes.
comptime _SUSPEND_HINT_NS = 10_000_000

comptime MBarrierRef[origin: MutOrigin] = Pointer[
    SharedMemBarrier, origin, address_space=.SHARED
]
"""An mbarrier in shared memory, of `origin`.

Parameters:
    origin: The barrier's origin.
"""


__enum MBarrierParity(BarrierPhase, Equatable, ImplicitlyCopyable):
    """The parity of an mbarrier phase. An mbarrier's phases alternate
    between even and odd, starting with an even phase."""

    # TODO(jtodd): Use explicit 0/1 cases for enum when available
    # A case's discriminant is its parity bit, which the wait passes to the
    # instruction; comparing cases instead adds a select at run time.
    case even
    """An even phase."""
    case odd
    """An odd phase."""

    comptime start: Self = .even
    """An initialized mbarrier is in its first phase, which is even."""

    def next(self) -> Self:
        """Returns the parity of the phase after this one.

        Returns:
            `odd` for an even phase and `even` for an odd one.
        """
        comptime assert _PARITY_IS_DISCRIMINANT, _PARITY_ENCODING_MSG
        return .odd if self._get_enum_discriminant() == 0 else .even


# An `__enum` cannot choose its case values yet, so this pins the encoding
# the wait passes to the instruction.
comptime _PARITY_IS_DISCRIMINANT = (
    MBarrierParity.even._get_enum_discriminant() == 0
    and MBarrierParity.odd._get_enum_discriminant() == 1
)
comptime _PARITY_ENCODING_MSG = "MBarrierParity must store even as 0, odd as 1"


@fieldwise_init
struct MBarrierTryWait(BarrierWaitAtom):
    """`mbarrier.try_wait.parity`: waits for a phase of an mbarrier."""

    comptime Ref[origin: MutOrigin] = MBarrierRef[origin]
    """The mbarrier the wait reads.

    Parameters:
        origin: The barrier's origin.
    """

    comptime Phase = MBarrierParity
    """The phase's parity."""

    @always_inline
    def __call__[
        origin: MutOrigin, //
    ](self, barrier: MBarrierRef[origin], phase: MBarrierParity):
        """Blocks until the mbarrier completes a phase of parity `phase`.

        Parameters:
            origin: The barrier's origin.

        Args:
            barrier: The mbarrier.
            phase: The phase's parity.
        """
        comptime assert _PARITY_IS_DISCRIMINANT, _PARITY_ENCODING_MSG
        mbarrier_try_wait_parity_shared(
            barrier, Int32(phase._get_enum_discriminant()), _SUSPEND_HINT_NS
        )


@fieldwise_init
struct MBarrierExpectTx(ArmAtom):
    """`mbarrier.arrive.expect_tx`: arrives on an mbarrier and expects
    transaction bytes in its current phase."""

    comptime Ref[origin: MutOrigin] = MBarrierRef[origin]
    """The mbarrier the arm updates.

    Parameters:
        origin: The barrier's origin.
    """

    comptime Amount = Int
    """The transaction bytes the phase expects."""

    @always_inline
    def __call__[
        origin: MutOrigin, //
    ](self, barrier: MBarrierRef[origin], amount: Int):
        """Arrives on the mbarrier and expects `amount` more transaction
        bytes in its current phase.

        Parameters:
            origin: The barrier's origin.

        Args:
            barrier: The mbarrier.
            amount: The bytes the phase expects.
        """
        mbarrier_arrive_expect_tx_shared(barrier, Int32(amount))
