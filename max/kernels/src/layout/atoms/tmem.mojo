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
"""Defines layout atoms for Blackwell Tensor Memory (TMEM).

`TMemLoad` and `TMemStore` wrap `tcgen05.ld` and `tcgen05.st` in the
`32x32b` shape as load and store atoms, issued through `TMemEngine`. A warp
issues each one together. In `tcgen05` addresses, thread `t` of warp
`w` reaches TMEM lane `32 (w % 4) + t`, and its value `v` is column `v` past
the address: the TMEM map fixes lane, warp and value bits, and nothing free
may reach the two lane bits the warp picks. The register side is `cols`
registers in each of the warp's 32 threads.

Neither atom waits. Each runs on an async engine with a `Counted` route:
`Tcgen05LdEngine`'s wait is `tcgen05.wait::ld` and `Tcgen05StEngine`'s is
`tcgen05.wait::st`, each of which waits for all of this thread's outstanding
loads or stores. As with
`TMemEngine.copy_to_async` and `copy_from_async`, the register operand must
be a register tile, a loaded fragment is valid only after the load wait, and
a stored fragment must not change before the store wait.

You can import these APIs from the `layout.atoms.tmem` module:

```mojo
from layout.atoms.tmem import TMemLoad, TMemStore
from layout.atoms.tmem import Tcgen05LdEngine, Tcgen05StEngine
```
"""

from layout import ImmTileTensor, MutTileTensor
from layout.atoms import (
    INDEX_LANE,
    INDEX_VALUE,
    INDEX_WARP,
    TMEM_COLUMN,
    TMEM_LANE,
    AddressEncoding,
    Counted,
    CounterWaitAtom,
    HasCount,
    LoadAtom,
    OperandMap,
    registers_map,
    StoreAtom,
    same_engine,
)
from layout.bit_layout import BitLayout, bits
from layout.tensor_engine import TensorEngine
from layout.layout import TensorLayout
from layout.tmem_engine import TMemEngine

from max.gpu.compute.arch.tcgen05 import tcgen05_load_wait, tcgen05_store_wait

from std.bit import count_trailing_zeros

comptime _WARP_LANE_BITS = 5
comptime _MAX_ASYNC_COLS = 64


def _log2(n: Int) -> Int:
    return Int(count_trailing_zeros(UInt64(n)))


def tmem_32x32b_map(cols: Int) -> OperandMap:
    """Returns the TMEM operand of a `32x32b` access of `cols` columns.

    Args:
        cols: The columns one issue moves, a power of two.

    Returns:
        The map from index bits to `tcgen05` address bits.
    """
    return {
        AddressEncoding.TMEM,
        BitLayout()
        .run(INDEX_LANE, _WARP_LANE_BITS, TMEM_LANE)
        .run(INDEX_WARP, 2, TMEM_LANE + _WARP_LANE_BITS)
        .run(INDEX_VALUE, _log2(cols), TMEM_COLUMN),
        bits(TMEM_LANE + _WARP_LANE_BITS, 2),
    }


@fieldwise_init
struct TMemLoadWait(CounterWaitAtom):
    """`tcgen05.wait::ld`: waits for all of this thread's TMEM loads."""

    comptime max_pending = 0
    """None: the wait covers every outstanding load."""

    @always_inline
    def __call__[pending: Int = 0](self):
        """Blocks until this thread's TMEM loads are done.

        Parameters:
            pending: Must be 0; the instruction waits for all of them.
        """
        comptime assert (
            0 <= pending <= Self.max_pending
        ), "tcgen05.wait::ld waits for all loads"
        tcgen05_load_wait()


@fieldwise_init
struct TMemStoreWait(CounterWaitAtom):
    """`tcgen05.wait::st`: waits for all of this thread's TMEM stores."""

    comptime max_pending = 0
    """None: the wait covers every outstanding store."""

    @always_inline
    def __call__[pending: Int = 0](self):
        """Blocks until this thread's TMEM stores are done.

        Parameters:
            pending: Must be 0; the instruction waits for all of them.
        """
        comptime assert (
            0 <= pending <= Self.max_pending
        ), "tcgen05.wait::st waits for all stores"
        tcgen05_store_wait()


struct TMemLoads(Counted):
    """`tcgen05.ld` issues: only the issuing thread waits, for all of them."""

    comptime Wait = TMemLoadWait
    """`tcgen05.wait::ld`."""


struct TMemStores(Counted):
    """`tcgen05.st` issues: only the issuing thread waits, for all of them."""

    comptime Wait = TMemStoreWait
    """`tcgen05.wait::st`."""


struct Tcgen05LdEngine(HasCount):
    """The unit that runs `tcgen05.ld`. Its loads complete out of order, and
    only the issuing thread waits on them."""

    comptime in_order = False
    """Loads may complete in any order."""

    comptime Count = TMemLoads
    """Wait for the thread's loads."""


struct Tcgen05StEngine(HasCount):
    """The unit that runs `tcgen05.st`. Its stores complete out of order,
    and only the issuing thread waits on them."""

    comptime in_order = False
    """Stores may complete in any order."""

    comptime Count = TMemStores
    """Wait for the thread's stores."""


@fieldwise_init
struct TMemLoad[dt: DType, cols: Int](LoadAtom):
    """`tcgen05.ld.32x32b.x{cols}`: a warp loads its TMEM quadrant's `cols`
    columns, each thread its own lane.

    Parameters:
        dt: A 4-byte element type.
        cols: The columns one issue moves, a power of two up to 64.
    """

    comptime shape_values = ParameterList.of[
        1 << _WARP_LANE_BITS, Self.cols
    ]().values
    """One issue covers 32 lanes by `cols` columns."""
    comptime dtype = Self.dt
    """The element type the instruction moves."""
    comptime AsyncEngine = Tcgen05LdEngine
    """`tcgen05.wait::ld` waits for the thread's TMEM loads."""
    comptime Engine = TMemEngine
    """The engine of the memory the instruction reads."""
    comptime mem_map = tmem_32x32b_map(Self.cols)
    """Where each index reads memory."""
    comptime reg_map = registers_map(Self.cols, 1 << _WARP_LANE_BITS)
    """Which register each index writes."""

    @always_inline
    def __call__[
        T: DType,
        LR: TensorLayout,
        ER: TensorEngine,
        LM: TensorLayout,
        EM: TensorEngine,
    ](
        self,
        dst: MutTileTensor[T, LR, Engine=ER, ...],
        src: ImmTileTensor[T, LM, Engine=EM, ...],
    ):
        """Issues the load without waiting for it.

        Parameters:
            T: The operands' element type.
            LR: The register view's layout.
            ER: The register view's engine.
            LM: The TMEM view's layout.
            EM: The TMEM view's engine.

        Args:
            dst: This thread's `cols` registers.
            src: The warp's quadrant, from the base every thread passes.
        """
        comptime assert same_engine[EM, TMemEngine](), "TMemLoad: src is TMEM"
        comptime assert (
            Self.cols <= _MAX_ASYNC_COLS
        ), "TMemLoad: one issue moves at most 64 columns"
        comptime assert (
            Self.cols & (Self.cols - 1) == 0
        ), "TMemLoad: columns are a power of two"
        TMemEngine._copy_out[
            self_address_space=type_of(src).address_space,
            other_address_space=type_of(dst).address_space,
            OtherEngine=ER,
            wait=False,
        ](
            (
                rebind[
                    TMemEngine.StorageType[
                        T, src.origin, type_of(src).address_space
                    ]
                ](src._storage),
                src.layout,
            ),
            (dst._storage, dst.layout),
        )


@fieldwise_init
struct TMemStore[dt: DType, cols: Int](StoreAtom):
    """`tcgen05.st.32x32b.x{cols}`: a warp stores `cols` columns into its
    TMEM quadrant, each thread its own lane.

    Parameters:
        dt: A 4-byte element type.
        cols: The columns one issue moves, a power of two up to 64.
    """

    comptime shape_values = ParameterList.of[
        1 << _WARP_LANE_BITS, Self.cols
    ]().values
    """One issue covers 32 lanes by `cols` columns."""
    comptime dtype = Self.dt
    """The element type the instruction moves."""
    comptime AsyncEngine = Tcgen05StEngine
    """`tcgen05.wait::st` waits for the thread's TMEM stores."""
    comptime Engine = TMemEngine
    """The engine of the memory the instruction writes."""
    comptime mem_map = tmem_32x32b_map(Self.cols)
    """Where each index writes memory."""
    comptime reg_map = registers_map(Self.cols, 1 << _WARP_LANE_BITS)
    """Which register each index reads."""

    @always_inline
    def __call__[
        T: DType,
        LM: TensorLayout,
        EM: TensorEngine,
        LR: TensorLayout,
        ER: TensorEngine,
    ](
        self,
        dst: MutTileTensor[T, LM, Engine=EM, ...],
        src: ImmTileTensor[T, LR, Engine=ER, ...],
    ):
        """Issues the store without waiting for it.

        Parameters:
            T: The operands' element type.
            LM: The TMEM view's layout.
            EM: The TMEM view's engine.
            LR: The register view's layout.
            ER: The register view's engine.

        Args:
            dst: The warp's quadrant, from the base every thread passes.
            src: This thread's `cols` registers.
        """
        comptime assert same_engine[EM, TMemEngine](), "TMemStore: dst is TMEM"
        comptime assert (
            Self.cols <= _MAX_ASYNC_COLS
        ), "TMemStore: one issue moves at most 64 columns"
        comptime assert (
            Self.cols & (Self.cols - 1) == 0
        ), "TMemStore: columns are a power of two"
        TMemEngine._copy_in[
            self_address_space=type_of(dst).address_space,
            other_address_space=type_of(src).address_space,
            OtherEngine=ER,
            wait=False,
        ](
            (
                rebind[
                    TMemEngine.StorageType[
                        T, dst.origin, type_of(dst).address_space
                    ]
                ](dst._storage),
                dst.layout,
            ),
            # `src` may be a GENERIC stack tile this thread just wrote, and
            # an immutable GENERIC view still loads invariant; read it
            # through a mutable handle so the loads stay after those writes.
            (src._unsafe_storage_cast[to_mut=True](), src.layout),
        )
