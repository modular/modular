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
"""Defines atoms: wrappers for single hardware instructions, typed on
`TileTensor` operands.

An atom issues one instruction. A data atom moves or computes elements and
describes its operands so that code above it can check and arrange them.
Each operand has an engine, the kind of storage it lives in, and an
`OperandMap`: the part of the instruction's index map that the hardware
fixes, from index bits to address bits in the operand's `AddressEncoding`.
The index bits encode one element of one issue as the threads that issue it
see it: bits from `INDEX_VALUE` pick the element within one thread's part,
bits from `INDEX_LANE` the thread within its warp, and bits from
`INDEX_WARP` the warp.

An operand map says nothing about the tensor in that storage. A tensor's
layout places logical coordinates at addresses; combining that placement
with an atom's maps gives the fragment each thread holds, and is the work of
the libraries built on top. Index bits a map leaves out are free: the
caller's addresses decide where they go, as long as they keep clear of the
map's `avoid` bits.

Orchestration is not an atom's job: tiling operands into issues, choosing
which threads issue, partitioning tiles among them, and completing
asynchronous instructions belong to the libraries built on top.

Data atoms and TileOps never synchronize; sync atoms are issued by a
building block, or by Moka on its behalf. A data atom returns nothing and
names the `AsyncEngine` that runs it. The async engine states whether it
runs in order and whether its reads can complete alone, and conforms to a
capability trait for each route its completions can take:

- `HasCount`, a `Counted` route: only the issuing thread waits, on a count
  of its issues, as with `tcgen05.wait::ld`.
- `HasGroups`, a `Grouped` route: only the issuing thread waits. `Commit`
  closes a group of its issues, as `cp.async.commit_group` does, and `Wait`
  counts groups.
- `HasBarrier`, a `BarrierRoute`: completions arrive on a barrier, and any
  thread waits on it. Under `Arrives`, each issue arrives on the barrier
  its atom was built with, and `Arm` arrives once and sets the amount a
  phase expects. Under `SealsToBarrier`, a `Seal` built with a barrier
  arrives on it once the thread's earlier issues have completed.
- `HasHandles`, a `Handled` route: a sealing variant of the data atom
  returns a handle, and the issuing thread waits by consuming it.

`Synchronous` has no routes: its results are ready when the call returns,
so nothing needs to wait for them.

Each wait, commit, seal and arm a route names is a sync atom that issues
one instruction. Code that waits picks a route by name, such as
`A.AsyncEngine.Count.Wait()()`, or calls `wait_issues`, `commit_batch` or
`wait_batches` on the engine. Generic code requires the capability it
uses, such as `E: HasGroups`, so an engine without that route fails to
compile. Where to seal is the caller's choice, since only the caller knows
which issue is the last before a handoff.

An engine whose seal is a mode of the data instruction, not a separate
instruction, has data atoms that conform to `SealsOnLastIssue`: their
sealing variants issue the last piece of a batch and return a `Token`, the
only value an atom returns.

Operand dtypes and engines are checked rather than pinned in signatures. A
view's engine is its parent's `OffsetResultType`, and an atom's associated
`dtype` reached through a generic parameter does not fold to the struct
parameter that set it, so pinning either would reject valid operands in
generic code.

Each instruction family is a submodule: `layout.atoms.pointer` for plain
pointer loads and stores, `layout.atoms.cp_async` for `cp.async`,
`layout.atoms.b200.cp_async` for the B200 bulk copy `cp.async.bulk`,
`layout.atoms.tmem` for Blackwell Tensor Memory, and `layout.atoms.mbarrier`
for the mbarrier waits and arms that barrier routes share.

You can import these APIs from the `layout.atoms` package:

```mojo
from layout.atoms import CastAtom, LoadAtom, MmaAtom, StoreAtom, TransferAtom
from layout.atoms import SealingTransferAtom, SealsOnLastIssue
from layout.atoms import AsyncEngine, HasBarrier, HasCount, HasGroups
from layout.atoms import HasHandles, HandleValue
from layout.atoms import Arrives, Counted, Grouped, Handled, SealsToBarrier
from layout.atoms import BarrierPhase
from layout.atoms import commit_batch, wait_batches, wait_issues
from layout.atoms import OperandMap, Synchronous, same_engine
from layout.atoms import AddressEncoding, registers_map, vector_map
```
"""

from layout import ImmTileTensor, MutTileTensor
from layout.bit_layout import BitLayout
from layout.layout import TensorLayout
from layout.tensor_engine import TensorEngine

from std.bit import count_trailing_zeros
from std.builtin.variadics import _MLIR

comptime INDEX_VALUE = 0
"""The lowest index bit that picks an element within one thread's part."""

comptime INDEX_LANE = 32
"""The lowest index bit that picks the thread within its warp."""

comptime INDEX_WARP = 38
"""The lowest index bit that picks the warp, leaving room for 64-lane AMD
wavefronts."""

comptime TMEM_COLUMN = 0
"""The lowest TMEM address bit that picks a column."""

comptime TMEM_LANE = 16
"""The lowest TMEM address bit that picks a lane."""


@fieldwise_init
struct AddressEncoding(Equatable, TrivialRegisterPassable, Writable):
    """How an operand's address bits encode a location in its storage."""

    var _value: Int

    comptime REGISTERS = Self(0)
    """Registers: the register number from bit `INDEX_VALUE` and the thread
    from `INDEX_LANE`, as in the index space."""

    comptime BYTES = Self(1)
    """Memory: the byte address, such as a shared or global offset."""

    comptime TMEM = Self(2)
    """Tensor Memory: columns from bit `TMEM_COLUMN` and lanes from bit
    `TMEM_LANE`, as a `tcgen05` address."""

    def __eq__(self, other: Self) -> Bool:
        """Returns whether two encodings are the same.

        Args:
            other: The encoding to compare with.

        Returns:
            True when they are the same encoding.
        """
        return self._value == other._value

    def write_to(self, mut writer: Some[Writer]):
        """Writes the encoding's name.

        Args:
            writer: The writer to write to.
        """
        if self == Self.REGISTERS:
            writer.write("registers")
        elif self == Self.BYTES:
            writer.write("bytes")
        else:
            writer.write("tmem")


@fieldwise_init
struct OperandMap(Equatable, TrivialRegisterPassable, Writable):
    """The part of an instruction's index map the hardware fixes for one
    operand."""

    var encoding: AddressEncoding
    """How `fixed` and `avoid` encode the operand's addresses."""

    var fixed: BitLayout
    """The index bits the hardware fixes, each to the address bits it
    reaches. Every other index bit is free."""

    var avoid: UInt64
    """The address bits no free index bit may reach, such as the bytes
    within one vector access."""

    def free(self, index: UInt64) -> UInt64:
        """Returns the index bits among `index` the hardware leaves to the
        caller's addresses.

        Args:
            index: The index bits an issue uses.

        Returns:
            `index` without the fixed bits.
        """
        return index & ~self.fixed.dom

    def __eq__(self, other: Self) -> Bool:
        """Returns whether two operand maps are the same.

        Args:
            other: The map to compare with.

        Returns:
            True when encoding, fixed bits and avoided bits agree.
        """
        return (
            self.encoding == other.encoding
            and self.fixed == other.fixed
            and self.avoid == other.avoid
        )

    def write_to(self, mut writer: Some[Writer]):
        """Writes the map as `encoding fixed avoid=mask`.

        Args:
            writer: The writer to write to.
        """
        writer.write(self.encoding, " ", self.fixed, " avoid=", hex(self.avoid))


def _log2(n: Int) -> Int:
    return Int(count_trailing_zeros(UInt64(n)))


def registers_map(values: Int, lanes: Int = 1) -> OperandMap:
    """Returns a register operand: value `v` of lane `l` is register `v` of
    thread `l`.

    Args:
        values: The registers each thread holds, a power of two.
        lanes: The threads that issue the instruction together, a power of
            two: 1 for a per-thread instruction, 32 for a warp-collective
            one.

    Returns:
        A map that fixes every value and lane bit.
    """
    return {
        AddressEncoding.REGISTERS,
        BitLayout()
        .run(INDEX_VALUE, _log2(values), INDEX_VALUE)
        .run(INDEX_LANE, _log2(lanes), INDEX_LANE),
        0,
    }


def vector_map(values: Int, element_bytes: Int) -> OperandMap:
    """Returns a memory operand of a per-lane vector access: value `v` is
    `v * element_bytes` bytes past the address the lane supplies.

    Every lane supplies its own address, so lane bits are free; they must
    keep clear of the bytes one access covers.

    Args:
        values: The elements one access moves, a power of two.
        element_bytes: The bytes per element, a power of two.

    Returns:
        A map that fixes the value bits.
    """
    var unit = _log2(values) + _log2(element_bytes)
    return {
        AddressEncoding.BYTES,
        BitLayout().run(INDEX_VALUE, _log2(values), _log2(element_bytes)),
        (UInt64(1) << UInt64(unit)) - 1,
    }


trait Atom:
    """Wraps one hardware instruction.

    A call issues the instruction and nothing else. A data atom names the
    `AsyncEngine` that runs it; the engine's routes name the sync atoms that
    complete it, and placing them is the caller's job.

    `Atom` is not `TrivialRegisterPassable`, so that commit atoms can be
    linear. Every other atom trait requires it.
    """

    pass


trait CounterWaitAtom(Atom, Defaultable, TrivialRegisterPassable):
    """Waits until at most a given number of this thread's issues on an
    engine are outstanding, counted in issues or in committed groups.

    Each wait checks at compile time that `pending` is in
    `[0, max_pending]`, so a direct call is checked as well as one through
    `wait_issues` or `wait_batches`.
    """

    comptime max_pending: Int
    """The largest number the wait can leave outstanding; 0 when it can
    only wait for all of them."""

    def __call__[pending: Int = 0](self):
        """Blocks until at most `pending` are outstanding.

        Parameters:
            pending: The issues or groups that may stay outstanding, at most
                `max_pending`.
        """
        ...


trait CommitAtom(Atom, Movable):
    """Seals what this thread has issued on an engine since its last seal.

    A group commit takes no operands. A seal onto a barrier is built with
    the barrier, so generic code takes one its caller built.

    A commit atom must not be `Deinitable`, and its call consumes it, so
    one that is built must be issued. Mojo cannot express a negative bound,
    so the tests enforce it.
    """

    def __call__(deinit self):
        """Seals this thread's open issues."""
        ...


trait BarrierPhase(TrivialRegisterPassable):
    """Names a phase of a barrier, so that generic code can start at a
    barrier's first phase and advance one phase at a time.

    Each target picks how many phases it tells apart: an NVIDIA mbarrier
    needs only the parity, and another barrier can count more phases.
    """

    comptime start: Self
    """The phase a newly initialized barrier is in."""

    def next(self) -> Self:
        """Returns the phase after this one.

        Returns:
            The phase the barrier is in once this one completes.
        """
        ...


trait BarrierWaitAtom(Atom, Defaultable, TrivialRegisterPassable):
    """Waits for a phase of a barrier to complete.

    Barrier routes are not specific to one target: an NVIDIA mbarrier in
    shared memory is one barrier, and a DMA multicast that completes on a
    syncpoint other cores wait on is another. Only the operands differ, so
    each target binds the barrier reference and the phase.
    """

    comptime Ref[origin: MutOrigin]: TrivialRegisterPassable
    """A reference to the barrier the wait reads.

    Parameters:
        origin: The barrier's origin. A target whose barriers are not
            memory ignores it.
    """

    comptime Phase: BarrierPhase
    """What names the phase to wait for, such as a parity."""

    def __call__[
        origin: MutOrigin, //
    ](self, barrier: Self.Ref[origin], phase: Self.Phase):
        """Blocks until the barrier completes `phase`.

        Parameters:
            origin: The barrier's origin.

        Args:
            barrier: The barrier.
            phase: The phase to wait for.
        """
        ...


trait ArmAtom(Atom, Defaultable, TrivialRegisterPassable):
    """Arrives once on a barrier and sets how much its current phase
    expects. The arm is one of the phase's arrivals, so the arming thread
    does not arrive again.

    As for `BarrierWaitAtom`, each target binds the barrier reference and
    the amount.
    """

    comptime Ref[origin: MutOrigin]: TrivialRegisterPassable
    """A reference to the barrier the arm updates.

    Parameters:
        origin: The barrier's origin. A target whose barriers are not
            memory ignores it.
    """

    comptime Amount: TrivialRegisterPassable
    """What the phase expects to arrive, such as a byte count."""

    def __call__[
        origin: MutOrigin, //
    ](self, barrier: Self.Ref[origin], amount: Self.Amount):
        """Arrives once on the barrier and expects `amount` more in its
        current phase.

        Parameters:
            origin: The barrier's origin.

        Args:
            barrier: The barrier.
            amount: What the phase expects.
        """
        ...


trait HandleValue(Movable):
    """A handle a sealing variant returns: it covers what the variant
    seals."""

    pass


trait HandleWaitAtom(Atom, Defaultable, TrivialRegisterPassable):
    """Waits for what a handle covers, consuming the handle."""

    comptime Handle: HandleValue
    """The handle the wait consumes."""

    def __call__(self, var handle: Self.Handle):
        """Blocks until everything the handle covers has completed.

        Args:
            handle: The handle.
        """
        ...


trait Counted:
    """Only the issuing thread waits, on a count of its issues. Nothing
    seals."""

    comptime Wait: CounterWaitAtom
    """The wait."""


trait Grouped:
    """Only the issuing thread waits. `Commit` closes a group of the
    thread's issues, and `Wait` counts groups."""

    comptime Commit: CommitAtom & Defaultable
    """Closes the thread's open group."""

    comptime Wait: CounterWaitAtom
    """Waits on committed groups."""


trait BarrierRoute:
    """Completions arrive on a barrier, and any thread waits on it. A
    consumer waits the same way whether the producer armed the phase or
    sealed it."""

    comptime Wait: BarrierWaitAtom
    """The wait, which also types the barrier reference."""


trait Arrives(BarrierRoute):
    """Each issue arrives on the barrier its atom was built with, and an
    arm sets how much a phase expects."""

    comptime Arm: ArmAtom
    """Arrives once and sets how much a phase expects, on the barrier
    `Wait` reads."""


trait SealsToBarrier(BarrierRoute):
    """A seal built with a barrier arrives on it once everything this
    thread issued on the engine before the seal has completed."""

    comptime Seal[origin: MutOrigin]: CommitAtom
    """The seal, built with a reference to the barrier `Wait` reads.

    Parameters:
        origin: The barrier reference's origin.
    """


trait Handled:
    """A sealing variant of the data atom returns a handle, and only the
    issuing thread waits, consuming the handle."""

    comptime Wait: HandleWaitAtom
    """The wait."""


trait AsyncEngine:
    """The hardware unit that runs a data atom: its facts, and through the
    capability traits, the routes its completions can take.

    Whether only the issuing thread observes completion follows from the
    routes: it does unless the engine conforms to `HasBarrier`.
    """

    comptime in_order: Bool
    """Whether one agent's operations read before any later operation
    writes, and operations on one location run in issue order."""

    comptime partial: Bool = False
    """Whether the issuer can wait for an operation's reads alone."""


trait HasCount(AsyncEngine):
    """The engine has a `Counted` route."""

    comptime Count: Counted
    """The route."""


trait HasGroups(AsyncEngine):
    """The engine has a `Grouped` route."""

    comptime Groups: Grouped
    """The route."""


trait HasBarrier(AsyncEngine):
    """The engine has a barrier route."""

    comptime Barrier: BarrierRoute
    """The route."""


trait HasHandles(AsyncEngine):
    """The engine has a `Handled` route."""

    comptime Handles: Handled
    """The route."""


@always_inline
def wait_issues[E: HasCount, pending: Int = 0]():
    """Blocks until at most `pending` of this thread's issues on `E` are
    outstanding, with the wait of `E`'s count route.

    Parameters:
        E: The async engine of the atoms to wait for.
        pending: The issues that may stay outstanding, at most
            `E.Count.Wait.max_pending`.
    """
    E.Count.Wait().__call__[pending]()


@always_inline
def commit_batch[E: HasGroups]():
    """Closes this thread's open group of issues on `E`, with the commit of
    `E`'s group route.

    Parameters:
        E: The async engine of the atoms the group collects.
    """
    E.Groups.Commit()()


@always_inline
def wait_batches[E: HasGroups, pending: Int = 0]():
    """Blocks until at most `pending` of this thread's committed groups on
    `E` are outstanding, with the wait of `E`'s group route.

    Parameters:
        E: The async engine of the atoms the groups collect.
        pending: The groups that may stay outstanding, at most
            `E.Groups.Wait.max_pending`.
    """
    E.Groups.Wait().__call__[pending]()


struct Synchronous(AsyncEngine):
    """No routes: the results are ready when the atom returns."""

    comptime in_order = True
    """Each operation completes before the next issues."""


trait DataAtom(Atom, TrivialRegisterPassable):
    """Wraps one hardware instruction that moves or computes elements of
    `TileTensor` operands. A call returns nothing."""

    comptime AsyncEngine: AsyncEngine
    """The unit that runs the instruction. Its routes say how the results
    can complete."""

    # The trait cannot require `ParameterList[type=Int, ...]` of an atom: a
    # wildcard list never unifies with a concrete one. Atoms set the list's
    # values, and `shape` is built from them.
    comptime shape_values: _MLIR.KGENParamListType[Int]
    """The values of `shape`, set as `ParameterList.of[...]().values`."""

    comptime shape: ParameterList[Self.shape_values] = ParameterList[
        Self.shape_values
    ]()
    """The instruction's shape, such as `(32, 16)` for
    `tcgen05.ld.32x32b.x16` or `(m, n, k)` for an MMA."""


trait LoadAtom(DataAtom):
    """Describes an instruction that loads elements from memory into
    registers."""

    comptime dtype: DType
    """The element type the instruction moves."""

    comptime Engine: TensorEngine
    """The engine of the memory the instruction reads."""

    comptime mem_map: OperandMap
    """Where each index reads memory."""

    comptime reg_map: OperandMap
    """Which register each index writes."""

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
        """Issues the instruction once.

        Parameters:
            T: The operands' element type.
            LR: The register view's layout.
            ER: The register view's engine.
            LM: The memory view's layout.
            EM: The memory view's engine.

        Args:
            dst: The registers `reg_map` describes.
            src: The memory `mem_map` describes.
        """
        ...


trait StoreAtom(DataAtom):
    """Describes an instruction that stores elements from registers into
    memory."""

    comptime dtype: DType
    """The element type the instruction moves."""

    comptime Engine: TensorEngine
    """The engine of the memory the instruction writes."""

    comptime mem_map: OperandMap
    """Where each index writes memory."""

    comptime reg_map: OperandMap
    """Which register each index reads."""

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
        """Issues the instruction once.

        Parameters:
            T: The operands' element type.
            LM: The memory view's layout.
            EM: The memory view's engine.
            LR: The register view's layout.
            ER: The register view's engine.

        Args:
            dst: The memory `mem_map` describes.
            src: The registers `reg_map` describes.
        """
        ...


trait TransferAtom(DataAtom):
    """Describes an instruction that moves elements from memory to memory,
    such as a DMA submit or `cp.async.bulk`, without passing through the
    issuer's registers."""

    comptime dtype: DType
    """The element type the instruction moves."""

    comptime SrcEngine: TensorEngine
    """The engine of the memory the instruction reads."""

    comptime DstEngine: TensorEngine
    """The engine of the memory the instruction writes."""

    comptime src_map: OperandMap
    """Where each index reads the source."""

    comptime dst_map: OperandMap
    """Where each index writes the destination."""

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
        """Issues the instruction once.

        Parameters:
            T: The operands' element type.
            LD: The destination view's layout.
            ED: The destination view's engine.
            LS: The source view's layout.
            ES: The source view's engine.

        Args:
            dst: The memory `dst_map` describes.
            src: The memory `src_map` describes.
        """
        ...


trait SealingTransferAtom(Atom, TrivialRegisterPassable):
    """A variant of a transfer instruction that also seals: the only kind of
    atom that returns something to wait on."""

    comptime Token: HandleValue
    """What the call returns, covering what the variant seals. The engine's
    `Handled` route waits on it.

    A token must not be `Deinitable`, so that it cannot be dropped without
    its wait. Mojo cannot express a negative bound, so the tests enforce
    it."""

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
    ) -> Self.Token:
        """Issues the instruction once and seals.

        Parameters:
            T: The operands' element type.
            LD: The destination view's layout.
            ED: The destination view's engine.
            LS: The source view's layout.
            ES: The source view's engine.

        Args:
            dst: The memory the transfer atom's `dst_map` describes.
            src: The memory the transfer atom's `src_map` describes.

        Returns:
            The token that covers what the variant seals.
        """
        ...


trait SealsOnLastIssue(TransferAtom):
    """The engine has no standalone seal: the last issue of a batch seals
    it, through a variant of this instruction.

    The variant is ordered or unordered. A copy split into several pieces
    seals its last piece ordered: the piece starts only after this thread's
    earlier issues on the engine complete, so its token covers them too. A
    copy of one piece has nothing earlier to order behind, so it seals
    unordered and its token covers that issue alone, without the stall.
    """

    # A DMA queue that seals this way marks a descriptor as ordered with an
    # attribute bit: earlier DMAs complete before that descriptor starts.
    # The queue rejects a zero-length descriptor, so the seal rides on a
    # piece that moves data; there is no zero-byte marker. Moka's DMA
    # backend and the copy building blocks set the bit only on the last
    # descriptor of a copy of more than one piece. One parametric variant
    # rather than two named ones lets a copy pick the form from its piece
    # count, as `sealing[ordered = pieces > 1]()`.
    comptime Sealing[ordered: Bool]: SealingTransferAtom
    """The variant that seals.

    Parameters:
        ordered: Whether its token also covers this thread's earlier issues
            on the engine.
    """

    def sealing[ordered: Bool](self) -> Self.Sealing[ordered]:
        """Returns a sealing variant of this atom.

        Parameters:
            ordered: Whether the variant waits for this thread's earlier
                issues on the engine, so that its token covers them.

        Returns:
            The variant.
        """
        ...


trait CastAtom(DataAtom):
    """Describes an instruction that converts elements from a source element
    type to a destination element type."""

    comptime src_dtype: DType
    """The element type the instruction reads."""

    comptime dst_dtype: DType
    """The element type the instruction writes."""

    comptime SrcEngine: TensorEngine
    """The engine of the storage the instruction reads."""

    comptime DstEngine: TensorEngine
    """The engine of the storage the instruction writes."""

    comptime src_map: OperandMap
    """Where each index reads the source."""

    comptime dst_map: OperandMap
    """Where each index writes the destination."""

    def __call__[
        TD: DType,
        LD: TensorLayout,
        ED: TensorEngine,
        TS: DType,
        LS: TensorLayout,
        ES: TensorEngine,
    ](
        self,
        dst: MutTileTensor[TD, LD, Engine=ED, ...],
        src: ImmTileTensor[TS, LS, Engine=ES, ...],
    ):
        """Issues the instruction once.

        Parameters:
            TD: The destination's element type.
            LD: The destination view's layout.
            ED: The destination view's engine.
            TS: The source's element type.
            LS: The source view's layout.
            ES: The source view's engine.

        Args:
            dst: The operand `dst_map` describes.
            src: The operand `src_map` describes.
        """
        ...


trait MmaAtom(DataAtom):
    """Describes a matrix multiply-accumulate instruction, `C += A @ B`.

    The accumulator is read and written in place, as UMMA and WGMMA do; an
    instruction with a separate C operand passes the same storage for both.
    An index multiplies the A element `a_map` reaches by the B element
    `b_map` reaches and adds the product into the accumulator `c_map`
    reaches.
    """

    comptime a_dtype: DType
    """The element type of the A operand."""

    comptime b_dtype: DType
    """The element type of the B operand."""

    comptime c_dtype: DType
    """The element type of the accumulator."""

    comptime AEngine: TensorEngine
    """The engine of the storage A lives in."""

    comptime BEngine: TensorEngine
    """The engine of the storage B lives in."""

    comptime CEngine: TensorEngine
    """The engine of the storage the accumulator lives in."""

    comptime a_map: OperandMap
    """Where each index reads A."""

    comptime b_map: OperandMap
    """Where each index reads B."""

    comptime c_map: OperandMap
    """Where each index updates the accumulator."""

    def __call__[
        TC: DType,
        LC: TensorLayout,
        EC: TensorEngine,
        TA: DType,
        LA: TensorLayout,
        EA: TensorEngine,
        TB: DType,
        LB: TensorLayout,
        EB: TensorEngine,
    ](
        self,
        c: MutTileTensor[TC, LC, Engine=EC, ...],
        a: ImmTileTensor[TA, LA, Engine=EA, ...],
        b: ImmTileTensor[TB, LB, Engine=EB, ...],
    ):
        """Issues the instruction once.

        Parameters:
            TC: The accumulator's element type.
            LC: The accumulator view's layout.
            EC: The accumulator view's engine.
            TA: A's element type.
            LA: A's view layout.
            EA: A's view engine.
            TB: B's element type.
            LB: B's view layout.
            EB: B's view engine.

        Args:
            c: The operand `c_map` describes.
            a: The operand `a_map` describes.
            b: The operand `b_map` describes.
        """
        ...


def same_engine[E: TensorEngine, F: TensorEngine]() -> Bool:
    """Returns whether two engines are one engine family.

    A view's engine is its parent's `OffsetResultType`, which generic code
    cannot fold back to the parent's type, so atoms compare families rather
    than types.

    Parameters:
        E: The first engine.
        F: The second engine.

    Returns:
        True when both engines have the same base type name.
    """
    return E._BASE_TYPE_NAME == F._BASE_TYPE_NAME
