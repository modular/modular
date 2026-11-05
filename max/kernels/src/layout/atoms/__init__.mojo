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
"""Defines atoms: wrappers for single hardware intrinsics as typed
operations on `TileTensor` operands.

An atom issues one instruction and describes its operands so that code
above it can check and arrange them. Each operand has an engine, the kind
of storage it lives in, and an `OperandMap`: the part of the instruction's
index map that the hardware fixes, from index bits to address bits in the
operand's `AddressEncoding`. The index bits encode one element of one issue
as the threads that issue it see it: bits from `INDEX_VALUE` pick the
element within one thread's part, bits from `INDEX_LANE` the thread within
its warp, and bits from `INDEX_WARP` the warp.

An operand map says nothing about the tensor in that storage. A tensor's
layout places logical coordinates at addresses; combining that placement
with an atom's maps gives the fragment each thread holds, and is the work of
the libraries built on top. Index bits a map leaves out are free: the
caller's addresses decide where they go, as long as they keep clear of the
map's `avoid` bits.

Orchestration is not an atom's job: tiling operands into issues, choosing
which threads issue, partitioning tiles among them, and waiting on
asynchronous instructions belong to the libraries built on top. An atom
only advertises how its results complete, through its `Completion` type,
which wraps the wait instructions of the atom's instruction family and is
one of:

- `Synchronous`: the results are ready when the call returns.
- A `BarrierCompletion`: the atom is built with a caller's barrier, such as
  an mbarrier for a TMA load, and the instruction signals it; anyone can
  wait on the barrier.
- A `HandleCompletion`: the instruction produces a handle, such as a DMA
  id, which the call returns as its `Result`; the target's wait instruction
  takes it.
- A `CounterCompletion`: the family's wait instruction is tied to the
  instruction type rather than to one call, such as `tcgen05.wait::ld` or
  `cp.async.wait_group`, and waits for all of this thread's outstanding
  issues or all but a given number.
- A `BatchCompletion`: issues join a batch the caller opens, without naming
  it; the caller commits the batch and waits on it, as with
  `cp.async.bulk.commit_group` and `cp.async.bulk.wait_group`.

An instruction family shares one completion type, so code that waits can
tell which issues one wait covers.

Operand dtypes and engines are checked rather than pinned in signatures. A
view's engine is its parent's `OffsetResultType`, and an atom's associated
`dtype` reached through a generic parameter does not fold to the struct
parameter that set it, so pinning either would reject valid operands in
generic code.

Each instruction family is a submodule: `layout.atoms.pointer` for plain
pointer loads and stores, `layout.atoms.cp_async` for `cp.async`,
`layout.atoms.b200.cp_async` for the B200 bulk copy `cp.async.bulk`, and
`layout.atoms.tmem` for Blackwell Tensor Memory.

You can import these APIs from the `layout.atoms` package:

```mojo
from layout.atoms import CastAtom, LoadAtom, MmaAtom, StoreAtom, TransferAtom
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

comptime INDEX_WARP = 37
"""The lowest index bit that picks the warp."""

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


trait AtomCompletion:
    """Advertises how an instruction family's results complete."""

    comptime Result: Movable
    """What each call returns: the instruction's handle for a
    `HandleCompletion`, else `NoneType`."""


struct Synchronous(AtomCompletion):
    """The results are ready when the call returns."""

    comptime Result = NoneType
    """Nothing: the call returns no handle."""


trait BarrierCompletion(AtomCompletion):
    """The atom is built with a caller's barrier, and the instruction
    signals it when its results are ready; anyone can wait on the barrier.
    """

    comptime Barrier: AnyType
    """The barrier type the atom takes from its caller."""


trait HandleCompletion(AtomCompletion):
    """The instruction produces a handle, which the call returns as its
    `Result`; the target's wait instruction takes it."""

    pass


trait CounterCompletion(AtomCompletion):
    """The family's wait instruction is tied to the instruction type, not to
    a call: it waits until at most a given number of this thread's issues
    of the family are outstanding."""

    comptime max_pending: Int
    """The largest number `wait` can leave outstanding; 0 when the family
    can only wait for all of its issues."""

    @staticmethod
    def wait[pending: Int = 0]():
        """Blocks until at most `pending` of the family's issues are
        outstanding.

        Parameters:
            pending: The issues that may stay outstanding, at most
                `max_pending`.
        """
        ...


trait BatchCompletion(AtomCompletion):
    """Issues join the batch the caller opened, without naming it; the
    caller commits the batch and waits on it."""

    comptime max_pending: Int
    """The largest number of committed batches `wait` can leave
    outstanding; 0 when it can only wait for all of them."""

    @staticmethod
    def open():
        """Opens a batch for the issues that follow."""
        ...

    @staticmethod
    def commit():
        """Closes the open batch."""
        ...

    @staticmethod
    def wait[pending: Int = 0]():
        """Blocks until at most `pending` committed batches are outstanding.

        Parameters:
            pending: The batches that may stay outstanding, at most
                `max_pending`.
        """
        ...


trait Atom(TrivialRegisterPassable):
    """Wraps one hardware intrinsic issued on `TileTensor` operands.

    A call issues the intrinsic and nothing else, and returns what the
    instruction produces: `Completion.Result`, which is `NoneType` unless
    the instruction yields a handle. Waiting for an asynchronous atom is the
    caller's job, with the instructions its `Completion` wraps.
    """

    comptime Completion: AtomCompletion
    """How the atom's results complete."""

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


trait LoadAtom(Atom):
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
    ) -> Self.Completion.Result:
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

        Returns:
            The instruction's handle for a `HandleCompletion`, else
            nothing.
        """
        ...


trait StoreAtom(Atom):
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
    ) -> Self.Completion.Result:
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

        Returns:
            The instruction's handle for a `HandleCompletion`, else
            nothing.
        """
        ...


trait TransferAtom(Atom):
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
    ) -> Self.Completion.Result:
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

        Returns:
            The instruction's handle for a `HandleCompletion`, else
            nothing.
        """
        ...


trait CastAtom(Atom):
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
    ) -> Self.Completion.Result:
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

        Returns:
            The instruction's handle for a `HandleCompletion`, else
            nothing.
        """
        ...


trait MmaAtom(Atom):
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
    ) -> Self.Completion.Result:
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

        Returns:
            The instruction's handle for a `HandleCompletion`, else
            nothing.
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
