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
"""Defines atoms for plain pointer loads and stores.

`PointerLoad` and `PointerStore` wrap one vector access through a pointer:
`width` contiguous elements, such as `ld.global.v4.f32` or
`st.shared.v2.b32` on NVIDIA GPUs, `global_load_dwordx4` or `ds_write_b128`
on AMD GPUs, and a vector load or store on a CPU. The access claims `align`
bytes of alignment, by default the whole vector's; a caller whose addresses
are only element aligned claims less, and the backend then picks accesses
that need no more. A load is never invariant: an immutable view only says
the access reads, and other agents may write the memory. Each thread issues its own access at the base of the memory
view it passes, so the memory map fixes the value bits and leaves lanes
free. The registers are ready, and the memory written, when the call
returns.

You can import these APIs from the `layout.atoms.pointer` module:

```mojo
from layout.atoms.pointer import PointerLoad, PointerStore
```
"""

from layout import ImmTileTensor, MutTileTensor
from layout.atoms import (
    LoadAtom,
    StoreAtom,
    Synchronous,
    registers_map,
    same_engine,
    vector_map,
)
from layout.tensor_engine import DefaultEngine, TensorEngine
from layout.layout import TensorLayout

from std.sys import size_of


@fieldwise_init
struct PointerLoad[dt: DType, width: Int, align: Int = width * size_of[dt]()](
    LoadAtom
):
    """A vector load of `width` contiguous elements into registers.

    Parameters:
        dt: The element type.
        width: The elements one load moves, a power of two.
        align: The bytes of alignment the access claims.
    """

    comptime shape_values = ParameterList.of[Self.width]().values
    """One issue moves `width` elements."""
    comptime Completion = Synchronous
    """Ready when the call returns."""
    comptime dtype = Self.dt
    """The element type the instruction moves."""
    comptime Engine = DefaultEngine[element_width=1]
    """The engine of the memory the instruction reads."""
    comptime mem_map = vector_map(Self.width, size_of[Self.dt]())
    """Where each index reads memory."""
    comptime reg_map = registers_map(Self.width)
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
        """Issues the load.

        Parameters:
            T: The operands' element type.
            LR: The register view's layout.
            ER: The register view's engine.
            LM: The memory view's layout.
            EM: The memory view's engine.

        Args:
            dst: This thread's `width` registers.
            src: The memory, from the vector's first element.
        """
        comptime assert T == Self.dt, "PointerLoad: operand dtype"
        comptime assert same_engine[EM, Self.Engine](), "PointerLoad: memory"
        comptime assert (
            Self.width & (Self.width - 1) == 0
        ), "PointerLoad: width is a power of two"
        var value = src.unsafe_ptr().unsafe_load[
            width=Self.width, alignment=Self.align, invariant=False
        ]()
        dst.unsafe_ptr().unsafe_store[alignment=Self.align](value)


@fieldwise_init
struct PointerStore[dt: DType, width: Int, align: Int = width * size_of[dt]()](
    StoreAtom
):
    """A vector store of `width` registers into contiguous elements.

    Parameters:
        dt: The element type.
        width: The elements one store moves, a power of two.
        align: The bytes of alignment the access claims.
    """

    comptime shape_values = ParameterList.of[Self.width]().values
    """One issue moves `width` elements."""
    comptime Completion = Synchronous
    """Ready when the call returns."""
    comptime dtype = Self.dt
    """The element type the instruction moves."""
    comptime Engine = DefaultEngine[element_width=1]
    """The engine of the memory the instruction writes."""
    comptime mem_map = vector_map(Self.width, size_of[Self.dt]())
    """Where each index writes memory."""
    comptime reg_map = registers_map(Self.width)
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
        """Issues the store.

        Parameters:
            T: The operands' element type.
            LM: The memory view's layout.
            EM: The memory view's engine.
            LR: The register view's layout.
            ER: The register view's engine.

        Args:
            dst: The memory, from the vector's first element.
            src: This thread's `width` registers.
        """
        comptime assert T == Self.dt, "PointerStore: operand dtype"
        comptime assert same_engine[EM, Self.Engine](), "PointerStore: memory"
        comptime assert (
            Self.width & (Self.width - 1) == 0
        ), "PointerStore: width is a power of two"
        var value = src.unsafe_ptr().unsafe_load[
            width=Self.width, alignment=Self.align, invariant=False
        ]()
        dst.unsafe_ptr().unsafe_store[alignment=Self.align](value)
