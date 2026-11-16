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
"""Defines the transfer atom for NVIDIA's `cp.async` global-to-shared copy.

`CpAsync` wraps `cp.async`: each thread copies 4, 8 or 16 bytes from global
to shared memory, and completes in batches: issues join the open
`cp.async` group, `cp.async.commit_group` closes it, and
`cp.async.wait_group` waits until at most a given number of groups are
outstanding.

The bulk copy, `cp.async.bulk`, is B200-specific and lives in
`layout.atoms.b200.cp_async`.

You can import these APIs from the `layout.atoms.cp_async` module:

```mojo
from layout.atoms.cp_async import CpAsync, CpAsyncGroups
```
"""

from layout import ImmTileTensor, MutTileTensor
from layout.atoms import BatchCompletion, TransferAtom, vector_map
from layout.tensor_engine import DefaultEngine, TensorEngine
from layout.layout import TensorLayout

from max.gpu.memory import (
    async_copy,
    async_copy_commit_group,
    async_copy_wait_all,
    async_copy_wait_group,
)

from std.sys import size_of


struct CpAsyncGroups(BatchCompletion):
    """`cp.async` groups: issues join the open group, `commit` closes it
    with `cp.async.commit_group`, and `wait` is `cp.async.wait_group`."""

    comptime Result = NoneType
    """Nothing: a `cp.async` returns no handle."""
    comptime max_pending = Int.MAX
    """Any number of committed groups can stay outstanding."""

    @staticmethod
    @always_inline
    def open():
        """Does nothing: a group is always open, and issues join it."""
        pass

    @staticmethod
    @always_inline
    def commit():
        """Closes the open group with `cp.async.commit_group`."""
        async_copy_commit_group()

    @staticmethod
    @always_inline
    def wait[pending: Int = 0]():
        """Waits with `cp.async.wait_group` until at most `pending`
        committed groups are outstanding.

        Parameters:
            pending: The groups that may stay outstanding.
        """
        comptime if pending == 0:
            async_copy_wait_all()
        else:
            async_copy_wait_group(Int32(pending))


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
    comptime Completion = CpAsyncGroups
    """Issues join the open `cp.async` group."""
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
        """Issues the copy; it joins the open `cp.async` group.

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
