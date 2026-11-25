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
"""Tests a transfer atom whose engine has no standalone seal: the last issue
of a batch goes through one of the atom's sealing variants, which returns a
token that only the engine's handle route consumes.

The atom is a host stand-in for such a DMA queue: it copies through
pointers, and its token records whether the issue was sealed ordered.
"""

from std.sys import size_of
from std.testing import assert_equal, assert_false, assert_true, TestSuite

from layout import ImmTileTensor, MutTileTensor, TileTensor, row_major
from layout.atoms import (
    HandleValue,
    HandleWaitAtom,
    Handled,
    HasBarrier,
    HasCount,
    HasGroups,
    HasHandles,
    SealingTransferAtom,
    SealsOnLastIssue,
    vector_map,
)
from layout.atoms.cp_async import CpAsync
from layout.layout import TensorLayout
from layout.owned_tensor import OwnedGPUTensor
from layout.tensor_engine import DefaultEngine, TensorEngine


struct _Ticket(not Deinitable, HandleValue):
    """Covers what its variant sealed; only `_TicketWait` closes it."""

    var ordered: Bool

    def __init__(out self, ordered: Bool):
        self.ordered = ordered

    def close(deinit self) -> Bool:
        return self.ordered


@fieldwise_init
struct _TicketWait(HandleWaitAtom):
    comptime Handle = _Ticket

    def __call__(self, var handle: _Ticket):
        _ = handle^.close()


struct _Tickets(Handled):
    comptime Wait = _TicketWait


struct _QueueEngine(HasHandles):
    comptime in_order = False
    comptime Handles = _Tickets


@fieldwise_init
struct _QueueCopy[dt: DType, width: Int](SealsOnLastIssue):
    comptime AsyncEngine = _QueueEngine
    comptime Sealing[ordered: Bool] = _QueueCopySealed[
        Self.dt, Self.width, ordered
    ]
    comptime shape_values = ParameterList.of[Self.width]().values
    comptime dtype = Self.dt
    comptime SrcEngine = DefaultEngine[element_width=1]
    comptime DstEngine = DefaultEngine[element_width=1]
    comptime src_map = vector_map(Self.width, size_of[Self.dt]())
    comptime dst_map = vector_map(Self.width, size_of[Self.dt]())

    def sealing[
        ordered: Bool
    ](self) -> _QueueCopySealed[Self.dt, Self.width, ordered]:
        return {}

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
        dst.unsafe_ptr().unsafe_store(
            src.unsafe_ptr().unsafe_load[width=Self.width]()
        )


@fieldwise_init
struct _QueueCopySealed[dt: DType, width: Int, ordered: Bool](
    SealingTransferAtom
):
    comptime Token = _Ticket

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
    ) -> _Ticket:
        _QueueCopy[Self.dt, Self.width]()(dst, src)
        return _Ticket(Self.ordered)


def _wait[E: HasHandles](var token: E.Handles.Wait.Handle):
    E.Handles.Wait()(token^)


def _last_piece[
    A: SealsOnLastIssue, pieces: Int
](atom: A) -> A.Sealing[pieces > 1]:
    return atom.sealing[ordered=pieces > 1]()


def test_routes() raises:
    comptime A = _QueueCopy[.float32, 4]
    assert_true(conforms_to(A, SealsOnLastIssue))
    assert_true(conforms_to(A.AsyncEngine, HasHandles))
    assert_false(conforms_to(A.AsyncEngine, HasCount))
    assert_false(conforms_to(A.AsyncEngine, HasGroups))
    assert_false(conforms_to(A.AsyncEngine, HasBarrier))
    # A dropped token would never be waited on, in either form.
    assert_false(conforms_to(A.Sealing[True].Token, Deinitable))
    assert_false(conforms_to(A.Sealing[False].Token, Deinitable))
    # An engine with a standalone seal has no sealing variant.
    assert_false(conforms_to(CpAsync[.float32, 16], SealsOnLastIssue))


def test_multi_piece_copy_seals_ordered() raises:
    var src0 = OwnedGPUTensor[.float32, .GENERIC](row_major[1, 4]())
    var src1 = OwnedGPUTensor[.float32, .GENERIC](row_major[1, 4]())
    var dst0 = OwnedGPUTensor[.float32, .GENERIC](row_major[1, 4]())
    var dst1 = OwnedGPUTensor[.float32, .GENERIC](row_major[1, 4]())
    for i in range(4):
        TileTensor(src0)[0, i] = Float32(i + 1)
        TileTensor(src1)[0, i] = Float32(i + 5)
    comptime A = _QueueCopy[.float32, 4]
    A()(TileTensor(dst0), TileTensor(src0))
    var token = _last_piece[A, 2](A())(TileTensor(dst1), TileTensor(src1))
    assert_true(token^.close())
    var again = A().sealing[ordered=True]()(TileTensor(dst1), TileTensor(src1))
    _wait[A.AsyncEngine](again^)
    for i in range(4):
        assert_equal(TileTensor(dst0)[0, i][0], Float32(i + 1))
        assert_equal(TileTensor(dst1)[0, i][0], Float32(i + 5))


def test_one_piece_copy_seals_unordered() raises:
    var src = OwnedGPUTensor[.float32, .GENERIC](row_major[1, 4]())
    var dst = OwnedGPUTensor[.float32, .GENERIC](row_major[1, 4]())
    for i in range(4):
        TileTensor(src)[0, i] = Float32(i + 1)
    comptime A = _QueueCopy[.float32, 4]
    var token = _last_piece[A, 1](A())(TileTensor(dst), TileTensor(src))
    assert_false(token^.close())
    var again = A().sealing[ordered=False]()(TileTensor(dst), TileTensor(src))
    _wait[A.AsyncEngine](again^)
    for i in range(4):
        assert_equal(TileTensor(dst)[0, i][0], Float32(i + 1))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
