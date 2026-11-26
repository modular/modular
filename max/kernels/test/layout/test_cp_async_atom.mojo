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
"""Host tests for the `cp.async` transfer atoms: their descriptions, their
async engines, and the PTX they and their routes' sync atoms issue when
compiled for sm_100a."""

from std.memory import unsafe_stack_allocation
from std.sys.info import _is_sm_100x_or_newer
from std.testing import assert_equal, assert_false, assert_true, TestSuite

from layout import TileTensor, row_major
from layout.atoms import (
    AddressEncoding,
    Arrives,
    AsyncEngine,
    BarrierPhase,
    HasBarrier,
    HasCount,
    HasGroups,
    HasHandles,
    SealsToBarrier,
    commit_batch,
    wait_batches,
)
from layout.atoms.b200.cp_async import CpAsyncBulk
from layout.atoms.cp_async import CpAsync
from layout.atoms.mbarrier import MBarrierParity
from layout.owned_tensor import OwnedGPUTensor
from layout.tma_async import SharedMemBarrier

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code


def cp_async_kernel(
    src: Pointer[Float32, ImmUntrackedOrigin, address_space=.GLOBAL]
):
    var smem = OwnedGPUTensor[.float32, .SHARED](row_major[1, 4]())
    comptime A = CpAsync[.float32, 16]
    A()(TileTensor(smem), TileTensor(src, row_major[1, 4]()))
    # The atom only issues; this kernel, standing in for a building block,
    # commits the group and waits on it.
    commit_batch[A.AsyncEngine]()
    wait_batches[A.AsyncEngine, 1]()
    A.AsyncEngine.Groups.Wait()()


def cp_async_sealed_kernel(
    src: Pointer[Float32, ImmUntrackedOrigin, address_space=.GLOBAL]
):
    comptime if _is_sm_100x_or_newer():
        comptime A = CpAsync[.float32, 16]
        var smem = OwnedGPUTensor[.float32, .SHARED](row_major[1, 4]())
        var barrier = unsafe_stack_allocation[
            1, SharedMemBarrier, address_space=.SHARED
        ]()
        A()(TileTensor(smem), TileTensor(src, row_major[1, 4]()))
        A.AsyncEngine.Barrier.Seal[MutUntrackedOrigin](barrier)()
        A.AsyncEngine.Barrier.Wait()(barrier, .even)


def cp_async_bulk_kernel(
    src: Pointer[Float32, ImmUntrackedOrigin, address_space=.GLOBAL]
):
    # The test binary also elaborates kernels for the host, where the
    # instruction's sm_100 check fails.
    comptime if _is_sm_100x_or_newer():
        var smem = OwnedGPUTensor[.float32, .SHARED](row_major[1, 64]())
        var barrier = unsafe_stack_allocation[
            1, SharedMemBarrier, address_space=.SHARED
        ]()
        comptime A = CpAsyncBulk[.float32, 256, MutUntrackedOrigin]
        A.AsyncEngine.Barrier.Arm()(barrier, 256)
        A(barrier)(TileTensor(smem), TileTensor(src, row_major[1, 64]()))
        A.AsyncEngine.Barrier.Wait()(barrier, .even)


def test_cp_async_description() raises:
    comptime A = CpAsync[.float32, 16]
    assert_equal(A.shape[0], 4)
    assert_true(A.src_map.encoding == AddressEncoding.BYTES)
    assert_equal(A.dst_map.fixed(UInt64(3)), UInt64(12))
    assert_equal(A.dst_map.avoid, UInt64(15))
    comptime E = A.AsyncEngine
    assert_true(conforms_to(E, HasGroups))
    assert_true(conforms_to(E, HasBarrier))
    assert_true(conforms_to(E.Barrier, SealsToBarrier))
    assert_false(conforms_to(E, HasCount))
    assert_false(conforms_to(E, HasHandles))
    assert_equal(E.Groups.Wait.max_pending, Int.MAX)


def test_cp_async_bulk_description() raises:
    comptime A = CpAsyncBulk[.float32, 256, MutUntrackedOrigin]
    assert_equal(A.shape[0], 64)
    assert_equal(A.src_map.fixed(UInt64(63)), UInt64(252))
    assert_equal(A.src_map.avoid, UInt64(255))
    comptime E = A.AsyncEngine
    assert_true(conforms_to(E, HasBarrier))
    assert_true(conforms_to(E.Barrier, Arrives))
    assert_false(conforms_to(E, HasCount))
    assert_false(conforms_to(E, HasGroups))
    assert_false(conforms_to(E, HasHandles))


def _assert_moka_row[
    E: AsyncEngine
](issuer_only: Bool, in_order: Bool, partial: Bool) raises:
    # Moka's SM100 engine table keeps these facts by hand; only the issuer
    # observes completion unless a route arrives on a barrier.
    assert_equal(not conforms_to(E, HasBarrier), issuer_only)
    assert_equal(E.in_order, in_order)
    assert_equal(E.partial, partial)


def test_engines_match_moka_rows() raises:
    _assert_moka_row[CpAsync[.float32, 16].AsyncEngine](False, False, False)
    _assert_moka_row[
        CpAsyncBulk[.float32, 256, MutUntrackedOrigin].AsyncEngine
    ](False, False, False)


@fieldwise_init
struct _ThreePhase(BarrierPhase):
    """A barrier phase that tells three phases apart, unlike a parity."""

    var index: Int

    comptime start: Self = Self(0)

    def next(self) -> Self:
        return Self((self.index + 1) % 3)


def _advance[P: BarrierPhase](rounds: Int) -> P:
    var phase = P.start
    for _ in range(rounds):
        phase = phase.next()
    return phase


def test_mbarrier_parity_flips() raises:
    assert_true(MBarrierParity.start == .even)
    assert_true(_advance[MBarrierParity](1) == .odd)
    assert_true(_advance[MBarrierParity](2) == .even)


def test_barrier_phase_counts_past_two() raises:
    assert_equal(_advance[_ThreePhase](1).index, 1)
    assert_equal(_advance[_ThreePhase](2).index, 2)
    assert_equal(_advance[_ThreePhase](3).index, 0)


def test_cp_async_ptx() raises:
    var asm = _compile_code[
        cp_async_kernel, target=get_gpu_target["sm_100a"]()
    ]().asm
    assert_true("cp.async.cg.shared.global" in asm)
    assert_true("cp.async.commit_group;" in asm)
    assert_true("cp.async.wait_group \t1;" in asm)
    assert_true("cp.async.wait_all;" in asm)


def test_cp_async_sealed_ptx() raises:
    var asm = _compile_code[
        cp_async_sealed_kernel, target=get_gpu_target["sm_100a"]()
    ]().asm
    assert_true("cp.async.mbarrier.arrive" in asm)
    assert_true("mbarrier.try_wait.parity" in asm)
    assert_false("cp.async.commit_group" in asm)


def test_cp_async_bulk_ptx() raises:
    var asm = _compile_code[
        cp_async_bulk_kernel, target=get_gpu_target["sm_100a"]()
    ]().asm
    assert_true(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes" in asm
    )
    assert_true("mbarrier.arrive.expect_tx" in asm)
    assert_true("mbarrier.try_wait.parity" in asm)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
