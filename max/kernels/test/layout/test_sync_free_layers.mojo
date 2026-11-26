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
"""Gates the layering rule: atoms and TileOps contain no synchronization.

Each kernel issues one atom, or one TileOp over atoms, compiled for sm_100a,
and its PTX must hold no synchronization instruction: no barrier, mbarrier
operation, fence, completion wait or commit, and no issuer election. Only
building blocks may emit those, through `layout.building_blocks`. The
planted kernels add a building block's wait or commit to the same issues
and must be caught, so the gate cannot pass vacuously.
"""

from std.memory import unsafe_stack_allocation
from std.sys.info import _is_sm_100x_or_newer
from std.testing import assert_equal, assert_true, TestSuite

from layout import TileTensor, coord, row_major
from layout.atoms.b200.cp_async import CpAsyncBulk
from layout.atoms.cp_async import CpAsync
from layout.atoms.pointer import PointerLoad, PointerStore
from layout.atoms.tmem import TMemLoad, TMemStore
from layout.building_blocks import commit_batch, wait_batches, wait_issues
from layout.layout import Layout as TileLayout
from layout.owned_tensor import OwnedGPUTensor
from layout.tileops import tiled_load, tiled_store
from layout.tma_async import SharedMemBarrier
from layout.tmem_engine import TMEM_NUM_LANES, TMemEngine, TMemStorage

from max.gpu import thread_idx
from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code

comptime AccumLayout = TileLayout(
    shape=coord[128, 32], stride=coord[1, TMEM_NUM_LANES]
)


__enum SyncInstruction(Equatable, ImplicitlyCopyable, Writable):
    """A PTX instruction that synchronizes or elects an issuer: what only a
    building block may issue."""

    case bar
    case barrier
    case mbarrier
    case membar
    case fence
    case griddepcontrol
    case elect
    case tcgen05_wait
    case tcgen05_fence
    case tcgen05_commit
    case cp_async_commit_group
    case cp_async_wait_group
    case cp_async_wait_all
    case cp_async_mbarrier_arrive
    case cp_async_bulk_commit_group
    case cp_async_bulk_wait_group

    def __eq__(self, other: Self) -> Bool:
        return self._get_enum_discriminant() == other._get_enum_discriminant()

    def mnemonic(self) -> StaticString:
        """The PTX text the instruction starts with."""
        match self:
            case .bar:
                return "bar."
            case .barrier:
                return "barrier."
            case .mbarrier:
                return "mbarrier."
            case .membar:
                return "membar."
            case .fence:
                return "fence."
            case .griddepcontrol:
                return "griddepcontrol."
            case .elect:
                return "elect.sync"
            case .tcgen05_wait:
                return "tcgen05.wait"
            case .tcgen05_fence:
                return "tcgen05.fence"
            case .tcgen05_commit:
                return "tcgen05.commit"
            case .cp_async_commit_group:
                return "cp.async.commit_group"
            case .cp_async_wait_group:
                return "cp.async.wait_group"
            case .cp_async_wait_all:
                return "cp.async.wait_all"
            case .cp_async_mbarrier_arrive:
                return "cp.async.mbarrier.arrive"
            case .cp_async_bulk_commit_group:
                return "cp.async.bulk.commit_group"
            case .cp_async_bulk_wait_group:
                return "cp.async.bulk.wait_group"


def _sync_instructions() -> List[SyncInstruction]:
    """Every `SyncInstruction`, in declaration order."""
    return [
        SyncInstruction.bar,
        SyncInstruction.barrier,
        SyncInstruction.mbarrier,
        SyncInstruction.membar,
        SyncInstruction.fence,
        SyncInstruction.griddepcontrol,
        SyncInstruction.elect,
        SyncInstruction.tcgen05_wait,
        SyncInstruction.tcgen05_fence,
        SyncInstruction.tcgen05_commit,
        SyncInstruction.cp_async_commit_group,
        SyncInstruction.cp_async_wait_group,
        SyncInstruction.cp_async_wait_all,
        SyncInstruction.cp_async_mbarrier_arrive,
        SyncInstruction.cp_async_bulk_commit_group,
        SyncInstruction.cp_async_bulk_wait_group,
    ]


def _sync_issued(asm: String) -> List[SyncInstruction]:
    """The sync instructions `asm` issues. An instruction follows a tab, or
    a predicate and a space; an operand such as `.mbarrier::complete_tx` of
    a bulk copy follows neither."""
    var found = List[SyncInstruction]()
    for i in _sync_instructions():
        var m = String(i.mnemonic())
        if "\t" + m in asm or " " + m in asm:
            found.append(i)
    return found^


def _assert_sync_free(asm: String, what: String) raises:
    var found = _sync_issued(asm)
    var names = String()
    for i in found:
        names += " " + String(i.mnemonic())
    assert_equal(len(found), 0, what + " synchronizes:" + names)


def pointer_tileops_kernel(
    dst: Pointer[Float32, MutUntrackedOrigin, address_space=.GLOBAL],
    src: Pointer[Float32, ImmUntrackedOrigin, address_space=.GLOBAL],
):
    var regs = OwnedGPUTensor[.float32, .LOCAL](row_major[4, 8]())
    tiled_load(
        PointerLoad[.float32, 4](),
        TileTensor(regs),
        TileTensor(src, row_major[4, 8]()),
    )
    tiled_store(
        PointerStore[.float32, 4](),
        TileTensor(dst, row_major[4, 8]()),
        TileTensor(regs),
    )


def cp_async_atom_kernel(
    src: Pointer[Float32, ImmUntrackedOrigin, address_space=.GLOBAL]
):
    var smem = OwnedGPUTensor[.float32, .SHARED](row_major[1, 4]())
    CpAsync[.float32, 16]()(
        TileTensor(smem), TileTensor(src, row_major[1, 4]())
    )


def cp_async_committed_kernel(
    src: Pointer[Float32, ImmUntrackedOrigin, address_space=.GLOBAL]
):
    var smem = OwnedGPUTensor[.float32, .SHARED](row_major[1, 4]())
    comptime A = CpAsync[.float32, 16]
    A()(TileTensor(smem), TileTensor(src, row_major[1, 4]()))
    commit_batch[A.Completion]()
    wait_batches[A.Completion]()


def cp_async_bulk_atom_kernel(
    src: Pointer[Float32, ImmUntrackedOrigin, address_space=.GLOBAL]
):
    # The test binary also elaborates kernels for the host, where the
    # instruction's sm_100 check fails.
    comptime if _is_sm_100x_or_newer():
        var smem = OwnedGPUTensor[.float32, .SHARED](row_major[1, 64]())
        var barrier = unsafe_stack_allocation[
            1, SharedMemBarrier, address_space=.SHARED
        ]()
        CpAsyncBulk[.float32, 256, MutUntrackedOrigin](barrier)(
            TileTensor(smem), TileTensor(src, row_major[1, 64]())
        )


def tmem_tileops_kernel(addr: UInt32):
    comptime if _is_sm_100x_or_newer():
        var tile = TileTensor[Engine=TMemEngine](
            TMemStorage[.float32, MutUntrackedOrigin](addr), AccumLayout
        )
        var regs = OwnedGPUTensor[.float32, .LOCAL](row_major[1, 32]())
        tiled_load[threads=128](
            TMemLoad[.float32, 16](), TileTensor(regs), tile, Int(thread_idx.x)
        )
        tiled_store[threads=128](
            TMemStore[.float32, 16](), tile, TileTensor(regs), Int(thread_idx.x)
        )


def tmem_waited_kernel(addr: UInt32):
    comptime if _is_sm_100x_or_newer():
        var tile = TileTensor[Engine=TMemEngine](
            TMemStorage[.float32, MutUntrackedOrigin](addr), AccumLayout
        )
        var regs = OwnedGPUTensor[.float32, .LOCAL](row_major[1, 32]())
        comptime A = TMemLoad[.float32, 16]
        tiled_load[threads=128](A(), TileTensor(regs), tile, Int(thread_idx.x))
        wait_issues[A.Completion]()


def test_pointer_tileops_are_sync_free() raises:
    var asm = _compile_code[
        pointer_tileops_kernel, target=get_gpu_target["sm_100a"]()
    ]().asm
    assert_true("ld.global.v" in asm)
    assert_true("st.global.v" in asm)
    _assert_sync_free(asm, "tiled_load and tiled_store over pointer atoms")


def test_cp_async_atom_is_sync_free() raises:
    var asm = _compile_code[
        cp_async_atom_kernel, target=get_gpu_target["sm_100a"]()
    ]().asm
    assert_true("cp.async.cg.shared.global" in asm)
    _assert_sync_free(asm, "CpAsync")


def test_cp_async_bulk_atom_is_sync_free() raises:
    var asm = _compile_code[
        cp_async_bulk_atom_kernel, target=get_gpu_target["sm_100a"]()
    ]().asm
    # The copy signals its barrier as part of its one instruction; arming
    # and waiting on the barrier are the building block's.
    assert_true(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes" in asm
    )
    _assert_sync_free(asm, "CpAsyncBulk")


def test_tmem_tileops_are_sync_free() raises:
    var asm = _compile_code[
        tmem_tileops_kernel, target=get_gpu_target["sm_100a"]()
    ]().asm
    assert_true("tcgen05.ld.sync.aligned.32x32b.x16.b32" in asm)
    assert_true("tcgen05.st.sync.aligned.32x32b.x16.b32" in asm)
    _assert_sync_free(asm, "tiled_load and tiled_store over TMEM atoms")


def test_gate_catches_a_building_block_wait() raises:
    var asm = _compile_code[
        tmem_waited_kernel, target=get_gpu_target["sm_100a"]()
    ]().asm
    var found = _sync_issued(asm)
    assert_equal(len(found), 1)
    assert_equal(found[0], SyncInstruction.tcgen05_wait)


def test_gate_catches_a_building_block_commit() raises:
    var asm = _compile_code[
        cp_async_committed_kernel, target=get_gpu_target["sm_100a"]()
    ]().asm
    var found = _sync_issued(asm)
    assert_equal(len(found), 2)
    assert_equal(found[0], SyncInstruction.cp_async_commit_group)
    assert_equal(found[1], SyncInstruction.cp_async_wait_all)


def test_every_sync_instruction_is_scanned() raises:
    assert_equal(
        len(_sync_instructions()),
        ParameterList[SyncInstruction._enum_case_names].length,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
