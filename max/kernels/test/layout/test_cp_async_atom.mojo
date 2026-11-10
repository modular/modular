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
"""Host tests for the `cp.async` transfer atoms: their descriptions, and the
PTX they issue when compiled for sm_100a."""

from std.memory import unsafe_stack_allocation
from std.sys.info import _is_sm_100x_or_newer
from std.testing import assert_equal, assert_false, assert_true, TestSuite

from layout import TileTensor, row_major
from layout.atoms import (
    AddressEncoding,
    BarrierCompletion,
    BatchCompletion,
    CounterCompletion,
)
from layout.atoms.b200.cp_async import CpAsyncBulk
from layout.atoms.cp_async import CpAsync
from layout.owned_tensor import OwnedGPUTensor
from layout.tma_async import SharedMemBarrier

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code


def cp_async_kernel(
    src: Pointer[Float32, ImmUntrackedOrigin, address_space=.GLOBAL]
):
    var smem = OwnedGPUTensor[.float32, .SHARED](row_major[1, 4]())
    CpAsync[.float32, 16]()(
        TileTensor(smem), TileTensor(src, row_major[1, 4]())
    )
    CpAsync[.float32, 16].Completion.commit()
    CpAsync[.float32, 16].Completion.wait[1]()
    CpAsync[.float32, 16].Completion.wait()


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
        CpAsyncBulk[.float32, 256, MutUntrackedOrigin](barrier)(
            TileTensor(smem), TileTensor(src, row_major[1, 64]())
        )


def test_cp_async_description() raises:
    comptime A = CpAsync[.float32, 16]
    assert_equal(A.shape[0], 4)
    assert_true(A.src_map.encoding == AddressEncoding.BYTES)
    assert_equal(A.dst_map.fixed(UInt64(3)), UInt64(12))
    assert_equal(A.dst_map.avoid, UInt64(15))
    assert_true(conforms_to(A.Completion, BatchCompletion))
    assert_false(conforms_to(A.Completion, CounterCompletion))
    assert_equal(A.Completion.max_pending, Int.MAX)


def test_cp_async_bulk_description() raises:
    comptime A = CpAsyncBulk[.float32, 256, MutUntrackedOrigin]
    assert_equal(A.shape[0], 64)
    assert_equal(A.src_map.fixed(UInt64(63)), UInt64(252))
    assert_equal(A.src_map.avoid, UInt64(255))
    assert_true(conforms_to(A.Completion, BarrierCompletion))


def test_cp_async_ptx() raises:
    var asm = _compile_code[
        cp_async_kernel, target=get_gpu_target["sm_100a"]()
    ]().asm
    assert_true("cp.async.cg.shared.global" in asm)
    assert_true("cp.async.commit_group;" in asm)
    assert_true("cp.async.wait_group \t1;" in asm)
    assert_true("cp.async.wait_all;" in asm)


def test_cp_async_bulk_ptx() raises:
    var asm = _compile_code[
        cp_async_bulk_kernel, target=get_gpu_target["sm_100a"]()
    ]().asm
    assert_true(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes" in asm
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
