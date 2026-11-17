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

from layout import TileTensor, row_major
from layout.atoms import (
    INDEX_LANE,
    AddressEncoding,
    BarrierCompletion,
    BatchCompletion,
    CounterCompletion,
    HandleCompletion,
    registers_map,
)
from layout.bit_layout import bits
from layout.owned_tensor import OwnedGPUTensor
from layout.atoms.pointer import PointerLoad, PointerStore
from layout.atoms.tmem import TMemLoad, TMemStore
from std.testing import assert_equal, assert_false, assert_true, TestSuite


def test_round_trip_through_registers() raises:
    var mem = OwnedGPUTensor[.float32, .GENERIC](row_major[1, 4]())
    var regs = OwnedGPUTensor[.float32, .GENERIC](row_major[1, 4]())
    var out = OwnedGPUTensor[.float32, .GENERIC](row_major[1, 4]())
    for i in range(4):
        TileTensor(mem)[0, i] = Float32(i + 1)
        TileTensor(out)[0, i] = 0.0
    PointerLoad[.float32, 4]()(regs, TileTensor(mem))
    PointerStore[.float32, 4]()(out, TileTensor(regs))
    for i in range(4):
        assert_equal(TileTensor(out)[0, i][0], Float32(i + 1))


def test_maps() raises:
    comptime load = PointerLoad[.float32, 4]
    assert_true(load.mem_map.encoding == AddressEncoding.BYTES)
    # Value v is 4 v bytes past the thread's address; lanes are free but
    # keep clear of the 16 bytes one access covers.
    assert_equal(load.mem_map.fixed(UInt64(3)), UInt64(12))
    assert_equal(
        load.mem_map.free(bits(INDEX_LANE, 5) | 3), bits(INDEX_LANE, 5)
    )
    assert_equal(load.mem_map.avoid, UInt64(15))
    comptime regs_match = load.reg_map == registers_map(4)
    assert_true(regs_match)
    comptime store_matches = PointerStore[.float32, 4].mem_map == load.mem_map
    assert_true(store_matches)


def test_shape() raises:
    comptime s = PointerLoad[.bfloat16, 8].shape
    assert_equal(s.length, 1)
    assert_equal(s[0], 8)


def test_completions() raises:
    # Pointer accesses are synchronous: no wait mechanism applies.
    comptime C = PointerLoad[.float32, 4].Completion
    assert_false(conforms_to(C, CounterCompletion))
    assert_false(conforms_to(C, BarrierCompletion))
    assert_false(conforms_to(C, HandleCompletion))
    assert_false(conforms_to(C, BatchCompletion))
    # tcgen05 loads and stores wait on their family's counter, for all of
    # this thread's outstanding issues.
    comptime L = TMemLoad[.float32, 16].Completion
    comptime S = TMemStore[.float32, 16].Completion
    assert_true(conforms_to(L, CounterCompletion))
    assert_true(conforms_to(S, CounterCompletion))
    assert_equal(L.max_pending, 0)
    assert_equal(S.max_pending, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
