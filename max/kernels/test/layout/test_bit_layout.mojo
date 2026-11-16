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

from layout.atoms import (
    INDEX_LANE,
    INDEX_VALUE,
    INDEX_WARP,
    TMEM_LANE,
    AddressEncoding,
    registers_map,
    vector_map,
)
from layout.bit_layout import BitLayout, bit, bits
from layout.atoms.tmem import TMemLoad, TMemStore, tmem_32x32b_map
from std.testing import assert_equal, assert_false, assert_true, TestSuite


def test_run_is_a_strided_mode() raises:
    # (8 : 4): three domain bits onto codomain bits 2 to 4.
    var m = BitLayout().run(0, 3, 2)
    for x in range(8):
        assert_equal(m(UInt64(x)), UInt64(4 * x))
    assert_equal(m.dom, bits(0, 3))


def test_swizzle_is_linear() raises:
    # A 3-bit XOR swizzle of a 64-element row-major tile: bits 3 to 5 also
    # flip bits 0 to 2.
    var m = BitLayout()
    for i in range(3):
        m = m.put(i, bit(i))
    for i in range(3):
        m = m.put(3 + i, bit(3 + i) | bit(i))
    for x in range(64):
        assert_equal(m(UInt64(x)), UInt64(x ^ ((x >> 3) & 7)))


def test_bits_outside_the_domain_are_ignored() raises:
    var m = BitLayout().run(0, 2, 0)
    assert_equal(m(UInt64(0b111)), UInt64(0b11))
    assert_false(m.has(2))


def test_dead_bits() raises:
    var m = BitLayout().run(0, 2, 0).kill(4)
    assert_true(m.has(4))
    assert_true(m.is_dead(4))
    assert_equal(m.live(), bits(0, 2))
    assert_equal(m(bit(4)), UInt64(0))
    # A later put revives the bit.
    assert_false(m.put(4, bit(7)).is_dead(4))


def test_equality_and_write() raises:
    var m = BitLayout().run(0, 2, 1).kill(5)
    assert_true(m == BitLayout().run(0, 2, 1).kill(5))
    assert_false(m == BitLayout().run(0, 2, 1))
    assert_equal(String(m), "{0: 0x2, 1: 0x4, 5: x}")


def test_tmem_32x32b_map() raises:
    # Thread l of warp w, value v is TMEM lane 32 (w % 4) + l, column v.
    comptime m = tmem_32x32b_map(16)
    assert_true(m.encoding == AddressEncoding.TMEM)
    for warp in range(4):
        for lane in range(32):
            for v in range(16):
                var index = (
                    (UInt64(warp) << UInt64(INDEX_WARP))
                    | (UInt64(lane) << UInt64(INDEX_LANE))
                    | UInt64(v)
                )
                var tmem_lane = UInt64(32 * warp + lane)
                assert_equal(
                    m.fixed(index),
                    (tmem_lane << UInt64(TMEM_LANE)) | UInt64(v),
                )
    assert_equal(m.avoid, bits(TMEM_LANE + 5, 2))
    comptime load_reads_it = TMemLoad[.float32, 16].mem_map == m
    comptime store_writes_it = TMemStore[.float32, 16].mem_map == m
    assert_true(load_reads_it)
    assert_true(store_writes_it)


def test_registers_map_fixes_lanes() raises:
    comptime m = registers_map(16, 32)
    assert_true(m.encoding == AddressEncoding.REGISTERS)
    assert_equal(m.fixed.dom, bits(INDEX_VALUE, 4) | bits(INDEX_LANE, 5))
    assert_equal(m.free(m.fixed.dom), UInt64(0))
    comptime load_writes_it = TMemLoad[.float32, 16].reg_map == m
    assert_true(load_writes_it)


def test_vector_map_frees_lanes() raises:
    comptime m = vector_map(4, 2)
    assert_true(m.encoding == AddressEncoding.BYTES)
    assert_equal(m.fixed(UInt64(3)), UInt64(6))
    assert_equal(m.free(bits(INDEX_LANE, 5) | 3), bits(INDEX_LANE, 5))
    assert_equal(m.avoid, UInt64(7))


def test_large_transfer_keeps_clear_of_lanes() raises:
    # A 4 MiB transfer of bytes needs 22 value bits, below the lane field.
    comptime m = vector_map(1 << 22, 1)
    assert_equal(m.fixed.dom, bits(INDEX_VALUE, 22))
    assert_equal(m.fixed.dom & bits(INDEX_LANE, 5), UInt64(0))
    assert_equal(m.fixed(UInt64((1 << 22) - 1)), UInt64((1 << 22) - 1))


def test_tmem_atom_shape() raises:
    comptime s = TMemLoad[.float32, 32].shape
    assert_equal(s.length, 2)
    assert_equal(s[0], 32)
    assert_equal(s[1], 32)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
