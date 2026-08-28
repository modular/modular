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

from std.os import (
    MemoryMap,
    page_size,
    Prot,
    MapFlags,
    PROT_READ,
    PROT_WRITE,
    MAP_SHARED,
    MAP_PRIVATE,
    MAP_ANONYMOUS,
)
from std.tempfile import NamedTemporaryFile
from std.testing import TestSuite, assert_equal, assert_true


def test_page_size() raises:
    var ps = page_size()
    assert_true(ps > 0)
    # page size is always a power of two
    assert_equal(ps & (ps - 1), 0)


def test_flags_compose() raises:
    assert_equal((PROT_READ | PROT_WRITE).value, 3)
    assert_equal(
        (MAP_PRIVATE | MAP_ANONYMOUS).value,
        MAP_PRIVATE.value | MAP_ANONYMOUS.value,
    )


def test_anonymous_read_write() raises:
    var n = page_size()
    var m = MemoryMap.anonymous(n)
    assert_equal(len(m), n)
    m.bytes()[0] = 42
    m.bytes()[n - 1] = 99
    assert_equal(Int(m.bytes()[0]), 42)
    assert_equal(Int(m.bytes()[n - 1]), 99)


def test_map_file_readonly() raises:
    var s = String("hello, memory map")
    var content = s.as_bytes()
    var tmp = NamedTemporaryFile("rw")
    tmp.write_bytes(content)

    with open(tmp.name, "r") as f:
        var m = MemoryMap.map(
            f, len(content), prot=PROT_READ, flags=MAP_PRIVATE
        )
        assert_equal(len(m), len(content))
        for i in range(len(content)):
            assert_equal(Int(m.bytes()[i]), Int(content[i]))
    tmp.close()


def test_map_file_offset_shim() raises:
    # A non-page-aligned offset must still land on the exact requested byte.
    var marker = UInt8(0xAB)
    var content = List[UInt8]()
    for i in range(128):
        content.append(marker if i == 64 else UInt8(0))
    var tmp = NamedTemporaryFile("rw")
    tmp.write_bytes(content)

    with open(tmp.name, "r") as f:
        var m = MemoryMap.map(
            f, 1, offset=64, prot=PROT_READ, flags=MAP_PRIVATE
        )
        assert_equal(Int(m.bytes()[0]), Int(marker))
    tmp.close()


def test_map_file_shared_write_flush() raises:
    var tmp = NamedTemporaryFile("rw")
    tmp.write_bytes([UInt8(0), UInt8(0), UInt8(0), UInt8(0)])

    with open(tmp.name, "rw") as f:
        var m = MemoryMap.map(
            f, 4, prot=PROT_READ | PROT_WRITE, flags=MAP_SHARED
        )
        m.bytes()[2] = 0x7F
        m.flush()

    # Re-open and confirm the write reached the file.
    with open(tmp.name, "r") as f2:
        var check = MemoryMap.map(f2, 4, prot=PROT_READ, flags=MAP_PRIVATE)
        assert_equal(Int(check.bytes()[2]), 0x7F)
    tmp.close()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
