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

from std.ffi import ErrNo, get_errno, _null
from std.sys._libc import (
    MAP_ANONYMOUS,
    MAP_PRIVATE,
    MS_SYNC,
    PROT_READ,
    PROT_WRITE,
    mmap,
    msync,
    munmap,
)
from std.testing import TestSuite, assert_equal, assert_true

comptime LENGTH = 4096


def test_anonymous_mapping_round_trip() raises:
    var mapping = mmap(
        _null[NoneType],
        LENGTH,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS,
        -1,
        0,
    )
    assert_true(Int(mapping) != -1)
    var bytes = Span(unsafe_ptr=mapping.unsafe_bitcast[Byte](), length=LENGTH)
    bytes.fill(42)
    for byte in bytes:
        assert_equal(byte, 42)
    assert_equal(msync(mapping, LENGTH, MS_SYNC), 0)
    assert_equal(munmap(mapping, LENGTH), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
