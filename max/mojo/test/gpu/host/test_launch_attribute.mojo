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

from std.sys import size_of
from std.testing import assert_equal, TestSuite

from max.gpu.host import Dim
from max.gpu.host import (
    LaunchAttribute,
    LaunchAttributeID,
    LaunchAttributeValue,
)


@inline(.never)
def _dirty_stack() -> Int:
    var junk = Array[UInt8, 4096](fill=0xAB)
    var total = 0
    for i in range(len(junk)):
        total += Int(junk[i])
    return total


@inline(.never)
def _pdl_attribute() -> LaunchAttribute:
    return LaunchAttribute(
        LaunchAttributeID.PROGRAMMATIC_STREAM_SERIALIZATION,
        LaunchAttributeValue(True),
    )


@inline(.never)
def _cluster_attribute() -> LaunchAttribute:
    return LaunchAttribute.from_cluster_dim(Dim(2, 1, 1))


def _bytes_of(attr: LaunchAttribute) -> List[UInt8]:
    var ptr = Pointer(to=attr).unsafe_bitcast[UInt8]()
    var bytes = List[UInt8]()
    for i in range(size_of[LaunchAttribute]()):
        bytes.append(ptr[unsafe_offset=i])
    return bytes^


def _assert_zero_from(bytes: List[UInt8], start: Int) raises:
    for i in range(start, len(bytes)):
        assert_equal(bytes[i], 0, String("byte ", i, " is not zero"))


# The launch-trace verifier hashes every byte of each launch attribute, so
# padding and the unused tail of the value union must be deterministic.
def test_pdl_attribute_bytes_are_deterministic() raises:
    _ = _dirty_stack()
    var bytes = _bytes_of(_pdl_attribute())
    assert_equal(bytes[8], 1)
    _assert_zero_from(bytes, 9)
    for i in range(size_of[LaunchAttributeID](), 8):
        assert_equal(bytes[i], 0)


def test_cluster_attribute_bytes_are_deterministic() raises:
    _ = _dirty_stack()
    var bytes = _bytes_of(_cluster_attribute())
    assert_equal(bytes[8], 2)
    assert_equal(bytes[12], 1)
    assert_equal(bytes[16], 1)
    _assert_zero_from(bytes, 24)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
