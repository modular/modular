# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from memory import bitcast, pack_bits
from testing import TestSuite
from testing import assert_equal


def test_bitcast():
    assert_equal(
        bitcast[DType.int8, 8](SIMD[DType.int16, 4](1, 2, 3, 4)),
        SIMD[DType.int8, 8](1, 0, 2, 0, 3, 0, 4, 0),
    )

    assert_equal(
        bitcast[DType.int32, 1](SIMD[DType.int8, 4](0xFF, 0x00, 0xFF, 0x55)),
        Int32(1442775295),
    )


def test_pack_bits():
    alias b1 = Scalar[DType.bool](True)
    assert_equal(pack_bits(b1).cast[DType.bool](), b1)
    assert_equal(pack_bits(b1).cast[DType.uint8](), UInt8(0b0000_0001))

    alias b2 = SIMD[DType.bool, 2](1, 0)
    assert_equal(pack_bits(b2).cast[DType.uint8](), UInt8(0b0000_0001))

    alias b4 = SIMD[DType.bool, 4](1, 1, 0, 1)
    assert_equal(pack_bits(b4).cast[DType.uint8](), UInt8(0b0000_1011))

    alias b8 = SIMD[DType.bool, 8](1, 1, 1, 0, 1, 0, 1, 0)
    assert_equal(pack_bits(b8), UInt8(0b0101_0111))

    alias b16 = SIMD[DType.bool, 16](
        1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1
    )
    assert_equal(pack_bits(b16), UInt16(0b1000_1010_0101_0111))
    assert_equal(
        pack_bits[DType.uint8, 2](b16),
        SIMD[DType.uint8, 2](0b0101_0111, 0b1000_1010),
    )


def main():
    var suite = TestSuite()

    suite.test[test_bitcast]()
    suite.test[test_pack_bits]()

    suite^.run()
