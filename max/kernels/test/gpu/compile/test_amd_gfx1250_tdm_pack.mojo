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
"""Host test for the gfx1250 TDM descriptor bit packing.

The expected words are not transcribed from the Mojo code: a host C++ program
filled AMD's own descriptor structs from
`hip/amd_detail/amd_gfx1250_TDM.h` and printed the words. The header needs a HIP
device context to include normally, so that program stubs `__device__` away and
uses only the header's unions and bitfields.

Two conversions the header does not perform, so the generator does them and this
test therefore pins them:

* Padding fields are stored in dwords, biased by one. The emulator's descriptor
  decoder (rocjitsu) reads `pad_interval = 1 << (field + 1)` dwords and
  `pad_amount = field + 1` dwords, i.e. `pad_interval_bytes` is
  `4 << (field + 1)`; the header gives only the fields' widths, so no public
  source confirms the units.
* Dim 0's stride is the implicit one element, so the row pitch goes in the field
  the header names `tensorDim0Stride` (descriptor bits 160..207), leaving the
  dim-2 stride zero. That is also how the emulator's decoder reads those bits,
  and it labels them the row stride.
"""

from std.memory import bitcast
from std.testing import assert_equal

from max.gpu.tdm import TDMDescriptor


def _assert_words(
    desc: TDMDescriptor,
    d0: SIMD[DType.uint32, 4],
    d1: SIMD[DType.uint32, 8],
    context: String,
) raises:
    assert_equal(
        bitcast[DType.uint32, 4](desc.d0), d0, String(t"{context} group 0")
    )
    assert_equal(
        bitcast[DType.uint32, 8](desc.d1), d1, String(t"{context} group 1")
    )


def test_no_pad() raises:
    """64x64 bf16 box, no padding, 48-bit global address."""
    var desc = TDMDescriptor.make_2d[DType.bfloat16, 64, 64](
        0x123456789ABC00, 0x4000, 1234, 567, 2048
    )
    _assert_words(
        desc,
        SIMD[DType.uint32, 4](0x00000001, 0x00004000, 0x789ABC00, 0x80123456),
        SIMD[DType.uint32, 8](
            0x00010000,
            0x04D20000,
            0x02370000,
            0x00400000,
            0x00000040,
            0x00000800,
            0x00000000,
            0x00000000,
        ),
        "no_pad:",
    )


def test_pad_256_16() raises:
    """Same box with 16 bytes of padding every 256 bytes of dense LDS stream."""
    var desc = TDMDescriptor.make_2d[
        DType.bfloat16, 64, 64, pad_interval_bytes=256, pad_amount_bytes=16
    ](0x123456789ABC00, 0x4000, 1234, 567, 2048)
    _assert_words(
        desc,
        SIMD[DType.uint32, 4](0x00000001, 0x00004000, 0x789ABC00, 0x80123456),
        SIMD[DType.uint32, 8](
            0x07510000,
            0x04D20000,
            0x02370000,
            0x00400000,
            0x00000040,
            0x00000800,
            0x00000000,
            0x00000000,
        ),
        "pad_256_16:",
    )


def test_ragged_extents() raises:
    """Box larger than the tensor: the extents are what hardware clamps."""
    var desc = TDMDescriptor.make_2d[DType.bfloat16, 64, 64](
        0x7FFF0000, 0, 100, 7, 100
    )
    _assert_words(
        desc,
        SIMD[DType.uint32, 4](0x00000001, 0x00000000, 0x7FFF0000, 0x80000000),
        SIMD[DType.uint32, 8](
            0x00010000,
            0x00640000,
            0x00070000,
            0x00400000,
            0x00000040,
            0x00000064,
            0x00000000,
            0x00000000,
        ),
        "ragged:",
    )


def test_multicast_mask() raises:
    """0x1111 as the 16-bit workgroup mask, plus a 33-bit global address."""
    var desc = TDMDescriptor.make_2d[
        DType.bfloat16, 128, 32, multicast_mask=0x1111
    ](0x100000000, 0x8000, 4096, 4096, 4096)
    _assert_words(
        desc,
        SIMD[DType.uint32, 4](0x00000001, 0x00008000, 0x00000000, 0x80000001),
        SIMD[DType.uint32, 8](
            0x00011111,
            0x10000000,
            0x10000000,
            0x00800000,
            0x00000020,
            0x00001000,
            0x00000000,
            0x00000000,
        ),
        "multicast_0x1111:",
    )


def test_f32_pad_128_8() raises:
    """A 4-byte element (data_size code 2) with 8 bytes of padding every 128."""
    var desc = TDMDescriptor.make_2d[
        DType.float32, 32, 16, pad_interval_bytes=128, pad_amount_bytes=8
    ](0xFFFFFFF0, 0x100, 640, 4096, 640)
    _assert_words(
        desc,
        SIMD[DType.uint32, 4](0x00000001, 0x00000100, 0xFFFFFFF0, 0x80000000),
        SIMD[DType.uint32, 8](
            0x03120000,
            0x02800000,
            0x10000000,
            0x00200000,
            0x00000010,
            0x00000280,
            0x00000000,
            0x00000000,
        ),
        "f32_pad_128_8:",
    )


def test_disabled() raises:
    """A count-0 descriptor keeps every other field, including the addresses."""
    var desc = TDMDescriptor.make_2d[DType.bfloat16, 64, 64](
        0x123456789ABC00, 0x4000, 1234, 567, 2048
    )
    _assert_words(
        desc.disabled(),
        SIMD[DType.uint32, 4](0x00000000, 0x00004000, 0x789ABC00, 0x80123456),
        SIMD[DType.uint32, 8](
            0x00010000,
            0x04D20000,
            0x02370000,
            0x00400000,
            0x00000040,
            0x00000800,
            0x00000000,
            0x00000000,
        ),
        "disabled:",
    )


def test_advance() raises:
    """`with_addr` + `with_extents` equals a rebuild at the new origin."""
    var desc = TDMDescriptor.make_2d[DType.bfloat16, 64, 64](
        0x7FFF0000, 0, 100, 7, 100
    )
    var advanced = desc.with_addr(0x7FFF0040, 0).with_extents(68, 6)
    _assert_words(
        advanced,
        SIMD[DType.uint32, 4](0x00000001, 0x00000000, 0x7FFF0040, 0x80000000),
        SIMD[DType.uint32, 8](
            0x00010000,
            0x00440000,
            0x00060000,
            0x00400000,
            0x00000040,
            0x00000064,
            0x00000000,
            0x00000000,
        ),
        "ragged_advanced:",
    )


def test_disabled_survives_addr() raises:
    """Re-addressing a disabled descriptor must not re-arm it."""
    var desc = TDMDescriptor.make_2d[DType.bfloat16, 64, 64](
        0x7FFF0000, 0, 100, 7, 100
    )
    var readdr = desc.disabled().with_addr(0x7FFF0040, 0x40)
    assert_equal(
        bitcast[DType.uint32, 4](readdr.d0)[0] & 0x3,
        UInt32(0),
        "re-addressed disabled descriptor must keep a zero count field",
    )
    assert_equal(
        bitcast[DType.uint32, 4](readdr.d0)[1],
        UInt32(0x40),
        "re-addressed disabled descriptor must still take the new LDS address",
    )


def main() raises:
    test_no_pad()
    test_pad_256_16()
    test_ragged_extents()
    test_multicast_mask()
    test_f32_pad_128_8()
    test_disabled()
    test_advance()
    test_disabled_survives_addr()
    print("tdm descriptor packing: ok")
