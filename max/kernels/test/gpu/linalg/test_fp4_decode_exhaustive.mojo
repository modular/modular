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
"""Exhaustive correctness test for the E2M1 -> float decode.

`cast_uint_to_fp4e2m1` decodes packed FP4 (two E2M1 nibbles per byte). There
are only 256 distinct input bytes, so this checks every one against the
`E2M1_TO_FLOAT32` ground-truth LUT for both the low and high nibble, for
float32 and bfloat16 outputs. Runs on the host (no GPU needed).
"""

from std.testing import assert_equal

from linalg.fp4_utils import (
    cast_uint_to_fp4e2m1,
    decode_fp4e2m1_marlin,
    E2M1_TO_FLOAT32,
    FP4E2M1_MARLIN_BIAS,
)


def _check_all_bytes[out_dtype: DType]() raises:
    # All 256 byte values at once: byte b -> [decode(b & 0xF), decode(b >> 4)].
    var bytes = SIMD[DType.uint8, 256]()
    comptime for b in range(256):
        bytes[b] = UInt8(b)

    var decoded = cast_uint_to_fp4e2m1[out_dtype=out_dtype, out_width=512](
        bytes
    )

    comptime for b in range(256):
        var lo = E2M1_TO_FLOAT32[b & 0x0F].cast[out_dtype]()
        var hi = E2M1_TO_FLOAT32[(b >> 4) & 0x0F].cast[out_dtype]()
        # Element layout is [lo(byte0), hi(byte0), lo(byte1), hi(byte1), ...].
        assert_equal(decoded[b * 2], lo)
        assert_equal(decoded[b * 2 + 1], hi)


def _check_marlin() raises:
    # decode_fp4e2m1_marlin returns 2^-14 of the true magnitude; multiplying by
    # FP4E2M1_MARLIN_BIAS recovers it exactly (every value is a small sum of
    # powers of two, exact in f32). Check all 256 bytes against the LUT.
    var bytes = SIMD[DType.uint8, 256]()
    comptime for b in range(256):
        bytes[b] = UInt8(b)

    var decoded = decode_fp4e2m1_marlin(bytes) * FP4E2M1_MARLIN_BIAS

    comptime for b in range(256):
        assert_equal(decoded[b * 2], E2M1_TO_FLOAT32[b & 0x0F])
        assert_equal(decoded[b * 2 + 1], E2M1_TO_FLOAT32[(b >> 4) & 0x0F])


def main() raises:
    _check_all_bytes[DType.float32]()
    _check_all_bytes[DType.bfloat16]()
    _check_marlin()
    print(
        "E2M1 decode exhaustive: all 256 bytes correct (arithmetic f32/bf16 +"
        " Marlin)"
    )
