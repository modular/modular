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
"""DeepSeek-V4's activation fake-quantizers and Hadamard rotation, per warp.

The reference snaps activations to the grid the model was trained against
(`act_quant` and `fp4_act_quant` in `inference/kernel.py`): FP8 e4m3 in groups
of 64 channels, or FP4 e2m1 in groups of 32 after a Hadamard rotation. A
group's scale is its absolute max over the format's max, floored, and rounded
up to a power of two (UE8M0). Both helpers keep a row spread over the lanes of
one warp, ``vec`` consecutive channels per lane.
"""

from std.bit import log2_floor
from std.math import sqrt

from max.gpu import lane_id
import max.gpu.primitives.warp as warp

from linalg.fp4_utils import cast_fp_to_fp4e2m1

comptime FP8_GROUP = 64
"""Channels per FP8 e4m3 scale group."""
comptime FP4_GROUP = 32
"""Channels per FP4 e2m1 scale group."""

comptime _FP8_MAX: Float32 = 448.0
comptime _FP8_AMAX_FLOOR: Float32 = 1e-4
comptime _FP4_MAX: Float32 = 6.0
comptime _FP4_AMAX_FLOOR = Float32(6.0) * Float32(1.1754943508222875e-38)
"""Six times the smallest normal float32."""


@inline(.always)
def fake_quantize[
    vec: Int, //, group: Int, fp4: Bool
](x: SIMD[.float32, vec]) -> type_of(x):
    """Snaps ``x`` to the FP4 or FP8 grid and back, a scale per group.

    A group of ``group`` channels spans ``group // vec`` consecutive lanes,
    every one of which must call this.

    Parameters:
        vec: Channels per lane (inferred).
        group: Channels per scale group.
        fp4: Snap to FP4 e2m1 rather than FP8 e4m3.

    Args:
        x: This lane's channels.

    Returns:
        The channels on the grid, rescaled.
    """
    comptime value_max = _FP4_MAX if fp4 else _FP8_MAX
    comptime amax_floor = _FP4_AMAX_FLOOR if fp4 else _FP8_AMAX_FLOOR
    # The reference multiplies by the reciprocal rounded to float32.
    comptime inv_max = Float32(1.0 / Float64(value_max))
    var amax = warp.lane_group_max[num_lanes=group // vec](abs(x).reduce_max())
    # The UE8M0 cast rounds up, which is the reference's power-of-two ceiling.
    var scale = (
        (max(amax, amax_floor) * inv_max)
        .cast[.float8_e8m0fnu]()
        .cast[.float32]()
    )
    var q = (x / scale).clamp(-value_max, value_max)
    comptime if fp4:
        q = cast_fp_to_fp4e2m1(q)
    else:
        q = q.cast[.float8_e4m3fn]().cast[.float32]()
    return q * scale


@inline(.always)
def hadamard[vec: Int, //, width: Int](x: SIMD[.float32, vec]) -> type_of(x):
    """``x`` times the orthonormal Sylvester Hadamard matrix of order ``width``.

    The row is spread over one warp, ``vec`` consecutive channels per lane.

    Parameters:
        vec: Channels per lane (inferred).
        width: The row's width.

    Args:
        x: This lane's channels.

    Returns:
        This lane's channels of the rotated row.
    """
    var y = x
    comptime for level in range(log2_floor(width)):
        comptime stride = 1 << level
        comptime if stride < vec:
            var z = y
            comptime for i in range(vec):
                comptime if i & stride == 0:
                    z[i] = y[i] + y[i + stride]
                else:
                    z[i] = y[i - stride] - y[i]
            y = z
        else:
            comptime lanes = stride // vec
            var other = type_of(y)()
            comptime for i in range(vec):
                other[i] = warp.shuffle_xor(y[i], UInt32(lanes))
            y = other - y if (lane_id() & lanes) != 0 else y + other
    comptime scale = Float32(1.0 / sqrt(Float64(width)))
    return y * scale
