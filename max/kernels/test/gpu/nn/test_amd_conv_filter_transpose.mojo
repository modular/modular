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
"""Exact check of the AMD conv RSCF -> K-padded FRSC filter repack.

The repack moves one 16-byte K-group per thread. Rows cover the whole-group
path (`R*S*C` a multiple of the group width) and the partial-group path (FP8,
where a group is 16 elements and `R*S*C` is only a multiple of 8), each with a
zero-filled K tail.
"""

from max.gpu.host import DeviceContext
from std.math import align_up, ceildiv
from std.sys import size_of

from layout import Idx, TileTensor, row_major

from nn.conv.gpu.amd.dispatch import _transpose_rscf_to_frsc_kpad


def _run[
    dtype: DType, R: Int, S: Int, C: Int, F: Int, bk: Int
](ctx: DeviceContext) raises:
    comptime K_real = R * S * C
    comptime K_padded = align_up(K_real, 2 * bk)
    comptime group = 16 // size_of[dtype]()
    var src_host = ctx.enqueue_create_host_buffer[dtype](K_real * F)
    for i in range(K_real * F):
        src_host[i] = Scalar[dtype](Float32((i * 7 + i // F) % 13 + 1))
    var src = ctx.enqueue_create_buffer[dtype](K_real * F)
    var dst = ctx.enqueue_create_buffer[dtype](F * K_padded)
    ctx.enqueue_copy(src, src_host)
    # A non-zero fill proves the padded tail is written, not just left alone.
    ctx.enqueue_memset(dst, Scalar[dtype](Float32(5)))

    var src_tt = TileTensor(src, row_major(Idx[R], Idx[S], Idx[C], Idx[F]))
    comptime kernel = _transpose_rscf_to_frsc_kpad[
        dtype,
        src_tt.LayoutType,
        ImmOrigin(src_tt.origin),
        src_tt.Engine,
        R,
        S,
        C,
        F,
        K_padded,
    ]
    ctx.enqueue_function[kernel](
        src_tt.as_imm(),
        dst.unsafe_ptr(),
        grid_dim=ceildiv(F * (K_padded // group), 256),
        block_dim=256,
    )
    var dst_host = ctx.enqueue_create_host_buffer[dtype](F * K_padded)
    ctx.enqueue_copy(dst_host, dst)
    ctx.synchronize()

    for f in range(F):
        for k in range(K_padded):
            var expected = Float32(0)
            if k < K_real:
                expected = Float32(src_host[k * F + f])
            var actual = Float32(dst_host[f * K_padded + k])
            if actual != expected:
                print(t"mismatch f={f} k={k} {actual} != {expected}")
                raise Error("filter repack mismatch")
    print(t"PASS {dtype} {R} {S} {C} {F} K_padded={K_padded}")


def main() raises:
    with DeviceContext() as ctx:
        # The WAN 96 -> 96 filter: K = 864, padded to 896.
        _run[.bfloat16, 3, 3, 96, 96, 64](ctx)
        _run[.float16, 3, 3, 96, 96, 64](ctx)
        _run[.bfloat16, 3, 3, 192, 96, 64](ctx)
        # Output channels that do not fill a thread block.
        _run[.bfloat16, 3, 3, 40, 24, 64](ctx)
        # FP8: 3 * 3 * 24 = 216 is not a multiple of the 16-element group.
        _run[.float8_e4m3fn, 3, 3, 24, 40, 128](ctx)
        _run[.float8_e4m3fn, 1, 1, 256, 128, 128](ctx)
