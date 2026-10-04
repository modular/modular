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
"""AMD W4A8 grouped GEMM against independent software-dequantized FP32 dots.

No reference MFMA or production decoder is used. Ragged routing, empty experts,
nonidentity IDs, ID -1, zero active experts, dynamic M and untouched output guard
rows are checked. Values exercise signs, both FP4 nibbles and nonunit E8M0 scales
varying by row and 32-element block. Packed expert weights remain packed.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.builtin._closure import __ownership_keepalive
from std.math import ceildiv
from std.memory import bitcast
from std.testing import assert_true
from std.utils.numerics import inf, isinf

from layout import Idx, TileTensor, row_major
from linalg.arch.amd.block_scaled_mma import CDNA4F8F6F4MatrixFormat
from linalg.matmul.gpu.amd.block_scaled_matmul_amd import _launch_block_scaled
from linalg.matmul.gpu.amd.block_scaled_grouped_matmul_amd import (
    block_scaled_grouped_matmul_amd,
    _launch_block_scaled_grouped,
)


def _mix(index: Int) -> UInt32:
    """Deterministic bit mixing avoids cancellation from short periodic fills.
    """
    var value = UInt32(index)
    value ^= value >> 16
    value *= UInt32(0x7FEB352D)
    value ^= value >> 15
    value *= UInt32(0x846CA68B)
    return value ^ (value >> 16)


@inline(.always)
def _fp4(code: UInt8) -> Float32:
    var values = SIMD[.float32, 8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    var value = values[Int(code & 7)]
    return -value if code & 8 else value


@inline(.always)
def _fp8(code: UInt8) -> Float32:
    var exponent = Int((code >> 3) & 15)
    var mantissa = Float32(code & 7)
    var value = mantissa / 512.0 if exponent == 0 else (
        (1.0 + mantissa / 8.0) * bitcast[.float32](UInt32(exponent + 120) << 23)
    )
    return -value if code & 128 else value


@inline(.always)
def _e8m0(code: UInt8) -> Float32:
    # Fixtures use exponent bytes 125..129 (scales .25..4).
    return bitcast[.float32](UInt32(code) << 23)


def _reference[
    N: Int, K: Int
](
    a: ImmPointer[UInt8, ImmutAnyOrigin],
    b: ImmPointer[UInt8, ImmutAnyOrigin],
    sa: ImmPointer[UInt8, ImmutAnyOrigin],
    sb: ImmPointer[UInt8, ImmutAnyOrigin],
    offsets: ImmPointer[UInt32, ImmutAnyOrigin],
    ids: ImmPointer[Int32, ImmutAnyOrigin],
    reference: MutPointer[Float32, MutAnyOrigin],
    magnitude: MutPointer[Float32, MutAnyOrigin],
    rows_dev: Int32,
    active_dev: Int32,
):
    var rows = Int(rows_dev)
    var active = Int(active_dev)
    var m = global_idx.x
    var n = global_idx.y
    if m >= rows or n >= N:
        return
    var expert = -1
    for slot in range(active):
        if m >= Int(offsets[slot]) and m < Int(offsets[slot + 1]):
            expert = Int(ids[slot])
            break
    if expert < 0:
        reference[m * N + n] = inf[.float32]()
        magnitude[m * N + n] = 0.0
        return
    var acc = Float32(0)
    var mag = Float32(0)
    for k in range(K):
        var av = _fp8(a[m * K + k]) * _e8m0(sa[m * (K // 32) + k // 32])
        var packed = b[(expert * N + n) * (K // 2) + k // 2]
        var nibble = (packed >> 4) if k % 2 else (packed & 15)
        var bv = _fp4(nibble) * _e8m0(
            sb[(expert * N + n) * (K // 32) + k // 32]
        )
        acc += av * bv
        mag += abs(av * bv)
    reference[m * N + n] = acc
    magnitude[m * N + n] = mag


def _check[
    N: Int,
    K: Int,
    out_dtype: DType = DType.bfloat16,
    forced_bk: Int = 0,
    dense_mma_32: Bool = False,
](
    ctx: DeviceContext, counts: List[Int], ids: List[Int], active: Int = -1
) raises:
    comptime experts = 5
    var rows = 0
    var max_rows = 0
    for count in counts:
        rows += count
        max_rows = max(max_rows, count)
    var num_active = len(counts) if active < 0 else active
    # Keep valid allocations even for zero-row/zero-active launches.
    var storage_rows = max(rows, 1)
    var ah = ctx.enqueue_create_host_buffer[.uint8](storage_rows * K)
    var bh = ctx.enqueue_create_host_buffer[.uint8](experts * N * (K // 2))
    var sah = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        storage_rows * (K // 32)
    )
    var sbh = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        experts * N * (K // 32)
    )
    var oh = ctx.enqueue_create_host_buffer[.uint32](len(counts) + 1)
    var ih = ctx.enqueue_create_host_buffer[.int32](len(counts))
    var values = SIMD[.uint8, 8](0x30, 0x38, 0x3C, 0x40, 0x48, 0xB0, 0xBC, 0xC0)
    for i in range(storage_rows * K):
        ah[i] = values[Int(_mix(i + 17) & 7)]
    for i in range(experts * N * (K // 2)):
        bh[i] = UInt8(_mix(i + 71) & 255)
    for i in range(storage_rows * (K // 32)):
        sah[i] = bitcast[.float8_e8m0fnu](UInt8(125 + Int(_mix(i + 101) % 5)))
    for i in range(experts * N * (K // 32)):
        sbh[i] = bitcast[.float8_e8m0fnu](UInt8(125 + Int(_mix(i + 157) % 5)))
    oh[0] = 0
    for i in range(len(counts)):
        oh[i + 1] = oh[i] + UInt32(counts[i])
        ih[i] = Int32(ids[i])
    var a = ctx.enqueue_create_buffer[.uint8](storage_rows * K)
    var b = ctx.enqueue_create_buffer[.uint8](experts * N * (K // 2))
    var sa = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        storage_rows * (K // 32)
    )
    var sb = ctx.enqueue_create_buffer[.float8_e8m0fnu](experts * N * (K // 32))
    var offsets = ctx.enqueue_create_buffer[.uint32](len(counts) + 1)
    var expert_ids = ctx.enqueue_create_buffer[.int32](len(counts))
    var c = ctx.enqueue_create_buffer[out_dtype]((storage_rows + 2) * N)
    var reference = ctx.enqueue_create_buffer[.float32](storage_rows * N)
    var mag = ctx.enqueue_create_buffer[.float32](storage_rows * N)
    ctx.enqueue_copy(a, ah)
    ctx.enqueue_copy(b, bh)
    ctx.enqueue_copy(sa, sah)
    ctx.enqueue_copy(sb, sbh)
    ctx.enqueue_copy(offsets, oh)
    ctx.enqueue_copy(expert_ids, ih)
    var sentinel = inf[out_dtype]()
    c.enqueue_fill(sentinel)
    var at = TileTensor(a, row_major((rows, Idx[K]))).as_imm()
    var bt = TileTensor(b, row_major[experts, N, K // 2]()).as_imm()
    var sat = TileTensor(sa, row_major((rows, Idx[K // 32]))).as_imm()
    var sbt = TileTensor(sb, row_major[experts, N, K // 32]()).as_imm()
    var ot = TileTensor(offsets, row_major(len(counts) + 1)).as_imm()
    var it = TileTensor(expert_ids, row_major(len(counts))).as_imm()
    var ct = TileTensor(c, row_major((rows, Idx[N])))
    comptime fp8 = CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3
    comptime fp4 = CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1
    comptime if dense_mma_32:
        assert_true(len(counts) == 1 and ids[0] == 0)
        var bd = TileTensor(b.unsafe_ptr(), row_major[N, K // 2]()).as_imm()
        var sbd = TileTensor(sb.unsafe_ptr(), row_major[N, K // 32]()).as_imm()
        _launch_block_scaled[
            BM=128,
            BN=128,
            BK_ELEMS=128,
            WM=64,
            WN=64,
            MMA_M=32,
            MMA_N=32,
            MMA_K=64,
            matrix_format=fp8,
            b_matrix_format=fp4,
        ](ct, at, bd, sat, sbd, rows, ctx)
    elif forced_bk:
        _launch_block_scaled_grouped[
            BM=64,
            BN=128,
            BK_ELEMS=forced_bk,
            WM=64,
            WN=64,
            matrix_format=fp8,
            b_matrix_format=fp4,
        ](ct, at, bt, sat, sbt, ot, it, max_rows, num_active, ctx)
    else:
        block_scaled_grouped_matmul_amd[matrix_format=fp8, b_matrix_format=fp4](
            ct, at, bt, sat, sbt, ot, it, max_rows, num_active, ctx
        )
    if rows > 0:
        ctx.enqueue_function[_reference[N, K]](
            a.unsafe_ptr().as_imm().unsafe_origin_cast[ImmutAnyOrigin](),
            b.unsafe_ptr().as_imm().unsafe_origin_cast[ImmutAnyOrigin](),
            sa.unsafe_ptr()
            .bitcast[UInt8]()
            .as_imm()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            sb.unsafe_ptr()
            .bitcast[UInt8]()
            .as_imm()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            offsets.unsafe_ptr().as_imm().unsafe_origin_cast[ImmutAnyOrigin](),
            expert_ids.unsafe_ptr()
            .as_imm()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            reference.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            mag.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            Int32(rows),
            Int32(num_active),
            grid_dim=(ceildiv(rows, 16), ceildiv(N, 16)),
            block_dim=(16, 16),
        )
    var ch = ctx.enqueue_create_host_buffer[out_dtype]((storage_rows + 2) * N)
    var rh = ctx.enqueue_create_host_buffer[.float32](storage_rows * N)
    var mh = ctx.enqueue_create_host_buffer[.float32](storage_rows * N)
    ctx.enqueue_copy(ch, c)
    ctx.enqueue_copy(rh, reference)
    ctx.enqueue_copy(mh, mag)
    ctx.synchronize()
    var max_error = Float32(0)
    var max_reference = Float32(0)
    var nonzero_reference = 0
    for i in range(rows * N):
        var got = ch[i].cast[.float32]()
        var want = rh[i]
        if isinf(want):
            assert_true(got == want, "rows without an active expert changed")
            continue
        if mh[i] > 0 and abs(want) > 0.1:
            nonzero_reference += 1
        max_reference = max(max_reference, abs(want))
        var error = abs(got - want)
        var tolerance = Float32(6e-5) * mh[i] + Float32(1e-5)
        comptime if out_dtype == DType.bfloat16:
            tolerance += Float32(0.004) * abs(want)
        assert_true(
            error <= tolerance,
            "mixed grouped GEMM must match independent FP32 dot",
        )
        max_error = max(max_error, error)
    if rows > 0 and num_active > 0:
        assert_true(
            nonzero_reference > 0, "reference must contain nonzero dots"
        )
    for i in range(rows * N, (storage_rows + 2) * N):
        assert_true(ch[i] == sentinel, "guard rows changed")
    __ownership_keepalive(a, b, sa, sb, offsets, expert_ids, c, reference, mag)
    print(
        "W4A8 N=",
        N,
        " K=",
        K,
        " rows=",
        rows,
        " active=",
        num_active,
        " forced_bk=",
        forced_bk,
        " dense_mma_32=",
        dense_mma_32,
        " max_abs_error=",
        max_error,
        " max_abs_reference=",
        max_reference,
        " nonzero_reference=",
        nonzero_reference,
        " PASS",
    )


def main() raises:
    with DeviceContext() as ctx:
        _check[128, 128, DType.float32](
            ctx, [1, 31, 0, 64, 65, 129], [3, 1, 4, 0, -1, 2]
        )
        _check[128, 384](ctx, [31, 0, 65], [4, 2, 1])
        _check[128, 384, DType.float32, 0, True](ctx, [65], [0])
        _check[768, 3584](ctx, [1, 31, 64, 0], [3, 1, 0, 4])
        _check[768, 3584, DType.float32, 128](ctx, [1, 31], [3, 1])
        _check[3584, 384](ctx, [1, 31, 0], [4, 2, 1])
        _check[128, 128](ctx, [31, 0], [4, 2], 0)
        _check[128, 128](ctx, [0, 0], [4, 2], 0)
